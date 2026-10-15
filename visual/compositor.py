#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import cv2
import numpy as np
from glob import glob

import subprocess

USE_FFMPEG_WRITER = os.environ.get("INPAINT_USE_FFMPEG", "0") == "1"


class _OpenCVWriter:
    def __init__(self, out_path: str, w: int, h: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not self.writer.isOpened():
            raise RuntimeError(f"failed to open VideoWriter: {out_path}")

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()


class _FFmpegWriter:
    def __init__(self, proc):
        self.proc = proc

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def _start_ffmpeg_writer(out_path: str, w: int, h: int, fps: float):
    # H.264 + yuv420p + faststart → browser/VS Code friendly
    return subprocess.Popen([
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", f"{fps:.6f}",
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ], stdin=subprocess.PIPE)


def _start_video_writer(out_path: str, w: int, h: int, fps: float):
    if USE_FFMPEG_WRITER:
        return _FFmpegWriter(_start_ffmpeg_writer(out_path, w, h, fps))
    return _OpenCVWriter(out_path, w, h, fps)

# ROOT_SEG = "/mnt/data2/dexmimic/datasets/egoengine/Aria/data/Segmentation"
ROOT_SEG = "/mnt/data2/dexmimic/workspace/egoengine/rl_tracking_dev"
ROOT_REM = "/mnt/data2/dexmimic/datasets/egoengine/Aria/data/Removed"

BG_CANDIDATES = [
    "removed_w_mask_5.mp4",  # preferred
    "mask_5.mp4",
    "w_mask_5.mp4",
]

MASK_VIDEO_CANDIDATES = [
    "mask_bin.mp4",  # preferred (binary)
    "mask.mp4",
]

def open_cap(path):
    cap = cv2.VideoCapture(path)
    return cap if cap.isOpened() else None

class VideoOrFolderMask:
    """
    Reads mask frames from a video file or a directory of numbered images (e.g., mask_15/000000.jpg).
    Output frames are single-channel 8-bit (0..255).
    """
    def __init__(self, video_path=None, folder_path=None):
        self.cap = None
        self.folder = None
        self.files = None
        self.idx = 0
        if video_path and os.path.isfile(video_path):
            cap = open_cap(video_path)
            if cap:
                self.cap = cap
                return
        if folder_path and os.path.isdir(folder_path):
            # Accept mask_*/ folders, sort numerically by filename
            self.folder = folder_path
            imgs = glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.png"))
            def k(s):
                m = re.search(r"(\d+)", os.path.basename(s))
                return int(m.group(1)) if m else int(1e12)
            self.files = sorted(imgs, key=k)

    def read(self):
        if self.cap:
            ok, frame = self.cap.read()
            if not ok:
                return False, None
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # binarize if not already
            _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
            return True, frame
        if self.files is not None:
            if self.idx >= len(self.files):
                return False, None
            img = cv2.imread(self.files[self.idx], cv2.IMREAD_GRAYSCALE)
            self.idx += 1
            if img is None:
                return False, None
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            return True, img
        return False, None

    def release(self):
        if self.cap:
            self.cap.release()

def find_background_video(rem_ep_dir):
    for name in BG_CANDIDATES:
        p = os.path.join(rem_ep_dir, name)
        if os.path.isfile(p):
            return p
    return None

def find_mask_source(seg_ep_dir, rem_ep_dir):
    # Prefer mask video in Segmentation/Egocentric_* dir
    for name in MASK_VIDEO_CANDIDATES:
        p = os.path.join(seg_ep_dir, p := name)
        if os.path.isfile(p):
            return VideoOrFolderMask(video_path=p)
    # Fall back to a mask_* folder in Removed_Videos (e.g., mask_15/)
    mask_folders = sorted(
        [d for d in os.listdir(rem_ep_dir) if d.startswith("mask_") and os.path.isdir(os.path.join(rem_ep_dir, d))]
    )
    for d in mask_folders:
        p = os.path.join(rem_ep_dir, d)
        if os.path.isdir(p):
            return VideoOrFolderMask(folder_path=p)
    return None

def _color_brightness_match(src_bgr, bg_bgr, mask):
    """
    Minimal: gentle per-frame attenuation to avoid over-saturation in the pasted region.
    """
    src = src_bgr.astype(np.float32)
    src *= 0.78
    return np.clip(src, 0, 255).astype(np.uint8)

def match(im):
    # very gentle channel balance without changing your look
    imf = im.astype(np.float32)
    p = 6
    eps = 1e-6
    m = np.power(np.power(imf, p).mean(axis=(0, 1)) + eps, 1.0 / p)  # per-channel “gray” level
    scale = (m.mean() / (m + eps))
    im_corr = np.clip(imf * scale, 0, 255)
    im = np.clip(im_corr, 0, 255).astype(np.uint8)
    return im

def composite_episode(seg_ep_dir, rem_ep_dir, out_name="inpainted_video.mp4", clone_mode="mixed",
                      src_name=None, mask_name=None, bg_name=None, out_dir=None, start_frame=0,
                      edge_blur=0, edge_erode=0, rim_noise=0.0, rim_blend=0.0, fg_noise=0.0):
    """
    Minimal change: allow explicit file names (src/mask/bg) if provided.
    Otherwise preserve the original logic (video.mp4 + discovered mask + discovered bg).
    """

    # --- Resolve source (src) ---
    if src_name:
        src_video = os.path.join(seg_ep_dir, src_name)
        if not os.path.isfile(src_video):
            print(f"  [skip] no src video: {src_video}")
            return False
    else:
        # original behavior: Segmentation/<episode>/video.mp4
        src_video = os.path.join(seg_ep_dir, "video.mp4")
        if not os.path.isfile(src_video):
            print(f"  [skip] no src video: {src_video}")
            return False

    # --- Resolve background (bg) ---
    if bg_name:
        bg_video = os.path.join(rem_ep_dir, bg_name)
        if not os.path.isfile(bg_video):
            print(f"  [skip] no background video: {bg_video}")
            return False
    else:
        bg_video = find_background_video(rem_ep_dir)
        if not bg_video:
            print(f"  [skip] no background video in {rem_ep_dir}")
            return False

    # --- Resolve mask source ---
    if mask_name:
        mask_src = VideoOrFolderMask(video_path=os.path.join(seg_ep_dir, mask_name))
        # If mask video missing, fail gracefully
        if mask_src.cap is None and mask_src.files is None:
            print(f"  [skip] no mask file: {os.path.join(seg_ep_dir, mask_name)}")
            return False
    else:
        mask_src = find_mask_source(seg_ep_dir, rem_ep_dir)
        if not mask_src:
            print(f"  [skip] no mask source for {seg_ep_dir}")
            return False

    cap_src = open_cap(src_video)
    cap_bg  = open_cap(bg_video)
    if cap_src is None or cap_bg is None:
        print("  [skip] failed to open one of the videos")
        if cap_src: cap_src.release()
        if cap_bg:  cap_bg.release()
        mask_src.release()
        return False

    # Output writer based on background size/fps
    fps = cap_bg.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap_bg.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    h   = int(cap_bg.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w == 0 or h == 0:
        ok_bg, frame_bg = cap_bg.read()
        if not ok_bg:
            print("  [skip] cannot read first bg frame")
            cap_src.release(); cap_bg.release(); mask_src.release()
            return False
        h, w = frame_bg.shape[:2]
        cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    start_frame = max(0, int(start_frame))
    if start_frame > 0:
        cap_bg.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_path = os.path.join(out_dir if out_dir else rem_ep_dir, out_name)
    writer = _start_video_writer(out_path, w, h, fps)


    mode_map = {
        "normal": cv2.NORMAL_CLONE,
        "mixed":  cv2.MIXED_CLONE,
        "monochrome": cv2.MONOCHROME_TRANSFER,
    }
    cv_mode = mode_map.get(clone_mode.lower(), cv2.MIXED_CLONE)

    frames = 0
    while True:
        ok_bg,  frame_bg  = cap_bg.read()
        ok_src, frame_src = cap_src.read()
        ok_m,   frame_m   = mask_src.read()
        if not (ok_bg and ok_src and ok_m):
            break

        # Resize src & mask to bg size
        if (frame_src.shape[1], frame_src.shape[0]) != (w, h):
            frame_src = cv2.resize(frame_src, (w, h), interpolation=cv2.INTER_LINEAR)
        if (frame_m.shape[1], frame_m.shape[0]) != (w, h):
            frame_m = cv2.resize(frame_m, (w, h), interpolation=cv2.INTER_NEAREST)

        # Single-channel uint8 mask (0/255)
        if frame_m.ndim == 3:
            frame_m = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(frame_m, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        if cv2.countNonZero(mask) == 0:
            writer.write(frame_bg)
            frames += 1
            continue

        # Compute clone center from mask centroid (fallback: image center)
        M = cv2.moments(mask, binaryImage=True)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = w // 2, h // 2
        center = (cx, cy)

        # Ensure 3 channels
        if frame_src.ndim == 2: frame_src = cv2.cvtColor(frame_src, cv2.COLOR_GRAY2BGR)
        if frame_bg.ndim  == 2: frame_bg  = cv2.cvtColor(frame_bg,  cv2.COLOR_GRAY2BGR)

        src_matched = frame_src

        # try:
            # blended = cv2.seamlessClone(src_matched, frame_bg, mask, center, cv_mode)
        # except cv2.error:
            # Safe fallback: hard composite            
        mask_work = mask.copy()
        edge_erode = max(0, int(edge_erode))
        if edge_erode > 0:
            kernel = np.ones((edge_erode * 2 + 1, edge_erode * 2 + 1), dtype=np.uint8)
            mask_work = cv2.erode(mask_work, kernel, iterations=1)

        m = mask_work.astype(np.float32) / 255.0
        edge_blur = max(0, int(edge_blur))
        if edge_blur > 0:
            if edge_blur % 2 == 0:
                edge_blur += 1
            m = cv2.GaussianBlur(m, (edge_blur, edge_blur), 0)
            m = np.clip(m, 0.0, 1.0)
        rim = np.clip(mask.astype(np.float32) / 255.0 - m, 0.0, 1.0)
        m = m[..., None]
        rim = rim[..., None]

        if rim_blend > 0.0:
            rim_blend = float(np.clip(rim_blend, 0.0, 1.0))
            src_f = src_matched.astype(np.float32)
            bg_f = frame_bg.astype(np.float32)
            src_f = src_f * (1.0 - rim * rim_blend) + bg_f * (rim * rim_blend)
        else:
            src_f = src_matched.astype(np.float32)
            bg_f = frame_bg.astype(np.float32)

        if rim_noise > 0.0:
            noise = np.random.normal(0.0, float(rim_noise), size=frame_bg.shape).astype(np.float32)
            src_f = np.clip(src_f + noise * rim, 0.0, 255.0)

        if fg_noise > 0.0:
            # Fine-grain monochrome Gaussian noise looks more like sensor grain and
            # avoids large colored blotches after compositing / video encoding.
            fg_sigma = float(fg_noise)
            grain = np.random.normal(0.0, fg_sigma, size=frame_bg.shape[:2]).astype(np.float32)
            src_f = np.clip(src_f + grain[..., None] * m, 0.0, 255.0)

        blended = (src_f * m + bg_f * (1.0 - m)).astype(np.uint8)

        writer.write(blended)
        frames += 1

    writer.close()
    cap_src.release()
    cap_bg.release()
    mask_src.release()

    if frames == 0:
        print("  [warn] produced 0 frames → removing empty output")
        try: os.remove(out_path)
        except OSError: pass
        return False

    print(f"  [OK] wrote {frames} frames → {out_path}")
    return True

def main():
    total = 0
    done = 0
    for tuple_name in os.listdir(ROOT_SEG):
        seg_tuple_dir = os.path.join(ROOT_SEG, tuple_name)
        if not os.path.isdir(seg_tuple_dir):
            continue

        for episode_name in os.listdir(seg_tuple_dir):
            seg_ep_dir = os.path.join(seg_tuple_dir, episode_name)
            if not os.path.isdir(seg_ep_dir):
                continue

            rem_ep_dir = os.path.join(ROOT_REM, tuple_name, episode_name)
            if not os.path.isdir(rem_ep_dir):
                # Only process when both sides exist
                continue

            total += 1
            print(f"\n>> {tuple_name}/{episode_name}")
            ok = composite_episode(seg_ep_dir, rem_ep_dir, out_name="inpainted_video.mp4")
            if ok:
                done += 1

    print(f"\nSummary: processed={total}, succeeded={done}, failed={total - done}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Backward compat: --seg-dir still works; you can also use --rgb-dir
    parser.add_argument("--seg-dir", type=str, help="Path to Segmentation/Egocentric episode dir")
    parser.add_argument("--rgb-dir", type=str, help="Alias for --seg-dir (use Egocentric_RGB_Videos/<id>/1)")
    parser.add_argument("--rem-dir", type=str, help="Path to Removed episode dir")
    # New optional overrides
    parser.add_argument("--src-name",  type=str, default=None, help="Source video filename (e.g., mask.mp4)")
    parser.add_argument("--mask-name", type=str, default=None, help="Mask video filename (e.g., mask_bin.mp4)")
    parser.add_argument("--bg-name",   type=str, default=None, help="Background video filename (e.g., removed_w_mask_5.mp4)")
    parser.add_argument("--out-name",  type=str, default="inpainted_video.mp4")
    parser.add_argument("--out-dir",   type=str, default=None, help="Output directory (defaults to rem-dir)")
    parser.add_argument("--clone-mode",type=str, default="mixed", choices=["normal","mixed","monochrome"])
    parser.add_argument("--edge-blur", type=int, default=0, help="Optional Gaussian blur kernel size for softening the blend boundary")
    parser.add_argument("--edge-erode", type=int, default=0, help="Optional inward erosion radius for shrinking the pasted foreground a little before feathering")
    parser.add_argument("--rim-noise", type=float, default=0.0, help="Optional Gaussian noise sigma applied only on the rim region")
    parser.add_argument("--rim-blend", type=float, default=0.0, help="Optional amount of extra background mixing on the rim only")
    parser.add_argument("--fg-noise", type=float, default=0.0, help="Optional fine monochrome Gaussian grain sigma applied on the foreground")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Background frame index to start from. Use this to align blending with a retarget/replay sequence trimmed at the front.",
    )

    args = parser.parse_args()

    seg_dir = args.rgb_dir if args.rgb_dir else args.seg_dir

    if seg_dir and args.rem_dir:
        composite_episode(
            seg_ep_dir=seg_dir,
            rem_ep_dir=args.rem_dir,
            out_name=args.out_name,
            clone_mode=args.clone_mode,
            src_name=args.src_name,
            mask_name=args.mask_name,
            bg_name=args.bg_name,
            out_dir=args.out_dir,
            start_frame=args.start_frame,
            edge_blur=args.edge_blur,
            edge_erode=args.edge_erode,
            rim_noise=args.rim_noise,
            rim_blend=args.rim_blend,
            fg_noise=args.fg_noise,
        )
    else:
        main()

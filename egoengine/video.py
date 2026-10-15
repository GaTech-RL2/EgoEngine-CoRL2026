"""Strict frame-preserving video I/O used by reproducible visual stages."""

import os
import tempfile
from pathlib import Path

import cv2


def video_info(path, decode=False):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot decode video: {path}")
    info = dict(
        width=int(cap.get(3)), height=int(cap.get(4)), fps=float(cap.get(5)), frames=int(cap.get(7))
    )
    if decode:
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        if count != info["frames"]:
            raise ValueError(f"Truncated video: {path}")
    cap.release()
    return info


def writer(path, width, height, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not w.isOpened():
        raise RuntimeError(f"Cannot create video: {path}")
    return w


def retime(source, output, fps, expected_frames, size=None):
    info = video_info(source)
    if info["frames"] != expected_frames:
        raise ValueError("Compositor lost or duplicated frames")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.stem + "_", suffix=".mp4", dir=output.parent)
    os.close(fd)
    cap = cv2.VideoCapture(str(source))
    width, height = size if size is not None else (info["width"], info["height"])
    out = writer(temporary, width, height, fps)
    try:
        for _ in range(expected_frames):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Truncated compositor output")
            if frame.shape[1::-1] != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            out.write(frame)
        out.release()
        Path(temporary).replace(output)
    finally:
        cap.release()
        out.release()
        Path(temporary).unlink(missing_ok=True)


def read_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Missing frame {index} in {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

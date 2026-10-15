import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation


def _assert_T(T: np.ndarray, name: str):
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} last row must be [0 0 0 1], got {T[3]}")


def _draw_triad(ax, T: np.ndarray, axis_len: float = 0.2, label: str | None = None):
    o = T[:3, 3]
    R = T[:3, :3]
    X = o + R[:, 0] * axis_len
    Y = o + R[:, 1] * axis_len
    Z = o + R[:, 2] * axis_len
    ax.plot([o[0], X[0]],[o[1], X[1]],[o[2], X[2]], c='red',   lw=2)
    ax.plot([o[0], Y[0]],[o[1], Y[1]],[o[2], Y[2]], c='green', lw=2)
    ax.plot([o[0], Z[0]],[o[1], Z[1]],[o[2], Z[2]], c='blue',  lw=2)
    if label:
        ax.text(o[0], o[1], o[2], label, color='k')


def _draw_triad_lines(ax, T: np.ndarray, axis_len: float = 0.2):
    o = T[:3, 3]
    R = T[:3, :3]
    X = o + R[:, 0] * axis_len
    Y = o + R[:, 1] * axis_len
    Z = o + R[:, 2] * axis_len
    lx, = ax.plot([o[0], X[0]],[o[1], X[1]],[o[2], X[2]], c='red',   lw=2)
    ly, = ax.plot([o[0], Y[0]],[o[1], Y[1]],[o[2], Y[2]], c='green', lw=2)
    lz, = ax.plot([o[0], Z[0]],[o[1], Z[1]],[o[2], Z[2]], c='blue',  lw=2)
    return lx, ly, lz


def _update_triad_lines(lines, T: np.ndarray, axis_len: float = 0.2):
    o = T[:3, 3]
    R = T[:3, :3]
    X = o + R[:, 0] * axis_len
    Y = o + R[:, 1] * axis_len
    Z = o + R[:, 2] * axis_len
    lines[0].set_data([o[0], X[0]],[o[1], X[1]]); lines[0].set_3d_properties([o[2], X[2]])
    lines[1].set_data([o[0], Y[0]],[o[1], Y[1]]); lines[1].set_3d_properties([o[2], Y[2]])
    lines[2].set_data([o[0], Z[0]],[o[1], Z[1]]); lines[2].set_3d_properties([o[2], Z[2]])


def _set_equal_3d(ax, P: np.ndarray, pad=0.1):
    mn = np.nanmin(P, axis=0)
    mx = np.nanmax(P, axis=0)
    ctr = (mn + mx) / 2.0
    rng = (mx - mn)
    r = float(np.max(rng) * (1.0 + pad)) if np.isfinite(rng).all() and np.max(rng) > 0 else 1.0
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass


def _align_cam2w_with_tag0(cam2w_raw: np.ndarray, T_world_tag: np.ndarray, T_tag_cam0: np.ndarray) -> np.ndarray:
    """Align a cam2world trajectory so that frame 0 matches tag-anchored world."""
    assert cam2w_raw.ndim == 3 and cam2w_raw.shape[1:] == (4,4)
    T_w_c_from_tag0 = T_world_tag @ T_tag_cam0
    H = T_w_c_from_tag0 @ np.linalg.inv(cam2w_raw[0])
    cam2w_aligned = np.einsum('ij,tjk->tik', H, cam2w_raw)
    return cam2w_aligned


def main():
    ap = argparse.ArgumentParser(description="Visualize camera and tag poses; align trajectory to world with tag@t0.")
    ap.add_argument("--tag_ref", default=str(Path("tag1_ref.txt")),
                    help="Path to 4x4 tag pose in world (T_world_tag)")
    ap.add_argument("--cam_in_tag", default=str(Path("outputs/tag1_T_cam_in_tag.txt")),
                    help="Path to 4x4 camera pose in tag frame for first frame (T_tag_cam)")
    ap.add_argument("--out", default=str(Path("outputs/pose_3d_in_world_first_frame.png")),
                    help="Output PNG path of first-frame pose")
    ap.add_argument("--axis_len", type=float, default=0.15, help="Axis length for triads")
    # Extrinsics sequence (uncalibrated world)
    ap.add_argument("--extrinsics", default="Egocentric_Camera_Parameters/201/1/egocentric_frame_extrinsic.npy",
                    help="Path to per-frame extrinsics (.npy) either cam2w or w2c. If omitted, tries outputs/*cam2w.npy or *w2c.npy")
    ap.add_argument("--extrinsics_is_w2c", action="store_true",
                    help="If set, treat --extrinsics as world→camera (invert to cam→world)")
    ap.add_argument("--gif_out", default=str(Path("outputs/camera_traj_world.gif")),
                    help="Output GIF path for camera trajectory in world")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    # Load world tag pose and first-frame camera-in-tag
    T_world_tag = np.loadtxt(args.tag_ref).astype(float)
    T_tag_cam = np.loadtxt(args.cam_in_tag).astype(float)
    _assert_T(T_world_tag, "T_world_tag")
    _assert_T(T_tag_cam, "T_tag_cam (camera pose in tag frame)")

    # First-frame camera pose in world
    T_world_cam0 = T_world_tag @ T_tag_cam

    # Static first-frame figure
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("First-Frame Poses in World Coordinates")
    ax.set_xlabel('Xw'); ax.set_ylabel('Yw'); ax.set_zlabel('Zw')
    _draw_triad(ax, np.eye(4), axis_len=args.axis_len, label='world')
    _draw_triad(ax, T_world_tag, axis_len=args.axis_len, label='tag')
    _draw_triad(ax, T_world_cam0, axis_len=args.axis_len, label='cam')
    P = np.vstack([np.zeros(3), T_world_tag[:3, 3], T_world_cam0[:3, 3]])
    _set_equal_3d(ax, P, pad=0.2)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)
    print(f"Saved -> {out_path}")

    # Load extrinsics if provided or discoverable, align to world with tag@t0, and animate
    extr_path = args.extrinsics
    if not extr_path:
        cand1 = Path("outputs/camera_pose_world_cam2w.npy")
        cand2 = Path("outputs/camera_extrinsics_world_w2c.npy")
        if cand1.exists():
            extr_path = str(cand1)
            args.extrinsics_is_w2c = False
        elif cand2.exists():
            extr_path = str(cand2)
            args.extrinsics_is_w2c = True

    if extr_path and Path(extr_path).exists():
        seq = np.load(extr_path).astype(float)
        if seq.ndim != 3 or seq.shape[1:] != (4,4):
            raise ValueError(f"Bad extrinsics shape {seq.shape} from {extr_path}")
        cam2w_raw = np.linalg.inv(seq) if args.extrinsics_is_w2c else seq

        cam2w_aligned = _align_cam2w_with_tag0(cam2w_raw, T_world_tag, T_tag_cam)

        out_dir = Path(extr_path).parent
        stem = Path(extr_path).stem  # original filename without extension
        out_dir.mkdir(parents=True, exist_ok=True)

        # Aligned cam->world (cam2w) sequence
        cam2w_save = out_dir / f"upd_{stem}_cam2w.npy"
        np.save(cam2w_save, cam2w_aligned.astype(np.float64))
        print(f"Saved updated cam2w -> {cam2w_save}")

        # Also save the corresponding world->cam (w2c) sequence
        # (np.linalg.inv supports stacked matrices shaped (T, 4, 4))
        w2c_aligned = np.linalg.inv(cam2w_aligned)
        w2c_save = out_dir / f"upd_{stem}_w2c.npy"
        np.save(w2c_save, w2c_aligned.astype(np.float64))
        print(f"Saved updated w2c -> {w2c_save}")

        T = cam2w_aligned.shape[0]
        cam_positions = cam2w_aligned[:, :3, 3]
        pts = np.vstack([np.zeros(3), T_world_tag[:3,3], cam_positions])

        figA = plt.figure(figsize=(7,7))
        axA = figA.add_subplot(111, projection='3d')
        axA.set_title("Camera Trajectory in World (aligned with tag@t0)")
        axA.set_xlabel('Xw'); axA.set_ylabel('Yw'); axA.set_zlabel('Zw')
        _set_equal_3d(axA, pts, pad=0.2)
        _draw_triad(axA, np.eye(4), axis_len=args.axis_len, label='world')
        _draw_triad(axA, T_world_tag, axis_len=args.axis_len, label='tag')
        path_line, = axA.plot([], [], [], c='k', lw=1, alpha=0.6)
        cam_lines = _draw_triad_lines(axA, cam2w_aligned[0], axis_len=args.axis_len)
        # Frame counter overlay (2D in axes coordinates)
        frame_text = axA.text2D(0.02, 0.95, "", transform=axA.transAxes)

        def init():
            path_line.set_data([], [])
            path_line.set_3d_properties([])
            _update_triad_lines(cam_lines, cam2w_aligned[0], axis_len=args.axis_len)
            frame_text.set_text("Frame 0")
            return (*cam_lines, path_line, frame_text)

        def update(i):
            p = cam_positions[:i+1]
            path_line.set_data(p[:,0], p[:,1])
            path_line.set_3d_properties(p[:,2])
            _update_triad_lines(cam_lines, cam2w_aligned[i], axis_len=args.axis_len)
            frame_text.set_text(f"Frame {i}")
            return (*cam_lines, path_line, frame_text)

        ani = animation.FuncAnimation(figA, update, frames=T, init_func=init, blit=True, interval=max(1,int(1000/ max(1,args.fps))))
        gif_path = Path(args.gif_out)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ani.save(str(gif_path), writer=animation.PillowWriter(fps=args.fps))
            print(f"Saved GIF -> {gif_path}")
        finally:
            plt.close(figA)
    else:
        if args.extrinsics:
            raise FileNotFoundError(f"Extrinsics not found: {args.extrinsics}")


if __name__ == "__main__":
    main()
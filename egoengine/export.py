"""Frame-aligned paired RGB/state/control dataset export."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np

from .data import aligned_trajectory
from .video import video_info


def export_dataset(case, video, output, action_path=None, image_size=256, visual_source="prepared"):
    trajectory = aligned_trajectory(case, action_path)
    visual_metadata = None
    sidecar = Path(video).with_suffix(".json")
    if sidecar.is_file():
        visual_metadata = json.loads(sidecar.read_text())
        if visual_metadata.get("case") != case.name:
            raise ValueError("Visual artifact belongs to another case")
        if visual_metadata.get("video_sha256") and (
            visual_metadata["video_sha256"] != hashlib.sha256(Path(video).read_bytes()).hexdigest()
        ):
            raise ValueError("Visual video differs from its provenance sidecar")
    n = len(trajectory["qpos"])
    if video_info(video, decode=True)["frames"] != n:
        raise ValueError("Video/action frame counts differ")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.stem + "_", suffix=".partial", dir=output.parent)
    os.close(fd)
    cap = cv2.VideoCapture(str(video))
    try:
        with h5py.File(temporary, "w") as f:
            group = f.create_group("data/demo_0")
            images = group.create_dataset(
                "obs/agentview_image",
                (n, image_size, image_size, 3),
                dtype="uint8",
                chunks=(1, image_size, image_size, 3),
                compression="gzip",
            )
            for i in range(n):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Video truncated while exporting")
                images[i] = cv2.cvtColor(
                    cv2.resize(frame, (image_size, image_size)), cv2.COLOR_BGR2RGB
                )
            group.create_dataset("actions", data=trajectory["ctrl"].astype(np.float32))
            group.create_dataset("obs/qpos", data=trajectory["qpos"].astype(np.float32))
            group.create_dataset("obs/qvel", data=trajectory["qvel"].astype(np.float32))
            if visual_metadata and visual_metadata.get("renderer") == "rby1":
                commands_path = (
                    Path(video).parent / visual_metadata["render_directory"] / "rby1_commands.npz"
                )
                with np.load(commands_path, allow_pickle=False) as commands:
                    if len(commands["joint_commands"]) != n:
                        raise ValueError("RB-Y1 visual command length mismatch")
                    group.create_dataset("obs/rby1_joint_commands", data=commands["joint_commands"])
                    group.attrs["rby1_joint_names"] = json.dumps(commands["joint_names"].tolist())
            if visual_metadata and visual_metadata.get("commands_file"):
                commands_path = (Path(video).parent / visual_metadata["render_directory"]
                                 / visual_metadata["commands_file"])
                with np.load(commands_path, allow_pickle=False) as commands:
                    values, names = commands["joint_commands"], commands["joint_names"]
                    if len(values) != n:
                        raise ValueError("Visual robot command length mismatch")
                    if values.ndim != 2 or len(names) != values.shape[1] or not np.isfinite(values).all():
                        raise ValueError("Invalid visual robot joint layout or values")
                    group.create_dataset("obs/robot_joint_commands", data=values)
                    group.attrs["robot_joint_names"] = json.dumps(names.tolist())
                    group.attrs["visual_robot"] = visual_metadata["robot_name"]
                    group.attrs["visual_robot_config_sha256"] = visual_metadata.get("robot_config_sha256", "")
            group.create_dataset(
                "obs/joint_pos",
                data=trajectory["qpos"][:, : trajectory["ctrl"].shape[1]].astype(np.float32),
            )
            poses, k = (
                case.camera(world_frame="action")
                if visual_metadata and visual_metadata.get("renderer") == "xhand"
                else case.camera()
            )
            group.create_dataset("obs/camera_to_world", data=poses)
            original = case.visual / (
                "cropped_video.mp4" if case.info["dataset"] == "Aria" else "color.mp4"
            )
            source_info = video_info(original)
            scaled_k = k.copy()
            scaled_k[0, :] *= image_size / source_info["width"]
            scaled_k[1, :] *= image_size / source_info["height"]
            group.create_dataset("camera_intrinsics", data=scaled_k)
            group.create_dataset("camera_intrinsics_original", data=k)
            group.create_dataset("timestamps", data=trajectory["timestamps"])
            group.create_dataset("dones", data=np.arange(n) == n - 1)
            group.attrs["num_samples"] = n
            group.attrs["action_source"] = trajectory["source"]
            group.attrs["visual_source"] = (
                visual_metadata["source"] if visual_metadata else visual_source
            )
            group.attrs["visual_backend"] = (
                visual_metadata.get("backend", "prepared_historical")
                if visual_metadata else "prepared_historical"
            )
            group.attrs["action_sha256"] = (
                hashlib.sha256(Path(action_path).read_bytes()).hexdigest() if action_path else ""
            )
            group.attrs["visual_retarget_sha256"] = (
                (visual_metadata.get("retarget_sha256") or "") if visual_metadata else ""
            )
            group.attrs["action_semantics"] = (
                "MuJoCo actuator position targets; virtual wrist XYZ/angles + XHand joints"
            )
            group.attrs["ref_dt"] = case.ref_dt
            group.attrs["case"] = case.name
            f["data"].attrs["total"] = n
            f.attrs["schema_version"] = "egoengine.prepared.v1"
            f.attrs["env_args"] = json.dumps(
                {
                    "env_name": "EgoEnginePreparedXHand",
                    "env_type": "mujoco",
                    "env_kwargs": {"case": case.name},
                }
            )
        Path(temporary).replace(output)
    finally:
        cap.release()
        Path(temporary).unlink(missing_ok=True)
    return output

"""Portable prepared inputs and explicit reference/physics time alignment."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "examples" / "prepared"


@dataclass
class Case:
    root: Path
    info: dict

    @property
    def name(self):
        return self.info["task"]

    @property
    def processed(self):
        return self.root / self.info["processed_dir"]

    @property
    def visual(self):
        return self.root / self.info["visual_dir"]

    @property
    def scene(self):
        return self.processed / "scene.xml"

    @property
    def ref_dt(self):
        return float(json.loads((self.processed / "task_info.json").read_text())["ref_dt"])

    def reference(self):
        with np.load(self.processed / "0/trajectory_kinematic.npz", allow_pickle=False) as f:
            values = {k: f[k].copy() for k in f.files}
        n = len(values["qpos"])
        for key in ("qpos", "qvel", "ctrl"):
            if values[key].ndim != 2 or len(values[key]) != n or not np.isfinite(values[key]).all():
                raise ValueError(f"Invalid {key} reference for {self.name}")
        return values

    def retarget_frames(self):
        """Recorded RB-Y1 configurations after the producer's warmup prefix."""
        retarget = json.loads((self.visual / "retarget.json").read_text())
        frames = retarget["traj"]
        count = len(self.reference()["qpos"])
        warmup = int(retarget.get("warm_up_steps", 0))
        if len(frames) == count + warmup:
            frames = frames[warmup:]
        if len(frames) != count:
            raise ValueError(f"Retarget/reference frame mismatch for {self.name}")
        return frames

    def camera(self, world_frame="visual"):
        """Normalize the two producers' camera formats, then trim recorded warmup."""
        retarget = json.loads((self.visual / "retarget.json").read_text())
        poses = np.asarray(retarget["camera_traj"]["target2camera"], dtype=np.float64)
        warmup = int(retarget.get("warm_up_steps", 0))
        reference = self.reference()
        count = len(reference["qpos"])
        convention = self.info.get("camera_convention")
        if convention is None:
            convention = (
                "camera_to_world" if "cam_K" in retarget["camera_traj"] else "target_to_camera"
            )
        if convention == "target_to_camera":
            if world_frame == "action" and (anchor := self.info.get("camera_anchor_joint")):
                # Camera-from-target is invariant to world-frame placement. Use
                # the prepared reference's target frame, not an older retarget
                # world's coordinates. Camera motion must not follow optimization.
                import mujoco

                model = mujoco.MjModel.from_xml_path(str(self.scene))
                data = mujoco.MjData(model)
                joint = model.joint(anchor)
                body_id = int(model.jnt_bodyid[joint.id])
                if len(poses) == count + warmup:
                    poses = poses[warmup:]
                targets = np.repeat(np.eye(4)[None], count, axis=0)
                for index, qpos in enumerate(reference["qpos"]):
                    data.qpos[:] = qpos
                    mujoco.mj_kinematics(model, data)
                    targets[index, :3, :3] = data.xmat[body_id].reshape(3, 3)
                    targets[index, :3, 3] = data.xpos[body_id]
            else:
                frames = self.retarget_frames()
                if len(poses) == count + warmup:
                    poses = poses[warmup:]
                targets = np.asarray([frame["target"] for frame in frames], dtype=np.float64)
            if targets.shape != poses.shape:
                raise ValueError("Target/camera track lengths differ")
            poses = targets @ np.linalg.inv(poses)
        elif convention != "camera_to_world":
            raise ValueError(f"Unknown camera convention: {convention}")
        if len(poses) == count + warmup:
            poses = poses[warmup:]
        if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
            raise ValueError(
                f"Camera/reference frame mismatch for {self.name}: {poses.shape}, {count}"
            )
        k = retarget["camera_traj"].get("cam_K", retarget["camera_traj"].get("intrinsic"))
        if k is None:
            k = np.loadtxt(self.visual / "egocentric_intrinsic.txt")
        return poses, np.asarray(k, dtype=np.float64)


def cases(root=DEFAULT_DATA_ROOT):
    root = Path(root).expanduser().resolve()
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Prepared examples missing: {manifest}; pass --data-root")
    return [Case(root, info) for info in json.loads(manifest.read_text())["cases"]]


def get_case(name, root=DEFAULT_DATA_ROOT):
    available = cases(root)
    for case in available:
        if case.name == name:
            return case
    raise ValueError(f"Unknown case {name!r}. Available: {', '.join(c.name for c in available)}")


def aligned_trajectory(case, action_path=None):
    """Return pre-action observations and controls on the reference-frame clock."""
    reference = case.reference()
    n = len(reference["qpos"])
    if action_path is None:
        return dict(
            qpos=reference["qpos"],
            qvel=reference["qvel"],
            ctrl=reference["ctrl"],
            timestamps=np.arange(n) * case.ref_dt,
            source="kinematic_reference",
            success=None,
        )
    with np.load(action_path, allow_pickle=False) as f:
        required = {
            "qpos",
            "qvel",
            "initial_qpos",
            "initial_qvel",
            "ctrl",
            "statistics_json",
            "success",
        }
        if missing := required.difference(f.files):
            raise ValueError(
                f"Action export lacks {sorted(missing)}; regenerate with the unified runner"
            )
        stats = json.loads(f["statistics_json"].item())
        if not bool(f["success"].item()):
            raise ValueError(
                "Action optimization is incomplete; refusing to export a successful paired dataset"
            )
        ratio = float(stats["sim_steps_per_reference"])
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("Invalid physics/reference timestep ratio")
        indices = np.rint(np.arange(n) * ratio).astype(int)
        state_qpos = np.concatenate((f["initial_qpos"][None], f["qpos"]), axis=0)
        state_qvel = np.concatenate((f["initial_qvel"][None], f["qvel"]), axis=0)
        if indices[-1] >= len(f["ctrl"]) or state_qpos.shape[1] != reference["qpos"].shape[1]:
            raise ValueError("Action/reference dimension or time alignment mismatch")
        result = dict(
            qpos=state_qpos[indices],
            qvel=state_qvel[indices],
            ctrl=f["ctrl"][indices],
            timestamps=indices / ratio * case.ref_dt,
            source="adaptive_physics",
            success=True,
        )
    if any(not np.isfinite(result[k]).all() for k in ("qpos", "qvel", "ctrl")):
        raise ValueError("Nonfinite action trajectory")
    return result

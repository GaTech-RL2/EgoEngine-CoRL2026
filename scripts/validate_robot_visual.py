"""Check actual visual-robot outputs against the unchanged prepared demonstration."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np

from egoengine.data import PROJECT_ROOT, get_case
from egoengine.robot_visual import base_pose, load_config, register_robot
from egoengine.video import video_info


def check_materials(config, placement):
    sys.path.insert(0, str(PROJECT_ROOT / "visual"))
    from robosuite.models.bases import robot_base_factory
    from robosuite.models.grippers import gripper_factory

    robot = register_robot(config, placement)(idn=0)
    robot.add_base(robot_base_factory("NullMount", idn=0))
    robot.add_gripper(gripper_factory(config["gripper_name"], idn="0_right"))
    robot.set_base_xpos(placement[:3, 3])
    model = mujoco.MjModel.from_xml_string(robot.get_xml())
    sources = (
        ("visual/robosuite/models/assets/robots/panda/robot.xml", "robot0_"),
        ("visual/robosuite/models/assets/grippers/x_right_hand_aria.xml", "gripper0_right_"),
    )
    checked = 0
    for filename, prefix in sources:
        for material in ET.parse(PROJECT_ROOT / filename).findall("./asset/material"):
            native = model.mat(prefix + material.get("name"))
            for key in ("rgba", "specular", "shininess"):
                if material.get(key) is not None:
                    np.testing.assert_allclose(
                        getattr(native, key), np.fromstring(material.get(key), sep=" "), atol=2e-7
                    )
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "outputs/franka_xhand")
    parser.add_argument(
        "--robot-config", type=Path, default=PROJECT_ROOT / "examples/robots/franka_xhand.json"
    )
    parser.add_argument("--cases", nargs="+", default=["mustard_1", "mustard_2", "mustard_3"])
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "docs/validation/FRANKA_XHAND.json"
    )
    args = parser.parse_args()
    config = load_config(args.robot_config)
    rows = []
    for name in args.cases:
        case = get_case(name)
        folder = args.root / name
        metadata = json.loads((folder / "render/render.json").read_text())
        visual_metadata = json.loads((folder / "visual.json").read_text())
        assert visual_metadata["renderer"] == "configured_robot"
        assert visual_metadata["robot_name"] == config["name"]
        assert visual_metadata["robot_config_sha256"] == metadata["robot_config_sha256"]
        source_path = case.visual / "retarget.json"
        retarget = json.loads(source_path.read_text())
        adapted = json.loads(
            (folder / "render/original_pipeline/Hand_Poses" / name / "1/retarget.json").read_text()
        )
        assert metadata["retarget_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert adapted["camera_traj"] == retarget["camera_traj"]
        assert len(adapted["traj"]) == len(retarget["traj"])
        for before, after in zip(retarget["traj"], adapted["traj"]):
            assert {k: v for k, v in before.items() if k != "robot_cfg"} == {
                k: v for k, v in after.items() if k != "robot_cfg"
            }
        with np.load(folder / "render/robot_commands.npz", allow_pickle=False) as commands:
            names = commands["joint_names"].tolist()
            values = commands["joint_commands"]
            hand_names = names[len(config["arm_joints"]) :]
            expected_hand = [
                [frame["robot_cfg"][joint] for joint in hand_names]
                for frame in case.retarget_frames()
            ]
            np.testing.assert_array_equal(values[:, len(config["arm_joints"]) :], expected_hand)
            with np.load(folder / "render/replay_states.npz", allow_pickle=False) as replay:
                np.testing.assert_allclose(
                    commands["actual_wrist_pose"], replay["wrist_pose"], atol=2e-6
                )
        n = len(case.reference()["qpos"])
        for video in (
            "render/scene.mp4",
            "render/foreground.mp4",
            "render/visibility_mask.mp4",
            "visual.mp4",
        ):
            assert video_info(folder / video, decode=True)["frames"] == n
        with h5py.File(folder / "dataset.hdf5") as dataset:
            demo = dataset["data/demo_0"]
            np.testing.assert_array_equal(
                demo["actions"][:], case.reference()["ctrl"].astype(np.float32)
            )
            np.testing.assert_array_equal(demo["obs/robot_joint_commands"][:], values)
            np.testing.assert_array_equal(demo["obs/camera_to_world"][:], case.camera()[0])
            assert len(demo["obs/agentview_image"]) == n
            assert demo.attrs["visual_robot"] == config["name"]
        assert not metadata["render_collision_mesh"]
        assert metadata["frames_with_visible_robot"] == n
        rows.append(
            dict(
                case=name,
                frames=n,
                original_hand_commands_identical=True,
                camera_and_object_tracks_unchanged=True,
                eef_actions_unchanged=True,
                rendered_fk_matches_mapping=True,
                collision_geometry_visible=False,
                visible_robot_frames=metadata["frames_with_visible_robot"],
                max_wrist_position_error_m=metadata["max_wrist_position_error_m"],
                max_wrist_orientation_error_rad=metadata["max_wrist_orientation_error_rad"],
            )
        )
    materials = check_materials(config, base_pose(config, retarget))
    result = dict(
        scope="Visual arm replacement and paired reference-EEF export; no new RL or hardware claim",
        robot=config["name"],
        materials_verified=materials,
        cases=rows,
        robot_config_sha256=hashlib.sha256(args.robot_config.read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

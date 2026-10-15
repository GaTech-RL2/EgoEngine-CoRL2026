"""Copy existing action inputs into a portable search set without editing the sources."""

import argparse
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taco-root", type=Path, required=True)
    parser.add_argument("--aria-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--asset-fallback", action="append", default=[], type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory; an existing search set is frozen")
    output.mkdir(parents=True, exist_ok=True)
    maps = [item.split("=", 1) for item in args.path_map]
    fallback = {}
    for folder in args.asset_fallback:
        for path in folder.iterdir():
            # Previous portable bundles prefix the original filename with its path hash.
            name = path.name.split("_", 1)[1] if len(path.name.split("_", 1)[0]) == 12 else path.name
            if path.is_file():
                fallback.setdefault(name, []).append(path)
    assets, cases, errors = {}, [], []

    def copy_xml(source, destination):
        tree = ET.parse(source)
        compiler = tree.getroot().find("compiler")
        compiler_args = {} if compiler is None else compiler.attrib
        for element in tree.getroot().iter():
            if "file" not in element.attrib:
                continue
            original = element.attrib["file"]
            path = Path(original)
            if not path.is_absolute():
                subdir = compiler_args.get("texturedir" if element.tag == "texture" else "meshdir", "")
                path = source.parent / subdir / path
            candidates = [path] + [Path(str(path).replace(old, new, 1))
                                   for old, new in maps if str(path).startswith(old)]
            actual = next((p for p in candidates if p.is_file()), None)
            if actual is None:
                options = fallback.get(path.name, [])
                if not options or len({digest(p) for p in options}) != 1:
                    raise FileNotFoundError(f"Missing or ambiguous asset: {original}")
                actual = options[0]
            sha = digest(actual)
            relative = Path("assets") / f"{sha[:16]}_{actual.name}"
            target = output / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(actual, target)
            assets[str(relative)] = {"source": str(actual), "sha256": sha, "bytes": actual.stat().st_size}
            element.attrib["file"] = os.path.relpath(target, destination.parent)
        if compiler is not None:
            compiler.attrib.pop("meshdir", None)
            compiler.attrib.pop("texturedir", None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tree.write(destination, encoding="utf-8", xml_declaration=True)

    inputs = [("Aria", "aria", "right", p.parent.name, p, p / "trajectory_kinematic.npz")
              for p in sorted(args.aria_root.glob("*/1"), key=lambda p: int(p.parent.name.rsplit("_", 1)[1]))
              if (p / "trajectory_kinematic.npz").is_file()]
    inputs += [("TACO", "gigahand", "bimanual", p.name, p, p / "0/trajectory_kinematic.npz")
               for p in sorted(args.taco_root.iterdir()) if (p / "0/trajectory_kinematic.npz").is_file()]
    for dataset, dataset_name, embodiment, task, source, trajectory in inputs:
        try:
            relative = Path("processed") / dataset_name / "xhand" / embodiment / task
            destination = output / relative
            for name in ("scene.xml", "scene_eq.xml"):
                copy_xml(source / name, destination / name)
            info = json.loads((source / "task_info.json").read_text())
            for key in ("right_object_visual_mesh", "right_object_collision_meshes",
                        "left_object_visual_mesh", "left_object_collision_meshes"):
                info.pop(key, None)
            (destination / "task_info.json").write_text(json.dumps(info, indent=2) + "\n")
            target = destination / "0/trajectory_kinematic.npz"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(trajectory, target)
            models = [mujoco.MjModel.from_xml_path(str(destination / name))
                      for name in ("scene.xml", "scene_eq.xml")]
            model = models[0]
            with np.load(target, allow_pickle=False) as data:
                qpos, qvel = data["qpos"], data["qvel"]
                if qpos.shape != (len(qpos), model.nq) or qvel.shape != (len(qpos), model.nv):
                    raise ValueError("Reference state layout does not match the scene")
                if not all(np.isfinite(data[k]).all() for k in ("qpos", "qvel")):
                    raise ValueError("Nonfinite reference states")
                frames = len(qpos)
            source_files = ("scene.xml", "scene_eq.xml", "task_info.json")
            cases.append({"dataset": dataset, "dataset_name": dataset_name, "robot_type": "xhand",
                          "embodiment_type": embodiment, "task": task, "data_id": 0,
                          "processed_dir": str(relative), "source_directory": str(source),
                          "action_inputs_complete": True, "reference_frames": frames,
                          "ref_dt": info["ref_dt"], "nq": model.nq, "nv": model.nv, "nu": model.nu,
                          "source_sha256": {**{name: digest(source / name) for name in source_files},
                                            "trajectory_kinematic.npz": digest(trajectory)},
                          "portable_sha256": {name: digest(destination / name) for name in source_files},
                          "trajectory_sha256": digest(target)})
            print(f"[INPUT] {task}: {frames} frames", flush=True)
        except Exception as exc:
            errors.append({"task": task, "dataset": dataset, "source_directory": str(source),
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"[INPUT ERROR] {task}: {exc}", flush=True)
    manifest = {"purpose": "Exploratory action sample search; sources unchanged; no visual inputs",
                "cases": cases, "input_errors": errors, "assets": assets,
                "candidate_count": len(inputs), "script_sha256": digest(Path(__file__))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"candidates": len(inputs), "valid": len(cases), "errors": len(errors)}), flush=True)


if __name__ == "__main__":
    main()

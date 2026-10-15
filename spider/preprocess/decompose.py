#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import trimesh
except ModuleNotFoundError:
    print("trimesh is required. Please install with `pip install trimesh`")
    raise SystemExit(1)

from pathlib import Path

import numpy as np
import tyro
from loguru import logger

MeshPart = tuple[np.ndarray, np.ndarray]


def fast_voxel_convex_decomp_from_pointcloud(
    points: np.ndarray, pitch: float = 0.02, min_points: int = 20
) -> list[MeshPart]:
    """Approximate convex decomposition from voxel clusters."""
    coords = np.floor(points / pitch).astype(int)
    unique_voxels, inverse = np.unique(coords, axis=0, return_inverse=True)

    hulls: list[MeshPart] = []
    for idx, _ in enumerate(unique_voxels):
        cluster_points = points[inverse == idx]
        if len(cluster_points) < min_points:
            continue
        cluster_mesh = trimesh.Trimesh(vertices=cluster_points, faces=[])
        hull = cluster_mesh.convex_hull
        vertices = np.asarray(hull.vertices)
        faces = np.asarray(hull.faces, dtype=int)
        hulls.append((vertices, faces))
    return hulls


def decompose_one_mesh(
    input_path: Path,
    output_path: Path,
    pitch: float,
    min_points: int,
) -> bool:
    mesh = trimesh.load(str(input_path), force="mesh", process=False, skip_materials=True)
    points = np.asarray(mesh.vertices)
    if points.size == 0:
        logger.warning("Skip empty mesh: {}", input_path)
        return False

    hulls = fast_voxel_convex_decomp_from_pointcloud(points, pitch=pitch, min_points=min_points)
    if not hulls:
        logger.warning("No convex hull parts generated: {}", input_path)
        return False

    parts = [trimesh.Trimesh(vs, fs, process=False) for vs, fs in hulls]
    merged = trimesh.util.concatenate(parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.export(str(output_path))
    logger.info("Saved {} parts -> {}", len(parts), output_path)
    return True


def main(
    meshes_root: str = "/home/ycl/projects/workspace/egoengine-mpc/robosuite/models/assets/objects/meshes",
    recursive: bool = True,
    pitch: float = 0.02,
    min_points: int = 20,
    overwrite: bool = False,
) -> None:
    root = Path(meshes_root).expanduser().resolve()
    if not root.exists():
        logger.error("Mesh root does not exist: {}", root)
        return

    pattern = "**/*_cm.obj" if recursive else "*_cm.obj"
    input_files = sorted(root.glob(pattern))
    if not input_files:
        logger.warning("No *_cm.obj found under {}", root)
        return

    ok_count = 0
    skip_count = 0
    fail_count = 0
    for input_path in input_files:
        output_path = input_path.with_name(input_path.stem.replace("_cm", "_convex") + ".obj")
        if output_path.exists() and not overwrite:
            logger.info("Skip existing: {}", output_path)
            skip_count += 1
            continue
        try:
            success = decompose_one_mesh(
                input_path=input_path,
                output_path=output_path,
                pitch=pitch,
                min_points=min_points,
            )
            if success:
                ok_count += 1
            else:
                fail_count += 1
        except Exception as exc:
            logger.exception("Failed {}: {}", input_path, exc)
            fail_count += 1

    logger.info(
        "Done. total={} success={} skipped={} failed={}",
        len(input_files),
        ok_count,
        skip_count,
        fail_count,
    )


if __name__ == "__main__":
    tyro.cli(main)

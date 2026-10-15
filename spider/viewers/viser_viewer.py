# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MuJoCo XML Viser visualizer.

This mirrors the Rerun viewer APIs so it can be used as a drop-in replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import loguru
import mujoco
import numpy as np
import trimesh

# -----------------------------
# Trace visualization defaults
# -----------------------------

DEFAULT_TRACE_COLOR = [204, 26, 204]  # ~ (0.8, 0.1, 0.8)
DEFAULT_TRACE_RADIUS = 0.002
DEFAULT_FLOOR_COLOR = [200, 200, 200]  # light grey


def _lazy_import_viser():
    try:
        import viser  # type: ignore

        return viser
    except ImportError as exc:
        raise ImportError(
            "viser is required for the Viser viewer. Install with `pip install viser`."
        ) from exc


@dataclass
class _ViserState:
    server: Any | None = None
    entity_root: str = "mujoco"
    body_handles: list[tuple[Any, int]] = field(default_factory=list)
    trace_handle: Any | None = None
    trace_colors: np.ndarray | None = None


_STATE = _ViserState()


# -----------------------------
# Geometry helpers
# -----------------------------


def _rgba_to_uint8(rgba: np.ndarray) -> np.ndarray:
    rgba_arr = np.asarray(rgba)
    if np.issubdtype(rgba_arr.dtype, np.floating):
        rgba_arr = np.clip(rgba_arr, 0.0, 1.0)
        rgba_arr = (rgba_arr * 255.0).astype(np.uint8)
    else:
        rgba_arr = rgba_arr.astype(np.uint8)
    if rgba_arr.size == 3:
        rgba_arr = np.concatenate([rgba_arr, np.array([255], dtype=np.uint8)])
    return rgba_arr


def _set_mesh_color(mesh: trimesh.Trimesh, rgba: np.ndarray) -> None:
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    rgba_int = _rgba_to_uint8(rgba)
    mesh.visual = TextureVisuals(
        material=PBRMaterial(
            baseColorFactor=rgba_int,
            main_color=rgba_int,
            metallicFactor=0.5,
            roughnessFactor=1.0,
            alphaMode="BLEND" if rgba_int[-1] < 255 else "OPAQUE",
        )
    )


def _trimesh_from_primitive(
    geom_type: int, size: np.ndarray, rgba: np.ndarray | None = None
) -> trimesh.Trimesh | None:
    t = mujoco.mjtGeom
    if geom_type == t.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2)
    elif geom_type == t.mjGEOM_CAPSULE:
        radius = float(size[0])
        length = float(2.0 * size[1])
        mesh = trimesh.creation.capsule(radius=radius, height=length)
    elif geom_type == t.mjGEOM_CYLINDER:
        radius = float(size[0])
        height = float(2.0 * size[1])
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
    elif geom_type == t.mjGEOM_BOX:
        extents = 2.0 * np.asarray(size[:3], dtype=np.float32)
        mesh = trimesh.creation.box(extents=extents)
    elif geom_type == t.mjGEOM_PLANE:
        mesh = trimesh.creation.box(extents=[20.0, 20.0, 0.01])
    else:
        return None

    if rgba is not None:
        _set_mesh_color(mesh, rgba)
    return mesh


def _mujoco_mesh_to_trimesh(
    model: mujoco.MjModel, geom_id: int
) -> trimesh.Trimesh | None:
    mesh_id = model.geom_dataid[geom_id]
    if mesh_id < 0:
        return None

    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    vertices = model.mesh_vert[vert_start : vert_start + vert_count]
    faces = model.mesh_face[face_start : face_start + face_count]

    if len(vertices) == 0 or len(faces) == 0:
        return None

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _get_mesh_file(spec: mujoco.MjSpec, geom: mujoco.MjsGeom) -> Path | None:
    try:
        meshname = geom.meshname
        if not meshname:
            return None
        mesh = spec.mesh(meshname)
        mesh_dir = spec.meshdir if spec.meshdir is not None else ""
        model_dir = spec.modelfiledir if spec.modelfiledir is not None else ""
        return (Path(model_dir) / mesh_dir / mesh.file).resolve()
    except Exception:
        return None


def _get_mesh_scale(spec: mujoco.MjSpec, geom: mujoco.MjsGeom) -> np.ndarray | None:
    try:
        mesh = spec.mesh(geom.meshname)
        scale = mesh.scale
        if scale is None:
            return None
        return np.asarray(scale, dtype=np.float32)
    except Exception:
        return None


def _ensure_names(spec: mujoco.MjSpec) -> None:
    geom_placeholder_idx = 0
    body_placeholder_idx = 0
    for body in spec.bodies[1:]:
        if not body.name:
            body.name = f"VISER_BODY_{body_placeholder_idx}"
            body_placeholder_idx += 1
        for geom in body.geoms:
            if not geom.name:
                geom.name = f"VISER_GEOM_{geom_placeholder_idx}"
                geom_placeholder_idx += 1


# -----------------------------
# Scene construction
# -----------------------------


def init_viser(app_name: str = "spider", spawn: bool | None = None) -> None:
    """Initialize Viser server (spawn unused, kept for drop-in compatibility)."""
    if _STATE.server is not None:
        return
    viser = _lazy_import_viser()
    _STATE.server = viser.ViserServer(label=app_name)


def _get_server() -> Any:
    if _STATE.server is None:
        init_viser()
    return _STATE.server


def build_and_log_scene_from_spec(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    xml_path: Path | None = None,
    entity_root: str = "mujoco",
) -> list[tuple[Any, int]]:
    """Build and log a Viser scene directly from a spec and compiled model."""
    _ensure_names(spec)
    server = _get_server()
    _STATE.entity_root = entity_root

    # Add a floor grid.
    try:
        server.scene.add_grid(
            f"{entity_root}/ground_plane",
            section_color=tuple(np.array(DEFAULT_FLOOR_COLOR) / 255.0),
            cell_color=tuple(np.array(DEFAULT_FLOOR_COLOR) / 255.0),
        )
    except Exception:
        pass

    body_entity_and_ids: list[tuple[Any, int]] = []

    for body in spec.bodies[1:]:
        body_name = body.name
        body_path = f"{entity_root}/{body_name}"
        body_handle = server.scene.add_frame(body_path, show_axes=False)

        try:
            body_id = model.body(body_name).id
        except Exception:
            body_id = body.id

        body_entity_and_ids.append((body_handle, body_id))

        for geom in body.geoms:
            geom_name = geom.name
            geom_path = f"{body_path}/geom_{geom_name}"

            model_geom = None
            try:
                model_geom = model.geom(geom.name)
            except Exception:
                model_geom = None

            rgba = None
            if model_geom is not None:
                try:
                    rgba = np.asarray(model_geom.rgba, dtype=np.float32)
                except Exception:
                    rgba = None
            if rgba is None:
                try:
                    rgba = np.asarray(geom.rgba, dtype=np.float32)
                except Exception:
                    rgba = None

            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                tm = None
                mesh_file = _get_mesh_file(spec, geom)
                mesh_scale = _get_mesh_scale(spec, geom)
                if mesh_file is not None and mesh_file.exists():
                    try:
                        tm = trimesh.load(str(mesh_file), force="mesh")
                        if isinstance(tm, trimesh.Scene):
                            tm = tm.to_mesh()
                    except Exception:
                        tm = None
                if tm is None:
                    try:
                        geom_id = model_geom.id if model_geom is not None else -1
                        tm = _mujoco_mesh_to_trimesh(model, geom_id)
                    except Exception:
                        tm = None
                if tm is None:
                    loguru.logger.warning(
                        f"Viser: failed to load mesh for geom '{geom_name}'"
                    )
                    continue
                if mesh_scale is not None:
                    try:
                        tm.apply_scale(mesh_scale)
                    except Exception:
                        pass
                if rgba is not None:
                    _set_mesh_color(tm, rgba)
            else:
                size = geom.size
                if model_geom is not None:
                    try:
                        model_size = model.geom_size[model_geom.id]
                        if np.any(np.asarray(size) == 0) or np.any(np.isnan(size)):
                            size = model_size
                    except Exception:
                        pass
                tm = _trimesh_from_primitive(geom.type, size, rgba=rgba)

            if tm is None:
                continue

            try:
                server.scene.add_mesh_trimesh(
                    geom_path,
                    tm,
                    position=np.asarray(geom.pos, dtype=np.float32),
                    wxyz=np.asarray(geom.quat, dtype=np.float32),
                )
            except Exception as exc:
                loguru.logger.warning(f"Viser: failed to add geom '{geom_name}': {exc}")

    _STATE.body_handles = body_entity_and_ids
    return body_entity_and_ids


def build_and_log_scene(
    xml_path: Path, entity_root: str = "mujoco"
) -> tuple[mujoco.MjSpec, mujoco.MjModel, list[tuple[Any, int]]]:
    """Load MJCF, create static geometry, and log it to Viser."""
    spec = mujoco.MjSpec.from_file(str(xml_path))
    _ensure_names(spec)
    model = spec.compile()
    body_entity_and_ids = build_and_log_scene_from_spec(
        spec=spec,
        model=model,
        xml_path=xml_path,
        entity_root=entity_root,
    )
    return spec, model, body_entity_and_ids


def log_scene_from_npz(npz_path: Path) -> list[tuple[Any, int]]:
    """Viser does not support baked .npz scenes; noop for compatibility."""
    loguru.logger.warning(
        f"Viser: log_scene_from_npz is not supported (requested {npz_path})."
    )
    return []


# -----------------------------
# Logging helpers
# -----------------------------


def log_frame(
    data: mujoco.MjData,
    sim_time: float,
    viewer_body_entity_and_ids: list[tuple[Any, int]] = [],
) -> None:
    del sim_time
    if _STATE.server is None or not viewer_body_entity_and_ids:
        return

    server = _STATE.server
    with server.atomic():
        for handle, bid in viewer_body_entity_and_ids:
            pos = np.asarray(data.xpos[bid], dtype=np.float32)
            quat = np.asarray(data.xquat[bid], dtype=np.float32)
            try:
                handle.position = tuple(pos)
                handle.wxyz = tuple(quat)
            except Exception:
                # Fallback for handles that store transform differently.
                try:
                    handle.position = pos
                    handle.wxyz = quat
                except Exception:
                    pass

    try:
        server.flush()
    except Exception:
        pass


def _compute_trace_colors(I: int, N: int, K: int) -> np.ndarray:
    colors = np.zeros([I, N, K, 3])
    white = np.array([255, 255, 255])
    red = np.array([255, 0, 0])
    blue = np.array([0, 0, 255])

    for i in range(I):
        for k in range(K):
            if I == 1:
                colors[i, :, k, :] = red if k < 1 else blue
            else:
                if k < 1:
                    colors[i, :, k, :] = (1 - i / (I - 1)) * white + (i / (I - 1)) * red
                else:
                    colors[i, :, k, :] = (1 - i / (I - 1)) * white + (i / (I - 1)) * blue
    return colors.reshape(I * N * K, 3).astype(np.uint8)


def log_traces_from_info(traces: np.ndarray, sim_time: float) -> None:
    del sim_time
    if _STATE.server is None:
        return

    a = np.asarray(traces, dtype=np.float32)
    if a.ndim != 5 or a.shape[-1] != 3:
        loguru.logger.warning(
            f"Viser: skip trace logging with incompatible shape {a.shape}"
        )
        return

    I, N, P, K, _ = a.shape
    if P < 2:
        return

    # Rearrange to (I, N, K, P, 3) then flatten strips.
    a = a.transpose(0, 1, 3, 2, 4)
    strips = a.reshape(I * N * K, P, 3)

    # Convert line strips to segments (S*(P-1), 2, 3).
    segments = np.stack([strips[:, :-1, :], strips[:, 1:, :]], axis=2)
    segments = segments.reshape(-1, 2, 3)

    colors = _compute_trace_colors(I, N, K)
    colors = np.repeat(colors, repeats=P - 1, axis=0)
    colors = np.repeat(colors[:, None, :], repeats=2, axis=1)

    server = _STATE.server
    if _STATE.trace_handle is None:
        _STATE.trace_handle = server.scene.add_line_segments(
            f"{_STATE.entity_root}/traces",
            segments,
            colors,
            line_width=4.0,
        )
    else:
        _STATE.trace_handle.points = segments
        if hasattr(_STATE.trace_handle, "colors"):
            _STATE.trace_handle.colors = colors

    try:
        server.flush()
    except Exception:
        pass

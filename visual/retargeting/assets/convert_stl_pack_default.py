#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import List

import trimesh


# -------------------------
# Mesh utilities
# -------------------------
def load_mesh(path: Path) -> trimesh.Trimesh:
    obj = trimesh.load(path, force="mesh")
    if isinstance(obj, trimesh.Scene):
        geoms = list(obj.geometry.values())
        if not geoms:
            raise RuntimeError(f"Empty scene: {path}")
        obj = trimesh.util.concatenate(geoms)
    if not isinstance(obj, trimesh.Trimesh) or obj.is_empty:
        raise RuntimeError(f"Failed to load mesh: {path}")
    return obj


def export_obj(mesh: trimesh.Trimesh, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)


def write_assets_xml_flat(out_dir: Path, mesh_files: List[Path], filename: str = "assets.xml") -> None:
    """
    <mujoco>
      <mesh file="..."/>
      ...
    </mujoco>
    """
    lines = ["<mujoco>"]
    for p in mesh_files:
        lines.append(f'  <mesh file="{p.name}"/>')
    lines.append("</mujoco>")
    lines.append("")
    (out_dir / filename).write_text("\n".join(lines), encoding="utf-8")


def clean_link_folder(dst_dir: Path) -> None:
    """
    Overwrite behavior: remove all .obj and assets.xml in the destination folder.
    Never delete any STL (shouldn't exist there anyway, but still safe).
    """
    if not dst_dir.exists():
        return
    for p in dst_dir.glob("*.obj"):
        if p.is_file():
            p.unlink(missing_ok=True)
    ax = dst_dir / "assets.xml"
    if ax.exists():
        ax.unlink(missing_ok=True)


def vhacd_collision_objs_from_mesh(mesh: trimesh.Trimesh, out_dir: Path, prefix: str) -> List[Path]:
    """
    Run pybullet VHACD with defaults and write ONLY final collision hull OBJs to out_dir:
      {prefix}_collision_0.obj, {prefix}_collision_1.obj, ...
    Temp artifacts are kept in a temp directory (no _vhacd_* files left in out_dir).
    """
    try:
        import pybullet as p
    except Exception as e:
        raise RuntimeError("pybullet not available; install with: pip install pybullet") from e

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tmp_in_obj = td / f"_{prefix}_vhacd_in.obj"
        tmp_out_obj = td / f"_{prefix}_vhacd_out.obj"
        tmp_log = td / f"_{prefix}_vhacd_log.txt"

        mesh.export(tmp_in_obj)

        cid = p.connect(p.DIRECT)
        try:
            p.vhacd(str(tmp_in_obj), str(tmp_out_obj), str(tmp_log))
        finally:
            p.disconnect(cid)

        if not tmp_out_obj.exists():
            raise RuntimeError(
                "VHACD did not produce an output OBJ.\n"
                f"Input OBJ: {tmp_in_obj}\n"
                f"Expected output: {tmp_out_obj}\n"
                f"Log: {tmp_log}\n"
                "Workaround:\n"
                "  - rerun with --no_collision\n"
            )

        out = trimesh.load(tmp_out_obj, force="scene")
        hulls: List[trimesh.Trimesh] = []
        if isinstance(out, trimesh.Scene):
            hulls = [g for g in out.geometry.values()]
        elif isinstance(out, trimesh.Trimesh):
            hulls = [out]
        else:
            raise RuntimeError(f"Unexpected VHACD output type: {type(out)}")

        if not hulls:
            raise RuntimeError("VHACD produced 0 hulls (see temp log).")

        out_paths: List[Path] = []
        for i, h in enumerate(hulls):
            fn = out_dir / f"{prefix}_collision_{i}.obj"
            h.export(fn)
            out_paths.append(fn)

        return out_paths


# -------------------------
# Builders for LINK_18/19/20 (LEFT ARM)
# -------------------------
def build_link18(
    src_dir: Path,
    dst_dir: Path,
    make_collision: bool,
    scale: float,
    clean: bool,
) -> None:
    """
    LINK_18 (left arm4):
      visuals:
        LINK_18_WY_1.stl -> LINK_18_WY_1.obj   (white/grey in MJCF)
        LINK_18_WY_2.stl -> LINK_18_WY_2.obj   (black in MJCF)
      collisions (optional):
        if LINK_18_WY.stl exists -> use it
        else -> concatenate _1 and _2 and VHACD that
        output: LINK_18_WY_collision_*.obj
      assets.xml lists visuals first in order: _1, _2, then collisions
    """
    stl1 = src_dir / "LINK_18_WY_1.stl"
    stl2 = src_dir / "LINK_18_WY_2.stl"
    stl_combined = src_dir / "LINK_18_WY.stl"

    if not stl1.exists() or not stl2.exists():
        raise FileNotFoundError(f"Missing LINK_18 STLs: {stl1} / {stl2}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        clean_link_folder(dst_dir)

    m1 = load_mesh(stl1)
    m2 = load_mesh(stl2)
    if scale != 1.0:
        m1.apply_scale(scale)
        m2.apply_scale(scale)

    v1 = dst_dir / "LINK_18_WY_1.obj"
    v2 = dst_dir / "LINK_18_WY_2.obj"
    export_obj(m1, v1)
    export_obj(m2, v2)

    collision_paths: List[Path] = []
    if make_collision:
        if stl_combined.exists():
            mc = load_mesh(stl_combined)
            if scale != 1.0:
                mc.apply_scale(scale)
        else:
            mc = trimesh.util.concatenate([m1, m2])
        collision_paths = vhacd_collision_objs_from_mesh(mc, dst_dir, prefix="LINK_18_WY")

    write_assets_xml_flat(dst_dir, [v1, v2] + collision_paths)
    print(f"[OK] LINK_18 -> {dst_dir}")
    print(f"  visuals: 2 (LINK_18_WY_1.obj, LINK_18_WY_2.obj)")
    print(f"  collisions: {len(collision_paths)}")


def build_single_link(
    *,
    src_stl: Path,
    dst_dir: Path,
    visual_prefix: str,
    make_collision: bool,
    scale: float,
    clean: bool,
) -> None:
    """
    LINK_19 / LINK_20 (single STL -> single visual OBJ)
    collisions (optional): {visual_prefix}_collision_*.obj
    """
    if not src_stl.exists():
        raise FileNotFoundError(src_stl)

    dst_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        clean_link_folder(dst_dir)

    m = load_mesh(src_stl)
    if scale != 1.0:
        m.apply_scale(scale)

    v = dst_dir / f"{visual_prefix}.obj"
    export_obj(m, v)

    collision_paths: List[Path] = []
    if make_collision:
        collision_paths = vhacd_collision_objs_from_mesh(m, dst_dir, prefix=visual_prefix)

    write_assets_xml_flat(dst_dir, [v] + collision_paths)
    print(f"[OK] {dst_dir.name} -> {dst_dir}")
    print(f"  visuals: 1 ({v.name})")
    print(f"  collisions: {len(collision_paths)}")


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Convert STL pack to Robosuite OBJ asset folders (LEFT ARM ONLY)")
    ap.add_argument("--src_dir", type=str, required=True, help="Folder containing source STLs (rbyxhand_modified)")
    ap.add_argument("--dst_link18_dir", type=str, required=True, help=".../assets/LINK_18")
    ap.add_argument("--dst_link19_dir", type=str, required=True, help=".../assets/LINK_19")
    ap.add_argument("--dst_link20_dir", type=str, required=True, help=".../assets/LINK_20")
    ap.add_argument("--no_collision", action="store_true", help="Skip VHACD collision generation")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--clean", action="store_true", help="Delete existing *.obj and assets.xml in dst link folders first")
    args = ap.parse_args()

    src_dir = Path(args.src_dir).expanduser().resolve()
    dst_link18_dir = Path(args.dst_link18_dir).expanduser().resolve()
    dst_link19_dir = Path(args.dst_link19_dir).expanduser().resolve()
    dst_link20_dir = Path(args.dst_link20_dir).expanduser().resolve()

    make_collision = not args.no_collision

    # LINK_18 (left arm4, two-color)
    build_link18(
        src_dir=src_dir,
        dst_dir=dst_link18_dir,
        make_collision=make_collision,
        scale=args.scale,
        clean=args.clean,
    )

    # LINK_19 (left arm5)
    build_single_link(
        src_stl=src_dir / "LINK_19_WP.stl",
        dst_dir=dst_link19_dir,
        visual_prefix="LINK_19_WP",
        make_collision=make_collision,
        scale=args.scale,
        clean=args.clean,
    )

    # LINK_20 (left arm6)
    build_single_link(
        src_stl=src_dir / "LINK_20_WR.stl",
        dst_dir=dst_link20_dir,
        visual_prefix="LINK_20_WR",
        make_collision=make_collision,
        scale=args.scale,
        clean=args.clean,
    )

    print("[DONE] Left-arm-only LINK_18 / LINK_19 / LINK_20 generated.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

# ---------- URDF parsing ----------
MESH_RE = re.compile(r'<mesh[^>]*filename="([^"]+)"')

def find_mesh_files_in_urdf(urdf_text: str) -> List[str]:
    return MESH_RE.findall(urdf_text)

def is_convertible_mesh(p: Path) -> bool:
    return p.suffix.lower() in {".dae", ".glb", ".gltf"}

# ---------- Conversion backends ----------
def convert_with_trimesh(src: Path, dst: Path) -> None:
    import trimesh  # noqa
    # force="mesh" helps if scene is returned; we handle scene below too.
    obj = trimesh.load(src, force=None)
    if obj is None:
        raise RuntimeError("trimesh.load returned None")

    if hasattr(obj, "geometry") and hasattr(obj, "dump"):
        # It's a Scene
        meshes = obj.dump()
        if not meshes:
            raise RuntimeError("Loaded scene has no meshes")
        merged = trimesh.util.concatenate(meshes)
        merged.export(dst)
    else:
        # It's a Trimesh
        obj.export(dst)

def convert_with_assimp(src: Path, dst: Path) -> None:
    # Requires assimp CLI installed (assimp export in out.stl)
    exe = shutil.which("assimp")
    if exe is None:
        raise RuntimeError("assimp not found in PATH")
    cmd = [exe, "export", str(src), str(dst)]
    subprocess.run(cmd, check=True)

def convert_mesh(src: Path, dst: Path, prefer_assimp: bool = False) -> Tuple[bool, str]:
    """
    Returns (ok, message).
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        if prefer_assimp:
            try:
                convert_with_assimp(src, dst)
                return True, "assimp"
            except Exception as e:
                # fall through to trimesh
                pass

        convert_with_trimesh(src, dst)
        return True, "trimesh"
    except Exception as e_trimesh:
        # last resort: assimp if not tried
        try:
            convert_with_assimp(src, dst)
            return True, "assimp"
        except Exception as e_assimp:
            return False, f"FAILED (trimesh: {e_trimesh}) (assimp: {e_assimp})"

# ---------- Main ----------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, required=True, help="Path to URDF")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory for STL. Default: alongside source mesh file.")
    ap.add_argument("--inplace", action="store_true",
                    help="Write STL next to each source mesh (default behavior if --out-dir not set).")
    ap.add_argument("--prefer-assimp", action="store_true",
                    help="Try assimp first (if available).")
    args = ap.parse_args()

    urdf_path: Path = args.urdf
    urdf_dir = urdf_path.parent
    urdf_text = urdf_path.read_text()

    mesh_files = find_mesh_files_in_urdf(urdf_text)
    if not mesh_files:
        print(f"No <mesh filename=...> found in {urdf_path}")
        return

    # Resolve and de-duplicate
    resolved = []
    for mf in mesh_files:
        src = (urdf_dir / mf).resolve() if not Path(mf).is_absolute() else Path(mf)
        resolved.append((mf, src))
    # unique by absolute path
    seen = set()
    uniq = []
    for mf, src in resolved:
        if src in seen:
            continue
        seen.add(src)
        uniq.append((mf, src))

    any_fail = False
    for mf, src in uniq:
        if not src.exists():
            print(f"[MISS] {mf} -> {src} does not exist")
            any_fail = True
            continue

        if not is_convertible_mesh(src):
            # skip non-dae/glb/gltf (already stl/obj)
            continue

        dst_name = src.with_suffix(".stl").name

        if args.out_dir is not None:
            dst = args.out_dir / dst_name
        else:
            # default: next to source
            dst = src.with_suffix(".stl") if (args.inplace or True) else src.with_suffix(".stl")

        ok, backend = convert_mesh(src, dst, prefer_assimp=args.prefer_assimp)
        if ok:
            print(f"[OK] {src.name} -> {dst} ({backend})")
        else:
            print(f"[FAIL] {src.name} -> {dst}: {backend}")
            any_fail = True

    if any_fail:
        raise SystemExit(2)

if __name__ == "__main__":
    main()

# ycb_loader.py

import os, sys, math, xml.etree.ElementTree as ET

import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf, Sdf

def load_ycb(
    stage,
    urdf_path,
    name=None,
    translate=(0.0, 0.0, 0.0),
    rotate=(0.0, 0.0, 0.0),
):
    """
    Manually load a single-link YCB URDF under /World/<id>, with full PhysX:
      1) Ensure /World/PhysicsScene with gravity.
      2) Parse URDF for mass, mesh file, visual origin.
      3) Create /World/<id> Xform and apply translate/rotate.
      4) Under it, make a 'visual' Xform, apply the mesh's origin.
      5) Reference the mesh file into that prim.
      6) Recursively apply CollisionAPI to the mesh prim.
      7) Apply RigidBodyAPI (dynamic) and MassAPI (correct mass).
    """
    # 1) Physics scene
    phys = UsdPhysics.Scene.Get(stage, "/World/PhysicsScene")
    if not phys:
        phys = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        phys.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1.0))
        phys.CreateGravityMagnitudeAttr(9.8)

    # 2) Parse URDF
    if not os.path.isfile(urdf_path):
        print(f"URDF not found: {urdf_path}", file=sys.stderr)
        sys.exit(1)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # 2a) Mass
    inert = root.find(".//inertial/mass")
    mass_val = float(inert.attrib.get("value","1.0")) if inert is not None else 1.0

    # 2b) Mesh filename
    link = root.find("link")
    if link is None:
        print("No <link> in URDF", file=sys.stderr); sys.exit(1)
    mesh_elem = link.find("visual/geometry/mesh")
    if mesh_elem is None or "filename" not in mesh_elem.attrib:
        print("No <mesh filename=> in URDF", file=sys.stderr); sys.exit(1)
    mesh_file = mesh_elem.attrib["filename"]
    if not os.path.isabs(mesh_file):
        mesh_file = os.path.join(os.path.dirname(urdf_path), mesh_file)

    # 2c) Visual origin
    orig = link.find("visual/origin")
    if orig is not None:
        xyz     = tuple(map(float, orig.attrib.get("xyz","0 0 0").split()))
        rpy_rad = tuple(map(float, orig.attrib.get("rpy","0 0 0").split()))
    else:
        xyz, rpy_rad = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    rpy_deg = tuple(math.degrees(a) for a in rpy_rad)

    # 3) Wrapper Xform
    obj_id       = name or os.path.splitext(os.path.basename(urdf_path))[0]
    wrapper_path = f"/World/{obj_id}"
    wrapper      = UsdGeom.Xform.Define(stage, wrapper_path)
    wapi         = UsdGeom.XformCommonAPI(wrapper)
    wapi.SetTranslate(translate)
    wapi.SetRotate(rotate)

    # 4) 'visual' child Xform
    vis_path = f"{wrapper_path}/visual"
    vis_xf   = UsdGeom.Xform.Define(stage, vis_path)
    vapi     = UsdGeom.XformCommonAPI(vis_xf)
    vapi.SetTranslate(xyz)
    vapi.SetRotate(rpy_deg)

    # 5) Reference the mesh
    prim = stage.GetPrimAtPath(vis_path)
    prim.GetReferences().AddReference(
        Sdf.Reference(assetPath=mesh_file)
    )

    # 6) Recursively tag for collision
    def _tag_collision(p):
        for c in p.GetChildren():
            if c.GetTypeName() == "Mesh":
                UsdPhysics.CollisionAPI.Apply(c)
            _tag_collision(c)
    _tag_collision(vis_xf.GetPrim())

    # 7a) Apply RigidBodyAPI (dynamic by default) so it’s driven by PhysX :contentReference[oaicite:0]{index=0}
    UsdPhysics.RigidBodyAPI.Apply(wrapper.GetPrim())

    # 7b) Apply MassAPI so it has the correct mass :contentReference[oaicite:1]{index=1}
    mass_api = UsdPhysics.MassAPI.Apply(wrapper.GetPrim())
    mass_api.CreateMassAttr().Set(mass_val)

    return wrapper_path

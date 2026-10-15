# desk_loader.py
from pxr import UsdGeom, Gf, UsdPhysics
import omni.kit.commands

def load_desk(
    stage,
    translate=(2.0, 0.0, 0.0),
    rotate_z=180.0,
    size=(1.0, 0.6, 0.05),
    color=(0.55, 0.27, 0.07),
    leg_height=0.75,
    leg_radius=0.02,
    leg_color=(0.55, 0.27, 0.07),
    margin=0.05,
):
    """
    :param stage:      USD Stage
    :param translate:  (x,y,z) of table root
    :param rotate_z:   yaw in degrees
    :param size:       (length, width, thickness)
    :param color:      tabletop RGB
    :param leg_height: height of each leg
    :param leg_radius: cylinder radius of legs
    :param leg_color:  RGB color of legs
    :param margin:     inset from table-edge to leg
    """
    # ─── root Xform ─────────────────────────────────────────────────────────
    table = UsdGeom.Xform.Define(stage, "/World/Table")
    table_xform = UsdGeom.XformCommonAPI(table)
    table_xform.SetTranslate(translate)
    table_xform.SetRotate((0.0, 0.0, rotate_z))

    # ─── make it a static rigid body ────────────────────────────────────────
    omni.kit.commands.execute(
        "PhysicsAddRigidBody",
        path=str(table.GetPrim().GetPath()),
        bodyType="static",
        collision=True,
        enableCCD=False,
        mass=0.0
    )

    # ─── tabletop ───────────────────────────────────────────────────────────
    top = UsdGeom.Cube.Define(stage, "/World/Table/TableTop")
    top_xform = UsdGeom.XformCommonAPI(top)
    top_xform.SetScale(size)
    top_xform.SetTranslate((size[0] * 0.5, 0.0, leg_height + size[2] * 0.5))
    top.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    # ensure collision matches visual
    UsdPhysics.CollisionAPI.Apply(top.GetPrim())

    # ─── legs ───────────────────────────────────────────────────────────────
    x_offs = [size[0] - margin, margin]
    y_offs = [ size[1] / 2 - margin, -(size[1] / 2 - margin) ]
    for i, (x, y) in enumerate((x, y) for x in x_offs for y in y_offs):
        leg = UsdGeom.Cylinder.Define(stage, f"/World/Table/Leg{i}")
        leg_xform = UsdGeom.XformCommonAPI(leg)
        leg.GetHeightAttr().Set(leg_height)
        leg.GetRadiusAttr().Set(leg_radius)
        leg_xform.SetTranslate((x, y, leg_height * 0.5))
        leg.CreateDisplayColorAttr().Set([Gf.Vec3f(*leg_color)])
        UsdPhysics.CollisionAPI.Apply(leg.GetPrim())

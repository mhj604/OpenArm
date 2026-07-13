import asyncio
import shutil
from datetime import datetime

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics


USD_PATH = "/home/mh/Openarm/openarm_test_v2.usd"
TABLE_ROOT = "/World/packing_table/OpenArmMountTable_v2"
SUPPORT_ROOT = "/World/openarm_base_support_table"


def set_vec3_attr(prim, attr_name, values):
    attr = prim.GetAttribute(attr_name)
    if not attr:
        raise RuntimeError(f"missing attr: {prim.GetPath()}.{attr_name}")
    attr.Set(Gf.Vec3d(*values))


def define_cube(stage, path, translate, scale, color):
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(Sdf.Path(path))
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*translate))
    cube.AddScaleOp().Set(Gf.Vec3d(*scale))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


async def main():
    app = omni.kit.app.get_app()
    for _ in range(5):
        await app.next_update_async()

    backup_path = f"/home/mh/Openarm/openarm_test_v2_before_image_style_desk_apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.usd"
    shutil.copy2(USD_PATH, backup_path)
    print(f"[image_desk] backup={backup_path}", flush=True)

    result, error = await omni.usd.get_context().open_stage_async(USD_PATH)
    print(f"[image_desk] open_stage result={result} error={error} path={USD_PATH}", flush=True)
    for _ in range(10):
        await app.next_update_async()

    stage = omni.usd.get_context().get_stage()
    table = stage.GetPrimAtPath(TABLE_ROOT)
    if not table.IsValid():
        raise RuntimeError(f"missing table root: {TABLE_ROOT}")

    support = stage.GetPrimAtPath(SUPPORT_ROOT)
    print(f"[image_desk] support_untouched={support.IsValid()} path={SUPPORT_ROOT}", flush=True)

    # Remove the high U-shaped strips around the robot. The image-style desk is a
    # lower work surface, while /World/openarm_base_support_table keeps supporting the robot.
    for child_name in (
        "TopSurface_Back",
        "TopSurface_Front",
        "TopSurface_Left",
        "TopSurface_Right",
        "TopSurface_Front_PositiveSide",
        "TopSurface_Back_PositiveSide",
    ):
        path = f"{TABLE_ROOT}/{child_name}"
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(Sdf.Path(path))
            print(f"[image_desk] removed_high_strip={path}", flush=True)

    wood = (0.72, 0.60, 0.42)
    dark_leg = (0.22, 0.22, 0.20)

    # Parent transform maps local y to the world front/back axis and local x to
    # the world left/right axis. These dimensions create a low, wide desk surface.
    define_cube(
        stage,
        f"{TABLE_ROOT}/ImageStyleDesktop",
        translate=(0.0, 0.20, 0.5350),
        scale=(2.80, 1.42, 0.060),
        color=wood,
    )

    # Four legs under the new desktop, kept away from the robot center.
    leg_specs = {
        "Leg_1": (-1.18, -0.50, 0.245),
        "Leg_2": (-1.18, 0.90, 0.245),
        "Leg_3": (1.18, -0.50, 0.245),
        "Leg_4": (1.18, 0.90, 0.245),
    }
    for name, translate in leg_specs.items():
        path = f"{TABLE_ROOT}/{name}"
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            set_vec3_attr(prim, "xformOp:translate", translate)
            set_vec3_attr(prim, "xformOp:scale", (0.075, 0.075, 0.490))
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*dark_leg)])
            print(f"[image_desk] adjusted_leg={path}", flush=True)
        else:
            define_cube(stage, path, translate, (0.075, 0.075, 0.490), dark_leg)
            print(f"[image_desk] created_leg={path}", flush=True)

    stage.GetRootLayer().Save()
    print(f"[image_desk] saved={USD_PATH}", flush=True)
    app.post_quit()


asyncio.ensure_future(main())

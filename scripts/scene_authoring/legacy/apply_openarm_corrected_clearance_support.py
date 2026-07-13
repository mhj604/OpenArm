import asyncio
import shutil
from datetime import datetime

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics


USD_PATH = "/home/mh/Openarm/openarm_test_v2.usd"

# Important: these USD names do not match the user's visual front/back.
# User visual front  == TopSurface_Back  -> keep it.
# User visual back   == TopSurface_Front -> remove it.
USER_VISUAL_BACK_STRIP = "/World/packing_table/OpenArmMountTable_v2/TopSurface_Front"

SIDE_TARGETS = {
    "/World/packing_table/OpenArmMountTable_v2/TopSurface_Left": {
        "translate": (-1.2167, 0.0, 0.9616),
        "scale": (0.3667, 0.68, 0.065),
    },
    "/World/packing_table/OpenArmMountTable_v2/TopSurface_Right": {
        "translate": (1.2167, 0.0, 0.9616),
        "scale": (0.3667, 0.68, 0.065),
    },
    "/World/packing_table/OpenArmMountTable_v2/Leg_1": {
        "translate": (-1.2167, -0.281, 0.46455),
    },
    "/World/packing_table/OpenArmMountTable_v2/Leg_2": {
        "translate": (-1.2167, 0.281, 0.46455),
    },
    "/World/packing_table/OpenArmMountTable_v2/Leg_3": {
        "translate": (1.2167, -0.281, 0.46455),
    },
    "/World/packing_table/OpenArmMountTable_v2/Leg_4": {
        "translate": (1.2167, 0.281, 0.46455),
    },
}

SUPPORT_ROOT = "/World/openarm_base_support_table"


def set_attr_vec3(prim, attr_name, values):
    attr = prim.GetAttribute(attr_name)
    if not attr:
        raise RuntimeError(f"missing attr {prim.GetPath()}.{attr_name}")
    attr.Set(Gf.Vec3d(*values))


def add_support_cube(stage, path, translate, scale, color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*translate))
    cube.AddScaleOp().Set(Gf.Vec3d(*scale))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def create_robot_support_table(stage):
    if stage.GetPrimAtPath(SUPPORT_ROOT).IsValid():
        stage.RemovePrim(Sdf.Path(SUPPORT_ROOT))

    UsdGeom.Xform.Define(stage, SUPPORT_ROOT)
    color = (0.28, 0.28, 0.26)

    # Robot bbox starts at z ~= 0.3500, so the top plate reaches that height.
    add_support_cube(
        stage,
        f"{SUPPORT_ROOT}/TopPlate",
        translate=(-0.03, 0.0, 0.33),
        scale=(0.36, 0.56, 0.04),
        color=color,
    )

    leg_positions = [
        (-0.17, -0.24, 0.155),
        (-0.17, 0.24, 0.155),
        (0.11, -0.24, 0.155),
        (0.11, 0.24, 0.155),
    ]
    for index, pos in enumerate(leg_positions, start=1):
        add_support_cube(
            stage,
            f"{SUPPORT_ROOT}/Leg_{index}",
            translate=pos,
            scale=(0.045, 0.045, 0.31),
            color=color,
        )


async def main():
    app = omni.kit.app.get_app()
    for _ in range(5):
        await app.next_update_async()

    backup_path = f"/tmp/openarm_test_v2_before_corrected_clearance_support_{datetime.now().strftime('%Y%m%d_%H%M%S')}.usd"
    shutil.copy2(USD_PATH, backup_path)
    print(f"[correct_clearance] backup={backup_path}", flush=True)

    result, error = await omni.usd.get_context().open_stage_async(USD_PATH)
    print(f"[correct_clearance] open_stage result={result} error={error} path={USD_PATH}", flush=True)
    for _ in range(10):
        await app.next_update_async()

    stage = omni.usd.get_context().get_stage()

    back_strip = stage.GetPrimAtPath(USER_VISUAL_BACK_STRIP)
    if back_strip.IsValid():
        removed = stage.RemovePrim(Sdf.Path(USER_VISUAL_BACK_STRIP))
    else:
        removed = False
    print(f"[correct_clearance] removed_user_visual_back={removed} path={USER_VISUAL_BACK_STRIP}", flush=True)

    for path, values in SIDE_TARGETS.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"missing side/leg prim: {path}")
        if "translate" in values:
            set_attr_vec3(prim, "xformOp:translate", values["translate"])
        if "scale" in values:
            set_attr_vec3(prim, "xformOp:scale", values["scale"])
        print(f"[correct_clearance] adjusted={path}", flush=True)

    create_robot_support_table(stage)
    print(f"[correct_clearance] support_created={SUPPORT_ROOT}", flush=True)

    stage.GetRootLayer().Save()
    print(f"[correct_clearance] saved={USD_PATH}", flush=True)
    app.post_quit()


asyncio.ensure_future(main())

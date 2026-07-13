import asyncio
import shutil
from datetime import datetime

import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics


USD_PATH = "/home/mh/Openarm/openarm_test_v2.usd"
TABLE_ROOT = "/World/packing_table/OpenArmMountTable_v2"
CONTAINER_PATH = "/World/packing_table/container_h20"
SUPPORT_ROOT = "/World/openarm_base_support_table"

# World-space design target:
# - Robot support table remains low: z 0.00 .. 0.35.
# - Basket table is separate and higher: tabletop top ~= 0.5965.
# - Basket is centered on the high tabletop.
HIGH_TABLE_WORLD_CENTER_X = 0.52
HIGH_TABLE_WORLD_CENTER_Y = 0.50
HIGH_TABLE_WORLD_SIZE_X = 0.60
HIGH_TABLE_WORLD_SIZE_Y = 0.70
HIGH_TABLE_LOCAL_Z = 0.9616
HIGH_TABLE_LOCAL_THICKNESS_Z = 0.065


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


def world_xy_to_table_local(world_x, world_y):
    # /World/packing_table has translate=(0.2, 0, 0), scale=(0.6, 1, 0.6),
    # orient=90deg around z. For direct children:
    # world_x = 0.2 - local_y
    # world_y = local_x * 0.6
    return (world_y / 0.6, 0.2 - world_x)


async def main():
    app = omni.kit.app.get_app()
    for _ in range(5):
        await app.next_update_async()

    backup_path = f"/home/mh/Openarm/openarm_test_v2_before_separate_high_table_apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.usd"
    shutil.copy2(USD_PATH, backup_path)
    print(f"[separate_layout] backup={backup_path}", flush=True)

    result, error = await omni.usd.get_context().open_stage_async(USD_PATH)
    print(f"[separate_layout] open_stage result={result} error={error} path={USD_PATH}", flush=True)
    for _ in range(10):
        await app.next_update_async()

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(SUPPORT_ROOT).IsValid():
        raise RuntimeError(f"missing low robot support table: {SUPPORT_ROOT}")
    print(f"[separate_layout] low_support_untouched={SUPPORT_ROOT}", flush=True)

    # Keep only the new high tabletop and legs under the table root.
    for child_name in (
        "TopSurface_Back",
        "TopSurface_Front",
        "TopSurface_Left",
        "TopSurface_Right",
        "TopSurface_Front_PositiveSide",
        "TopSurface_Back_PositiveSide",
        "ImageStyleDesktop",
    ):
        path = f"{TABLE_ROOT}/{child_name}"
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(Sdf.Path(path))
            print(f"[separate_layout] removed_old_table_piece={path}", flush=True)

    wood = (0.72, 0.60, 0.42)
    dark_leg = (0.22, 0.22, 0.20)

    local_x, local_y = world_xy_to_table_local(HIGH_TABLE_WORLD_CENTER_X, HIGH_TABLE_WORLD_CENTER_Y)
    define_cube(
        stage,
        f"{TABLE_ROOT}/HighBasketTabletop",
        translate=(local_x, local_y, HIGH_TABLE_LOCAL_Z),
        scale=(HIGH_TABLE_WORLD_SIZE_Y / 0.6, HIGH_TABLE_WORLD_SIZE_X, HIGH_TABLE_LOCAL_THICKNESS_Z),
        color=wood,
    )
    print(f"[separate_layout] created_high_tabletop={TABLE_ROOT}/HighBasketTabletop", flush=True)

    # Leg top reaches the tabletop underside, matching the old high table height.
    leg_world_positions = {
        "Leg_1": (0.25, 0.20),
        "Leg_2": (0.79, 0.20),
        "Leg_3": (0.25, 0.80),
        "Leg_4": (0.79, 0.80),
    }
    for name, (world_x, world_y) in leg_world_positions.items():
        leg_local_x, leg_local_y = world_xy_to_table_local(world_x, world_y)
        path = f"{TABLE_ROOT}/{name}"
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            set_vec3_attr(prim, "xformOp:translate", (leg_local_x, leg_local_y, 0.46455))
            set_vec3_attr(prim, "xformOp:scale", (0.075, 0.075, 0.9291))
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*dark_leg)])
            print(f"[separate_layout] adjusted_high_table_leg={path}", flush=True)
        else:
            define_cube(stage, path, (leg_local_x, leg_local_y, 0.46455), (0.075, 0.075, 0.9291), dark_leg)
            print(f"[separate_layout] created_high_table_leg={path}", flush=True)

    # Move the existing basket onto the high tabletop. Geometry bottom remains
    # at z ~= 0.5955, matching tabletop top ~= 0.5965.
    container = stage.GetPrimAtPath(CONTAINER_PATH)
    if not container.IsValid():
        raise RuntimeError(f"missing container: {CONTAINER_PATH}")
    basket_local_x, basket_local_y = world_xy_to_table_local(HIGH_TABLE_WORLD_CENTER_X, HIGH_TABLE_WORLD_CENTER_Y)
    set_vec3_attr(container, "xformOp:translate", (basket_local_x, basket_local_y, 0.9925656570552239))
    print(f"[separate_layout] moved_basket={CONTAINER_PATH}", flush=True)

    stage.GetRootLayer().Save()
    print(f"[separate_layout] saved={USD_PATH}", flush=True)
    app.post_quit()


asyncio.ensure_future(main())

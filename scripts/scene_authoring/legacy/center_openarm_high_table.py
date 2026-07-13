import asyncio
import shutil
from datetime import datetime

import omni.kit.app
import omni.usd
from pxr import Gf


USD_PATH = "/home/mh/Openarm/openarm_test_v2.usd"
TABLE_ROOT = "/World/packing_table/OpenArmMountTable_v2"
CONTAINER_PATH = "/World/packing_table/container_h20"
SUPPORT_ROOT = "/World/openarm_base_support_table"

# Keep the current forward gap and width. Only center the high table on robot y=0.
TABLE_CENTER_X = 0.48
TABLE_CENTER_Y = 0.0
TABLE_SIZE_X = 0.60
TABLE_SIZE_Y = 1.10

# Preserve the basket's offset on robot-left side after centering the table.
BASKET_CENTER_X = 0.48
BASKET_CENTER_Y = 0.20


def world_xy_to_table_local(world_x, world_y):
    # /World/packing_table has translate=(0.2, 0, 0), scale=(0.6, 1, 0.6),
    # orient=90deg around z. For direct children:
    # world_x = 0.2 - local_y
    # world_y = local_x * 0.6
    return (world_y / 0.6, 0.2 - world_x)


def set_vec3_attr(prim, attr_name, values):
    attr = prim.GetAttribute(attr_name)
    if not attr:
        raise RuntimeError(f"missing attr: {prim.GetPath()}.{attr_name}")
    attr.Set(Gf.Vec3d(*values))


async def main():
    app = omni.kit.app.get_app()
    for _ in range(5):
        await app.next_update_async()

    backup_path = f"/home/mh/Openarm/openarm_test_v2_before_center_high_table_apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.usd"
    shutil.copy2(USD_PATH, backup_path)
    print(f"[center_table] backup={backup_path}", flush=True)

    result, error = await omni.usd.get_context().open_stage_async(USD_PATH)
    print(f"[center_table] open_stage result={result} error={error} path={USD_PATH}", flush=True)
    for _ in range(10):
        await app.next_update_async()

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(SUPPORT_ROOT).IsValid():
        raise RuntimeError(f"missing low robot support table: {SUPPORT_ROOT}")

    tabletop = stage.GetPrimAtPath(f"{TABLE_ROOT}/HighBasketTabletop")
    if not tabletop.IsValid():
        raise RuntimeError(f"missing high tabletop: {TABLE_ROOT}/HighBasketTabletop")

    table_local_x, table_local_y = world_xy_to_table_local(TABLE_CENTER_X, TABLE_CENTER_Y)
    set_vec3_attr(tabletop, "xformOp:translate", (table_local_x, table_local_y, 0.9616))
    set_vec3_attr(tabletop, "xformOp:scale", (TABLE_SIZE_Y / 0.6, TABLE_SIZE_X, 0.065))
    print("[center_table] centered_high_tabletop", flush=True)

    leg_world_positions = {
        "Leg_1": (TABLE_CENTER_X - 0.27, TABLE_CENTER_Y - 0.50),
        "Leg_2": (TABLE_CENTER_X + 0.27, TABLE_CENTER_Y - 0.50),
        "Leg_3": (TABLE_CENTER_X - 0.27, TABLE_CENTER_Y + 0.50),
        "Leg_4": (TABLE_CENTER_X + 0.27, TABLE_CENTER_Y + 0.50),
    }
    for name, (world_x, world_y) in leg_world_positions.items():
        prim = stage.GetPrimAtPath(f"{TABLE_ROOT}/{name}")
        if not prim.IsValid():
            raise RuntimeError(f"missing high table leg: {TABLE_ROOT}/{name}")
        leg_local_x, leg_local_y = world_xy_to_table_local(world_x, world_y)
        set_vec3_attr(prim, "xformOp:translate", (leg_local_x, leg_local_y, 0.46455))
        set_vec3_attr(prim, "xformOp:scale", (0.075, 0.075, 0.9291))
        print(f"[center_table] centered_leg={prim.GetPath()}", flush=True)

    basket = stage.GetPrimAtPath(CONTAINER_PATH)
    if not basket.IsValid():
        raise RuntimeError(f"missing basket: {CONTAINER_PATH}")
    basket_local_x, basket_local_y = world_xy_to_table_local(BASKET_CENTER_X, BASKET_CENTER_Y)
    set_vec3_attr(basket, "xformOp:translate", (basket_local_x, basket_local_y, 0.9925656570552239))
    print(f"[center_table] moved_basket_with_table={CONTAINER_PATH}", flush=True)

    stage.GetRootLayer().Save()
    print(f"[center_table] saved={USD_PATH}", flush=True)
    app.post_quit()


asyncio.ensure_future(main())

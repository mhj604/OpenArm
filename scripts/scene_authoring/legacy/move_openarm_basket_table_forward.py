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

# Low robot support table bbox currently ends at world x ~= 0.1500.
# Put the high basket table at world x min ~= 0.1800, leaving about 3 cm gap.
HIGH_TABLE_CENTER_X = 0.48
HIGH_TABLE_CENTER_Y = 0.0
HIGH_TABLE_SIZE_X = 0.60
HIGH_TABLE_SIZE_Y = 0.70


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

    backup_path = f"/home/mh/Openarm/openarm_test_v2_before_forward_adjust_apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.usd"
    shutil.copy2(USD_PATH, backup_path)
    print(f"[move_forward] backup={backup_path}", flush=True)

    result, error = await omni.usd.get_context().open_stage_async(USD_PATH)
    print(f"[move_forward] open_stage result={result} error={error} path={USD_PATH}", flush=True)
    for _ in range(10):
        await app.next_update_async()

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(SUPPORT_ROOT).IsValid():
        raise RuntimeError(f"missing low robot support table: {SUPPORT_ROOT}")

    tabletop = stage.GetPrimAtPath(f"{TABLE_ROOT}/HighBasketTabletop")
    if not tabletop.IsValid():
        raise RuntimeError(f"missing high tabletop: {TABLE_ROOT}/HighBasketTabletop")

    local_x, local_y = world_xy_to_table_local(HIGH_TABLE_CENTER_X, HIGH_TABLE_CENTER_Y)
    set_vec3_attr(tabletop, "xformOp:translate", (local_x, local_y, 0.9616))
    set_vec3_attr(tabletop, "xformOp:scale", (HIGH_TABLE_SIZE_Y / 0.6, HIGH_TABLE_SIZE_X, 0.065))
    print("[move_forward] moved_high_tabletop", flush=True)

    leg_world_positions = {
        "Leg_1": (HIGH_TABLE_CENTER_X - 0.27, HIGH_TABLE_CENTER_Y - 0.30),
        "Leg_2": (HIGH_TABLE_CENTER_X + 0.27, HIGH_TABLE_CENTER_Y - 0.30),
        "Leg_3": (HIGH_TABLE_CENTER_X - 0.27, HIGH_TABLE_CENTER_Y + 0.30),
        "Leg_4": (HIGH_TABLE_CENTER_X + 0.27, HIGH_TABLE_CENTER_Y + 0.30),
    }
    for name, (world_x, world_y) in leg_world_positions.items():
        prim = stage.GetPrimAtPath(f"{TABLE_ROOT}/{name}")
        if not prim.IsValid():
            raise RuntimeError(f"missing high table leg: {TABLE_ROOT}/{name}")
        leg_local_x, leg_local_y = world_xy_to_table_local(world_x, world_y)
        set_vec3_attr(prim, "xformOp:translate", (leg_local_x, leg_local_y, 0.46455))
        set_vec3_attr(prim, "xformOp:scale", (0.075, 0.075, 0.9291))
        print(f"[move_forward] moved_leg={prim.GetPath()}", flush=True)

    container = stage.GetPrimAtPath(CONTAINER_PATH)
    if not container.IsValid():
        raise RuntimeError(f"missing basket: {CONTAINER_PATH}")
    basket_local_x, basket_local_y = world_xy_to_table_local(HIGH_TABLE_CENTER_X, HIGH_TABLE_CENTER_Y)
    set_vec3_attr(container, "xformOp:translate", (basket_local_x, basket_local_y, 0.9925656570552239))
    print(f"[move_forward] moved_basket={CONTAINER_PATH}", flush=True)

    stage.GetRootLayer().Save()
    print(f"[move_forward] saved={USD_PATH}", flush=True)
    app.post_quit()


asyncio.ensure_future(main())

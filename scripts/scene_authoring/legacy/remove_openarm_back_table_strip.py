import asyncio
import shutil
from datetime import datetime

import omni.kit.app
import omni.usd
from pxr import Sdf


USD_PATH = "/home/mh/Openarm/openarm_test_v2.usd"
REMOVE_PATH = "/World/packing_table/OpenArmMountTable_v2/TopSurface_Back"


async def main():
    app = omni.kit.app.get_app()
    for _ in range(5):
        await app.next_update_async()

    backup_path = f"/tmp/openarm_test_v2_before_remove_back_strip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.usd"
    shutil.copy2(USD_PATH, backup_path)
    print(f"[remove_back_strip] backup={backup_path}", flush=True)

    result, error = await omni.usd.get_context().open_stage_async(USD_PATH)
    print(f"[remove_back_strip] open_stage result={result} error={error} path={USD_PATH}", flush=True)
    for _ in range(10):
        await app.next_update_async()

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(REMOVE_PATH)
    if not prim.IsValid():
        print(f"[remove_back_strip] missing={REMOVE_PATH}", flush=True)
        app.post_quit()
        return

    removed = stage.RemovePrim(Sdf.Path(REMOVE_PATH))
    stage.GetRootLayer().Save()
    print(f"[remove_back_strip] removed={removed} path={REMOVE_PATH}", flush=True)
    print(f"[remove_back_strip] saved={USD_PATH}", flush=True)
    app.post_quit()


asyncio.ensure_future(main())

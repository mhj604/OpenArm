"""
Convert existing DP3 zarr (action = leader qpos, 8-dim)
into a new zarr where:
    agent_pos (state) = follower qpos                    (8)
    action            = follower [qpos, qvel, torque]    (24), shifted by +1 step

MEMORY-EFFICIENT: processes one episode at a time, streams img directly
from source zarr to destination zarr without loading everything into RAM.
"""

import argparse
from pathlib import Path
import numpy as np
import zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src = zarr.open(str(args.src), mode="r")
    src_data = src["data"]
    src_meta = src["meta"]

    state    = src_data["state"][:]
    velocity = src_data["velocity"][:]
    torque   = src_data["torque"][:]
    ends     = src_meta["episode_ends"][:]
    N = len(state)

    src_pcd = src_data["point_cloud"]
    src_img = src_data["img"]

    print(f"[src] frames={N} episodes={len(ends)}")

    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open(str(dst_path), mode="w")
    d = dst.require_group("data")
    m = dst.require_group("meta")

    pcd_shape = src_pcd.shape[1:]
    img_shape = src_img.shape[1:]

    d_state  = d.zeros("state",  shape=(0, 8),  chunks=(1, 8),  dtype=np.float32)
    d_action = d.zeros("action", shape=(0, 24), chunks=(1, 24), dtype=np.float32)
    d_pcd    = d.zeros("point_cloud", shape=(0,) + pcd_shape,
                       chunks=(1,) + pcd_shape, dtype=np.float32)
    d_img    = d.zeros("img", shape=(0,) + img_shape,
                       chunks=(1,) + img_shape, dtype=np.uint8)

    new_ends = []
    cursor = 0
    start = 0

    for ei, e in enumerate(ends):
        end = int(e)
        L = end - start
        if L < 2:
            start = end
            continue

        obs_pos = state[start:end-1]
        act = np.concatenate([
            state[start+1:end],
            velocity[start+1:end],
            torque[start+1:end],
        ], axis=-1).astype(np.float32)

        obs_pcd = src_pcd[start:end-1]
        obs_img = src_img[start:end-1]

        n = L - 1
        d_state.resize((cursor + n, 8));         d_state[cursor:]  = obs_pos
        d_action.resize((cursor + n, 24));       d_action[cursor:] = act
        d_pcd.resize((cursor + n,) + pcd_shape); d_pcd[cursor:]    = obs_pcd
        d_img.resize((cursor + n,) + img_shape); d_img[cursor:]    = obs_img

        cursor += n
        new_ends.append(cursor)
        start = end
        print(f"  episode {ei}: {L} -> {n} frames (total {cursor})")

    m.array("episode_ends", np.array(new_ends, dtype=np.int64),
            chunks=(1,), dtype=np.int64)

    print(f"[dst] frames={cursor} episodes={len(new_ends)}")
    print(f"[dst] action (.., 24)  qpos8 + qvel8 + torque8")
    print(f"[done] saved to {dst_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc


def create_array(group, name, tail, dtype, chunks, compressor):
    return group.zeros(name, shape=(0,) + tuple(tail), chunks=chunks, dtype=dtype, compressor=compressor)


def append(arr, values):
    start = arr.shape[0]
    arr.resize((start + len(values),) + arr.shape[1:])
    arr[start:] = values


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw bimanual collection to DP3 PTV zarr")
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--action-mode", choices=("qpos", "qpos_velocity_torque"), default="qpos")
    parser.add_argument("--batch-frames", type=int, default=128)
    args = parser.parse_args()
    if args.dst.exists():
        raise SystemExit(f"destination already exists; refusing to overwrite: {args.dst}")

    src = zarr.open_group(str(args.src), mode="r")
    data = src["data"]
    ends = np.asarray(src["meta"]["episode_ends"][:], dtype=np.int64)
    for key in ("state", "velocity", "torque", "point_cloud"):
        if key not in data:
            raise SystemExit(f"source is missing data/{key}")
    state_dim = int(data["state"].shape[1])
    action_dim = state_dim if args.action_mode == "qpos" else state_dim * 3
    pc_tail = data["point_cloud"].shape[1:]

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open_group(str(args.dst), mode="w")
    d = dst.require_group("data")
    m = dst.require_group("meta")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    d_state = create_array(d, "state", (state_dim,), np.float32, (256, state_dim), compressor)
    d_action = create_array(d, "action", (action_dim,), np.float32, (256, action_dim), compressor)
    d_pcd = create_array(d, "point_cloud", pc_tail, np.float32, (1,) + pc_tail, compressor)
    d_time = None
    if "timestamp" in data:
        d_time = create_array(d, "timestamp", (), np.float64, (1024,), compressor)

    new_ends = []
    cursor = 0
    start = 0
    for episode, raw_end in enumerate(ends):
        end = int(raw_end)
        if end - start < 2:
            start = end
            continue
        for batch_start in range(start, end - 1, args.batch_frames):
            batch_end = min(end - 1, batch_start + args.batch_frames)
            obs_state = np.asarray(data["state"][batch_start:batch_end], dtype=np.float32)
            nxt = slice(batch_start + 1, batch_end + 1)
            if args.action_mode == "qpos":
                action = np.asarray(data["state"][nxt], dtype=np.float32)
            else:
                action = np.concatenate((data["state"][nxt], data["velocity"][nxt], data["torque"][nxt]), axis=-1).astype(np.float32)
            append(d_state, obs_state)
            append(d_action, action)
            append(d_pcd, np.asarray(data["point_cloud"][batch_start:batch_end], dtype=np.float32))
            if d_time is not None:
                append(d_time, np.asarray(data["timestamp"][batch_start:batch_end], dtype=np.float64))
            cursor += batch_end - batch_start
        new_ends.append(cursor)
        print(f"episode {episode}: {end - start} -> {end - start - 1} frames")
        start = end

    m.array("episode_ends", np.asarray(new_ends, dtype=np.int64), chunks=(128,), dtype=np.int64)
    dst.attrs.update({
        "schema": "openarm_bimanual_dp3_ptv_v1",
        "source": str(args.src.resolve()),
        "action_mode": args.action_mode,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_order": src.attrs.get("state_order", "unknown"),
    })
    print(f"wrote {len(new_ends)} episodes, {cursor} frames, state={state_dim}, action={action_dim}: {args.dst}")


if __name__ == "__main__":
    main()


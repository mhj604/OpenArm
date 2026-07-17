#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an OpenArm bimanual zarr dataset")
    parser.add_argument("path", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = zarr.open_group(str(args.path), mode="r")
    data = root["data"]
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    schema = root.attrs.get("schema", "unknown")
    required = {"state", "action", "point_cloud"}
    if "raw" in schema:
        required |= {"velocity", "torque", "img", "img_hand", "timestamp", "sensor_age_ms"}
    missing = required - set(data.keys())
    errors = []
    if missing:
        errors.append(f"missing arrays: {sorted(missing)}")
    lengths = {name: int(data[name].shape[0]) for name in required if name in data}
    if len(set(lengths.values())) > 1:
        errors.append(f"length mismatch: {lengths}")
    n = min(lengths.values()) if lengths else 0
    if len(ends) == 0 or np.any(np.diff(ends) <= 0) or (len(ends) and int(ends[-1]) != n):
        errors.append(f"invalid episode_ends={ends.tolist()} for n={n}")

    print(f"schema={schema} episodes={len(ends)} frames={n}")
    for name in sorted(data.keys()):
        print(f"data/{name}: shape={data[name].shape} dtype={data[name].dtype}")
    if n:
        sample_indices = np.unique(np.linspace(0, n - 1, min(n, 256)).astype(int))
        for name in ("state", "action", "velocity", "torque", "point_cloud", "sensor_age_ms"):
            if name not in data:
                continue
            values = np.asarray(data[name].get_orthogonal_selection((sample_indices,)))
            if not np.isfinite(values).all():
                errors.append(f"data/{name} contains non-finite sampled values")
            print(f"  {name}: min={float(np.nanmin(values)):.4g} max={float(np.nanmax(values)):.4g}")
        if "timestamp" in data and n > 1:
            ts = np.asarray(data["timestamp"][:], dtype=np.float64)
            dt = np.diff(ts)
            positive = dt[dt > 0]
            if len(positive):
                print(f"  effective_fps_median={1.0 / np.median(positive):.2f}")
        episode_lengths = np.diff(np.concatenate(([0], ends))) if len(ends) else np.array([])
        if len(episode_lengths):
            print(f"  episode_frames min/median/max={episode_lengths.min()}/{np.median(episode_lengths):.1f}/{episode_lengths.max()}")

    if args.strict and len(ends) < 2:
        errors.append("strict validation requires at least two episodes")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(1)
    print("[OK] dataset structure and sampled values are valid")


if __name__ == "__main__":
    main()


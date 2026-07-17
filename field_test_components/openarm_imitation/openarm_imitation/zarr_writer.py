from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc


class EpisodeZarrWriter:
    """Crash-recoverable append-only writer for the DP3 collection schema."""

    def __init__(self, path: str | Path, cfg: dict[str, Any]):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        rec = cfg["recording"]
        pc_n = int(cfg["point_cloud"]["points"])
        img_shape = (int(rec["image_height"]), int(rec["image_width"]), 3)
        clevel = int(rec.get("compression_level", 3))
        numeric_comp = Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.BITSHUFFLE)
        image_comp = Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.NOSHUFFLE)

        self.root = zarr.open_group(str(self.path), mode="a")
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")
        self.arrays = {
            "state": self._array("state", (16,), np.float32, (256, 16), numeric_comp),
            "velocity": self._array("velocity", (16,), np.float32, (256, 16), numeric_comp),
            "torque": self._array("torque", (16,), np.float32, (256, 16), numeric_comp),
            "action": self._array("action", (16,), np.float32, (256, 16), numeric_comp),
            "exo_action_raw": self._array("exo_action_raw", (16,), np.float32, (256, 16), numeric_comp),
            "point_cloud": self._array("point_cloud", (pc_n, 6), np.float32, (1, pc_n, 6), numeric_comp),
            "img": self._array("img", img_shape, np.uint8, (1,) + img_shape, image_comp),
            "img_hand": self._array("img_hand", img_shape, np.uint8, (1,) + img_shape, image_comp),
            "timestamp": self._array("timestamp", (), np.float64, (1024,), numeric_comp),
            "sensor_age_ms": self._array("sensor_age_ms", (5,), np.float32, (256, 5), numeric_comp),
        }
        if rec.get("save_depth", False):
            depth_shape = img_shape[:2]
            self.arrays["depth_chest_mm"] = self._array(
                "depth_chest_mm", depth_shape, np.uint16, (1,) + depth_shape, numeric_comp
            )
            self.arrays["depth_hand_mm"] = self._array(
                "depth_hand_mm", depth_shape, np.uint16, (1,) + depth_shape, numeric_comp
            )

        self.episode_ends = self._meta_array("episode_ends", np.int64)
        self._repair_lengths()
        if "in_progress_start" in self.root.attrs:
            self._truncate(int(self.root.attrs["in_progress_start"]))
            del self.root.attrs["in_progress_start"]

        clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        self.root.attrs.update({
            "schema": "openarm_bimanual_dp3_raw_v1",
            "state_order": "right_q1..q7,right_gripper,left_q1..q7,left_gripper",
            "action_semantics": "exo_joint_command_raw; convert before DP3 training",
            "config_json": json.dumps(clean_cfg, ensure_ascii=False),
        })
        self._episode_start: int | None = None

    def _array(self, name, tail, dtype, chunks, compressor):
        if name in self.data:
            arr = self.data[name]
            if arr.shape[1:] != tuple(tail):
                raise ValueError(f"existing {name} shape {arr.shape} conflicts with configured tail {tail}")
            return arr
        return self.data.zeros(name, shape=(0,) + tuple(tail), chunks=chunks, dtype=dtype, compressor=compressor)

    def _meta_array(self, name: str, dtype):
        if name in self.meta:
            return self.meta[name]
        return self.meta.zeros(name, shape=(0,), chunks=(128,), dtype=dtype)

    @property
    def frame_count(self) -> int:
        return int(self.arrays["state"].shape[0])

    @property
    def episode_count(self) -> int:
        return int(self.episode_ends.shape[0])

    def _repair_lengths(self) -> None:
        lengths = [int(arr.shape[0]) for arr in self.arrays.values()]
        target = min(lengths)
        if len(set(lengths)) != 1:
            self._truncate(target)
        while self.episode_ends.shape[0] and int(self.episode_ends[-1]) > target:
            self.episode_ends.resize((self.episode_ends.shape[0] - 1,))

    def _truncate(self, length: int) -> None:
        for arr in self.arrays.values():
            arr.resize((length,) + arr.shape[1:])

    def begin_episode(self) -> None:
        if self._episode_start is not None:
            raise RuntimeError("an episode is already in progress")
        self._episode_start = self.frame_count
        self.root.attrs["in_progress_start"] = self._episode_start

    def append(self, sample: dict[str, np.ndarray | float]) -> None:
        if self._episode_start is None:
            raise RuntimeError("begin_episode() must be called first")
        start = self.frame_count
        missing = set(self.arrays) - set(sample)
        if missing:
            raise KeyError(f"sample is missing fields: {sorted(missing)}")
        try:
            for name, arr in self.arrays.items():
                arr.resize((start + 1,) + arr.shape[1:])
                arr[start] = sample[name]
        except Exception:
            self._truncate(start)
            raise

    def current_episode_frames(self) -> int:
        return 0 if self._episode_start is None else self.frame_count - self._episode_start

    def commit_episode(self, minimum_frames: int) -> int:
        frames = self.current_episode_frames()
        if self._episode_start is None:
            raise RuntimeError("no episode is in progress")
        if frames < minimum_frames:
            raise ValueError(f"episode has {frames} frames; minimum is {minimum_frames}")
        idx = self.episode_ends.shape[0]
        self.episode_ends.resize((idx + 1,))
        self.episode_ends[idx] = self.frame_count
        self._episode_start = None
        del self.root.attrs["in_progress_start"]
        return frames

    def discard_episode(self) -> int:
        frames = self.current_episode_frames()
        if self._episode_start is not None:
            self._truncate(self._episode_start)
            self._episode_start = None
            if "in_progress_start" in self.root.attrs:
                del self.root.attrs["in_progress_start"]
        return frames


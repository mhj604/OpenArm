#!/usr/bin/env python3
from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from openarm_imitation.can_reader import PassiveCANReader
from openarm_imitation.config import load_config
from openarm_imitation.exo_subscriber import ExoJointSubscriber
from openarm_imitation.pointcloud import depth_to_pointcloud, resize_rgb_nearest
from openarm_imitation.realsense_reader import RealSenseReader, list_devices, resolve_camera_serials
from openarm_imitation.zarr_writer import EpisodeZarrWriter


class Keyboard:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("interactive collection requires a terminal on stdin")
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def __exit__(self, *_args):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def _depth_mm(depth_m: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = resize_rgb_nearest(depth_m, width, height)
    return np.clip(np.rint(resized * 1000.0), 0, 65535).astype(np.uint16)


def synthetic_collect(cfg: dict, output: Path, episodes: int, frames: int, task: str) -> None:
    writer = EpisodeZarrWriter(output, cfg)
    writer.root.attrs["task"] = task
    rec = cfg["recording"]
    h, w = int(rec["image_height"]), int(rec["image_width"])
    n_points = int(cfg["point_cloud"]["points"])
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:h, 0:w]

    for episode in range(episodes):
        writer.begin_episode()
        for frame in range(frames):
            t = frame / float(cfg["recording"]["fps"])
            phase = np.arange(16, dtype=np.float32) * 0.17 + episode * 0.1
            state = (30.0 * np.sin(t + phase)).astype(np.float32)
            velocity = (30.0 * np.cos(t + phase)).astype(np.float32)
            torque = (0.5 * np.sin(0.5 * t + phase)).astype(np.float32)
            xyz = rng.uniform((-0.35, -0.25, 0.25), (0.35, 0.25, 1.0), size=(n_points, 3)).astype(np.float32)
            rgb_pc = rng.random((n_points, 3), dtype=np.float32)
            pcd = np.concatenate((xyz, rgb_pc), axis=1)
            chest = np.stack(((xx + frame) % 256, (yy * 2) % 256, np.full_like(xx, episode * 20)), axis=-1).astype(np.uint8)
            hand = np.stack(((yy + frame) % 256, (xx * 2) % 256, np.full_like(xx, 80)), axis=-1).astype(np.uint8)
            sample = {
                "state": state,
                "velocity": velocity,
                "torque": torque,
                "action": state.copy(),
                "exo_action_raw": np.radians(state).astype(np.float32),
                "point_cloud": pcd,
                "img": chest,
                "img_hand": hand,
                "timestamp": time.time() + t,
                "sensor_age_ms": np.zeros(5, dtype=np.float32),
            }
            if rec.get("save_depth", False):
                sample["depth_chest_mm"] = np.full((h, w), 600, dtype=np.uint16)
                sample["depth_hand_mm"] = np.full((h, w), 250, dtype=np.uint16)
            writer.append(sample)
        count = writer.commit_episode(int(rec["minimum_episode_frames"]))
        print(f"[synthetic] episode {writer.episode_count - 1}: {count} frames")
    print(f"[synthetic] wrote {writer.episode_count} episodes to {output}")


def hardware_collect(cfg: dict, output: Path, episodes: int, task: str) -> None:
    robot = cfg["robot"]
    rec = cfg["recording"]
    pc_cfg = cfg["point_cloud"]
    devices = list_devices()
    serials = resolve_camera_serials(cfg, devices)
    print("[camera] resolved:", serials)
    print("[safety] CAN readers are receive-only; this process never enables or commands a motor.")

    writer = EpisodeZarrWriter(output, cfg)
    writer.root.attrs["task"] = task
    rng = np.random.default_rng()
    recording = False
    last_frame_numbers = (-1, -1)
    interval = 1.0 / float(rec["fps"])
    max_age = float(rec["max_sensor_age_ms"])
    min_frames = int(rec["minimum_episode_frames"])

    with ExitStack() as stack:
        right = stack.enter_context(PassiveCANReader(robot["right_can"], "right"))
        left = stack.enter_context(PassiveCANReader(robot["left_can"], "left"))
        exo = stack.enter_context(ExoJointSubscriber(robot["exo_topic"], int(robot["exo_expected_positions"])))
        chest = stack.enter_context(RealSenseReader("chest", serials["chest"], cfg["cameras"]["chest"]))
        hand = stack.enter_context(RealSenseReader("hand", serials["hand"], cfg["cameras"]["hand"]))

        print("[preflight] waiting for 8 motor feedback IDs on each arm and a 16-position ExoArm message...")
        if not right.wait_until_ready(12.0):
            raise RuntimeError(f"no complete right-arm feedback on {robot['right_can']}; start the existing bridge first")
        if not left.wait_until_ready(12.0):
            raise RuntimeError(f"no complete left-arm feedback on {robot['left_can']}; start the existing bridge first")
        if not exo.wait_until_ready(12.0):
            raise RuntimeError(f"no valid {robot['exo_topic']} message; start ExoArm ROS2/HMI first")

        print("\nSPACE: start/commit episode | d: discard current | q: quit")
        next_tick = time.monotonic()
        with Keyboard() as keyboard:
            while writer.episode_count < episodes:
                key = keyboard.read()
                if key == " ":
                    if not recording:
                        writer.begin_episode()
                        recording = True
                        print(f"\n[REC] episode {writer.episode_count} started")
                    else:
                        try:
                            count = writer.commit_episode(min_frames)
                        except ValueError as exc:
                            print(f"\n[REC] {exc}; continue or press d")
                        else:
                            recording = False
                            print(f"\n[SAVED] {count} frames ({writer.episode_count}/{episodes})")
                elif key in ("d", "D"):
                    if recording:
                        count = writer.discard_episode()
                        recording = False
                        print(f"\n[DISCARDED] {count} frames")
                elif key in ("q", "Q"):
                    if recording:
                        if writer.current_episode_frames() >= min_frames:
                            count = writer.commit_episode(min_frames)
                            print(f"\n[SAVED] {count} frames before quit")
                        else:
                            count = writer.discard_episode()
                            print(f"\n[DISCARDED] short episode ({count} frames)")
                    break

                now = time.monotonic()
                if now < next_tick:
                    time.sleep(min(0.002, next_tick - now))
                    continue
                next_tick += interval
                if next_tick < now - interval:
                    next_tick = now + interval

                rsnap, lsnap, esnap = right.snapshot(), left.snapshot(), exo.snapshot()
                csnap, hsnap = chest.snapshot(), hand.snapshot()
                if csnap is None or hsnap is None:
                    continue
                ages = np.asarray([
                    (now - rsnap.timestamp) * 1000,
                    (now - lsnap.timestamp) * 1000,
                    (now - esnap.timestamp) * 1000,
                    (now - csnap.timestamp) * 1000,
                    (now - hsnap.timestamp) * 1000,
                ], dtype=np.float32)
                valid = (
                    np.isfinite(rsnap.position).all()
                    and np.isfinite(lsnap.position).all()
                    and np.isfinite(esnap.position).all()
                    and float(ages.max()) <= max_age
                )
                status = "REC" if recording else "READY"
                print(
                    f"\r[{status}] ep={writer.episode_count}/{episodes} "
                    f"frames={writer.current_episode_frames():4d} ages(ms)={np.round(ages).astype(int).tolist()}",
                    end="", flush=True,
                )
                if not recording or not valid:
                    continue
                frame_numbers = (csnap.frame_number, hsnap.frame_number)
                if frame_numbers == last_frame_numbers:
                    continue
                last_frame_numbers = frame_numbers

                pcd = depth_to_pointcloud(
                    csnap.depth_m, csnap.rgb, csnap.intrinsics,
                    points=int(pc_cfg["points"]), bounds=pc_cfg["bounds_m"],
                    stride=int(pc_cfg["source_stride"]), rng=rng,
                )
                iw, ih = int(rec["image_width"]), int(rec["image_height"])
                state = np.concatenate((rsnap.position, lsnap.position)).astype(np.float32)
                velocity = np.concatenate((rsnap.velocity, lsnap.velocity)).astype(np.float32)
                torque = np.concatenate((rsnap.torque, lsnap.torque)).astype(np.float32)
                sample = {
                    "state": state,
                    "velocity": velocity,
                    "torque": torque,
                    "action": esnap.position.astype(np.float32),
                    "exo_action_raw": esnap.position.astype(np.float32),
                    "point_cloud": pcd,
                    "img": resize_rgb_nearest(csnap.rgb, iw, ih),
                    "img_hand": resize_rgb_nearest(hsnap.rgb, iw, ih),
                    "timestamp": time.time(),
                    "sensor_age_ms": ages,
                }
                if rec.get("save_depth", False):
                    sample["depth_chest_mm"] = _depth_mm(csnap.depth_m, iw, ih)
                    sample["depth_hand_mm"] = _depth_mm(hsnap.depth_m, iw, ih)
                writer.append(sample)
        print()
    print(f"[done] {writer.episode_count} episodes, {writer.frame_count} frames: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Passive bimanual OpenArm-v1 DP3 collector")
    parser.add_argument("--config", default="/home/mh/openarm_imitation/config/hardware.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--task", default="unspecified_task")
    parser.add_argument("--synthetic", action="store_true", help="generate data without touching hardware")
    parser.add_argument("--frames-per-episode", type=int, default=60, help="synthetic mode only")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.synthetic:
        synthetic_collect(cfg, args.output, args.num_episodes, args.frames_per_episode, args.task)
    else:
        hardware_collect(cfg, args.output, args.num_episodes, args.task)


if __name__ == "__main__":
    main()


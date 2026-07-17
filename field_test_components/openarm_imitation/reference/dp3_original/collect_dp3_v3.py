"""
DP3 Data Collection Script for OpenArm (PASSIVE READ version)
- Listens to CAN bus passively (no handshake, no motor control)
- Works alongside OpenArm teleop which controls the motors
- Reads follower state (qpos, qvel, torque) + leader action (qpos)
- Reads RGB + Depth from RealSense cameras
- Saves to zarr format for DP3 training

Usage:
    python collect_dp3_passive.py --output_dir ~/dp3_collect/datasets/pick_place --num_episodes 50

Controls (terminal focus required):
    SPACE : Start/Stop episode recording
    d     : Discard current episode
    q     : Quit (auto-saves if recording)
"""

import argparse
import threading
import time
import sys
import select
import termios
import tty
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import zarr
import can
import pyrealsense2 as rs


# ─── Configuration ────────────────────────────────────────────────────────────

FOLLOWER_PORT = "can2"
LEADER_PORT   = "can0"
CONTROL_HZ    = 30
POINT_CLOUD_N = 1024

# 카메라 시리얼 (--camera 플래그로 선택)
CAMERA_PRESETS = {
    "l515": {
        "serial":       "f1320728",
        "color_w":      640,  "color_h": 480,   # 640x480 @ 30Hz 지원 확인
        "depth_w":      640,  "depth_h": 480,
        "img_w":        640,  "img_h":   480,
    },
    "d435": {
        "serial":       "919122072515",
        "color_w":      640,  "color_h": 480,
        "depth_w":      640,  "depth_h": 480,
        "img_w":        640,  "img_h":   480,
    },
}
IMG_W, IMG_H = 640, 480  # 저장 기본값 (preset에서 덮어씀)

# Auto-start recording when any joint moves more than this (degrees)
# Eliminates the stationary period between pressing SPACE and starting motion.
AUTO_START_THRESH_DEG = 2.0

# motor_name: (send_id, recv_id, motor_type)
# recv_id is the feedback frame ID we listen for on the CAN bus
MOTOR_CONFIG = {
    "joint_1": (0x01, 0x11, "dm8009"),
    "joint_2": (0x02, 0x12, "dm8009"),
    "joint_3": (0x03, 0x13, "dm4340"),
    "joint_4": (0x04, 0x14, "dm4340"),
    "joint_5": (0x05, 0x15, "dm4310"),
    "joint_6": (0x06, 0x16, "dm4310"),
    "joint_7": (0x07, 0x17, "dm4310"),
    "gripper": (0x08, 0x18, "dm4310"),
}

# motor_type: (pmax, vmax, tmax)  -- from LeRobot tables.py MOTOR_LIMIT_PARAMS
MOTOR_LIMITS = {
    "dm8009": (12.5, 45.0, 54.0),
    "dm4340": (12.5, 8.0,  28.0),
    "dm4310": (12.5, 30.0, 10.0),
}

JOINT_NAMES = list(MOTOR_CONFIG.keys())
RECV_ID_TO_IDX = {MOTOR_CONFIG[n][1]: i for i, n in enumerate(JOINT_NAMES)}
RECV_ID_TO_TYPE = {MOTOR_CONFIG[n][1]: MOTOR_CONFIG[n][2] for n in JOINT_NAMES}


# ─── CAN Decode (mirrors LeRobot _decode_motor_state) ──────────────────────────

def uint_to_float(x, x_min, x_max, bits):
    span = x_max - x_min
    return float(x) / ((1 << bits) - 1) * span + x_min


def decode_motor_state(data, motor_type):
    """Returns (position_deg, velocity_deg_s, torque)."""
    if len(data) < 8:
        return None
    q_uint   = (data[1] << 8) | data[2]
    dq_uint  = (data[3] << 4) | (data[4] >> 4)
    tau_uint = ((data[4] & 0x0F) << 8) | data[5]
    pmax, vmax, tmax = MOTOR_LIMITS[motor_type]
    pos_rad = uint_to_float(q_uint,  -pmax, pmax, 16)
    vel_rad = uint_to_float(dq_uint, -vmax, vmax, 12)
    torque  = uint_to_float(tau_uint, -tmax, tmax, 12)
    return np.degrees(pos_rad), np.degrees(vel_rad), torque


# ─── Passive CAN Reader ─────────────────────────────────────────────────────────

class PassiveCANReader:
    """Listens to CAN bus passively and decodes motor feedback frames."""

    def __init__(self, port, name):
        self.port = port
        self.name = name
        self.pos = np.zeros(8, dtype=np.float32)
        self.vel = np.zeros(8, dtype=np.float32)
        self.tor = np.zeros(8, dtype=np.float32)
        self.timestamp = 0.0
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._bus = None

    def connect(self):
        # Open CAN bus in listen mode (no sending, no handshake)
        self._bus = can.interface.Bus(
            channel=self.port,
            interface="socketcan",
            fd=True,
        )
        print(f"[{self.name}] Listening on {self.port} (passive)")

    def disconnect(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._bus:
            self._bus.shutdown()
        print(f"[{self.name}] Stopped")

    def start(self):
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                msg = self._bus.recv(timeout=0.05)
                if msg is None:
                    continue
                rid = msg.arbitration_id
                if rid in RECV_ID_TO_IDX:
                    idx = RECV_ID_TO_IDX[rid]
                    mtype = RECV_ID_TO_TYPE[rid]
                    decoded = decode_motor_state(msg.data, mtype)
                    if decoded is not None:
                        pos, vel, tor = decoded
                        with self.lock:
                            self.pos[idx] = pos
                            self.vel[idx] = vel
                            self.tor[idx] = tor
                            self.timestamp = time.time()
            except Exception as e:
                print(f"[{self.name}] recv error: {e}")

    def get_state(self):
        with self.lock:
            return self.pos.copy(), self.vel.copy(), self.tor.copy(), self.timestamp


# ─── RealSense Camera Reader ───────────────────────────────────────────────────

class CameraReader:
    def __init__(self, serial, name, width=IMG_W, height=IMG_H,
                 color_w=None, color_h=None, fps=30):
        self.serial = serial
        self.name = name
        self.width = width
        self.height = height
        self.color_w = color_w or width   # 카메라에서 받는 color 해상도
        self.color_h = color_h or height
        self.fps = fps
        self.rgb = None
        self.depth = None
        self.timestamp = 0.0
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._pipeline = None
        self._intrinsics = None

    def connect(self):
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        # color_w/color_h: 카메라에서 받는 해상도 (L515는 640x480 color 미지원)
        # width/height:    저장 해상도 (crop 후)
        config.enable_stream(rs.stream.color, self.color_w, self.color_h, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width,   self.height,  rs.format.z16,  self.fps)
        profile = self._pipeline.start(config)
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self._intrinsics = depth_profile.get_intrinsics()
        print(f"[{self.name}] Connected (serial={self.serial})")
        print(f"[{self.name}] Intrinsics: fx={self._intrinsics.fx:.1f} fy={self._intrinsics.fy:.1f} "
              f"cx={self._intrinsics.ppx:.1f} cy={self._intrinsics.ppy:.1f}")
        print(f"[{self.name}] Color stream: {self.color_w}x{self.color_h} -> crop to {self.width}x{self.height}")
        for _ in range(30):  # L515는 warm-up이 더 오래 걸림
            self._pipeline.wait_for_frames(timeout_ms=3000)

    def disconnect(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._pipeline:
            self._pipeline.stop()
        print(f"[{self.name}] Disconnected")

    def start(self):
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        align        = rs.align(rs.stream.color)
        hole_filling = rs.hole_filling_filter()
        spatial      = rs.spatial_filter()    # 공간 평활화
        temporal     = rs.temporal_filter()   # 시간 평활화
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=200)
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if depth_frame:
                    depth_frame = spatial.process(depth_frame)
                    depth_frame = temporal.process(depth_frame)
                    depth_frame = hole_filling.process(depth_frame)
                if not color_frame or not depth_frame:
                    continue
                rgb   = np.asanyarray(color_frame.get_data()).astype(np.uint8)
                depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
                # color/depth가 저장 해상도보다 크면 center crop (L515용)
                if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
                    h, w = rgb.shape[:2]
                    y0 = (h - self.height) // 2
                    x0 = (w - self.width)  // 2
                    rgb   = rgb  [y0:y0+self.height, x0:x0+self.width]
                    depth = depth[y0:y0+self.height, x0:x0+self.width]
                with self.lock:
                    self.rgb = rgb
                    self.depth = depth
                    self.timestamp = time.time()
            except Exception as e:
                print(f"[{self.name}] frame error: {e}")

    def get_frame(self):
        with self.lock:
            if self.rgb is None:
                return None, None, None
            return self.rgb.copy(), self.depth.copy(), self.timestamp

    @property
    def intrinsics(self):
        return self._intrinsics


# ─── Point Cloud ──────────────────────────────────────────────────────────────

def depth_to_pointcloud(depth, rgb, intrinsics, n_points=POINT_CLOUD_N):
    h, w = depth.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    z = depth.astype(np.float32) / 1000.0
    mask = (z > 0.3) & (z < 1.3)        # 작업공간 crop (컵 ~95cm)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    xyz = np.stack([x[mask], y[mask], z[mask]], axis=-1)
    rgb_masked = rgb[mask].astype(np.float32) / 255.0
    points = np.concatenate([xyz, rgb_masked], axis=-1)
    if len(points) >= n_points:
        idx = np.random.choice(len(points), n_points, replace=False)
    else:
        idx = np.random.choice(len(points), n_points, replace=True)
    return points[idx].astype(np.float32)


# ─── Zarr Storage ─────────────────────────────────────────────────────────────

class ZarrWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zarr_path = self.output_dir / "dataset.zarr"
        self._root = zarr.open(str(self.zarr_path), mode="a")
        self._episode_idx = self._get_next_episode_idx()
        print(f"[Zarr] Saving to {self.zarr_path} (next episode: {self._episode_idx})")

    def _get_next_episode_idx(self):
        if "meta" not in self._root:
            return 0
        if "episode_ends" not in self._root["meta"]:
            return 0
        return len(self._root["meta"]["episode_ends"])

    def save_episode(self, episode):
        T = len(episode["state"])
        if T == 0:
            print("[Zarr] Empty episode, skipping")
            return
        data = self._root.require_group("data")
        meta = self._root.require_group("meta")
        for key, arr in episode.items():
            arr = np.array(arr)
            if key not in data:
                chunks = (1,) + arr.shape[1:]
                z = data.zeros(key, shape=arr.shape, chunks=chunks, dtype=arr.dtype)
                z[:] = arr
            else:
                old_len = data[key].shape[0]
                new_shape = (old_len + T,) + data[key].shape[1:]
                data[key].resize(new_shape)
                data[key][old_len:] = arr
        current_len = int(data["state"].shape[0])
        if "episode_ends" not in meta:
            meta.array("episode_ends", np.array([current_len], dtype=np.int64), chunks=(1,), dtype=np.int64)
        else:
            old_ends = meta["episode_ends"][:]
            new_ends = np.append(old_ends, current_len).astype(np.int64)
            del meta["episode_ends"]
            meta.array("episode_ends", new_ends, chunks=(1,), dtype=np.int64)
        self._episode_idx += 1
        print(f"[Zarr] Saved episode {self._episode_idx - 1} ({T} steps)")


class EpisodeRecorder:
    def __init__(self):
        self.state = []; self.action = []; self.velocity = []
        self.torque = []; self.point_cloud = []; self.img = []
        self.depth = []

    def add(self, state, action, velocity, torque, point_cloud, img, depth):
        self.state.append(state); self.action.append(action)
        self.velocity.append(velocity); self.torque.append(torque)
        self.point_cloud.append(point_cloud); self.img.append(img)
        self.depth.append(depth)

    def to_dict(self):
        return {
            "state":       np.array(self.state,       dtype=np.float32),
            "action":      np.array(self.action,      dtype=np.float32),
            "velocity":    np.array(self.velocity,    dtype=np.float32),
            "torque":      np.array(self.torque,      dtype=np.float32),
            "point_cloud": np.array(self.point_cloud, dtype=np.float32),
            "img":         np.array(self.img,         dtype=np.uint8),
            "depth":       np.array(self.depth,       dtype=np.uint16),
        }

    def __len__(self):
        return len(self.state)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./datasets/pick_place")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--camera", choices=["l515", "d435"], default="l515",
                        help="카메라 종류 선택 (기본: l515)")
    args = parser.parse_args()

    cam = CAMERA_PRESETS[args.camera]
    print(f"[Camera] {args.camera.upper()} (serial={cam['serial']})")

    print("=" * 60)
    print("DP3 Data Collection (PASSIVE READ) for OpenArm")
    print("=" * 60)
    print("NOTE: Run OpenArm teleop in another terminal first.")
    print("      This script only LISTENS to the CAN bus.")
    print("-" * 60)
    print("Controls:  SPACE=start/stop  d=discard  q=quit")
    print("=" * 60)

    follower = PassiveCANReader(FOLLOWER_PORT, "Follower")
    leader   = PassiveCANReader(LEADER_PORT,   "Leader")
    chest    = CameraReader(
        cam["serial"], "Chest",
        width=cam["img_w"], height=cam["img_h"],
        color_w=cam["color_w"], color_h=cam["color_h"],
    )
    writer   = ZarrWriter(args.output_dir)

    try:
        print("\n[Init] Opening CAN buses (passive)...")
        follower.connect(); leader.connect()
        follower.start();   leader.start()

        print("[Init] Connecting camera...")
        chest.connect(); chest.start()

        # Give passive readers a moment to catch first frames
        time.sleep(1.0)
        print("\n[Ready] Listening. Move the arm via teleop, press SPACE to record.\n")

        recording = False
        armed     = False
        arm_ref_pos = None
        episode   = None
        ep_count  = writer._episode_idx
        interval  = 1.0 / CONTROL_HZ

        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while ep_count < args.num_episodes:
                loop_start = time.perf_counter()

                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key == ' ':
                        if not recording and not armed:
                            # ARM: wait for motion, then auto-start recording
                            armed = True
                            arm_ref_pos = follower.get_state()[0]
                            episode = EpisodeRecorder()
                            print(f"\n[ARMED] Episode {ep_count} — move the arm to auto-start...")
                        elif recording:
                            recording = False
                            if len(episode) > 0:
                                writer.save_episode(episode.to_dict())
                                ep_count += 1
                                print(f"[SAVE] {ep_count}/{args.num_episodes}")
                            else:
                                print("[WARN] Empty episode")
                        elif armed:
                            armed = False
                            episode = None
                            print("\n[DISARM]")
                    elif key == 'd' and (recording or armed):
                        recording = False
                        armed = False
                        episode = None
                        print("[DISCARD]")
                    elif key == 'q':
                        if recording and episode and len(episode) > 0:
                            writer.save_episode(episode.to_dict())
                            ep_count += 1
                            print(f"\n[SAVE] Auto-saved before quit")
                        print("[QUIT]")
                        break

                f_pos, f_vel, f_tor, _ = follower.get_state()
                l_pos, _, _, _ = leader.get_state()
                rgb, depth, _ = chest.get_frame()

                # auto-start: armed and motion detected -> start recording
                if armed and not recording:
                    if np.any(np.abs(f_pos - arm_ref_pos) > AUTO_START_THRESH_DEG):
                        recording = True
                        armed = False
                        print(f"\n[REC] Motion detected — recording started!")

                if recording and rgb is not None and depth is not None:
                    pcd = depth_to_pointcloud(depth, rgb, chest.intrinsics)
                    episode.add(f_pos, l_pos, f_vel, f_tor, pcd, rgb, depth)

                status = "REC" if recording else ("ARM" if armed else "---")
                steps = len(episode) if episode else 0
                print(f"\r[{status}] ep={ep_count}/{args.num_episodes} steps={steps:4d} "
                      f"| f_pos={f_pos[0]:6.2f} | l_pos={l_pos[0]:6.2f}", end="", flush=True)

                elapsed = time.perf_counter() - loop_start
                if (st := interval - elapsed) > 0:
                    time.sleep(st)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    finally:
        print("\n[Cleanup] Disconnecting...")
        follower.disconnect(); leader.disconnect(); chest.disconnect()

    print(f"\n[Done] Collected {ep_count} episodes → {writer.zarr_path}")


if __name__ == "__main__":
    main()

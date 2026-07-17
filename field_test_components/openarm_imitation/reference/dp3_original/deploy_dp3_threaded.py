"""
DP3 Real-Robot Deployment for OpenArm — THREADED (inference / motor 분리) — POS+VEL+TORQUE (action=24)
PURE python-can version (no lerobot dependency, runs in dp3 env)

- Loads DP3 policy (EMA) from checkpoint
- Reads follower qpos by passively decoding feedback frames
- Sends MIT control (pos/vel/torque) frames directly via python-can

MIT protocol constants are ported from LeRobot's Damiao driver:
    MOTOR_LIMITS (pmax,vmax,tmax), MIT_KP_RANGE, MIT_KD_RANGE
    _float_to_uint / encode / decode

IMPORTANT: This CONTROLS the motors. Do NOT run teleop simultaneously.
           Keep E-stop reachable. Start with USE_TORQUE=False.

Usage:
    python deploy_dp3_can.py \
        --ckpt /home/fieldro_alpha17/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/2026.06.10/17.47.14_train_simple_dp3_openarm_pick_ptv/checkpoints/latest.ckpt \
        --config-dir /home/fieldro_alpha17/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config \
        --config-name simple_dp3 --task openarm_pick_ptv

Controls: SPACE=start/pause  q=quit
"""

import argparse, threading, time, sys, select, termios, tty
from collections import deque
import numpy as np
import torch
import can
import pyrealsense2 as rs


# ─── Config ─────────────────────────────────────────────────────────────────

FOLLOWER_PORT = "can2"
CONTROL_HZ    = 30
POINT_CLOUD_N = 1024
CHEST_SERIAL  = "919122072515"
IMG_W, IMG_H  = 640, 480
N_OBS_STEPS   = 2
DEBUG         = True  # True로 하면 액션 시퀀스 출력
MOTOR_HZ      = 30   # 모터 스레드 주파수 (수집 주기와 동일)
CHUNK_STEP_HZ = 30   # 액션 청크 1스텝을 소비하는 속도 (수집 30Hz와 동일)

# safety
MAX_STEP_DEG = 8.0
MAX_TORQUE   = 10.0   # 데이터 토크 최대 9.4 커버 (중력보상)
SMOOTH_ALPHA = 0.5    # 모터 타겟 평활화 (1.0=평활화 없음, 낮을수록 부드러움/느림)
USE_TORQUE   = True    # START FALSE. enable after verifying motion is safe
USE_VELOCITY = True

# MIT gains (per-joint: joint_1..7 + gripper)
# big motors (dm8009/4340) high, small (dm4310) lower, gripper lowest to avoid oscillation
DEPLOY_KP = 50.0   # default (used by move_to)
DEPLOY_KD = 1.0
KP_PER_JOINT = [150.0, 150.0, 150.0, 150.0, 24.0, 31.0, 25.0, 25.0]
KD_PER_JOINT = [5.0, 5.0, 5.0, 5.0, 0.5, 0.5, 0.5, 1.5]

# Training start pose (mean of episode first frames, degrees)
# Policy expects to START from this pose. We move here before running.
INIT_POSE = [109.0, 3.5, 6.6, 2.4, -6.0, -1.8, -14.5, -4.1]  # pick_v2 단일컵

# MIT protocol constants (from LeRobot tables.py)
MIT_KP_RANGE = (0.0, 500.0)
MIT_KD_RANGE = (0.0, 5.0)
# motor_type: (pmax, vmax, tmax)
MOTOR_LIMITS = {
    "dm8009": (12.5, 45.0, 54.0),
    "dm4340": (12.5, 8.0,  28.0),
    "dm4310": (12.5, 30.0, 10.0),
}
CAN_CMD_ENABLE = 0xFC   # enable motor command byte (Damiao)

# motor_name: (send_id, recv_id, motor_type)
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
JOINT_NAMES = list(MOTOR_CONFIG.keys())
RECV_ID_TO_IDX  = {MOTOR_CONFIG[n][1]: i for i, n in enumerate(JOINT_NAMES)}
RECV_ID_TO_TYPE = {MOTOR_CONFIG[n][1]: MOTOR_CONFIG[n][2] for n in JOINT_NAMES}


# ─── MIT encode / decode (ported from LeRobot) ──────────────────────────────

def float_to_uint(x, x_min, x_max, bits):
    x = max(x_min, min(x_max, x))
    span = x_max - x_min
    return int((x - x_min) / span * ((1 << bits) - 1))

def uint_to_float(x, x_min, x_max, bits):
    span = x_max - x_min
    return float(x) / ((1 << bits) - 1) * span + x_min

def encode_mit_packet(motor_type, kp, kd, pos_deg, vel_deg_s, torque):
    pos_rad = np.radians(pos_deg)
    vel_rad = np.radians(vel_deg_s)
    pmax, vmax, tmax = MOTOR_LIMITS[motor_type]
    kp_u  = float_to_uint(kp, *MIT_KP_RANGE, 12)
    kd_u  = float_to_uint(kd, *MIT_KD_RANGE, 12)
    q_u   = float_to_uint(pos_rad, -pmax, pmax, 16)
    dq_u  = float_to_uint(vel_rad, -vmax, vmax, 12)
    tau_u = float_to_uint(torque, -tmax, tmax, 12)
    d = [0]*8
    d[0] = (q_u >> 8) & 0xFF
    d[1] = q_u & 0xFF
    d[2] = dq_u >> 4
    d[3] = ((dq_u & 0xF) << 4) | ((kp_u >> 8) & 0xF)
    d[4] = kp_u & 0xFF
    d[5] = kd_u >> 4
    d[6] = ((kd_u & 0xF) << 4) | ((tau_u >> 8) & 0xF)
    d[7] = tau_u & 0xFF
    return d

def decode_motor_state(data, motor_type):
    if len(data) < 8:
        return None
    q_u   = (data[1] << 8) | data[2]
    dq_u  = (data[3] << 4) | (data[4] >> 4)
    tau_u = ((data[4] & 0x0F) << 8) | data[5]
    pmax, vmax, tmax = MOTOR_LIMITS[motor_type]
    pos = uint_to_float(q_u,  -pmax, pmax, 16)
    vel = uint_to_float(dq_u, -vmax, vmax, 12)
    tor = uint_to_float(tau_u, -tmax, tmax, 12)
    return np.degrees(pos), np.degrees(vel), tor


# ─── Motor Bus (python-can, read + MIT control) ─────────────────────────────

class CanMotorBus:
    def __init__(self, port):
        self.port = port
        self._bus = None
        self.pos = np.zeros(8, np.float32)
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def connect(self):
        self._bus = can.interface.Bus(channel=self.port, interface="socketcan", fd=True)
        print(f"[Motor] CAN open on {self.port}")
        # enable all motors (MIT enable frame: 0xFF*7 + CMD)
        for name, (sid, rid, mt) in MOTOR_CONFIG.items():
            data = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, CAN_CMD_ENABLE]
            self._bus.send(can.Message(arbitration_id=sid, data=data,
                                       is_extended_id=False, is_fd=True))
            time.sleep(0.005)
        print("[Motor] Enable frames sent")

    def start_reader(self):
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stop.is_set():
            msg = self._bus.recv(timeout=0.05)
            if msg is None:
                continue
            rid = msg.arbitration_id
            if rid in RECV_ID_TO_IDX:
                dec = decode_motor_state(msg.data, RECV_ID_TO_TYPE[rid])
                if dec:
                    with self.lock:
                        self.pos[RECV_ID_TO_IDX[rid]] = dec[0]

    def read_positions(self):
        with self.lock:
            return self.pos.copy()

    def send_pos_vel_torque(self, pos, vel, tor, kp=None, kd=None):
        for name, (sid, rid, mt) in MOTOR_CONFIG.items():
            i = RECV_ID_TO_IDX[rid]
            # per-joint gains unless explicit kp/kd override given
            jkp = KP_PER_JOINT[i] if kp is None else kp
            jkd = KD_PER_JOINT[i] if kd is None else kd
            data = encode_mit_packet(mt, jkp, jkd, pos[i], vel[i], tor[i])
            self._bus.send(can.Message(arbitration_id=sid, data=data,
                                       is_extended_id=False, is_fd=True))

    def move_to(self, target_pos, steps=60):
        """Smoothly interpolate from current pos to target_pos over `steps`."""
        start = self.read_positions()
        for s in range(1, steps + 1):
            alpha = s / steps
            interp = start * (1 - alpha) + np.array(target_pos) * alpha
            self.send_pos_vel_torque(interp, np.zeros(8), np.zeros(8))  # per-joint gains
            time.sleep(0.02)  # 50Hz
        print("[Motor] move_to done")

    def disconnect(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._bus:
            self._bus.shutdown()
        print("[Motor] CAN closed")


# ─── Camera ─────────────────────────────────────────────────────────────────

class CameraReader:
    def __init__(self, serial, width=IMG_W, height=IMG_H, fps=30):
        self.serial=serial; self.width=width; self.height=height; self.fps=fps
        self.rgb=None; self.depth=None
        self.lock=threading.Lock(); self._stop=threading.Event()
        self._thread=None; self._pipeline=None; self._intr=None

    def connect(self):
        self._pipeline = rs.pipeline()
        cfg = rs.config(); cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16,  self.fps)
        prof = self._pipeline.start(cfg)
        self._intr = prof.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        for _ in range(10):
            self._pipeline.wait_for_frames(timeout_ms=1000)
        print(f"[Camera] Connected ({self.serial})")

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True); self._thread.start()

    def _loop(self):
        align = rs.align(rs.stream.color)
        while not self._stop.is_set():
            try:
                f = align.process(self._pipeline.wait_for_frames(timeout_ms=200))
                c=f.get_color_frame(); d=f.get_depth_frame()
                if not c or not d: continue
                with self.lock:
                    self.rgb = np.asanyarray(c.get_data()).astype(np.uint8)
                    self.depth = np.asanyarray(d.get_data()).astype(np.uint16)
            except Exception as e:
                print(f"[Camera] {e}")

    def get_frame(self):
        with self.lock:
            if self.rgb is None: return None, None
            return self.rgb.copy(), self.depth.copy()

    def disconnect(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=2.0)
        if self._pipeline: self._pipeline.stop()

    @property
    def intrinsics(self): return self._intr


def depth_to_pointcloud(depth, rgb, intr, n_points=POINT_CLOUD_N):
    h, w = depth.shape
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32) / 1000.0
    mask = (z > 0.3) & (z < 1.3)        # 수집과 동일 crop (pick_v2)
    x = (uu - cx) * z / fx; y = (vv - cy) * z / fy
    xyz = np.stack([x[mask], y[mask], z[mask]], axis=-1)
    rgbm = rgb[mask].astype(np.float32) / 255.0
    pts = np.concatenate([xyz, rgbm], axis=-1)
    idx = np.random.choice(len(pts), n_points, replace=len(pts) < n_points)
    return pts[idx].astype(np.float32)



# ─── Motor Tracker Thread ───────────────────────────────────────────────────
# 추론과 독립적으로 30Hz로 모터를 부드럽게 제어하는 스레드.
# 추론이 set_chunk()로 새 액션 청크(16스텝)를 주면,
# tracker가 30Hz로 한 스텝씩 소비하며 스텝 사이를 보간해 전송한다.

class MotorTracker:
    def __init__(self, motor):
        self.motor = motor
        self.lock = threading.Lock()
        self.chunk = None        # (n_steps, 24) 현재 소비 중인 청크
        self.chunk_idx = 0.0     # 청크 내 진행 위치 (소수 = 보간)
        self.active = False
        self._last_pos = None   # EMA 평활화용
        self._stop = threading.Event()
        self._thread = None

    def set_chunk(self, chunk):
        with self.lock:
            self.chunk = chunk.copy()
            self.chunk_idx = 0.0

    def set_active(self, active):
        with self.lock:
            self.active = active
            if not active:
                self.chunk = None
                self._last_pos = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        dt = 1.0 / MOTOR_HZ
        idx_per_tick = CHUNK_STEP_HZ / MOTOR_HZ   # 보통 1.0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self.lock:
                chunk = self.chunk
                idx = self.chunk_idx
                active = self.active
            if active and chunk is not None:
                n = len(chunk)
                i0 = int(min(idx, n - 1))
                i1 = int(min(i0 + 1, n - 1))
                frac = min(idx, n - 1) - i0
                a0, a1 = chunk[i0], chunk[i1]
                ak = a0 * (1 - frac) + a1 * frac
                tgt_pos = ak[0:8]
                tgt_vel = ak[8:16] if USE_VELOCITY else np.zeros(8, np.float32)
                tgt_tor = np.clip(ak[16:24], -MAX_TORQUE, MAX_TORQUE) if USE_TORQUE else np.zeros(8, np.float32)
                cur = self.motor.read_positions()
                safe_pos = cur + np.clip(tgt_pos - cur, -MAX_STEP_DEG, MAX_STEP_DEG)
                # 진동 억제: 직전 전송 타겟과 EMA로 평활화
                if self._last_pos is None:
                    self._last_pos = safe_pos.copy()
                smooth_pos = SMOOTH_ALPHA * safe_pos + (1 - SMOOTH_ALPHA) * self._last_pos
                self._last_pos = smooth_pos.copy()
                self.motor.send_pos_vel_torque(smooth_pos, tgt_vel, tgt_tor)
                with self.lock:
                    self.chunk_idx = idx + idx_per_tick
            el = time.perf_counter() - t0
            if (st := dt - el) > 0:
                time.sleep(st)


# ─── Policy loader ──────────────────────────────────────────────────────────

def load_policy(ckpt, cdir, cname, task):
    from hydra import initialize_config_dir, compose
    import dill
    with initialize_config_dir(config_dir=cdir, version_base=None):
        cfg = compose(config_name=cname, overrides=[f"task={task}"])
    sys.path.insert(0, cdir.replace("/diffusion_policy_3d/config", ""))
    from train import TrainDP3Workspace
    ws = TrainDP3Workspace(cfg)
    payload = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location="cpu")
    ws.load_payload(payload)
    policy = ws.ema_model if (cfg.training.use_ema and ws.ema_model is not None) else ws.model
    print("[Policy] EMA" if policy is ws.ema_model else "[Policy] model")
    policy.eval().cuda()
    return policy, cfg


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--config-name", default="simple_dp3")
    ap.add_argument("--task", default="openarm_pick_ptv")
    args = ap.parse_args()

    print("="*60)
    print("DP3 Deployment (POS+VEL+TORQUE, python-can) — MOTOR ACTIVE")
    print(f"USE_VELOCITY={USE_VELOCITY} USE_TORQUE={USE_TORQUE} MAX_TORQUE={MAX_TORQUE}")
    print("Do NOT run teleop. Keep E-stop reachable.")
    print("Controls: SPACE=start/pause  q=quit")
    print("="*60)

    print("\n[Init] Loading policy...")
    policy, cfg = load_policy(args.ckpt, args.config_dir, args.config_name, args.task)
    dev = "cuda"

    motor = CanMotorBus(FOLLOWER_PORT)
    chest = CameraReader(CHEST_SERIAL)
    tracker = MotorTracker(motor)

    try:
        motor.connect(); motor.start_reader()
        chest.connect(); chest.start()
        time.sleep(1.0)
        tracker.start()   # 모터 스레드 시작 (active 전까지 대기)

        pc_hist  = deque(maxlen=N_OBS_STEPS)
        pos_hist = deque(maxlen=N_OBS_STEPS)
        running = False
        interval = 1.0 / CONTROL_HZ

        # save home position (current pose at startup)
        time.sleep(0.3)
        home_pos = motor.read_positions()
        print(f"[Home] saved: {np.round(home_pos, 1)}")

        old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        print("\n[Ready] SPACE to start.\n")

        try:
            while True:
                t0 = time.perf_counter()
                if select.select([sys.stdin], [], [], 0)[0]:
                    k = sys.stdin.read(1)
                    if k == ' ':
                        if not running:
                            # move to training start pose before running
                            print("\n[Init] Moving to training start pose...")
                            motor.move_to(np.array(INIT_POSE), steps=100)
                            pc_hist.clear(); pos_hist.clear()
                            running = True
                            tracker.set_active(True)
                            print("[RUN]")
                        else:
                            running = False
                            tracker.set_active(False)
                            print("\n[PAUSE]")
                    elif k == 'q':
                        print("\n[QUIT]"); break

                cur = motor.read_positions()
                rgb, depth = chest.get_frame()
                if rgb is None:
                    continue
                pc = depth_to_pointcloud(depth, rgb, chest.intrinsics)
                pc_hist.append(pc); pos_hist.append(cur)

                print(f"\r[{'RUN' if running else '---'}] pos0={cur[0]:6.2f}", end="", flush=True)

                if not running or len(pc_hist) < N_OBS_STEPS:
                    time.sleep(interval); continue

                pc_arr  = np.stack(list(pc_hist),  0)[None]
                pos_arr = np.stack(list(pos_hist), 0)[None]
                obs = {
                    "point_cloud": torch.from_numpy(pc_arr).float().to(dev),
                    "agent_pos":   torch.from_numpy(pos_arr).float().to(dev),
                }
                _t_inf = time.perf_counter()
                with torch.no_grad():
                    out = policy.predict_action(obs)
                infer_dt = time.perf_counter() - _t_inf
                full_action = out["action"][0].cpu().numpy()   # (n_action_steps, 24)
                a0 = full_action[0]
                if DEBUG:
                    print(f"\n  j1_seq={np.round(full_action[:,0],1)}")
                    print(f"  grip_seq={np.round(full_action[:,7],1)}")
                    diff = a0[:8] - cur
                    print(f"  cur={np.round(cur[:4],1)} tgt={np.round(a0[:4],1)} diff={np.round(diff[:4],1)}")

                # 새 청크를 tracker에 전달 → 모터 스레드가 30Hz로 소비
                tracker.set_chunk(full_action)

                loop_dt = time.perf_counter() - t0
                actual_hz = 1.0 / loop_dt if loop_dt > 0 else 0
                print(f"  [infer-loop {actual_hz:4.1f}Hz  infer={infer_dt*1000:.0f}ms]", end="")
                # 추론 루프는 sleep 최소 — 다음 관측/추론 바로
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
    finally:
        print("\n[Cleanup] stopping tracker, returning to home...")
        tracker.set_active(False)
        tracker.stop()
        try:
            motor.move_to(home_pos, steps=60)
        except Exception as e:
            print(f"[Home] return skipped: {e}")
        motor.disconnect(); chest.disconnect()


if __name__ == "__main__":
    main()

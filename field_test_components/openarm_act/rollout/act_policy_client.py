#!/usr/bin/env python3
"""Run the trained ACT policy on two RealSense RGB streams and live OpenArm state."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors


LOGGER = logging.getLogger("openarm_act.policy")
VECTOR_SIZE = 16


class StopRequested:
    value = False


def request_stop(_signum: int, _frame: Any) -> None:
    StopRequested.value = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chest-serial", required=True)
    parser.add_argument("--wrist-left-serial", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, required=True)
    parser.add_argument("--state-port", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until Ctrl+C")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-file", type=Path, required=True)
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
    )


def tensor_image(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(torch.float32) / 255.0


def receive_state(sock: socket.socket, timeout_s: float) -> tuple[int, list[float]]:
    deadline = time.monotonic() + timeout_s
    newest: tuple[int, list[float]] | None = None
    while time.monotonic() < deadline:
        sock.settimeout(max(0.001, deadline - time.monotonic()))
        try:
            raw, address = sock.recvfrom(65535)
        except socket.timeout:
            break
        if address[0] != "127.0.0.1":
            continue
        packet = json.loads(raw.decode("utf-8"))
        if packet.get("type") != "state" or packet.get("unit") != "radian":
            continue
        state = [float(value) for value in packet["state"]]
        if len(state) != VECTOR_SIZE:
            continue
        newest = (int(packet["sequence"]), state)
        sock.setblocking(False)
        while True:
            try:
                raw, address = sock.recvfrom(65535)
            except BlockingIOError:
                break
            packet = json.loads(raw.decode("utf-8"))
            if address[0] == "127.0.0.1" and packet.get("type") == "state":
                candidate = [float(value) for value in packet["state"]]
                if len(candidate) == VECTOR_SIZE:
                    newest = (int(packet["sequence"]), candidate)
        sock.setblocking(True)
        break
    if newest is None:
        raise TimeoutError("no live OpenArm state received")
    return newest


def send_packet(sock: socket.socket, address: tuple[str, int], packet: dict[str, Any]) -> None:
    sock.sendto(json.dumps(packet, separators=(",", ":")).encode("utf-8"), address)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_file)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    LOGGER.info("Loading ACT checkpoint: %s", args.checkpoint)
    policy_config = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_config.device = args.device
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(args.checkpoint, config=policy_config).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    camera_options = dict(fps=30, width=640, height=480, warmup_s=2, use_rgb=True, use_depth=False)
    chest = RealSenseCamera(RealSenseCameraConfig(serial_number_or_name=args.chest_serial, **camera_options))
    wrist_left = RealSenseCamera(
        RealSenseCameraConfig(serial_number_or_name=args.wrist_left_serial, **camera_options)
    )

    state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    state_socket.bind((args.host, args.state_port))
    action_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    motor_address = (args.host, args.action_port)

    try:
        chest.connect()
        wrist_left.connect()
        LOGGER.warning("Cameras and ACT ready; requesting active motor connection")
        send_packet(action_socket, motor_address, {"schema_version": 1, "type": "hello"})
        _sequence, state = receive_state(state_socket, timeout_s=10.0)

        start = time.monotonic()
        next_tick = start
        policy_sequence = 0
        last_report = start
        while not StopRequested.value and (args.duration <= 0 or time.monotonic() - start < args.duration):
            _state_sequence, state = receive_state(state_socket, timeout_s=0.5)
            observation = {
                "observation.state": torch.tensor(state, dtype=torch.float32),
                "observation.images.chest": tensor_image(chest.read_latest(max_age_ms=500)),
                "observation.images.wrist_left": tensor_image(wrist_left.read_latest(max_age_ms=500)),
            }
            with torch.inference_mode():
                processed = preprocessor(observation)
                action = postprocessor(policy.select_action(processed)).squeeze(0).detach().cpu()
            if action.shape != (VECTOR_SIZE,) or not bool(torch.isfinite(action).all()):
                raise RuntimeError("ACT produced an invalid action")

            send_packet(
                action_socket,
                motor_address,
                {
                    "schema_version": 1,
                    "type": "action",
                    "sequence": policy_sequence,
                    "monotonic_ns": time.monotonic_ns(),
                    "unit": "radian",
                    "action": [float(value) for value in action],
                },
            )
            policy_sequence += 1

            now = time.monotonic()
            if now - last_report >= 1.0:
                if args.duration <= 0:
                    LOGGER.info("active rollout %.1fs, commands=%d", now - start, policy_sequence)
                else:
                    LOGGER.info(
                        "active rollout %.1f/%.1fs, commands=%d",
                        now - start,
                        args.duration,
                        policy_sequence,
                    )
                last_report = now
            next_tick += 1.0 / args.fps
            time.sleep(max(0.0, next_tick - time.monotonic()))
        return 0
    finally:
        try:
            send_packet(action_socket, motor_address, {"schema_version": 1, "type": "stop"})
        except Exception:
            pass
        if chest.is_connected:
            chest.disconnect()
        if wrist_left.is_connected:
            wrist_left.disconnect()
        state_socket.close()
        action_socket.close()
        LOGGER.warning("ACT policy client stopped")


if __name__ == "__main__":
    raise SystemExit(main())

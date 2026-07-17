#!/usr/bin/env python3
"""OpenArm-v1 bimanual motor endpoint for the trained ACT policy.

This process deliberately uses the same Python binding, Classic-CAN mode,
motor IDs, and MIT gains as the known-good ExoArm bridge. It waits for a local
policy handshake before opening either CAN interface or enabling motors.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import socket
import time
from pathlib import Path
from typing import Any, Sequence

import openarm_can as oa


LOGGER = logging.getLogger("openarm_act.motor")
ARM_KP = (20.0, 20.0, 12.0, 12.0, 5.0, 5.0, 4.0)
ARM_KD = (1.0, 1.0, 0.8, 0.8, 0.3, 0.3, 0.3)
GRIPPER_KP = 5.0
GRIPPER_KD = 0.3
VECTOR_SIZE = 16


class StopRequested:
    value = False


def request_stop(_signum: int, _frame: Any) -> None:
    StopRequested.value = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-can", required=True)
    parser.add_argument("--right-can", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, required=True)
    parser.add_argument("--state-port", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stale-timeout", type=float, default=0.75)
    parser.add_argument("--max-arm-delta", type=float, default=0.04)
    parser.add_argument("--max-gripper-delta", type=float, default=0.06)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
    )


def decode_message(raw: bytes) -> dict[str, Any]:
    message = json.loads(raw.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("control packet must be a JSON object")
    return message


def load_action_bounds(path: Path) -> tuple[list[float], list[float]]:
    stats = json.loads(path.read_text(encoding="utf-8"))["action"]
    lower = [float(value) for value in stats["min"]]
    upper = [float(value) for value in stats["max"]]
    if len(lower) != VECTOR_SIZE or len(upper) != VECTOR_SIZE:
        raise ValueError("training action bounds must contain 16 values")
    return lower, upper


def make_arm(interface: str) -> oa.OpenArm:
    arm = oa.OpenArm(interface, False)
    motor_types = [
        oa.MotorType.DM8009,
        oa.MotorType.DM8009,
        oa.MotorType.DM4340,
        oa.MotorType.DM4340,
        oa.MotorType.DM4310,
        oa.MotorType.DM4310,
        oa.MotorType.DM4310,
    ]
    arm.init_arm_motors(motor_types, [1, 2, 3, 4, 5, 6, 7], [17, 18, 19, 20, 21, 22, 23])
    arm.init_gripper_motor(oa.MotorType.DM4310, 8, 24)
    arm.set_callback_mode_all(oa.CallbackMode.STATE)
    return arm


def zero_gain_query(arm: oa.OpenArm) -> None:
    arm.get_arm().mit_control_all([oa.MITParam(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(7)])
    arm.get_gripper().mit_control_all([oa.MITParam(0.0, 0.0, 0.0, 0.0, 0.0)])


def positions(arm: oa.OpenArm) -> list[float]:
    return [
        *[float(motor.get_position()) for motor in arm.get_arm().get_motors()],
        *[float(motor.get_position()) for motor in arm.get_gripper().get_motors()],
    ]


def read_bimanual_state(left: oa.OpenArm, right: oa.OpenArm) -> list[float]:
    return [*positions(left), *positions(right)]


def send_arm_target(arm: oa.OpenArm, values: Sequence[float]) -> None:
    arm.get_arm().mit_control_all(
        [oa.MITParam(ARM_KP[index], ARM_KD[index], float(values[index]), 0.0, 0.0) for index in range(7)]
    )
    arm.get_gripper().mit_control_all(
        [oa.MITParam(GRIPPER_KP, GRIPPER_KD, float(values[7]), 0.0, 0.0)]
    )


def bounded_target(
    requested: Sequence[float],
    previous: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    max_arm_delta: float,
    max_gripper_delta: float,
) -> list[float]:
    if len(requested) != VECTOR_SIZE or not all(math.isfinite(float(value)) for value in requested):
        raise ValueError("ACT action must contain 16 finite radian values")
    result: list[float] = []
    for index, raw_value in enumerate(requested):
        clipped = min(max(float(raw_value), lower[index]), upper[index])
        is_gripper = index in (7, 15)
        limit = max_gripper_delta if is_gripper else max_arm_delta
        delta = min(max(clipped - previous[index], -limit), limit)
        result.append(float(previous[index] + delta))
    return result


def send_state(sock: socket.socket, address: tuple[str, int], sequence: int, state: Sequence[float]) -> None:
    packet = {
        "schema_version": 1,
        "type": "state",
        "sequence": sequence,
        "monotonic_ns": time.monotonic_ns(),
        "unit": "radian",
        "state": list(state),
    }
    sock.sendto(json.dumps(packet, separators=(",", ":")).encode("utf-8"), address)


def wait_for_policy(sock: socket.socket) -> None:
    LOGGER.info("Waiting for local ACT policy handshake; CAN is not open yet")
    while not StopRequested.value:
        try:
            raw, address = sock.recvfrom(65535)
        except socket.timeout:
            continue
        message = decode_message(raw)
        if address[0] == "127.0.0.1" and message.get("type") == "hello":
            LOGGER.info("Policy handshake received")
            return
    raise KeyboardInterrupt


def main() -> int:
    args = parse_args()
    configure_logging(args.log_file)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    lower, upper = load_action_bounds(args.stats_json)
    control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_socket.bind((args.host, args.action_port))
    control_socket.settimeout(0.1)
    state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_address = (args.host, args.state_port)

    left: oa.OpenArm | None = None
    right: oa.OpenArm | None = None
    try:
        wait_for_policy(control_socket)
        LOGGER.warning("Opening CAN and enabling both OpenArm arms")
        right = make_arm(args.right_can)
        left = make_arm(args.left_can)
        right.enable_all()
        left.enable_all()

        for _ in range(5):
            zero_gain_query(left)
            zero_gain_query(right)
            left.recv_all(2000)
            right.recv_all(2000)

        current = read_bimanual_state(left, right)
        if len(current) != VECTOR_SIZE or not all(math.isfinite(value) for value in current):
            raise RuntimeError("failed to obtain 16 finite motor positions")
        if max(abs(value) for value in current) < 1e-4:
            raise RuntimeError("motor feedback remained all-zero; refusing to create a zero-position jump")

        last_target = list(current)
        send_arm_target(left, last_target[:8])
        send_arm_target(right, last_target[8:])
        left.recv_all(2000)
        right.recv_all(2000)
        LOGGER.info("Current-pose hold active; waiting for first ACT action")

        period = 1.0 / args.fps
        sequence = 0
        last_action_time: float | None = None
        startup_deadline = time.monotonic() + 5.0
        control_socket.setblocking(False)

        while not StopRequested.value:
            tick_started = time.monotonic()
            latest_action: list[float] | None = None
            while True:
                try:
                    raw, address = control_socket.recvfrom(65535)
                except BlockingIOError:
                    break
                message = decode_message(raw)
                if address[0] != "127.0.0.1":
                    continue
                if message.get("type") == "stop":
                    StopRequested.value = True
                    break
                if message.get("type") == "action" and message.get("unit") == "radian":
                    source_age = (time.monotonic_ns() - int(message["monotonic_ns"])) / 1_000_000_000
                    if 0.0 <= source_age <= args.stale_timeout:
                        latest_action = [float(value) for value in message["action"]]

            if StopRequested.value:
                break

            if latest_action is not None:
                last_target = bounded_target(
                    latest_action,
                    last_target,
                    lower,
                    upper,
                    args.max_arm_delta,
                    args.max_gripper_delta,
                )
                last_action_time = time.monotonic()
            elif last_action_time is None:
                if time.monotonic() > startup_deadline:
                    raise TimeoutError("no ACT action arrived during startup")
            elif time.monotonic() - last_action_time > args.stale_timeout:
                raise TimeoutError("ACT action stream became stale")

            send_arm_target(left, last_target[:8])
            send_arm_target(right, last_target[8:])
            left.recv_all(2000)
            right.recv_all(2000)
            state = read_bimanual_state(left, right)
            send_state(state_socket, state_address, sequence, state)
            sequence += 1
            time.sleep(max(0.0, period - (time.monotonic() - tick_started)))

        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        LOGGER.warning("Stopping active rollout and disabling OpenArm motors")
        if left is not None:
            try:
                left.disable_all()
            except Exception:
                LOGGER.exception("Failed to disable left arm")
        if right is not None:
            try:
                right.disable_all()
            except Exception:
                LOGGER.exception("Failed to disable right arm")
        control_socket.close()
        state_socket.close()


if __name__ == "__main__":
    raise SystemExit(main())

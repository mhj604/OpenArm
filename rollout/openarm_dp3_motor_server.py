#!/usr/bin/env python3
"""DP3 motor endpoint with verified post-enable gripper feedback acquisition.

The known-good ACT endpoint is imported for CAN construction, motor mapping,
limits, command bounding, and shutdown behavior. This DP3-specific process adds
one startup invariant: enable acknowledgements cannot be used as gripper state.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import signal
import socket
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import openarm_can as oa

from openarm_dp3.motor_safety import gripper_feedback_violations

BASE_SERVER = Path("/home/mh/openarm_act/rollout/openarm_act_motor_server.py")
LOGGER = logging.getLogger("openarm_dp3.motor")


def load_base_server(path: Path = BASE_SERVER) -> ModuleType:
    spec = importlib.util.spec_from_file_location("openarm_act_motor_server_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load existing OpenArm motor server: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LOGGER = LOGGER
    return module


BASE = load_base_server()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-can", required=True)
    parser.add_argument("--right-can", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, required=True)
    parser.add_argument("--state-port", type=int, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--stale-timeout", type=float, default=0.75)
    parser.add_argument("--max-arm-delta", type=float, default=0.02)
    parser.add_argument("--max-gripper-delta", type=float, default=0.03)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--feedback-timeout", type=float, default=3.0)
    parser.add_argument("--feedback-margin", type=float, default=0.20)
    parser.add_argument("--feedback-settle", type=float, default=0.010)
    parser.add_argument("--log-file", type=Path, required=True)
    return parser.parse_args()


def query_zero_gain(arm: Any, *, settle_s: float) -> None:
    """Request fresh state without applying position stiffness or torque."""
    arm.recv_all(0)
    arm.get_arm().mit_control_all([oa.MITParam(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(7)])
    arm.get_gripper().mit_control_all([oa.MITParam(0.0, 0.0, 0.0, 0.0, 0.0)])
    time.sleep(settle_s)
    arm.recv_all(20_000)


def acquire_verified_feedback(
    left: Any,
    right: Any,
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    timeout_s: float,
    margin: float,
    settle_s: float,
) -> list[float]:
    if timeout_s <= 0 or settle_s <= 0:
        raise ValueError("feedback timeout and settle time must be positive")

    # Let all enable acknowledgements arrive, then consume them before asking
    # for state. Without this separation, 0x0055 can be decoded as -12.4676 rad.
    time.sleep(0.050)
    left.recv_all(0)
    right.recv_all(0)

    deadline = time.monotonic() + timeout_s
    attempts = 0
    last_violations: list[str] = []
    while time.monotonic() < deadline and not BASE.StopRequested.value:
        query_zero_gain(left, settle_s=settle_s)
        query_zero_gain(right, settle_s=settle_s)
        state = BASE.read_bimanual_state(left, right)
        last_violations = gripper_feedback_violations(state, lower, upper, margin=margin)
        attempts += 1
        if not last_violations:
            LOGGER.info("Verified fresh gripper feedback after %d query cycle(s)", attempts)
            return state
        if attempts == 1 or attempts % 10 == 0:
            LOGGER.warning(
                "Rejecting non-physical gripper feedback: %s", "; ".join(last_violations)
            )

    details = "; ".join(last_violations) or "no valid gripper state received"
    raise TimeoutError(f"fresh gripper feedback did not arrive within {timeout_s:.1f}s: {details}")


def main() -> int:
    args = parse_args()
    BASE.configure_logging(args.log_file)
    BASE.StopRequested.value = False
    signal.signal(signal.SIGINT, BASE.request_stop)
    signal.signal(signal.SIGTERM, BASE.request_stop)

    lower, upper = BASE.load_action_bounds(args.stats_json)
    control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_socket.bind((args.host, args.action_port))
    control_socket.settimeout(0.1)
    state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_address = (args.host, args.state_port)

    left: Any | None = None
    right: Any | None = None
    try:
        BASE.wait_for_policy(control_socket)
        LOGGER.warning("Opening CAN and enabling both OpenArm arms")
        right = BASE.make_arm(args.right_can)
        left = BASE.make_arm(args.left_can)
        right.enable_all()
        left.enable_all()

        current = acquire_verified_feedback(
            left,
            right,
            lower,
            upper,
            timeout_s=args.feedback_timeout,
            margin=args.feedback_margin,
            settle_s=args.feedback_settle,
        )
        if len(current) != BASE.VECTOR_SIZE or not all(math.isfinite(value) for value in current):
            raise RuntimeError("failed to obtain 16 finite motor positions")
        if max(abs(value) for value in current) < 1e-4:
            raise RuntimeError("motor feedback remained all-zero; refusing a zero-position jump")

        last_target = list(current)
        BASE.send_arm_target(left, last_target[:8])
        BASE.send_arm_target(right, last_target[8:])
        left.recv_all(2_000)
        right.recv_all(2_000)
        LOGGER.info("Verified current-pose hold active; waiting for first DP3 action")

        period = 1.0 / args.fps
        sequence = 0
        last_action_time: float | None = None
        startup_deadline = time.monotonic() + 5.0
        control_socket.setblocking(False)

        while not BASE.StopRequested.value:
            tick_started = time.monotonic()
            latest_action: list[float] | None = None
            while True:
                try:
                    raw, address = control_socket.recvfrom(65535)
                except BlockingIOError:
                    break
                message = BASE.decode_message(raw)
                if address[0] != "127.0.0.1":
                    continue
                if message.get("type") == "stop":
                    BASE.StopRequested.value = True
                    break
                if message.get("type") == "action" and message.get("unit") == "radian":
                    source_age = (
                        time.monotonic_ns() - int(message["monotonic_ns"])
                    ) / 1_000_000_000
                    if 0.0 <= source_age <= args.stale_timeout:
                        latest_action = [float(value) for value in message["action"]]

            if BASE.StopRequested.value:
                break

            if latest_action is not None:
                last_target = BASE.bounded_target(
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
                    raise TimeoutError("no DP3 action arrived during startup")
            elif time.monotonic() - last_action_time > args.stale_timeout:
                raise TimeoutError("DP3 action stream became stale")

            BASE.send_arm_target(left, last_target[:8])
            BASE.send_arm_target(right, last_target[8:])
            left.recv_all(2_000)
            right.recv_all(2_000)
            state = BASE.read_bimanual_state(left, right)
            BASE.send_state(state_socket, state_address, sequence, state)
            sequence += 1
            time.sleep(max(0.0, period - (time.monotonic() - tick_started)))

        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        LOGGER.warning("Stopping DP3 rollout and disabling OpenArm motors")
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

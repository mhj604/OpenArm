#!/usr/bin/env python3
"""Passively mirror OpenArm command/state CAN frames to localhost UDP.

This process opens receive-only SocketCAN RAW sockets. It never transmits a CAN
frame, enables a motor, changes a zero, or configures an interface.
"""

from __future__ import annotations

import argparse
import json
import math
import selectors
import socket
import struct
import time
from dataclasses import dataclass, field

CAN_FRAME = struct.Struct("=IB3x8s")
CAN_SFF_MASK = 0x7FF
CAN_EFF_MASK = 0x1FFFFFFF
COMMAND_IDS = tuple(range(1, 9))
STATE_IDS = tuple(range(17, 25))
MOTOR_POSITION_MAX_RADIAN = 12.5


def decode_uint16_position(high: int, low: int) -> float:
    encoded = (int(high) << 8) | int(low)
    return encoded / 65535.0 * (2.0 * MOTOR_POSITION_MAX_RADIAN) - MOTOR_POSITION_MAX_RADIAN


def is_special_motor_command(data: bytes) -> bool:
    return len(data) >= 8 and data[:7] == b"\xff" * 7 and data[7] in (0xFC, 0xFD, 0xFE)


def open_receive_only_can(interface: str) -> socket.socket:
    can_socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    filters = b"".join(struct.pack("=II", can_id, CAN_SFF_MASK) for can_id in (*COMMAND_IDS, *STATE_IDS))
    can_socket.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
    can_socket.bind((interface,))
    can_socket.setblocking(False)
    return can_socket


@dataclass
class ArmTelemetry:
    command: list[float | None] = field(default_factory=lambda: [None] * 8)
    state: list[float | None] = field(default_factory=lambda: [None] * 8)
    command_time_ns: list[int] = field(default_factory=lambda: [0] * 8)
    state_time_ns: list[int] = field(default_factory=lambda: [0] * 8)

    @property
    def complete(self) -> bool:
        return all(value is not None for value in self.command) and all(
            value is not None for value in self.state
        )

    def update(self, arbitration_id: int, data: bytes, timestamp_ns: int) -> None:
        if arbitration_id in COMMAND_IDS:
            if len(data) < 8 or is_special_motor_command(data):
                return
            index = arbitration_id - 1
            self.command[index] = decode_uint16_position(data[0], data[1])
            self.command_time_ns[index] = timestamp_ns
        elif arbitration_id in STATE_IDS:
            if len(data) < 8:
                return
            index = arbitration_id - 17
            self.state[index] = decode_uint16_position(data[1], data[2])
            self.state_time_ns[index] = timestamp_ns

    def packet_fields(self, side: str) -> dict[str, object]:
        if not self.complete:
            raise RuntimeError(f"{side} telemetry is incomplete")
        command = [float(value) for value in self.command if value is not None]
        state = [float(value) for value in self.state if value is not None]
        if not all(math.isfinite(value) for value in (*command, *state)):
            raise RuntimeError(f"{side} telemetry contains NaN or Inf")
        return {
            f"{side}_command": command,
            f"{side}_state": state,
            f"{side}_command_time_ns": self.command_time_ns,
            f"{side}_state_time_ns": self.state_time_ns,
        }


class PassiveCANMirror:
    def __init__(self, *, left_interface: str, right_interface: str, host: str, port: int, publish_hz: float):
        if left_interface == right_interface:
            raise ValueError("left and right CAN interfaces must differ")
        self.left = ArmTelemetry()
        self.right = ArmTelemetry()
        self.left_socket = open_receive_only_can(left_interface)
        self.right_socket = open_receive_only_can(right_interface)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.left_socket, selectors.EVENT_READ, self.left)
        self.selector.register(self.right_socket, selectors.EVENT_READ, self.right)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Keep UDP unconnected so starting telemetry before the LeRobot receiver
        # does not surface an ICMP port-unreachable as ConnectionRefusedError.
        self.udp_target = (host, port)
        self.publish_period_s = 1.0 / publish_hz
        self.sequence = 0

    @staticmethod
    def _drain(can_socket: socket.socket, telemetry: ArmTelemetry) -> None:
        while True:
            try:
                raw = can_socket.recv(CAN_FRAME.size)
            except BlockingIOError:
                return
            if len(raw) != CAN_FRAME.size:
                continue
            raw_id, data_length, data = CAN_FRAME.unpack(raw)
            arbitration_id = raw_id & CAN_EFF_MASK
            telemetry.update(arbitration_id, data[:data_length], time.monotonic_ns())

    def run(self) -> None:
        next_publish = time.monotonic()
        waiting_reported = False
        while True:
            timeout = max(0.0, next_publish - time.monotonic())
            for key, _mask in self.selector.select(timeout):
                self._drain(key.fileobj, key.data)

            now = time.monotonic()
            if now < next_publish:
                continue
            next_publish = now + self.publish_period_s
            if not self.left.complete or not self.right.complete:
                if not waiting_reported:
                    print("Waiting for all 16 OpenArm command and state frames...", flush=True)
                    waiting_reported = True
                continue

            waiting_reported = False
            packet = {
                "schema_version": 1,
                "sequence": self.sequence,
                "timestamp_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "unit": "radian",
                **self.left.packet_fields("left"),
                **self.right.packet_fields("right"),
            }
            self.udp.sendto(
                json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8"),
                self.udp_target,
            )
            self.sequence += 1

    def close(self) -> None:
        self.selector.close()
        self.left_socket.close()
        self.right_socket.close()
        self.udp.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-interface", required=True)
    parser.add_argument("--right-interface", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29502)
    parser.add_argument("--publish-hz", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535 or args.publish_hz <= 0:
        raise ValueError("port must be valid and publish-hz must be positive")
    mirror = PassiveCANMirror(
        left_interface=args.left_interface,
        right_interface=args.right_interface,
        host=args.host,
        port=args.port,
        publish_hz=args.publish_hz,
    )
    try:
        mirror.run()
    except KeyboardInterrupt:
        pass
    finally:
        mirror.close()


if __name__ == "__main__":
    main()

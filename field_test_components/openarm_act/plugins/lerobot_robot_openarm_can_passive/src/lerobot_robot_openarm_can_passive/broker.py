from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass
from typing import Any

from .features import motor_values


class TelemetryError(RuntimeError):
    pass


@dataclass(frozen=True)
class TelemetrySnapshot:
    sequence: int
    received_monotonic: float
    state: dict[str, float]
    action: dict[str, float]
    maximum_source_age_s: float


class TelemetryBroker:
    def __init__(self, host: str, port: int, stale_timeout_s: float):
        self.host = host
        self.port = port
        self.stale_timeout_s = stale_timeout_s
        self.socket: socket.socket | None = None
        self.snapshot: TelemetrySnapshot | None = None
        self.last_sequence = -1

    @property
    def is_connected(self) -> bool:
        return self.socket is not None

    def connect(self) -> None:
        if self.socket is not None:
            raise ConnectionError("telemetry broker is already connected")
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp.bind((self.host, self.port))
        udp.setblocking(False)
        self.socket = udp

    @staticmethod
    def _float_vector(packet: dict[str, Any], key: str) -> list[float]:
        value = packet.get(key)
        if not isinstance(value, list) or len(value) != 8:
            raise TelemetryError(f"{key} must contain eight values")
        result = [float(item) for item in value]
        if not all(math.isfinite(item) for item in result):
            raise TelemetryError(f"{key} contains NaN or Inf")
        return result

    @staticmethod
    def _time_vector(packet: dict[str, Any], key: str) -> list[int]:
        value = packet.get(key)
        if not isinstance(value, list) or len(value) != 8:
            raise TelemetryError(f"{key} must contain eight timestamps")
        result = [int(item) for item in value]
        if any(item <= 0 for item in result):
            raise TelemetryError(f"{key} contains an invalid timestamp")
        return result

    def _decode(self, raw: bytes) -> TelemetrySnapshot:
        try:
            packet = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TelemetryError("invalid telemetry JSON") from exc
        return self._decode_json(packet)

    def _decode_json(self, packet: Any) -> TelemetrySnapshot:
        """Validate an already-decoded packet; split out for deterministic tests."""
        if not isinstance(packet, dict) or packet.get("schema_version") != 1:
            raise TelemetryError("telemetry schema_version must be 1")
        if packet.get("unit") != "radian":
            raise TelemetryError("telemetry unit must be radian")
        sequence = packet.get("sequence")
        if not isinstance(sequence, int) or sequence < 0:
            raise TelemetryError("telemetry sequence must be a non-negative integer")

        left_state = self._float_vector(packet, "left_state")
        right_state = self._float_vector(packet, "right_state")
        left_action = self._float_vector(packet, "left_command")
        right_action = self._float_vector(packet, "right_command")
        sample_times = [
            *self._time_vector(packet, "left_state_time_ns"),
            *self._time_vector(packet, "right_state_time_ns"),
            *self._time_vector(packet, "left_command_time_ns"),
            *self._time_vector(packet, "right_command_time_ns"),
        ]
        now_ns = time.monotonic_ns()
        if any(timestamp > now_ns for timestamp in sample_times):
            raise TelemetryError("telemetry contains a future monotonic timestamp")
        maximum_age_s = max(now_ns - timestamp for timestamp in sample_times) / 1_000_000_000

        return TelemetrySnapshot(
            sequence=sequence,
            received_monotonic=time.monotonic(),
            state=motor_values(left_state, right_state),
            action=motor_values(left_action, right_action),
            maximum_source_age_s=maximum_age_s,
        )

    def receive_latest(self) -> TelemetrySnapshot:
        if self.socket is None:
            raise ConnectionError("telemetry broker is not connected")
        while True:
            try:
                raw, _address = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            candidate = self._decode(raw)
            if candidate.sequence > self.last_sequence:
                self.snapshot = candidate
                self.last_sequence = candidate.sequence
        return self.require_fresh()

    def require_fresh(self) -> TelemetrySnapshot:
        if self.snapshot is None:
            raise ConnectionError("no complete OpenArm CAN telemetry packet has been received")
        packet_age_s = time.monotonic() - self.snapshot.received_monotonic
        effective_age_s = max(packet_age_s, self.snapshot.maximum_source_age_s)
        if effective_age_s > self.stale_timeout_s:
            raise TimeoutError(
                f"OpenArm telemetry is stale ({effective_age_s:.3f}s > {self.stale_timeout_s:.3f}s)"
            )
        return self.snapshot

    def disconnect(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        self.snapshot = None
        self.last_sequence = -1


_BROKERS: dict[str, TelemetryBroker] = {}


def register_broker(name: str, broker: TelemetryBroker) -> None:
    if name in _BROKERS:
        raise RuntimeError(f"telemetry broker {name!r} is already registered")
    _BROKERS[name] = broker


def get_broker(name: str) -> TelemetryBroker:
    try:
        return _BROKERS[name]
    except KeyError as exc:
        raise ConnectionError(f"telemetry broker {name!r} is not connected by the passive robot") from exc


def unregister_broker(name: str, broker: TelemetryBroker) -> None:
    if _BROKERS.get(name) is broker:
        del _BROKERS[name]

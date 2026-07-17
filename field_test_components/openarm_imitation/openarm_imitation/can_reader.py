from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


MOTOR_TYPES = (
    "dm8009", "dm8009", "dm4340", "dm4340",
    "dm4310", "dm4310", "dm4310", "dm4310",
)
RECV_IDS = tuple(range(0x11, 0x19))
MOTOR_LIMITS = {
    "dm8009": (12.5, 45.0, 54.0),
    "dm4340": (12.5, 8.0, 28.0),
    "dm4310": (12.5, 30.0, 10.0),
}


def uint_to_float(value: int, lower: float, upper: float, bits: int) -> float:
    return float(value) / ((1 << bits) - 1) * (upper - lower) + lower


def decode_motor_state(data: bytes | bytearray, motor_type: str) -> tuple[float, float, float] | None:
    """Decode a Damiao MIT feedback frame into degrees, degrees/s, and Nm."""
    if len(data) < 6:
        return None
    q_uint = (data[1] << 8) | data[2]
    dq_uint = (data[3] << 4) | (data[4] >> 4)
    tau_uint = ((data[4] & 0x0F) << 8) | data[5]
    pmax, vmax, tmax = MOTOR_LIMITS[motor_type]
    position = uint_to_float(q_uint, -pmax, pmax, 16)
    velocity = uint_to_float(dq_uint, -vmax, vmax, 12)
    torque = uint_to_float(tau_uint, -tmax, tmax, 12)
    return float(np.degrees(position)), float(np.degrees(velocity)), torque


@dataclass(frozen=True)
class ArmSnapshot:
    position: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray
    timestamp: float
    seen_ids: tuple[int, ...]


class PassiveCANReader:
    """Receive-only SocketCAN reader. This class deliberately exposes no send API."""

    def __init__(self, interface: str, label: str):
        self.interface = interface
        self.label = label
        self._position = np.full(8, np.nan, dtype=np.float32)
        self._velocity = np.full(8, np.nan, dtype=np.float32)
        self._torque = np.full(8, np.nan, dtype=np.float32)
        self._timestamp = 0.0
        self._seen: set[int] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bus = None
        self.error: Exception | None = None

    def start(self) -> None:
        import can

        filters = [{"can_id": rid, "can_mask": 0x7FF, "extended": False} for rid in RECV_IDS]
        self._bus = can.interface.Bus(
            channel=self.interface,
            bustype="socketcan",
            can_filters=filters,
            receive_own_messages=False,
        )
        self._thread = threading.Thread(target=self._read_loop, name=f"can-{self.label}", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        try:
            while not self._stop.is_set():
                msg = self._bus.recv(timeout=0.1)
                if msg is None or msg.is_error_frame or msg.is_remote_frame:
                    continue
                rid = int(msg.arbitration_id)
                if rid not in RECV_IDS:
                    continue
                idx = rid - 0x11
                decoded = decode_motor_state(msg.data, MOTOR_TYPES[idx])
                if decoded is None:
                    continue
                with self._lock:
                    self._position[idx], self._velocity[idx], self._torque[idx] = decoded
                    self._timestamp = time.monotonic()
                    self._seen.add(rid)
        except Exception as exc:  # surfaced by snapshot/preflight
            self.error = exc

    def snapshot(self) -> ArmSnapshot:
        if self.error is not None:
            raise RuntimeError(f"{self.label} CAN reader failed: {self.error}") from self.error
        with self._lock:
            return ArmSnapshot(
                self._position.copy(), self._velocity.copy(), self._torque.copy(),
                self._timestamp, tuple(sorted(self._seen)),
            )

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snap = self.snapshot()
            if len(snap.seen_ids) == 8 and np.isfinite(snap.position).all():
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None

    def __enter__(self) -> "PassiveCANReader":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.close()


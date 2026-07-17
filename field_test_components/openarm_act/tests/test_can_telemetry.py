from __future__ import annotations

import math
import socket
import time
from typing import Any

import pytest
from lerobot_robot_openarm_can_passive.broker import (
    TelemetryBroker,
    TelemetryError,
    TelemetrySnapshot,
)
from lerobot_robot_openarm_can_passive.features import MOTOR_NAMES, motor_features, motor_values

from ros2_adapter.openarm_can_telemetry_udp import (
    ArmTelemetry,
    decode_uint16_position,
    is_special_motor_command,
    open_receive_only_can,
)


def encode_position(position: float) -> tuple[int, int]:
    encoded = round((position + 12.5) / 25.0 * 65535.0)
    return (encoded >> 8) & 0xFF, encoded & 0xFF


def make_packet(*, sequence: int = 7, value: float = 0.25) -> dict[str, Any]:
    now_ns = time.monotonic_ns()
    vector = [value + index * 0.01 for index in range(8)]
    timestamps = [now_ns] * 8
    return {
        "schema_version": 1,
        "sequence": sequence,
        "timestamp_ns": time.time_ns(),
        "monotonic_ns": now_ns,
        "unit": "radian",
        "left_command": vector,
        "right_command": vector,
        "left_state": vector,
        "right_state": vector,
        "left_command_time_ns": timestamps,
        "right_command_time_ns": timestamps,
        "left_state_time_ns": timestamps,
        "right_state_time_ns": timestamps,
    }


def test_decode_uint16_position_endpoints_and_midpoint() -> None:
    assert decode_uint16_position(0x00, 0x00) == pytest.approx(-12.5)
    assert decode_uint16_position(0xFF, 0xFF) == pytest.approx(12.5)
    assert decode_uint16_position(0x80, 0x00) == pytest.approx(0.0, abs=2e-4)


def test_command_and_state_ids_preserve_left_right_motor_order() -> None:
    left = ArmTelemetry()
    right = ArmTelemetry()
    timestamp_ns = time.monotonic_ns()
    for index in range(8):
        left_position = -0.7 + index * 0.1
        right_position = 0.7 - index * 0.1
        left_high, left_low = encode_position(left_position)
        right_high, right_low = encode_position(right_position)

        left.update(index + 1, bytes((left_high, left_low, 0, 0, 0, 0, 0, 0)), timestamp_ns)
        left.update(index + 17, bytes((index + 1, left_high, left_low, 0, 0, 0, 0, 0)), timestamp_ns)
        right.update(index + 1, bytes((right_high, right_low, 0, 0, 0, 0, 0, 0)), timestamp_ns)
        right.update(index + 17, bytes((index + 1, right_high, right_low, 0, 0, 0, 0, 0)), timestamp_ns)

    assert left.complete and right.complete
    left_fields = left.packet_fields("left")
    right_fields = right.packet_fields("right")
    assert len(left_fields["left_command"]) == 8
    assert len(left_fields["left_state"]) == 8
    assert len(right_fields["right_command"]) == 8
    assert len(right_fields["right_state"]) == 8
    assert left_fields["left_command"][0] == pytest.approx(-0.7, abs=4e-4)
    assert left_fields["left_command"][7] == pytest.approx(0.0, abs=4e-4)
    assert right_fields["right_command"][0] == pytest.approx(0.7, abs=4e-4)
    assert right_fields["right_command"][7] == pytest.approx(0.0, abs=4e-4)


def test_special_enable_disable_zero_commands_are_never_recorded_as_actions() -> None:
    telemetry = ArmTelemetry()
    for command in (0xFC, 0xFD, 0xFE):
        data = b"\xff" * 7 + bytes((command,))
        assert is_special_motor_command(data)
        telemetry.update(1, data, time.monotonic_ns())
    assert telemetry.command[0] is None


def test_feature_schema_is_exactly_left_eight_then_right_eight() -> None:
    expected = tuple(f"{side}_{motor}.pos" for side in ("left", "right") for motor in MOTOR_NAMES)
    assert tuple(motor_features()) == expected
    values = motor_values(list(range(8)), list(range(8, 16)))
    assert tuple(values) == expected
    assert tuple(values.values()) == tuple(float(value) for value in range(16))


def test_broker_rejects_nan_and_wrong_unit() -> None:
    broker = TelemetryBroker("127.0.0.1", 29502, stale_timeout_s=0.1)
    packet = make_packet()
    packet["left_command"][3] = math.nan
    with pytest.raises(TelemetryError, match="NaN or Inf"):
        broker._decode_json(packet)

    packet = make_packet()
    packet["unit"] = "degree"
    with pytest.raises(TelemetryError, match="unit must be radian"):
        broker._decode_json(packet)


def test_broker_rejects_stale_snapshot() -> None:
    broker = TelemetryBroker("127.0.0.1", 29502, stale_timeout_s=0.01)
    broker.snapshot = TelemetrySnapshot(
        sequence=1,
        received_monotonic=time.monotonic() - 1.0,
        state=motor_values([0.0] * 8, [0.0] * 8),
        action=motor_values([0.0] * 8, [0.0] * 8),
        maximum_source_age_s=0.0,
    )
    with pytest.raises(TimeoutError, match="telemetry is stale"):
        broker.require_fresh()


def test_can_socket_setup_has_no_transmit_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []

    class FakeSocket:
        def setsockopt(self, *args: Any) -> None:
            calls.append(("setsockopt", args))

        def bind(self, address: Any) -> None:
            calls.append(("bind", address))

        def setblocking(self, enabled: bool) -> None:
            calls.append(("setblocking", enabled))

    fake_socket = FakeSocket()
    monkeypatch.setattr(socket, "socket", lambda *_args: fake_socket)
    result = open_receive_only_can("can-test")

    assert result is fake_socket
    assert [name for name, _value in calls] == ["setsockopt", "bind", "setblocking"]
    assert calls[1] == ("bind", ("can-test",))
    assert calls[2] == ("setblocking", False)

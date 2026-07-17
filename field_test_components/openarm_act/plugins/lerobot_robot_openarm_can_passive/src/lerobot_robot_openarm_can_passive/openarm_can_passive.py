from __future__ import annotations

import logging
import math
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .broker import TelemetryBroker, register_broker, unregister_broker
from .config_openarm_can_passive import OpenArmCANPassiveConfig
from .features import motor_features

logger = logging.getLogger(__name__)


class OpenArmCANPassive(Robot):
    """Dataset-only robot facade for an independently running OpenArm bridge."""

    config_class = OpenArmCANPassiveConfig
    name = "openarm_can_passive"

    def __init__(self, config: OpenArmCANPassiveConfig):
        super().__init__(config)
        self.config = config
        self.broker = TelemetryBroker(config.telemetry_host, config.telemetry_port, config.stale_timeout_s)
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = dict(motor_features())
        for key, camera in self.config.cameras.items():
            if getattr(camera, "use_rgb", True):
                features[key] = (camera.height, camera.width, 3)
            if getattr(camera, "use_depth", False):
                features[f"{key}_depth"] = (camera.height, camera.width, 1)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return motor_features()

    @property
    def is_connected(self) -> bool:
        return self.broker.is_connected and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = False) -> None:
        del calibrate
        self.broker.connect()
        try:
            for camera in self.cameras.values():
                camera.connect()
            register_broker(self.config.broker_id, self.broker)
        except Exception:
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
            self.broker.disconnect()
            raise
        logger.info("Passive OpenArm recorder connected; no CAN transmit socket was opened")

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def setup_motors(self) -> None:
        raise PermissionError("passive recorder cannot configure motors")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        snapshot = self.broker.receive_latest()
        observation: RobotObservation = dict(snapshot.state)
        for key, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                observation[key] = camera.read_latest()
            if getattr(camera, "use_depth", False):
                observation[f"{key}_depth"] = camera.read_latest_depth()
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        expected = set(self.action_features)
        if set(action) != expected:
            raise ValueError(
                f"passive action schema mismatch: expected {sorted(expected)}, got {sorted(action)}"
            )
        if not all(math.isfinite(float(value)) for value in action.values()):
            raise ValueError("passive action contains NaN or Inf")
        # The existing bridge already sent this command. Dataset recording must not transmit it again.
        return {key: float(value) for key, value in action.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        unregister_broker(self.config.broker_id, self.broker)
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        self.broker.disconnect()
        logger.info("Passive OpenArm recorder disconnected")

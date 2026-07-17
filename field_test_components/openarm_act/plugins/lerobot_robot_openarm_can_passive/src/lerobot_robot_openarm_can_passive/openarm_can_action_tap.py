from __future__ import annotations

from functools import cached_property
from typing import Any

from lerobot.teleoperators import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .broker import TelemetryBroker, get_broker
from .config_openarm_can_action_tap import OpenArmCANActionTapConfig
from .features import motor_features


class OpenArmCANActionTap(Teleoperator):
    """Reads the exact outgoing CAN target already sent by the existing bridge."""

    config_class = OpenArmCANActionTapConfig
    name = "openarm_can_action_tap"

    def __init__(self, config: OpenArmCANActionTapConfig):
        super().__init__(config)
        self.config = config
        self.broker: TelemetryBroker | None = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        return motor_features()

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.broker is not None and self.broker.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = False) -> None:
        del calibrate
        self.broker = get_broker(self.config.broker_id)

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        assert self.broker is not None
        return dict(self.broker.require_fresh().action)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        self.broker = None

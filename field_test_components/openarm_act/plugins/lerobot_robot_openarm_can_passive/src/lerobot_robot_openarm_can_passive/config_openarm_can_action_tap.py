from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("openarm_can_action_tap")
@dataclass(kw_only=True)
class OpenArmCANActionTapConfig(TeleoperatorConfig):
    id: str | None = "openarm_can_action_tap"
    broker_id: str = "openarm_v1_bimanual"

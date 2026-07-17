from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("openarm_can_passive")
@dataclass(kw_only=True)
class OpenArmCANPassiveConfig(RobotConfig):
    id: str | None = "openarm_v1_bimanual_passive"
    broker_id: str = "openarm_v1_bimanual"
    telemetry_host: str = "127.0.0.1"
    telemetry_port: int = 29502
    stale_timeout_s: float = 0.1
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 1 <= self.telemetry_port <= 65535:
            raise ValueError("telemetry_port must be in 1..65535")
        if self.stale_timeout_s <= 0:
            raise ValueError("stale_timeout_s must be positive")

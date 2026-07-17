from pathlib import Path

import pytest

from openarm_dp3.config import ConfigError, HardwareSettings, WorkspaceSettings

ROOT = Path("/home/mh/openarm_dp3")


def test_hardware_has_only_two_real_cameras() -> None:
    hardware = HardwareSettings.load(ROOT / "configs/hardware.yaml")
    assert hardware.chest.serial == "261722072049"
    assert hardware.wrist_left.serial == "260322271498"
    assert hardware.chest.mount == "chest_fixed"
    assert hardware.wrist_left.mount == "left_end_effector"


def test_uncalibrated_workspace_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
    path.write_text(
        "schema_version: 1\ncalibrated: false\ncamera: chest\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="not calibrated"):
        WorkspaceSettings.load(path)

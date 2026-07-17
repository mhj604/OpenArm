from pathlib import Path

import yaml


def test_exactly_two_expected_cameras() -> None:
    with Path("/home/mh/openarm_act/configs/cameras.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert set(config["cameras"]) == {"chest", "wrist_left"}
    assert config["cameras"]["chest"]["model"] == "Intel RealSense D435ICB"
    assert config["cameras"]["chest"]["serial_number"] == "261722072049"
    assert config["cameras"]["wrist_left"]["model"] == "Intel RealSense D405"
    assert config["cameras"]["wrist_left"]["serial_number"] == "260322271498"
    assert all(camera["use_depth"] is False for camera in config["cameras"].values())

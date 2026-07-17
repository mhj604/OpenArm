from __future__ import annotations

import numpy as np

from openarm_imitation.can_reader import MOTOR_TYPES, decode_motor_state
from openarm_imitation.config import load_config
from openarm_imitation.pointcloud import Intrinsics, depth_to_pointcloud, resize_rgb_nearest


def test_config_is_classic_can():
    cfg = load_config("/home/mh/openarm_imitation/config/hardware.yaml")
    assert cfg["robot"]["can_fd"] is False
    assert cfg["robot"]["right_can"] == "can0"
    assert cfg["robot"]["left_can"] == "can1"


def test_decode_neutral_frame_is_finite():
    decoded = decode_motor_state(bytes([1, 0x80, 0, 0x80, 0, 0x80, 0, 0]), MOTOR_TYPES[0])
    assert decoded is not None
    assert np.isfinite(decoded).all()


def test_pointcloud_shape_and_bounds():
    depth = np.full((48, 64), 0.6, dtype=np.float32)
    rgb = np.full((48, 64, 3), 127, dtype=np.uint8)
    intr = Intrinsics(60.0, 60.0, 32.0, 24.0, 64, 48)
    cloud = depth_to_pointcloud(
        depth, rgb, intr, points=128,
        bounds={"x": [-1, 1], "y": [-1, 1], "z": [0.2, 1.0]},
        stride=2, rng=np.random.default_rng(1),
    )
    assert cloud.shape == (128, 6)
    assert np.allclose(cloud[:, 2], 0.6)


def test_resize_rgb():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    assert resize_rgb_nearest(rgb, 320, 240).shape == (240, 320, 3)


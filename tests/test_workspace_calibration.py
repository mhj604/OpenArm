import cv2
import numpy as np

from openarm_dp3.calibrate_workspace import detect_red_workspace


def test_detects_center_rectangle_and_left_target_point() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(rgb, (300, 180), (500, 340), (255, 0, 0), 8)
    cv2.circle(rgb, (170, 270), 16, (255, 0, 0), -1)

    roi, preview = detect_red_workspace(rgb)

    assert roi[0] < 170 < roi[2]
    assert roi[0] < 300 < 500 < roi[2]
    assert roi[1] < 270 < roi[3]
    assert preview.shape == rgb.shape

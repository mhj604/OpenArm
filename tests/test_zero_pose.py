import json
from pathlib import Path

import numpy as np

from openarm_dp3.rollout_profile import RolloutProfile
from openarm_dp3.zero_pose import STATE_ORDER, ZeroPose


def test_zero_pose_uses_zero_arm_targets_and_open_gripper_bounds(tmp_path: Path) -> None:
    path = tmp_path / "zero.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "unit": "radian",
                "source": "test bridge",
                "state_order": list(STATE_ORDER),
                "target": [0.0] * 16,
            }
        ),
        encoding="utf-8",
    )
    lower = np.full(16, -1.0)
    upper = np.full(16, 1.0)
    upper[[7, 15]] = -0.0002
    profile = RolloutProfile(
        action_min=lower,
        action_max=upper,
        start_min=lower,
        start_max=upper,
        start_mean=np.zeros(16),
    )
    target = ZeroPose.load(path).action_target(profile)
    np.testing.assert_allclose(target[[*range(7), *range(8, 15)]], 0.0)
    np.testing.assert_allclose(target[[7, 15]], -0.0002)

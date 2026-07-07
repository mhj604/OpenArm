"""Gym registrations for OpenArm TestUSD pick-and-place."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-PickPlace-OpenArm-TestUSD-Bi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.testusd_pick_place_env_cfg:OpenArmTestUsdPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmTestUsdPickPlacePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-PickPlace-OpenArm-TestUSD-Bi-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.testusd_pick_place_env_cfg:OpenArmTestUsdPickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmTestUsdPickPlacePPORunnerCfg",
    },
)


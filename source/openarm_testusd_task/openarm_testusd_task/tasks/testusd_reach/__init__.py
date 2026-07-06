"""Gym registrations for original TestUSD OpenArm bimanual reach."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Reach-OpenArm-TestUSD-Bi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.testusd_reach_env_cfg:OpenArmTestUsdReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmTestUsdReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-OpenArm-TestUSD-Bi-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.testusd_reach_env_cfg:OpenArmTestUsdReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmTestUsdReachPPORunnerCfg",
    },
)

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class OpenArmTestUsdPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 25
    experiment_name = "openarm_testusd_bi_pick_place"
    run_name = ""
    resume = False
    clip_actions = 1.0
    empirical_normalization = None
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.30,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.25,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0001,
        num_learning_epochs=2,
        num_mini_batches=8,
        learning_rate=5.0e-6,
        schedule="fixed",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.25,
    )


@configclass
class OpenArmTestUsdPickPlaceStagePPORunnerCfg(OpenArmTestUsdPickPlacePPORunnerCfg):
    num_steps_per_env = 48
    max_iterations = 5000
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.15,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.25,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0001,
        num_learning_epochs=2,
        num_mini_batches=24,
        learning_rate=5.0e-6,
        schedule="fixed",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.25,
    )

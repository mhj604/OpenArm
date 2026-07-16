from __future__ import annotations

import inspect

import torch
from rsl_rl.algorithms import PPO


class AuthorityMaskedPPO(PPO):
    """Train the actor only on transitions where its arm target controls the robot."""

    def __init__(
        self,
        *args,
        authority_observation_group: str = "authority",
        **kwargs,
    ):
        ppo_arguments = inspect.signature(PPO.__init__).bind(None, *args, **kwargs)
        ppo_arguments.apply_defaults()
        if ppo_arguments.arguments["entropy_coef"] != 0.0:
            raise ValueError(
                "AuthorityMaskedPPO requires entropy_coef=0 because entropy is not "
                "masked by actor authority"
            )
        if ppo_arguments.arguments["schedule"] != "fixed":
            raise ValueError(
                "AuthorityMaskedPPO requires schedule='fixed' because adaptive KL "
                "includes non-authoritative transitions"
            )
        if ppo_arguments.arguments["symmetry_cfg"] is not None:
            raise ValueError(
                "AuthorityMaskedPPO requires symmetry_cfg=None because symmetry "
                "actor losses are not masked by actor authority"
            )
        super().__init__(*args, **kwargs)
        if self.normalize_advantage_per_mini_batch:
            raise ValueError(
                "AuthorityMaskedPPO requires rollout-wide authority-aware advantage normalization"
            )
        self.authority_observation_group = authority_observation_group
        self.actor_authority_fraction = 0.0
        actor_parameters = list(self.policy.actor.parameters())
        for noise_parameter_name in ("std", "log_std"):
            noise_parameter = getattr(self.policy, noise_parameter_name, None)
            if isinstance(noise_parameter, torch.nn.Parameter) and all(
                noise_parameter is not parameter for parameter in actor_parameters
            ):
                actor_parameters.append(noise_parameter)
        self._actor_parameters = tuple(actor_parameters)

    def compute_returns(self, obs) -> None:
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=False,
        )

        if self.authority_observation_group not in self.storage.observations.keys():
            raise KeyError(
                "AuthorityMaskedPPO requires the observation group "
                f"'{self.authority_observation_group}'"
            )
        authority = self.storage.observations[self.authority_observation_group]
        if authority.shape[-1] != 1:
            raise ValueError(
                "Actor authority observation must have one scalar per transition, "
                f"got shape {tuple(authority.shape)}"
            )
        authority_mask = authority[..., 0] > 0.5
        advantages = self.storage.advantages
        authoritative_advantages = advantages[authority_mask].clone()
        advantages.zero_()
        local_count = torch.tensor(
            float(authoritative_advantages.numel()), device=advantages.device
        )
        local_sum = authoritative_advantages.sum()
        local_sum_squared = torch.square(authoritative_advantages).sum()
        global_stats = torch.stack((local_count, local_sum, local_sum_squared))
        if self.is_multi_gpu:
            torch.distributed.all_reduce(global_stats, op=torch.distributed.ReduceOp.SUM)
        global_count, global_sum, global_sum_squared = global_stats
        total_count = authority_mask.numel() * self.gpu_world_size
        self.actor_authority_fraction = (
            global_count / max(total_count, 1)
        ).item()
        if global_count.item() == 0:
            return
        global_mean = global_sum / global_count
        global_variance = torch.clamp(
            global_sum_squared / global_count - torch.square(global_mean),
            min=0.0,
        )
        normalized = (authoritative_advantages - global_mean) / (
            torch.sqrt(global_variance) + 1.0e-8
        )
        advantages[authority_mask] = normalized

    def update(self) -> dict[str, float]:
        actor_parameters_before = tuple(
            parameter.detach().clone() for parameter in self._actor_parameters
        )
        loss_dict = super().update()
        loss_dict["actor_authority_fraction"] = self.actor_authority_fraction
        with torch.no_grad():
            squared_delta = sum(
                torch.sum(torch.square(parameter - before))
                for parameter, before in zip(
                    self._actor_parameters, actor_parameters_before, strict=True
                )
            )
        loss_dict["actor_parameter_delta_norm"] = torch.sqrt(squared_delta).item()
        return loss_dict


def register_openarm_rsl_rl_classes() -> None:
    """Register task-local classes in the namespace used by RSL-RL runner eval()."""

    import rsl_rl.runners.on_policy_runner as runner_module

    for name, cls in (("AuthorityMaskedPPO", AuthorityMaskedPPO),):
        existing = getattr(runner_module, name, None)
        if existing is not None and existing is not cls:
            raise RuntimeError(f"RSL-RL runner class name already registered: {name}")
        setattr(runner_module, name, cls)

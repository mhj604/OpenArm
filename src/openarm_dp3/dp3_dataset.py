from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer


class OpenArmDP3Dataset(BaseDataset):
    """DP3 dataset containing chest XYZ, 16-radian state, and 16-radian action."""

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: int | None = None,
        task_name: str | None = None,
    ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=["state", "action", "point_cloud"],
        )
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self) -> OpenArmDP3Dataset:
        validation = copy.copy(self)
        validation.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        validation.train_mask = ~self.train_mask
        return validation

    def get_normalizer(self, mode: str = "limits", **kwargs: Any) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "action": self.replay_buffer["action"],
                "agent_pos": self.replay_buffer["state"],
                "point_cloud": self.replay_buffer["point_cloud"],
            },
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    @staticmethod
    def _sample_to_data(sample: dict[str, np.ndarray]) -> dict[str, Any]:
        state = sample["state"].astype(np.float32)
        action = sample["action"].astype(np.float32)
        point_cloud = sample["point_cloud"].astype(np.float32)
        if state.shape[-1] != 16 or action.shape[-1] != 16:
            raise ValueError("OpenArm state and action must each have 16 channels")
        if point_cloud.shape[-2:] != (1024, 3):
            raise ValueError("OpenArm DP3 point cloud must be shaped (..., 1024, 3)")
        return {
            "obs": {"point_cloud": point_cloud, "agent_pos": state},
            "action": action,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(index)
        return dict_apply(self._sample_to_data(sample), torch.from_numpy)

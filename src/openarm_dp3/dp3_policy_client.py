from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import socket
import sys
import time
import unicodedata
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import dill
import hydra
import numpy as np
import torch

from .config import CollectionSettings, HardwareSettings, WorkspaceSettings
from .learned_control import (
    amplify_learned_chunk,
    slew_zero_target,
    stabilize_learned_chunk,
)
from .pointcloud import sample_aligned_vertices, voxel_sample_fixed
from .realsense import RealSenseRGBD, RGBDFrame
from .rollout_profile import VECTOR_SIZE, RolloutProfile
from .workspace_reference import (
    StableBooleanGate,
    WorkspaceReferenceError,
    WorkspaceReferences,
    load_workspace_references,
)
from .zero_pose import ZeroPose

LOGGER = logging.getLogger("openarm_dp3.policy")
STOP_REQUESTED = Event()
ARM_INDICES = np.asarray([*range(7), *range(8, 15)], dtype=np.int64)
LEFT_GRIPPER_INDEX = 7


@dataclass(frozen=True)
class PolicyBundle:
    policy: Any
    observation_steps: int
    action_steps: int
    prediction_steps: int
    point_count: int
    point_channels: int


@dataclass(frozen=True)
class LearnedCycleResult:
    sequence: int
    state: np.ndarray
    final_target: np.ndarray
    elapsed_s: float
    command_count: int


def request_stop(_signum: int, _frame: Any) -> None:
    STOP_REQUESTED.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run learned DP3 pick/place cycles from the configured OpenArm zero pose."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--zero-pose", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, default=0)
    parser.add_argument("--state-port", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-timeout", type=float, default=0.75)
    parser.add_argument("--selection-frames", type=int, default=10)
    parser.add_argument("--start-centroid-frames", type=int, default=5)
    parser.add_argument("--goal-centroid-frames", type=int, default=15)
    parser.add_argument("--minimum-yellow-points", type=int, default=8)
    parser.add_argument("--start-region-margin", type=float, default=0.015)
    parser.add_argument("--maximum-goal-distance", type=float, default=0.050)
    parser.add_argument("--goal-minimum-elapsed", type=float, default=3.0)
    parser.add_argument("--goal-stable-frames", type=int, default=8)
    parser.add_argument("--goal-open-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-gripper-open-seconds", type=float, default=2.5)
    parser.add_argument("--maximum-initial-zero-delta", type=float, default=1.50)
    parser.add_argument("--zero-tolerance", type=float, default=0.08)
    parser.add_argument("--zero-stable-frames", type=int, default=5)
    parser.add_argument("--zero-timeout", type=float, default=25.0)
    parser.add_argument("--maximum-tracking-error", type=float, default=0.90)
    parser.add_argument("--tracking-error-frames", type=int, default=10)
    parser.add_argument("--restart-stable-frames", type=int, default=15)
    parser.add_argument("--restart-cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--restart-report-seconds", type=float, default=5.0)
    parser.add_argument("--inference-seed", type=int, default=42)
    parser.add_argument("--inference-samples", type=int, default=5)
    parser.add_argument("--execute-full-horizon", action="store_true")
    parser.add_argument("--arm-action-gain", type=float, default=1.0)
    parser.add_argument("--initial-ramp-seconds", type=float, default=1.0)
    parser.add_argument("--initial-action-alpha", type=float, default=0.25)
    parser.add_argument("--action-alpha", type=float, default=0.70)
    parser.add_argument("--initial-max-arm-step", type=float, default=0.025)
    parser.add_argument("--policy-max-arm-step", type=float, default=0.08)
    parser.add_argument("--zero-max-arm-step", type=float, default=0.03)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file", type=Path, required=True)
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
    )


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            pickle_module=dill,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu", pickle_module=dill)
    if not isinstance(payload, dict):
        raise ValueError("DP3 checkpoint payload must be a mapping")
    return payload


def load_policy(checkpoint: Path, device: str) -> PolicyBundle:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing DP3 checkpoint: {checkpoint}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable to PyTorch. On this host, run "
            "'/sbin/ub-device-create --verbose' if NVIDIA device nodes are missing."
        )
    LOGGER.info("Loading DP3 checkpoint: %s", checkpoint)
    payload = _torch_load(checkpoint)
    cfg = payload.get("cfg")
    state_dicts = payload.get("state_dicts")
    if cfg is None or not isinstance(state_dicts, dict):
        raise ValueError("DP3 checkpoint is missing cfg or state_dicts")
    weights = state_dicts.get("ema_model") or state_dicts.get("model")
    if not isinstance(weights, dict):
        raise ValueError("DP3 checkpoint contains no EMA/model weights")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(weights, strict=True)
    policy.to(device).eval()
    policy.requires_grad_(False)

    point_shape = tuple(int(value) for value in cfg.task.shape_meta.obs.point_cloud.shape)
    action_shape = tuple(int(value) for value in cfg.task.shape_meta.action.shape)
    observation_steps = int(cfg.n_obs_steps)
    action_steps = int(cfg.n_action_steps)
    prediction_steps = int(cfg.horizon)
    if len(point_shape) != 2 or action_shape != (VECTOR_SIZE,):
        raise ValueError(
            f"unsupported checkpoint shapes: point_cloud={point_shape}, action={action_shape}"
        )
    if observation_steps <= 0 or action_steps <= 0 or prediction_steps <= 0:
        raise ValueError("checkpoint observation/action/prediction step counts must be positive")
    if observation_steps > prediction_steps:
        raise ValueError("checkpoint observation steps exceed its prediction horizon")
    return PolicyBundle(
        policy=policy,
        observation_steps=observation_steps,
        action_steps=action_steps,
        prediction_steps=prediction_steps,
        point_count=point_shape[0],
        point_channels=point_shape[1],
    )


def send_packet(sock: socket.socket, address: tuple[str, int], packet: dict[str, Any]) -> None:
    encoded = json.dumps(packet, separators=(",", ":")).encode("utf-8")
    sock.sendto(encoded, address)


def send_action(
    sock: socket.socket,
    address: tuple[str, int],
    sequence: int,
    action: np.ndarray,
) -> None:
    send_packet(
        sock,
        address,
        {
            "schema_version": 1,
            "type": "action",
            "sequence": sequence,
            "monotonic_ns": time.monotonic_ns(),
            "unit": "radian",
            "action": [float(value) for value in action],
        },
    )


def _decode_state(
    raw: bytes,
    address: tuple[str, int],
    host: str,
) -> tuple[int, np.ndarray, float] | None:
    if address[0] != host:
        return None
    packet = json.loads(raw.decode("utf-8"))
    if packet.get("type") != "state" or packet.get("unit") != "radian":
        return None
    state = np.asarray(packet.get("state"), dtype=np.float32)
    if state.shape != (VECTOR_SIZE,) or not np.isfinite(state).all():
        return None
    sequence = int(packet["sequence"])
    age_s = (time.monotonic_ns() - int(packet["monotonic_ns"])) / 1_000_000_000
    return sequence, state, age_s


def receive_state(
    sock: socket.socket,
    *,
    host: str,
    timeout_s: float,
) -> tuple[int, np.ndarray]:
    deadline = time.monotonic() + timeout_s
    newest: tuple[int, np.ndarray, float] | None = None
    while time.monotonic() < deadline:
        sock.settimeout(max(0.001, deadline - time.monotonic()))
        try:
            raw, address = sock.recvfrom(65535)
        except TimeoutError:
            break
        candidate = _decode_state(raw, address, host)
        if candidate is not None:
            newest = candidate
        sock.setblocking(False)
        try:
            while True:
                try:
                    raw, address = sock.recvfrom(65535)
                except BlockingIOError:
                    break
                candidate = _decode_state(raw, address, host)
                if candidate is not None:
                    newest = candidate
        finally:
            sock.setblocking(True)
        if newest is not None:
            break
    if newest is None:
        raise TimeoutError("no live OpenArm state received")
    sequence, state, age_s = newest
    if age_s < 0 or age_s > timeout_s:
        raise TimeoutError(f"OpenArm state packet is stale ({age_s:.3f}s)")
    return sequence, state


def build_point_cloud(
    frame: RGBDFrame,
    *,
    collection: CollectionSettings,
    workspace: WorkspaceSettings,
    point_count: int,
    point_channels: int,
    seed: int,
) -> np.ndarray:
    include_color = point_channels == 6
    if point_channels not in {3, 6}:
        raise ValueError(f"unsupported point-cloud channel count: {point_channels}")
    if include_color != collection.use_point_color:
        raise ValueError("checkpoint point-cloud channels differ from collection config")
    raw = sample_aligned_vertices(
        frame.vertices_xyz,
        height=frame.depth_raw.shape[0],
        width=frame.depth_raw.shape[1],
        pixel_roi=workspace.pixel_roi,
        depth_range_m=workspace.depth_range_m,
        point_count=collection.chest_raw_points,
        minimum_valid_points=collection.minimum_valid_depth_points,
        seed=seed,
        rgb=frame.rgb,
        include_color=include_color,
    )
    return voxel_sample_fixed(
        raw,
        point_count=point_count,
        voxel_size_m=collection.voxel_size_m,
        seed=seed,
    )


def predict_action_chunk(
    bundle: PolicyBundle,
    point_history: Sequence[np.ndarray],
    state_history: Sequence[np.ndarray],
    *,
    device: str,
    inference_seed: int,
    sample_count: int = 1,
    full_prediction_horizon: bool = False,
) -> np.ndarray:
    if (
        len(point_history) != bundle.observation_steps
        or len(state_history) != bundle.observation_steps
    ):
        raise ValueError("observation history does not match checkpoint n_obs_steps")
    if sample_count <= 0:
        raise ValueError("DP3 inference sample count must be positive")
    point_cloud = torch.from_numpy(np.stack(point_history)).unsqueeze(0).to(device)
    agent_pos = torch.from_numpy(np.stack(state_history)).unsqueeze(0).to(device)
    if sample_count > 1:
        point_cloud = point_cloud.expand(sample_count, *point_cloud.shape[1:]).contiguous()
        agent_pos = agent_pos.expand(sample_count, *agent_pos.shape[1:]).contiguous()
    cuda_devices = list(range(torch.cuda.device_count())) if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(inference_seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(inference_seed)
        with torch.inference_mode():
            prediction = bundle.policy.predict_action(
                {"point_cloud": point_cloud, "agent_pos": agent_pos}
            )
    if full_prediction_horizon:
        full = prediction.get("action_pred")
        if not isinstance(full, torch.Tensor):
            raise RuntimeError("DP3 policy did not expose its full action prediction")
        start = bundle.observation_steps - 1
        result = full[:, start:]
        expected_steps = bundle.prediction_steps - start
    else:
        result = prediction.get("action")
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("DP3 policy did not expose its action chunk")
        expected_steps = bundle.action_steps
    expected_samples = (sample_count, expected_steps, VECTOR_SIZE)
    if tuple(result.shape) != expected_samples or not torch.isfinite(result).all():
        raise RuntimeError(f"DP3 produced an invalid sampled action tensor: {tuple(result.shape)}")
    reduced = torch.median(result, dim=0).values if sample_count > 1 else result[0]
    action = reduced.detach().to("cpu").numpy().astype(np.float32, copy=False)
    expected = (expected_steps, VECTOR_SIZE)
    if action.shape != expected or not np.isfinite(action).all():
        raise RuntimeError(f"DP3 produced an invalid action chunk: {action.shape}")
    return action


def _fill_history(history: deque[np.ndarray], value: np.ndarray, size: int) -> None:
    while len(history) < size:
        history.appendleft(value.copy())


def capture_point_clouds(
    camera: RealSenseRGBD,
    *,
    collection: CollectionSettings,
    workspace: WorkspaceSettings,
    point_count: int,
    point_channels: int,
    frame_count: int,
    seed_start: int,
) -> list[np.ndarray]:
    return [
        build_point_cloud(
            camera.read(),
            collection=collection,
            workspace=workspace,
            point_count=point_count,
            point_channels=point_channels,
            seed=seed_start + index,
        )
        for index in range(frame_count)
    ]


def combine_point_clouds(clouds: Sequence[np.ndarray]) -> np.ndarray:
    if not clouds:
        raise ValueError("at least one point cloud is required")
    return np.ascontiguousarray(np.concatenate(tuple(clouds), axis=0), dtype=np.float32)


def normalize_active_confirmation(value: str) -> str:
    """Remove terminal paste/formatting artifacts without weakening START confirmation."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\x1b[200~", "").replace("\x1b[201~", "")
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) not in {"Cc", "Cf"}
    )
    return normalized.strip()


def _confirm_active_rollout(duration: float, *, continuous: bool) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("active DP3 rollout requires an interactive terminal")
    if STOP_REQUESTED.is_set():
        raise KeyboardInterrupt
    task_limit = "no task timeout" if duration <= 0 else f"a {duration:.1f}s task timeout"
    print("\nLEARNED DP3 IS READY. The next confirmation enables and moves BOTH OpenArm arms.")
    print("Each cycle: ZERO POSE -> DP3 PICK/PLACE -> ZERO POSE -> wait for the next cube.")
    if continuous:
        print(f"Continuous mode is active with {task_limit}; it stays alive until Ctrl+C.")
    else:
        print(f"One learned cycle will run with {task_limit}.")
    print("Recorded action trajectories are not replayed.")
    print("Emergency stop: Ctrl+C or physical power/E-stop.")
    raw_confirmation = input("Type START exactly to begin: ")
    confirmation = normalize_active_confirmation(raw_confirmation)
    if STOP_REQUESTED.is_set():
        raise KeyboardInterrupt
    if confirmation != "START":
        raise RuntimeError(
            "active rollout was not confirmed; motors were not opened; "
            f"received={raw_confirmation!r}"
        )


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.fps,
        args.state_timeout,
        args.selection_frames,
        args.start_centroid_frames,
        args.goal_centroid_frames,
        args.minimum_yellow_points,
        args.maximum_goal_distance,
        args.goal_stable_frames,
        args.maximum_initial_zero_delta,
        args.zero_tolerance,
        args.zero_stable_frames,
        args.zero_timeout,
        args.maximum_tracking_error,
        args.tracking_error_frames,
        args.restart_stable_frames,
        args.restart_report_seconds,
        args.inference_samples,
        args.arm_action_gain,
        args.initial_action_alpha,
        args.action_alpha,
        args.initial_max_arm_step,
        args.policy_max_arm_step,
        args.zero_max_arm_step,
    )
    if any(float(value) <= 0 or not math.isfinite(float(value)) for value in positive):
        raise ValueError("DP3 timing, count, and limit arguments must be finite and positive")
    nonnegative = (
        args.duration,
        args.start_region_margin,
        args.goal_minimum_elapsed,
        args.minimum_gripper_open_seconds,
        args.restart_cooldown_seconds,
        args.initial_ramp_seconds,
    )
    if any(float(value) < 0 or not math.isfinite(float(value)) for value in nonnegative):
        raise ValueError("DP3 duration, margins, and cooldowns cannot be negative")
    if not 0.0 < args.goal_open_fraction <= 1.0:
        raise ValueError("goal open fraction must be in (0, 1]")
    if args.initial_action_alpha > 1.0 or args.action_alpha > 1.0:
        raise ValueError("learned action alpha must not exceed 1")
    if not args.dry_run:
        if not 1 <= args.action_port <= 65535 or not 1 <= args.state_port <= 65535:
            raise ValueError("active rollout requires valid action and state ports")
        if args.action_port == args.state_port:
            raise ValueError("action and state UDP ports must differ")


def zero_pose_error(state: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(state[ARM_INDICES] - target[ARM_INDICES])))


def align_to_zero_pose(
    *,
    action_socket: socket.socket,
    state_socket: socket.socket,
    motor_address: tuple[str, int],
    target: np.ndarray,
    initial_state: np.ndarray,
    host: str,
    timeout_s: float,
    state_timeout_s: float,
    tolerance_rad: float,
    stable_frames_required: int,
    maximum_arm_step_rad: float,
    sequence: int,
) -> tuple[int, np.ndarray]:
    deadline = time.monotonic() + timeout_s
    stable_frames = 0
    last_state: np.ndarray | None = None
    command = np.asarray(initial_state, dtype=np.float32).copy()
    report_counter = 0
    while time.monotonic() < deadline and not STOP_REQUESTED.is_set():
        command = slew_zero_target(
            command,
            target,
            maximum_arm_step_rad=maximum_arm_step_rad,
        )
        send_action(action_socket, motor_address, sequence, command)
        sequence += 1
        _state_sequence, last_state = receive_state(
            state_socket, host=host, timeout_s=state_timeout_s
        )
        error = zero_pose_error(last_state, target)
        stable_frames = stable_frames + 1 if error <= tolerance_rad else 0
        report_counter += 1
        if report_counter == 1 or report_counter % 15 == 0:
            LOGGER.info(
                "zero-pose alignment max arm error=%.3frad stable=%d/%d",
                error,
                stable_frames,
                stable_frames_required,
            )
        if stable_frames >= stable_frames_required:
            return sequence, last_state
    if STOP_REQUESTED.is_set():
        raise KeyboardInterrupt
    error = math.inf if last_state is None else zero_pose_error(last_state, target)
    raise TimeoutError(
        f"zero-pose alignment did not settle within {timeout_s:.1f}s; "
        f"last maximum arm error={error:.3f}rad"
    )


def _left_gripper_is_open(
    state: np.ndarray,
    profile: RolloutProfile,
    *,
    open_fraction: float,
) -> bool:
    lower = float(profile.action_min[LEFT_GRIPPER_INDEX])
    upper = float(profile.action_max[LEFT_GRIPPER_INDEX])
    threshold = upper - (upper - lower) * open_fraction
    return float(state[LEFT_GRIPPER_INDEX]) >= threshold


def apply_initial_gripper_open_guard(
    action: np.ndarray,
    profile: RolloutProfile,
    *,
    elapsed_s: float,
    minimum_open_s: float,
) -> np.ndarray:
    """Keep the left gripper open during the demonstrated approach-only interval."""
    guarded = np.asarray(action, dtype=np.float32).copy()
    if guarded.shape != (VECTOR_SIZE,) or not np.isfinite(guarded).all():
        raise ValueError("learned action must contain 16 finite values")
    if elapsed_s < 0 or minimum_open_s < 0:
        raise ValueError("gripper guard timing cannot be negative")
    if elapsed_s < minimum_open_s:
        guarded[LEFT_GRIPPER_INDEX] = profile.action_max[LEFT_GRIPPER_INDEX]
    return guarded


def run_learned_cycle(
    *,
    camera: RealSenseRGBD,
    collection: CollectionSettings,
    workspace: WorkspaceSettings,
    references: WorkspaceReferences,
    profile: RolloutProfile,
    bundle: PolicyBundle,
    action_socket: socket.socket,
    state_socket: socket.socket,
    motor_address: tuple[str, int],
    zero_target: np.ndarray,
    initial_state: np.ndarray,
    initial_point_history: Sequence[np.ndarray],
    initial_sequence: int,
    args: argparse.Namespace,
    cycle_index: int,
) -> LearnedCycleResult:
    points: deque[np.ndarray] = deque(
        initial_point_history[-bundle.observation_steps :], maxlen=bundle.observation_steps
    )
    if not points:
        raise ValueError("learned cycle requires an initial D435 point-cloud history")
    _fill_history(points, points[-1], bundle.observation_steps)
    states: deque[np.ndarray] = deque(
        [initial_state.copy() for _index in range(bundle.observation_steps)],
        maxlen=bundle.observation_steps,
    )
    actions: deque[np.ndarray] = deque()
    goal_history: deque[np.ndarray] = deque(maxlen=args.selection_frames)
    goal_gate = StableBooleanGate(args.goal_stable_frames)
    current_state = initial_state
    final_target = initial_state.copy()
    sequence = initial_sequence
    command_count = 0
    plan_count = 0
    excessive_tracking_frames = 0
    previous_policy_target = initial_state.copy()
    started = time.monotonic()
    next_tick = started
    last_report = started

    LOGGER.warning("Learned DP3 cycle %d started from verified zero pose", cycle_index)
    while not STOP_REQUESTED.is_set():
        elapsed = time.monotonic() - started
        if args.duration > 0 and elapsed >= args.duration:
            raise TimeoutError(
                f"learned DP3 cycle {cycle_index} did not reach the goal within "
                f"{args.duration:.1f}s"
            )
        current_points = build_point_cloud(
            camera.read(),
            collection=collection,
            workspace=workspace,
            point_count=bundle.point_count,
            point_channels=bundle.point_channels,
            seed=sequence,
        )
        points.append(current_points)
        states.append(current_state.copy())
        _fill_history(points, current_points, bundle.observation_steps)
        _fill_history(states, current_state, bundle.observation_steps)

        if not actions:
            inference_started = time.monotonic()
            raw_chunk = predict_action_chunk(
                bundle,
                list(points),
                list(states),
                device=args.device,
                inference_seed=args.inference_seed,
                sample_count=args.inference_samples,
                full_prediction_horizon=args.execute_full_horizon,
            )
            amplified_chunk = amplify_learned_chunk(
                raw_chunk,
                zero_target,
                arm_gain=args.arm_action_gain,
            )
            amplified_chunk = np.clip(
                amplified_chunk, profile.action_min, profile.action_max
            ).astype(
                np.float32, copy=False
            )
            chunk = stabilize_learned_chunk(
                amplified_chunk,
                previous_policy_target,
                elapsed_s=time.monotonic() - started,
                fps=args.fps,
                initial_ramp_s=args.initial_ramp_seconds,
                initial_alpha=args.initial_action_alpha,
                normal_alpha=args.action_alpha,
                initial_max_arm_step_rad=args.initial_max_arm_step,
                normal_max_arm_step_rad=args.policy_max_arm_step,
            )
            actions.extend(chunk)
            plan_count += 1
            raw_target_delta = np.max(np.abs(raw_chunk - states[-1]), axis=0)
            amplified_target_delta = np.max(
                np.abs(amplified_chunk - states[-1]), axis=0
            )
            stabilized_steps = np.diff(np.vstack((previous_policy_target, chunk)), axis=0)
            LOGGER.info(
                "DP3 plan=%d samples=%d full_horizon=%s generated=%d "
                "gain=%.2f inference=%.3fs raw target delta left=%.3f right=%.3f; "
                "amplified target delta left=%.3f right=%.3f; "
                "stabilized step left=%.3f right=%.3frad",
                plan_count,
                args.inference_samples,
                args.execute_full_horizon,
                len(chunk),
                args.arm_action_gain,
                time.monotonic() - inference_started,
                float(np.max(raw_target_delta[:7])),
                float(np.max(raw_target_delta[8:15])),
                float(np.max(amplified_target_delta[:7])),
                float(np.max(amplified_target_delta[8:15])),
                float(np.max(np.abs(stabilized_steps[:, :7]))),
                float(np.max(np.abs(stabilized_steps[:, 8:15]))),
            )

        final_target = apply_initial_gripper_open_guard(
            actions.popleft(),
            profile,
            elapsed_s=time.monotonic() - started,
            minimum_open_s=args.minimum_gripper_open_seconds,
        )
        send_action(action_socket, motor_address, sequence, final_target)
        previous_policy_target = final_target.copy()
        sequence += 1
        command_count += 1
        next_tick += 1.0 / args.fps
        time.sleep(max(0.0, next_tick - time.monotonic()))
        _state_sequence, current_state = receive_state(
            state_socket,
            host=args.host,
            timeout_s=args.state_timeout,
        )

        tracking_error = float(
            np.max(np.abs(current_state[ARM_INDICES] - final_target[ARM_INDICES]))
        )
        excessive_tracking_frames = (
            excessive_tracking_frames + 1
            if tracking_error > args.maximum_tracking_error
            else 0
        )
        if excessive_tracking_frames >= args.tracking_error_frames:
            raise RuntimeError(
                "OpenArm did not track learned DP3 targets: "
                f"maximum arm error={tracking_error:.3f}rad"
            )

        goal_history.append(current_points)
        goal_condition = False
        goal_distance = math.inf
        if (
            time.monotonic() - started >= args.goal_minimum_elapsed
            and _left_gripper_is_open(
                current_state, profile, open_fraction=args.goal_open_fraction
            )
        ):
            try:
                observation = references.observe_cube(
                    combine_point_clouds(goal_history),
                    minimum_yellow_points=args.minimum_yellow_points,
                    start_region_margin_m=args.start_region_margin,
                    maximum_goal_distance_m=args.maximum_goal_distance,
                )
                goal_distance = observation.nearest_goal_distance_m
                goal_condition = goal_distance <= args.maximum_goal_distance
            except WorkspaceReferenceError:
                goal_condition = False
        if goal_gate.observe(goal_condition):
            elapsed = time.monotonic() - started
            LOGGER.warning(
                "Learned DP3 cycle %d completed: goal stable=%d frames distance=%.1fmm",
                cycle_index,
                goal_gate.stable_frames,
                goal_distance * 1000.0,
            )
            return LearnedCycleResult(
                sequence=sequence,
                state=current_state,
                final_target=final_target,
                elapsed_s=elapsed,
                command_count=command_count,
            )

        now = time.monotonic()
        if now - last_report >= 1.0:
            LOGGER.info(
                "learned cycle=%d elapsed=%.1fs commands=%d plans=%d "
                "tracking_error=%.3frad goal_stable=%d/%d",
                cycle_index,
                now - started,
                command_count,
                plan_count,
                tracking_error,
                goal_gate.stable_frames,
                args.goal_stable_frames,
            )
            last_report = now
    raise KeyboardInterrupt


def wait_for_cube_in_start_box(
    *,
    camera: RealSenseRGBD,
    collection: CollectionSettings,
    workspace: WorkspaceSettings,
    references: WorkspaceReferences,
    bundle: PolicyBundle,
    action_socket: socket.socket,
    state_socket: socket.socket,
    motor_address: tuple[str, int],
    zero_target: np.ndarray,
    current_state: np.ndarray,
    sequence: int,
    args: argparse.Namespace,
) -> tuple[int, np.ndarray]:
    history: deque[np.ndarray] = deque(maxlen=args.selection_frames)
    gate = StableBooleanGate(args.restart_stable_frames)
    cooldown_until = time.monotonic() + args.restart_cooldown_seconds
    last_report = 0.0
    LOGGER.warning(
        "OpenArm is holding zero pose; waiting for the yellow cube inside the start box"
    )
    while not STOP_REQUESTED.is_set():
        send_action(action_socket, motor_address, sequence, zero_target)
        sequence += 1
        _state_sequence, current_state = receive_state(
            state_socket, host=args.host, timeout_s=args.state_timeout
        )
        history.append(
            build_point_cloud(
                camera.read(),
                collection=collection,
                workspace=workspace,
                point_count=bundle.point_count,
                point_channels=bundle.point_channels,
                seed=sequence,
            )
        )
        now = time.monotonic()
        in_start = False
        observation = None
        if now >= cooldown_until and len(history) == args.selection_frames:
            try:
                observation = references.observe_cube(
                    combine_point_clouds(history),
                    minimum_yellow_points=args.minimum_yellow_points,
                    start_region_margin_m=args.start_region_margin,
                    maximum_goal_distance_m=args.maximum_goal_distance,
                )
                in_start = observation.in_start_region
            except WorkspaceReferenceError:
                in_start = False
        if gate.observe(in_start):
            assert observation is not None
            LOGGER.warning(
                "Start-box cube detected for %d stable frames at %s; next learned cycle starts",
                gate.stable_frames,
                observation.centroid_m.round(4).tolist(),
            )
            return sequence, current_state
        if now - last_report >= args.restart_report_seconds:
            if observation is None:
                LOGGER.info("Waiting for a visible yellow cube in the start box")
            else:
                LOGGER.info(
                    "cube=%s in_start=%s stable=%d/%d",
                    observation.centroid_m.round(4).tolist(),
                    observation.in_start_region,
                    gate.stable_frames,
                    args.restart_stable_frames,
                )
            last_report = now
    raise KeyboardInterrupt


def _preview_summary(
    *,
    references: WorkspaceReferences,
    clouds: Sequence[np.ndarray],
    chunk: np.ndarray,
    zero_target: np.ndarray,
    minimum_yellow_points: int,
    start_region_margin_m: float,
    maximum_goal_distance_m: float,
    minimum_gripper_open_seconds: float,
    arm_action_gain: float,
    full_prediction_horizon: bool,
) -> dict[str, object]:
    observation = references.observe_cube(
        combine_point_clouds(clouds),
        minimum_yellow_points=minimum_yellow_points,
        start_region_margin_m=start_region_margin_m,
        maximum_goal_distance_m=maximum_goal_distance_m,
    )
    delta = np.max(np.abs(chunk - zero_target), axis=0)
    transitions = np.diff(np.vstack((zero_target, chunk)), axis=0)
    return {
        "mode": "learned_dp3_preview_no_can_no_motor",
        "cube_centroid_m": observation.centroid_m.round(5).tolist(),
        "cube_in_start_region": observation.in_start_region,
        "model_generated_actions": int(len(chunk)),
        "full_prediction_horizon": full_prediction_horizon,
        "arm_action_gain": arm_action_gain,
        "maximum_left_arm_delta_rad": float(np.max(delta[:7])),
        "maximum_right_arm_delta_rad": float(np.max(delta[8:15])),
        "left_arm_step_max_rad": np.max(np.abs(transitions[:, :7]), axis=1).round(5).tolist(),
        "right_arm_step_max_rad": np.max(np.abs(transitions[:, 8:15]), axis=1)
        .round(5)
        .tolist(),
        "left_arm_targets_rad": chunk[:, :7].round(5).tolist(),
        "first_left_gripper_target_rad": float(chunk[0, 7]),
        "runtime_left_gripper_held_open_seconds": minimum_gripper_open_seconds,
        "recorded_action_replay": False,
    }


def main() -> int:
    args = parse_args()
    configure_logging(args.log_file)
    _validate_args(args)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    hardware = HardwareSettings.load(args.hardware)
    collection = CollectionSettings.load(args.collection)
    workspace = WorkspaceSettings.load(args.workspace)
    assert workspace is not None
    if workspace.camera_serial != hardware.chest.serial:
        raise ValueError("workspace calibration and configured D435 serial differ")
    profile = RolloutProfile.load(args.profile)
    zero_pose = ZeroPose.load(args.zero_pose)
    zero_target = zero_pose.action_target(profile)
    references = load_workspace_references(
        args.dataset,
        start_centroid_frames=args.start_centroid_frames,
        goal_centroid_frames=args.goal_centroid_frames,
        minimum_yellow_points=args.minimum_yellow_points,
    )
    bundle = load_policy(args.checkpoint, args.device)
    if bundle.point_count != collection.training_points:
        raise ValueError("checkpoint point count differs from collection config")

    chest = RealSenseRGBD(hardware.chest)
    state_socket: socket.socket | None = None
    action_socket: socket.socket | None = None
    motor_address = (args.host, args.action_port)
    try:
        chest.start()
        initial_clouds = capture_point_clouds(
            chest,
            collection=collection,
            workspace=workspace,
            point_count=bundle.point_count,
            point_channels=bundle.point_channels,
            frame_count=args.selection_frames,
            seed_start=0,
        )
        initial_observation = references.observe_cube(
            combine_point_clouds(initial_clouds),
            minimum_yellow_points=args.minimum_yellow_points,
            start_region_margin_m=args.start_region_margin,
            maximum_goal_distance_m=args.maximum_goal_distance,
        )
        if not initial_observation.in_start_region:
            raise WorkspaceReferenceError(
                "the yellow cube is visible but is not inside the learned start box"
            )
        point_history: deque[np.ndarray] = deque(
            initial_clouds[-bundle.observation_steps :], maxlen=bundle.observation_steps
        )
        _fill_history(point_history, initial_clouds[-1], bundle.observation_steps)
        warmup_started = time.monotonic()
        warmup_chunk = predict_action_chunk(
            bundle,
            list(point_history),
            [zero_target] * bundle.observation_steps,
            device=args.device,
            inference_seed=args.inference_seed,
            sample_count=args.inference_samples,
            full_prediction_horizon=args.execute_full_horizon,
        )
        LOGGER.info("DP3 warm-up inference completed in %.3fs", time.monotonic() - warmup_started)
        if args.dry_run:
            preview_amplified = amplify_learned_chunk(
                warmup_chunk,
                zero_target,
                arm_gain=args.arm_action_gain,
            )
            preview_chunk = stabilize_learned_chunk(
                np.clip(preview_amplified, profile.action_min, profile.action_max),
                zero_target,
                elapsed_s=0.0,
                fps=args.fps,
                initial_ramp_s=args.initial_ramp_seconds,
                initial_alpha=args.initial_action_alpha,
                normal_alpha=args.action_alpha,
                initial_max_arm_step_rad=args.initial_max_arm_step,
                normal_max_arm_step_rad=args.policy_max_arm_step,
            )
            print(
                json.dumps(
                    _preview_summary(
                        references=references,
                        clouds=initial_clouds,
                        chunk=preview_chunk,
                        zero_target=zero_target,
                        minimum_yellow_points=args.minimum_yellow_points,
                        start_region_margin_m=args.start_region_margin,
                        maximum_goal_distance_m=args.maximum_goal_distance,
                        minimum_gripper_open_seconds=args.minimum_gripper_open_seconds,
                        arm_action_gain=args.arm_action_gain,
                        full_prediction_horizon=args.execute_full_horizon,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        _confirm_active_rollout(args.duration, continuous=args.continuous)
        state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        state_socket.bind((args.host, args.state_port))
        action_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        LOGGER.warning("Active confirmation received; requesting motor connection")
        send_packet(action_socket, motor_address, {"schema_version": 1, "type": "hello"})
        _state_sequence, current_state = receive_state(
            state_socket,
            host=args.host,
            timeout_s=10.0,
        )
        initial_error = zero_pose_error(current_state, zero_target)
        if initial_error > args.maximum_initial_zero_delta:
            raise RuntimeError(
                "live robot is too far from the configured zero pose: "
                f"maximum arm delta={initial_error:.3f}rad, "
                f"limit={args.maximum_initial_zero_delta:.3f}rad"
            )

        sequence = 0
        cycle_index = 1
        while not STOP_REQUESTED.is_set():
            sequence, current_state = align_to_zero_pose(
                action_socket=action_socket,
                state_socket=state_socket,
                motor_address=motor_address,
                target=zero_target,
                initial_state=current_state,
                host=args.host,
                timeout_s=args.zero_timeout,
                state_timeout_s=args.state_timeout,
                tolerance_rad=args.zero_tolerance,
                stable_frames_required=args.zero_stable_frames,
                maximum_arm_step_rad=args.zero_max_arm_step,
                sequence=sequence,
            )
            start_clouds = capture_point_clouds(
                chest,
                collection=collection,
                workspace=workspace,
                point_count=bundle.point_count,
                point_channels=bundle.point_channels,
                frame_count=args.selection_frames,
                seed_start=sequence,
            )
            start_observation = references.observe_cube(
                combine_point_clouds(start_clouds),
                minimum_yellow_points=args.minimum_yellow_points,
                start_region_margin_m=args.start_region_margin,
                maximum_goal_distance_m=args.maximum_goal_distance,
            )
            if not start_observation.in_start_region:
                raise WorkspaceReferenceError(
                    "cube left the start box while OpenArm returned to zero pose"
                )
            LOGGER.info(
                "Cycle %d start verified: zero error=%.3frad cube=%s",
                cycle_index,
                zero_pose_error(current_state, zero_target),
                start_observation.centroid_m.round(4).tolist(),
            )
            result = run_learned_cycle(
                camera=chest,
                collection=collection,
                workspace=workspace,
                references=references,
                profile=profile,
                bundle=bundle,
                action_socket=action_socket,
                state_socket=state_socket,
                motor_address=motor_address,
                zero_target=zero_target,
                initial_state=current_state,
                initial_point_history=start_clouds,
                initial_sequence=sequence,
                args=args,
                cycle_index=cycle_index,
            )
            sequence, current_state = align_to_zero_pose(
                action_socket=action_socket,
                state_socket=state_socket,
                motor_address=motor_address,
                target=zero_target,
                initial_state=result.state,
                host=args.host,
                timeout_s=args.zero_timeout,
                state_timeout_s=args.state_timeout,
                tolerance_rad=args.zero_tolerance,
                stable_frames_required=args.zero_stable_frames,
                maximum_arm_step_rad=args.zero_max_arm_step,
                sequence=result.sequence,
            )
            LOGGER.warning(
                "Cycle %d finished in %.1fs with %d learned commands; OpenArm returned to zero",
                cycle_index,
                result.elapsed_s,
                result.command_count,
            )
            if not args.continuous:
                return 0
            sequence, current_state = wait_for_cube_in_start_box(
                camera=chest,
                collection=collection,
                workspace=workspace,
                references=references,
                bundle=bundle,
                action_socket=action_socket,
                state_socket=state_socket,
                motor_address=motor_address,
                zero_target=zero_target,
                current_state=current_state,
                sequence=sequence,
                args=args,
            )
            cycle_index += 1
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        return 130
    finally:
        if action_socket is not None:
            try:
                send_packet(action_socket, motor_address, {"schema_version": 1, "type": "stop"})
            except OSError:
                pass
        chest.stop()
        if state_socket is not None:
            state_socket.close()
        if action_socket is not None:
            action_socket.close()
        LOGGER.warning("DP3 policy client stopped")


if __name__ == "__main__":
    raise SystemExit(main())

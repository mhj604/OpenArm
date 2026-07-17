from __future__ import annotations

import argparse
import logging
import math
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np
import zarr

from .dp3_policy_client import (
    normalize_active_confirmation,
    receive_state,
    send_action,
    send_packet,
)
from .rollout_profile import VECTOR_SIZE, RolloutProfile

LOGGER = logging.getLogger("openarm_dp3.replay")
STOP_REQUESTED = Event()
ARM_INDICES = np.asarray([*range(7), *range(8, 15)], dtype=np.int64)
GRIPPER_INDICES = np.asarray([7, 15], dtype=np.int64)


class ReplayError(RuntimeError):
    """Raised when a recorded demonstration cannot be replayed safely."""


@dataclass(frozen=True)
class ReplayEpisode:
    path: Path
    episode_index: int
    sample_fps: float
    state: np.ndarray
    action: np.ndarray

    @property
    def duration_s(self) -> float:
        return len(self.action) / self.sample_fps


def request_stop(_signum: int, _frame: object) -> None:
    STOP_REQUESTED.set()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
    )


def load_replay_episode(root: Path, episode_index: int) -> ReplayEpisode:
    if episode_index < 0:
        raise ReplayError("episode index must be non-negative")
    path = root / f"episode_{episode_index:06d}.zarr"
    if not path.is_dir():
        raise ReplayError(f"recorded episode does not exist: {path}")
    group = zarr.open_group(str(path), mode="r")
    if group.attrs.get("complete") is not True or group.attrs.get("accepted") is not True:
        raise ReplayError("only complete, operator-accepted episodes may be replayed")
    if group.attrs.get("state_unit") != "radian" or group.attrs.get("action_unit") != "radian":
        raise ReplayError("recorded state/action units must be radian")
    if group.attrs.get("action_semantics") != "outgoing OpenArm CAN target position":
        raise ReplayError("episode action is not an outgoing OpenArm position target")

    state = np.asarray(group["data/state"][:], dtype=np.float32)
    action = np.asarray(group["data/action"][:], dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != VECTOR_SIZE or state.shape != action.shape:
        raise ReplayError(f"state/action must share shape (T, {VECTOR_SIZE})")
    if len(action) < 2 or not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ReplayError("recorded state/action must contain at least two finite frames")
    sample_fps = float(group.attrs.get("sample_fps", 0.0))
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ReplayError("recorded sample_fps must be finite and positive")
    return ReplayEpisode(
        path=path,
        episode_index=episode_index,
        sample_fps=sample_fps,
        state=np.ascontiguousarray(state),
        action=np.ascontiguousarray(action),
    )


def validate_episode_bounds(episode: ReplayEpisode, profile: RolloutProfile) -> None:
    tolerance = 1e-5
    action_below = episode.action < profile.action_min - tolerance
    action_above = episode.action > profile.action_max + tolerance
    if np.any(action_below | action_above):
        frame, channel = np.argwhere(action_below | action_above)[0]
        raise ReplayError(
            "recorded action is outside the active motor profile: "
            f"frame={int(frame)} channel={int(channel)} value={episode.action[frame, channel]:.4f}"
        )


def replay_start_target(episode: ReplayEpisode, profile: RolloutProfile) -> np.ndarray:
    """Convert measured first-frame feedback into an achievable position target."""
    return np.ascontiguousarray(
        np.clip(episode.state[0], profile.action_min, profile.action_max),
        dtype=np.float32,
    )


def required_replay_deltas(
    episode: ReplayEpisode,
    profile: RolloutProfile,
    *,
    margin: float = 1e-4,
) -> tuple[float, float]:
    """Return per-frame limits that preserve every recorded raw target."""
    if margin < 0 or not math.isfinite(margin):
        raise ValueError("replay delta margin must be finite and non-negative")
    start_target = replay_start_target(episode, profile)
    transitions = np.concatenate(
        (
            (episode.action[0] - start_target)[None, :],
            np.diff(episode.action, axis=0),
        ),
        axis=0,
    )
    arm_delta = float(np.max(np.abs(transitions[:, ARM_INDICES]))) + margin
    gripper_delta = float(np.max(np.abs(transitions[:, GRIPPER_INDICES]))) + margin
    return arm_delta, gripper_delta


def slew_vector(current: np.ndarray, target: np.ndarray, maximum_step: float) -> np.ndarray:
    if maximum_step <= 0 or not math.isfinite(maximum_step):
        raise ValueError("maximum replay alignment step must be finite and positive")
    current_values = np.asarray(current, dtype=np.float32)
    target_values = np.asarray(target, dtype=np.float32)
    if current_values.shape != (VECTOR_SIZE,) or target_values.shape != (VECTOR_SIZE,):
        raise ValueError("replay alignment vectors must contain 16 values")
    if not np.isfinite(current_values).all() or not np.isfinite(target_values).all():
        raise ValueError("replay alignment vectors must be finite")
    return np.ascontiguousarray(
        current_values + np.clip(target_values - current_values, -maximum_step, maximum_step),
        dtype=np.float32,
    )


def wait_until(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def align_to_recorded_start(
    *,
    action_socket: socket.socket,
    state_socket: socket.socket,
    motor_address: tuple[str, int],
    host: str,
    current_state: np.ndarray,
    target: np.ndarray,
    fps: float,
    state_timeout_s: float,
    maximum_initial_delta: float,
    maximum_step: float,
    tolerance: float,
    stable_frames_required: int,
    timeout_s: float,
    sequence: int,
) -> tuple[int, np.ndarray]:
    initial_error = float(np.max(np.abs(current_state[ARM_INDICES] - target[ARM_INDICES])))
    if initial_error > maximum_initial_delta:
        raise ReplayError(
            "live robot is too far from this demonstration start: "
            f"maximum arm delta={initial_error:.3f}rad, limit={maximum_initial_delta:.3f}rad"
        )
    command = current_state.copy()
    stable = 0
    started = time.monotonic()
    next_tick = started
    while not STOP_REQUESTED.is_set():
        if time.monotonic() - started >= timeout_s:
            raise TimeoutError("robot did not settle at the recorded start pose")
        command = slew_vector(command, target, maximum_step)
        send_action(action_socket, motor_address, sequence, command)
        sequence += 1
        _state_sequence, current_state = receive_state(
            state_socket, host=host, timeout_s=state_timeout_s
        )
        error = float(np.max(np.abs(current_state - target)))
        stable = stable + 1 if error <= tolerance else 0
        if sequence == 1 or sequence % max(1, round(fps)) == 0:
            LOGGER.info(
                "recorded-start alignment max error=%.3frad stable=%d/%d",
                error,
                stable,
                stable_frames_required,
            )
        if stable >= stable_frames_required:
            return sequence, current_state
        next_tick += 1.0 / fps
        wait_until(next_tick)
    raise KeyboardInterrupt


def replay_actions(
    *,
    episode: ReplayEpisode,
    action_socket: socket.socket,
    state_socket: socket.socket,
    motor_address: tuple[str, int],
    host: str,
    state_timeout_s: float,
    maximum_tracking_error: float,
    tracking_error_frames: int,
    final_hold_s: float,
    sequence: int,
) -> int:
    excessive_tracking_frames = 0
    started = time.monotonic()
    for frame_index, target in enumerate(episode.action):
        if STOP_REQUESTED.is_set():
            raise KeyboardInterrupt
        wait_until(started + frame_index / episode.sample_fps)
        send_action(action_socket, motor_address, sequence, target)
        sequence += 1
        _state_sequence, current_state = receive_state(
            state_socket, host=host, timeout_s=state_timeout_s
        )
        command_tracking_error = float(
            np.max(np.abs(current_state[ARM_INDICES] - target[ARM_INDICES]))
        )
        trajectory_error = float(
            np.max(
                np.abs(
                    current_state[ARM_INDICES]
                    - episode.state[frame_index, ARM_INDICES]
                )
            )
        )
        excessive_tracking_frames = (
            excessive_tracking_frames + 1
            if command_tracking_error > maximum_tracking_error
            else 0
        )
        if excessive_tracking_frames >= tracking_error_frames:
            raise ReplayError(
                "recorded replay tracking error remained excessive: "
                f"command error={command_tracking_error:.3f}rad, "
                f"recorded-state deviation={trajectory_error:.3f}rad"
            )
        if frame_index == 0 or (frame_index + 1) % max(1, round(episode.sample_fps)) == 0:
            LOGGER.info(
                "replay frame=%d/%d elapsed=%.1fs command_error=%.3frad "
                "recorded_state_deviation=%.3frad",
                frame_index + 1,
                len(episode.action),
                time.monotonic() - started,
                command_tracking_error,
                trajectory_error,
            )

    hold_frames = max(0, math.ceil(final_hold_s * episode.sample_fps))
    for hold_index in range(hold_frames):
        if STOP_REQUESTED.is_set():
            raise KeyboardInterrupt
        send_action(action_socket, motor_address, sequence, episode.action[-1])
        sequence += 1
        receive_state(state_socket, host=host, timeout_s=state_timeout_s)
        wait_until(started + (len(episode.action) + hold_index) / episode.sample_fps)
    LOGGER.warning(
        "Recorded episode %d replay completed: %d frames in %.1fs",
        episode.episode_index,
        len(episode.action),
        time.monotonic() - started,
    )
    return sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--action-port", type=int, required=True)
    parser.add_argument("--state-port", type=int, required=True)
    parser.add_argument("--state-timeout", type=float, default=0.75)
    parser.add_argument("--maximum-initial-delta", type=float, default=1.50)
    parser.add_argument("--start-max-step", type=float, default=0.03)
    parser.add_argument("--start-tolerance", type=float, default=0.08)
    parser.add_argument("--start-stable-frames", type=int, default=5)
    parser.add_argument("--start-timeout", type=float, default=25.0)
    parser.add_argument("--maximum-tracking-error", type=float, default=0.90)
    parser.add_argument("--tracking-error-frames", type=int, default=10)
    parser.add_argument("--final-hold-seconds", type=float, default=1.0)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--print-required-deltas", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.state_timeout,
        args.maximum_initial_delta,
        args.start_max_step,
        args.start_tolerance,
        args.start_stable_frames,
        args.start_timeout,
        args.maximum_tracking_error,
        args.tracking_error_frames,
    )
    if any(float(value) <= 0 or not math.isfinite(float(value)) for value in positive):
        raise ValueError("replay limits and timeouts must be finite and positive")
    if args.final_hold_seconds < 0 or not math.isfinite(args.final_hold_seconds):
        raise ValueError("final hold duration must be finite and non-negative")


def confirm_replay(episode: ReplayEpisode) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("active demonstration replay requires an interactive terminal")
    print("\nRECORDED DEMONSTRATION REPLAY IS READY.")
    print("This does not use DP3 inference or either camera.")
    print(
        f"Episode {episode.episode_index}: {len(episode.action)} frames, "
        f"{episode.duration_s:.1f}s at {episode.sample_fps:.0f}Hz."
    )
    print("BOTH OpenArm arms first move to the recorded start state, then replay raw CAN targets.")
    print("Emergency stop: Ctrl+C or physical power/E-stop.")
    raw_confirmation = input("Type REPLAY exactly to enable and move both arms: ")
    if normalize_active_confirmation(raw_confirmation) != "REPLAY":
        raise RuntimeError(
            "demonstration replay was not confirmed; motors were not opened; "
            f"received={raw_confirmation!r}"
        )


def main() -> int:
    args = parse_args()
    validate_args(args)
    episode = load_replay_episode(args.raw_root, args.episode_index)
    profile = RolloutProfile.load(args.profile)
    validate_episode_bounds(episode, profile)
    start_target = replay_start_target(episode, profile)
    if args.print_required_deltas:
        arm_delta, gripper_delta = required_replay_deltas(episode, profile)
        print(f"{arm_delta:.9f} {gripper_delta:.9f}")
        return 0

    configure_logging(args.log_file)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    confirm_replay(episode)

    action_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_socket.bind((args.host, args.state_port))
    motor_address = (args.host, args.action_port)
    try:
        LOGGER.warning("Replay confirmation received; requesting motor connection")
        send_packet(action_socket, motor_address, {"schema_version": 1, "type": "hello"})
        _state_sequence, current_state = receive_state(
            state_socket, host=args.host, timeout_s=10.0
        )
        sequence, _current_state = align_to_recorded_start(
            action_socket=action_socket,
            state_socket=state_socket,
            motor_address=motor_address,
            host=args.host,
            current_state=current_state,
            target=start_target,
            fps=episode.sample_fps,
            state_timeout_s=args.state_timeout,
            maximum_initial_delta=args.maximum_initial_delta,
            maximum_step=args.start_max_step,
            tolerance=args.start_tolerance,
            stable_frames_required=args.start_stable_frames,
            timeout_s=args.start_timeout,
            sequence=0,
        )
        LOGGER.warning("Recorded start pose verified; beginning raw action replay")
        replay_actions(
            episode=episode,
            action_socket=action_socket,
            state_socket=state_socket,
            motor_address=motor_address,
            host=args.host,
            state_timeout_s=args.state_timeout,
            maximum_tracking_error=args.maximum_tracking_error,
            tracking_error_frames=args.tracking_error_frames,
            final_hold_s=args.final_hold_seconds,
            sequence=sequence,
        )
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        try:
            send_packet(action_socket, motor_address, {"schema_version": 1, "type": "stop"})
        except OSError:
            LOGGER.exception("Failed to send replay stop packet")
        action_socket.close()
        state_socket.close()


if __name__ == "__main__":
    raise SystemExit(main())

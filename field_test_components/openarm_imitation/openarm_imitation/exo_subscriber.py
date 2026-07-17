from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExoSnapshot:
    position: np.ndarray
    timestamp: float


class ExoJointSubscriber:
    def __init__(self, topic: str, expected_positions: int = 16):
        self.topic = topic
        self.expected_positions = expected_positions
        self._lock = threading.Lock()
        self._position = np.full(expected_positions, np.nan, dtype=np.float32)
        self._timestamp = 0.0
        self._thread: threading.Thread | None = None
        self._executor = None
        self._node = None

    def start(self) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        if not rclpy.ok():
            rclpy.init(args=None)
        node = Node("openarm_imitation_exo_listener")

        def callback(msg: JointState) -> None:
            if len(msg.position) < self.expected_positions:
                return
            with self._lock:
                self._position[:] = np.asarray(msg.position[: self.expected_positions], dtype=np.float32)
                self._timestamp = time.monotonic()

        node.create_subscription(JointState, self.topic, callback, 10)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        self._node = node
        self._executor = executor
        self._thread = threading.Thread(target=executor.spin, name="exo-ros2", daemon=True)
        self._thread.start()

    def snapshot(self) -> ExoSnapshot:
        with self._lock:
            return ExoSnapshot(self._position.copy(), self._timestamp)

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snap = self.snapshot()
            if snap.timestamp > 0 and np.isfinite(snap.position).all():
                return True
            time.sleep(0.05)
        return False

    def close(self) -> None:
        import rclpy

        if self._executor is not None:
            self._executor.shutdown(timeout_sec=2.0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def __enter__(self) -> "ExoJointSubscriber":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.close()


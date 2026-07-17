# Demonstration Protocol

## Target behavior

1. Start from the same safe robot pose.
2. Locate the cube at a randomized position inside the center red rectangle.
3. Approach without contacting the table or rectangle boundary.
4. Close the gripper with a consistent grasp.
5. Lift the cube clear of the table.
6. Move to the red target point on the left.
7. Center the cube over that point, place it, and release it.
8. End recording within one second after release.

## Coverage

Use the center rectangle as a 3x3 grid. Balance demonstrations across all nine
cells instead of sampling only near the middle. Randomize cube yaw. Keep the
final target fixed: the cube center should finish over the left red point.

- Three demonstrations are supported only for a pipeline overfit test.
- First meaningful training: at least 30 successful demonstrations.
- Recommended first dataset: 40 successful demonstrations.
- Current generalization dataset target: 50 successful demonstrations.

Keep the chest camera, center red rectangle, left red point, robot base, control
mapping, and initial robot pose fixed for the first dataset. Change only the
cube's starting position and yaw.

## Quality rules

- Start motion within one second of the recording cue.
- Do not leave a long idle prefix or suffix.
- Save successful demonstrations only for the first training set.
- Preserve failed attempts as rejected episodes; do not silently delete them.
- Keep grasp, lift height, path, and release style reasonably consistent.
- Avoid correcting a failed grasp inside an accepted episode for the first
  baseline. Recovery demonstrations can be a separate later dataset.

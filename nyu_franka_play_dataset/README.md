# NYU Franka Play Dataset
Unstructured robot play demonstrations on a toy kitchen. Each episode contains some arbitrary task demonstrations. Raw RGB-D data is recorded at 720p @ 30fps from two stationary Realsense D435 cameras, along with robot joint angles, EE pose, and gripper state.

## Preprocessing
- Dropped corrupted trajectories
- Dropped idle frames such that there is always a large enough cumulative joint position change between frames
- Cropped and resized images to 128x128

## Observation space
- RGB (128x128x3 uint8), left camera view
- RGB (128x128x3 uint8), right camera view
- Depth (128x128x1 int32), left camera view
- Depth (128x128x1 int32), right camera view
- Robot state (13)
    - Robot arm joint angles (7)
    - Robot EE translation, base frame (3)
    - Robot EE roll/pitch/yaw, base frame (3)

## Action space
Arm action is calculated as a one-step forward difference on the arm state.
- Robot state delta (13)
    - Robot arm joint angle deltas (7)
    - Robot EE translation deltas, base frame (3)
    - Robot EE roll/pitch/yaw deltas, base frame (3)
- Gripper command (1) (-1 close, 1 open)
- Terminate episode (1) (1 last frame, 0 other frames)

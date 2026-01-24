# IK-Based Retargeting Debug Notes

## Goal
Debug and fix the GMR IK-based retargeting by visualizing end-effector positions.

## End-Effector Positions to Track

| End-Effector | MediaPipe Landmark | Robot Link | Notes |
|--------------|-------------------|------------|-------|
| Left Foot | LEFT_ANKLE (27) | left_ankle_roll_link | Ankle position |
| Right Foot | RIGHT_ANKLE (28) | right_ankle_roll_link | Ankle position |
| Left Palm | LEFT_WRIST (15) | left_wrist_yaw_link | Wrist is proxy for palm |
| Right Palm | RIGHT_WRIST (16) | right_wrist_yaw_link | Wrist is proxy for palm |
| Neck | mid(LEFT_SHOULDER, RIGHT_SHOULDER) | mid(shoulder links) | Mid-shoulder point |

## Pelvis Reference

**MediaPipe Pelvis**: Midpoint of LEFT_HIP (23) and RIGHT_HIP (24)

**Robot Pelvis**: `pelvis` link in MuJoCo model

All end-effector positions should be computed **relative to pelvis** for:
1. Translation invariance (person can stand anywhere in camera view)
2. Scale normalization (different human heights)

## Visualization Plan

### Step 1: Extract Human End-Effectors from Captured Poses
```python
# From captured pose JSON:
skeleton_3d = np.array(pose_data["skeleton_3d"])

# Compute pelvis
pelvis = (skeleton_3d[23] + skeleton_3d[24]) / 2

# End-effectors relative to pelvis
left_foot = skeleton_3d[27] - pelvis   # or 31 for foot index
right_foot = skeleton_3d[28] - pelvis
left_hand = skeleton_3d[15] - pelvis
right_hand = skeleton_3d[16] - pelvis
head = skeleton_3d[0] - pelvis  # nose
```

### Step 2: Load Robot Model and Get End-Effector Positions
```python
import mujoco

model = mujoco.MjModel.from_xml_path("assets/g1/g1_mocap_29dof.xml")
data = mujoco.MjData(model)

# Set joint angles from robot pose JSON
data.qpos[7:36] = robot_joint_angles  # 29 DOFs

# Forward kinematics
mujoco.mj_forward(model, data)

# Get body positions
left_foot_robot = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")]
right_foot_robot = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")]
left_hand_robot = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")]
right_hand_robot = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")]
pelvis_robot = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
```

### Step 3: Visualize Side-by-Side
- Plot human skeleton with end-effectors highlighted
- Plot robot skeleton with end-effectors highlighted
- Show pelvis-relative positions numerically
- Compute error = distance between corresponding end-effectors (after scaling)

## Questions to Answer

1. **Scale Factor**: What's the height ratio between human and robot?
   - Human height: distance from ankle to head
   - Robot height: same measurement from MuJoCo
   - Scale = robot_height / human_height

2. **Coordinate Frame**: Are MediaPipe and MuJoCo using same axes?
   - MediaPipe: X=right, Y=down(?), Z=forward(?)
   - MuJoCo G1: X=forward, Y=left, Z=up (typical robotics convention)

3. **T-Pose Offset**: Is there a calibration offset between human T-pose and robot T-pose?

4. **End-Effector Error**: After applying robot joint angles, how far are robot end-effectors from target human end-effectors?

## Existing Files to Review

- `mediapipe_to_g1_gmr.py` - Main GMR wrapper
- `mediapipe_gmr_adapter.py` - Converts MediaPipe to GMR format
- `tpose_calibration.py` - T-pose calibration logic
- `calibration/tpose_calibration.json` - Saved calibration data

## Data Available

- **Human Poses**: `calibration/captured_poses/*.json` (20 poses)
  - Contains: `skeleton_3d` (33x3 positions), `joint_angles` (computed angles)
  
- **Robot Poses**: `calibration/robot_poses/*.json` (9 matched poses)
  - Contains: `joint_angles_rad`, `joint_angles_deg`, `human_pose_ref`

## Tomorrow's Tasks

1. [ ] Create `visualize_end_effectors.py` script
2. [ ] Load a human pose and extract end-effector positions
3. [ ] Load corresponding robot pose and compute FK
4. [ ] Visualize both skeletons with end-effectors
5. [ ] Compute and display position errors
6. [ ] Identify coordinate frame issues
7. [ ] Test with GMR IK and compare results

## Useful Commands

```bash
# Activate environment
conda activate gmr

# View captured poses
python view_captured_poses.py

# Run calibration UI
python calibration_ui.py

# Check robot model
python -c "import mujoco; m = mujoco.MjModel.from_xml_path('../assets/g1/g1_mocap_29dof.xml'); print([m.body(i).name for i in range(m.nbody)])"
```

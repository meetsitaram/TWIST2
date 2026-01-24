# IK-Based Retargeting Debug Notes

## Goal
Implement a simpler **end-effector IK approach** for motion retargeting:
- Extract 5 end-effector positions + orientations from human pose
- Scale to robot proportions
- Use IK to find robot joint angles that achieve those targets

## Why End-Effector IK Instead of GMR?

**Problem with GMR:**
- Uses orientations from ALL body parts (14 parts)
- Complex retargeting considering whole kinematic chain
- MediaPipe doesn't provide shoulder yaw, causing issues

**End-Effector IK approach:**
- Only 5 targets to match (feet, hands, neck)
- Positions + orientations for each
- Let IK solver figure out intermediate joints (elbows, knees)
- Simpler and more controllable

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

### GMR Retargeting (current approach)
- `mediapipe_to_g1_gmr.py` - Main GMR wrapper
- `mediapipe_gmr_adapter.py` - Converts MediaPipe to GMR format (enhanced with hand/foot orientation)
- `tpose_calibration.py` - T-pose calibration logic
- `calibration/tpose_calibration.json` - Saved calibration data

### Pose Capture & Visualization Tools (use these for debugging)

**`capture_pose_simple.py`** - Capture human poses from multi-camera setup
```bash
python capture_pose_simple.py
# Press SPACE to capture, Q to quit
# Saves to calibration/captured_poses/*.json with skeleton_3d data
```
- Outputs 33-landmark MediaPipe skeleton in 3D (triangulated from multiple cameras)
- Includes computed joint angles (shoulder pitch/roll/yaw, elbow, hip, knee)
- Use this to capture new test poses for debugging

**`view_captured_poses.py`** - View captured 3D skeletons
```bash
python view_captured_poses.py
# Left/Right arrows to navigate, R to rename pose
```
- 3D Matplotlib visualization of captured skeletons
- Auto-aligns skeleton upright (corrects for camera tilt)
- Shows joint angles numerically
- Useful for verifying captured poses look correct

**`visualize_end_effectors.py`** - Compare human vs robot end-effectors
```bash
python visualize_end_effectors.py --pose 1_20260123_162606
python visualize_end_effectors.py --compare  # Compare all matched pairs
```
- Extracts 5 end-effector positions + orientations from human skeleton
- Loads robot pose and computes FK to get robot end-effectors
- Shows both side-by-side with orientation axes (red=forward, green=up, blue=right)
- Prints position errors between human and robot end-effectors
- **Key tool for debugging IK retargeting**

**`test_captured_poses_mujoco.py`** - Test GMR retargeting in MuJoCo
```bash
python test_captured_poses_mujoco.py --auto  # Cycle through all poses
python test_captured_poses_mujoco.py --pose 1_20260123_162606
```
- Loads captured human poses
- Runs GMR retargeting to get robot joint angles
- Visualizes robot in MuJoCo viewer
- Use to see how GMR performs on your captured poses

## Data Available

- **Human Poses**: `calibration/captured_poses/*.json` (20 poses)
  - Contains: `skeleton_3d` (33x3 positions), `joint_angles` (computed angles)
  
- **Robot Poses**: `calibration/robot_poses/*.json` (9 matched poses)
  - Contains: `joint_angles_rad`, `joint_angles_deg`, `human_pose_ref`

## Tomorrow's Tasks

1. [x] Create `visualize_end_effectors.py` script
2. [x] Load a human pose and extract end-effector positions
3. [x] Load corresponding robot pose and compute FK
4. [x] Visualize both skeletons with end-effectors + orientations
5. [x] Compute and display position errors
6. [ ] Identify coordinate frame issues
7. [ ] Test with GMR IK and compare results

## Updates Made

### Enhanced Orientation Computation (mediapipe_gmr_adapter.py)

Added improved orientation computation for hands and feet:

**Hand Orientation** (`_compute_hand_orientation`):
- Uses wrist, index finger, and pinky landmarks
- Z-axis (forward): wrist → index (finger pointing direction)
- X-axis (lateral): index → pinky (palm width direction)
- Y-axis (normal): cross product (palm facing direction)

**Foot Orientation** (`_compute_foot_orientation`):
- Uses ankle, heel, and toe landmarks
- Z-axis (forward): heel → toe (foot pointing direction)
- Uses ankle-to-heel to define up direction
- Y-axis (up): perpendicular to foot sole

This gives full 3-DOF orientation for hands/feet instead of just bone direction.

## End-Effector IK Approach

### The Flow

```
Human Pose (MediaPipe skeleton)
    ↓
Extract 5 EE positions + orientations (visualize_end_effectors.py has this)
    - left_hand:  position + palm orientation (from wrist/index/pinky)
    - right_hand: position + palm orientation
    - left_foot:  position + foot orientation (from ankle/heel/toe)
    - right_foot: position + foot orientation
    - neck:       position + torso orientation (from shoulders)
    ↓
Scale positions to robot proportions
    - scale = robot_height / human_height
    - positions *= scale
    - orientations stay the same
    ↓
MuJoCo IK solver (via mink library)
    - Target: left_wrist_yaw_link  → (scaled_left_hand_pos, left_hand_quat)
    - Target: right_wrist_yaw_link → (scaled_right_hand_pos, right_hand_quat)
    - Target: left_ankle_roll_link → (scaled_left_foot_pos, left_foot_quat)
    - Target: right_ankle_roll_link → (scaled_right_foot_pos, right_foot_quat)
    - Target: torso_link           → (scaled_neck_pos, neck_quat)
    ↓
Robot joint angles (29 DOFs)
```

### How IK Works

1. **Define end-effector targets:**
   ```python
   targets = {
       "left_wrist_yaw_link": (position_3d, quaternion),
       "right_wrist_yaw_link": (position_3d, quaternion),
       "left_ankle_roll_link": (position_3d, quaternion),
       "right_ankle_roll_link": (position_3d, quaternion),
       "torso_link": (position_3d, quaternion),
   }
   ```

2. **The robot model (MJCF) defines the kinematic chain:**
   - Which joints connect which bodies
   - Joint limits, ranges, etc.

3. **The IK solver uses the Jacobian:**
   - Jacobian tells how each joint angle change affects each end-effector
   - e.g., "moving shoulder_pitch by 0.1 rad moves wrist by [dx, dy, dz]"

4. **Optimization loop:**
   ```python
   while error > threshold:
       current_ee_poses = forward_kinematics(joint_angles)
       error = target_poses - current_ee_poses
       delta_joints = Jacobian_inverse @ error
       joint_angles += delta_joints
   ```

### Using mink for MuJoCo IK

`mink` is a library that provides IK solvers for MuJoCo models:

```python
import mink
import mujoco

model = mujoco.MjModel.from_xml_path("g1_mocap_29dof.xml")
data = mujoco.MjData(model)

# Define IK tasks for each end-effector
tasks = [
    mink.FrameTask(
        frame_name="left_wrist_yaw_link",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=0.5,  # Can weight position vs orientation
    ),
    # ... more tasks for other end-effectors
]

# Set target poses
tasks[0].set_target(target_position, target_quaternion)

# Solve IK
configuration = mink.Configuration(model, data)
velocity = mink.solve_ik(configuration, tasks, dt=0.01)
mujoco.mj_integratePos(model, data.qpos, velocity, dt)
```

### Differences from GMR

| Aspect | GMR | End-Effector IK |
|--------|-----|-----------------|
| Inputs | 14 body parts (positions + orientations) | 5 end-effectors only |
| Method | Complex retargeting | Simple IK optimization |
| Shoulder yaw | Required (not from MediaPipe) | Not needed (IK figures it out) |
| Elbow/knee | Explicitly matched | IK determines naturally |
| Control | Less direct | More controllable |

## Implementation TODO

1. [x] Create `end_effector_ik_retarget.py`:
   - Extract EE positions + orientations from skeleton
   - Scale to robot proportions  
   - Set up mink IK tasks
   - Solve and return joint angles

2. [x] Test with captured poses:
   - Load human pose
   - Run end-effector IK
   - Results: avg error 0.34, min 0.24, max 0.58

3. [ ] Live retargeting:
   - Integrate with multicam_pose_streamer
   - Real-time IK solving
   - Send to robot

## Session Notes - Jan 24, 2026

### Created `end_effector_ik_retarget.py`

Key implementation details:

**Coordinate Frame Alignment:**
- MediaPipe skeleton is first aligned upright (spine → Z-up)
- Then rotated so person faces X+ direction (MuJoCo convention: X=forward, Y=left, Z=up)
- Left shoulder should be at positive Y, right shoulder at negative Y

**Scaling:**
- Human height = ankle-to-shoulder distance
- Robot height = same measurement from MuJoCo model (~1.05m)
- Scale factor = robot_height / human_height
- End-effector positions are scaled, then offset to robot pelvis world position

**IK Setup:**
- Uses mink library for MuJoCo IK
- 5 FrameTasks: left_wrist_yaw_link, right_wrist_yaw_link, left_ankle_roll_link, right_ankle_roll_link, torso_link
- Position weight: 1.0, Orientation weight: 0.5
- Solver: daqp, Damping: 0.5

**Orientation Computation:**
- Hands: Forward = wrist→index, Right = index→pinky, Up = cross product
- Feet: Forward = heel→toe, Right = cross(forward, world_up), Up = cross product
- Torso: Up = spine direction, Right = shoulder line, Forward = cross product

**Test Results (19 valid poses):**
```
Average error: 0.3446
Min error: 0.2387
Max error: 0.5823
Average iterations: 44.2
```

**Usage:**
```bash
# Test single pose
python end_effector_ik_retarget.py --pose 1_20260123_162606

# Test all poses
python end_effector_ik_retarget.py --all

# Visualize in MuJoCo
python end_effector_ik_retarget.py --pose 1_20260123_162606 --viz
```

## Useful Commands

```bash
# Activate environment
conda activate gmr

# View captured poses with end-effectors
python visualize_end_effectors.py --pose 1_20260123_162606

# Compare all matched poses
python visualize_end_effectors.py --compare

# Test GMR retargeting in MuJoCo
python test_captured_poses_mujoco.py --auto

# View captured poses (3D skeleton)
python view_captured_poses.py

# Run calibration UI
python calibration_ui.py

# Check robot model bodies
python -c "import mujoco; m = mujoco.MjModel.from_xml_path('../assets/g1/g1_mocap_29dof.xml'); print([m.body(i).name for i in range(m.nbody)])"
```

## Session Notes - Jan 24, 2026 (Continued)

### Per-Limb Scaling Implementation

Changed from uniform height-based scaling to per-limb scaling:
- **Arm length**: Measured as upper arm + forearm (segment lengths, not straight-line)
- **Leg length**: Measured as thigh + shin (segment lengths, not straight-line)
- This ensures consistent measurements regardless of joint bend angles

### Waist Constraint (Keep Robot Upright)

Added hard constraint to prevent robot from tilting and falling:
- `waist_roll` and `waist_pitch` clamped to ±5° after each IK iteration
- `waist_yaw` left free for turning
- This ensures stable poses that won't cause the real robot to fall

### Live Streaming Integration - `stream_ik_teleop.py`

Created live teleoperation script with:
- Multi-camera 3D skeleton capture via `MultiCamPoseStreamer`
- End-effector IK retargeting in real-time
- MuJoCo passive viewer (kinematics only, no physics)
- ENTER key to start with 10-second countdown
- Dynamic pelvis height (crouches when human crouches)
- Fixed X, Y position (robot stays in place)

**Usage:**
```bash
cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
conda activate gmr
python stream_ik_teleop.py
```

### Test Results with Waist Constraint (19 valid poses)

```
Average error: 0.1047
Min error: 0.0283
Max error: 0.1999
```

### Key Implementation Details

**Fixed Base Teleop:**
- X, Y position: Fixed at origin
- Z height: Dynamic (adjusts for crouching/standing)
- Base orientation: Fixed (upright)
- Joint angles: All copied from IK solution

**IK Parameters:**
- Max iterations: 50 (30 for real-time streaming)
- Position weight: 1.0
- Orientation weight: 0.5
- Waist tilt limit: ±5°

## Files Created/Modified

| File | Purpose |
|------|---------|
| `visualize_end_effectors.py` | Visualize human + robot EE positions with orientations |
| `NOTES_IK_DEBUG.md` | This file - debug notes and implementation plan |
| `mediapipe_gmr_adapter.py` | Enhanced hand/foot orientation from finger/heel landmarks |
| `capture_pose_simple.py` | Added shoulder yaw computation |
| `view_captured_poses.py` | 3D skeleton viewer with upright alignment |
| `test_captured_poses_mujoco.py` | Test captured poses with GMR in MuJoCo |
| `end_effector_ik_retarget.py` | End-effector IK retargeting with mink |
| `stream_ik_teleop.py` | Live IK teleoperation with MuJoCo viewer |

## Current Status (Jan 24, 2026)

✅ **WORKING**: Whole-body IK teleoperation in kinematics mode
- Hands and feet tracked via end-effector IK
- Waist constrained to stay upright (±5°)
- Dynamic pelvis height for crouching
- Real-time multi-camera capture
- MuJoCo passive viewer for safe testing

## Next Steps

1. Add temporal smoothing for smoother motion
2. Add relative XY locomotion tracking
3. Deploy to real robot (send joint angles)
4. Record teleoperated motions for playback/training

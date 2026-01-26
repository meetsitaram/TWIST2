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

✅ **WORKING**: Arm-only IK teleoperation with fixed base
- Hands tracked via end-effector IK
- Elbows used as soft hints (5% weight)
- Base frozen during IK (no pelvis cheating)
- Fixed Z height at 0.75m
- Waist roll/pitch constrained to ±5°
- Waist yaw free for future whole-body
- Real-time multi-camera capture
- MuJoCo passive viewer for safe testing

## Session Summary - Jan 24, 2026 (Evening)

### Major Improvements Made

**1. Base Freezing (DofFreezingTask)**
- Added `mink.DofFreezingTask` to lock all 6 floating base DOFs during IK
- Prevents the IK solver from "cheating" by moving the pelvis
- Z position now stays fixed at 0.75m
- Conditional on `fixed_base` parameter for future whole-body teleop

**2. Per-Joint Posture Regularization**
- Added per-DOF costs to `mink.PostureTask`:
  - Shoulder/elbow: 0.05 (prevents hitting joint limits)
  - Wrist joints: 0.1 (keeps neutral since not tracked)
  - Waist joints: 0.05 (keeps facing forward)
  - Leg joints: 0.05 (keeps stable)
  - Root/base: 0.001 (minimal)

**3. Elbow Tracking**
- Added elbow positions as soft IK targets (5% weight)
- Helps guide arm pose without fighting hand tracking
- Orientation disabled (position only)

**4. Hand Tracking Error Metric**
- Added "HAND POSITION TRACKING ERROR" to analysis tool
- Computes actual robot hand position vs scaled human target
- Quality ratings: EXCELLENT (<5cm), GOOD (5-10cm), FAIR (10-20cm), POOR (>20cm)
- Added Figure 7 visualization with time series, histogram, rolling average

**5. Code Organization**
- Refactored `_setup_ik()` into modular functions:
  - `_setup_end_effector_tasks()` - hands/elbows
  - `_setup_posture_regularization()` - per-joint costs
  - `_setup_waist_constraints()` - waist clamping
  - `_setup_base_freezing()` - DofFreezingTask

**6. Directory Reorganization**
- Moved teleop episodes to `datasets/teleop_episodes/`
- Each episode now has its own subdirectory
- Moved charuco boards to `assets/charuco_board/`
- Updated .gitignore for new structure

### Known Issues

1. **Hand tracking accuracy**: ~56cm mean error (POOR) in last recording
   - Posture regularization may be too strong
   - Could reduce shoulder/elbow posture cost to 0.02
   - Could increase hand tracking weight to 2.0

2. **Shoulder joint limits**: Right shoulder roll occasionally hits -129° limit
   - Need better joint limit avoidance

3. **Cross-arm coupling**: ~0.6 correlation between left/right arm movement
   - Arms not fully independent

### Analysis Commands

```bash
# Analyze episode with hand tracking metric
python teleop_jitter_analysis.py --episode elbow_track_007 --verbose

# Save plots to episode directory
python teleop_jitter_analysis.py --episode elbow_track_007 --save --timeseries

# List all episodes
python teleop_jitter_analysis.py --list
```

### Key Files Modified

| File | Changes |
|------|---------|
| `end_effector_ik_retarget.py` | Base freezing, per-joint posture costs, modular setup |
| `stream_ik_teleop.py` | fixed_base=True for arm-only teleop |
| `teleop_jitter_analysis.py` | Hand tracking error metric + Figure 7 |
| `teleop_episode_recorder.py` | Episodes in subdirectories |

## Next Steps - Whole Body Teleop

### Phase 1: Improve Hand Tracking
1. [ ] Reduce posture regularization for shoulders (0.05 → 0.02)
2. [ ] Increase hand tracking weight (1.0 → 2.0)
3. [ ] Add soft joint limits to prevent hitting hard limits
4. [ ] Target: <10cm mean hand tracking error

### Phase 2: Enable Lower Body
1. [ ] Set `fixed_base=False` in stream_ik_teleop.py
2. [ ] Re-enable foot tracking in `ROBOT_EE_BODIES`
3. [ ] Tune leg posture regularization (may need lower cost)
4. [ ] Handle pelvis height from skeleton foot positions
5. [ ] Add ground contact constraint

### Phase 3: Full Whole-Body Teleop
1. [ ] Enable waist yaw tracking from human torso rotation
2. [ ] Add temporal smoothing (One Euro filter on skeleton)
3. [ ] Add velocity limiting on robot joints
4. [ ] Test crouching/standing transitions
5. [ ] Deploy to real robot

### Tuning Parameters Reference

```python
# In end_effector_ik_retarget.py

# End-effector task weights
HAND_POSITION_WEIGHT = 1.0       # Primary targets
ELBOW_POSITION_WEIGHT = 0.05    # Soft hints
# (Orientation weights all 0.0 - position only)

# Posture regularization per-DOF costs (in _setup_posture_regularization)
costs[0:6] = 0.001   # Root (base frozen anyway)
costs[6:18] = 0.05   # Legs (keep stable)
costs[18:21] = 0.05  # Waist (keep facing forward)
costs[21:25] = 0.05  # Left shoulder/elbow
costs[25:28] = 0.1   # Left wrist (keep neutral)
costs[28:32] = 0.05  # Right shoulder/elbow
costs[32:35] = 0.1   # Right wrist (keep neutral)

# Waist clamping (in retarget loop)
max_waist_tilt = 5.0°  # Roll and pitch only, yaw is free
```

## Episode Workflow Scripts

### 1. Recording Episodes (`stream_ik_teleop.py`)

Live IK teleoperation with optional recording:

```bash
# Just run teleop (no recording)
python stream_ik_teleop.py

# Record 60-second episode with One Euro smoothing
python stream_ik_teleop.py --record --name episode_name --duration 60 --smoothing one_euro

# Record with video from cameras
python stream_ik_teleop.py --record --name episode_name --duration 60 --video
```

**What gets saved:**
- `datasets/teleop_episodes/episode_name/episode_name.npz` - Skeleton + robot data
- `datasets/teleop_episodes/episode_name/episode_name_cam0.mp4` - Camera video (if --video)

**Controls:**
- ENTER: Start teleop (10-second countdown)
- ESC: Stop and exit

### 2. Analyzing Episodes (`teleop_jitter_analysis.py`)

Comprehensive jitter and tracking quality analysis:

```bash
# List all available episodes
python teleop_jitter_analysis.py --list

# Quick analysis with metrics
python teleop_jitter_analysis.py --episode elbow_track_007

# Detailed numerical analysis
python teleop_jitter_analysis.py --episode elbow_track_007 --verbose

# Generate and save all plots
python teleop_jitter_analysis.py --episode elbow_track_007 --save --timeseries

# Compare multiple episodes
python teleop_jitter_analysis.py --compare baseline_001 elbow_track_007
```

**Key Metrics Reported:**
- Velocity variance per joint group
- Acceleration RMS
- Max joint velocity
- Z position stability
- **Hand position tracking error** (the key quality metric!)
- IK error statistics

**Plots Generated (with --save --timeseries):**
1. `01_skeleton_positions.png` - Human skeleton landmarks over time
2. `02_skeleton_deltas.png` - Frame-to-frame skeleton changes
3. `03_robot_joints.png` - Robot joint angles over time
4. `04_joint_velocities.png` - Joint velocities and accelerations
5. `05_end_effector.png` - End-effector positions
6. `06_summary.png` - Summary statistics
7. `07_hand_tracking.png` - **Hand tracking error time series**

### 3. Episode Data Format

Each episode `.npz` contains:

```python
import numpy as np
data = np.load("datasets/teleop_episodes/episode_name/episode_name.npz", allow_pickle=True)

# Available arrays:
data['t_ms']           # (N,) Timestamps in milliseconds
data['human_skeleton'] # (N, 33, 3) MediaPipe skeleton per frame
data['robot_qpos']     # (N, 36) Robot qpos per frame [x,y,z,qw,qx,qy,qz,joints...]
data['ik_error']       # (N,) IK error per frame
data['metadata']       # JSON string with episode info
```

### 4. Episode Recorder API (`teleop_episode_recorder.py`)

For programmatic use:

```python
from teleop_episode_recorder import TeleopEpisodeRecorder, TeleopEpisode

# Create recorder
recorder = TeleopEpisodeRecorder(
    name="my_episode",
    fps=30,
    smoothing="one_euro",
    smoothing_params={"min_cutoff": 1.0, "beta": 0.007},
)

# Record frames
recorder.start()
recorder.add_frame(
    human_skeleton=skeleton_3d,  # (33, 3) array
    robot_qpos=qpos,             # (36,) array
    ik_error=error,              # float
)
recorder.stop()
filepath = recorder.save()

# Load episode
episode = TeleopEpisodeRecorder.load(filepath)
print(f"Frames: {episode.num_frames}, Duration: {episode.duration_sec}s")

# List all episodes
episodes = TeleopEpisodeRecorder.list_episodes()
```

### 5. Directory Structure

```
TWIST2/
├── datasets/
│   └── teleop_episodes/
│       ├── baseline_001/
│       │   ├── baseline_001.npz
│       │   ├── baseline_001_cam0.mp4
│       │   └── baseline_001_analysis/
│       │       ├── 01_skeleton_positions.png
│       │       └── ...
│       ├── elbow_track_007/
│       │   ├── elbow_track_007.npz
│       │   └── elbow_track_007_analysis/
│       └── ...
└── deploy_real/
    ├── stream_ik_teleop.py        # Recording
    ├── teleop_jitter_analysis.py  # Analysis
    └── teleop_episode_recorder.py # Core API
```

## Next Steps - Whole Body Teleop Pipeline (Jan 24, 2026 - Evening Planning)

### Overview

Goal: Mobile teleop (walking + upper body) with crouching/sitting support.
Approach: Record IK-based episodes → Train with RL (TWIST2/legged_gym style).

Key insight: The RL policy learns to track recorded motions stably—it's not learning 
from reward signals about hand tracking error, it's mimicking the IK solution. This means:
- IK doesn't need to be perfect, just physically plausible
- Robot won't fall if IK looks stable in recording
- Lower body can be approximate—RL figures out stable execution

### Step 1: Two-Stage IK (Decouple Upper/Lower Body)

**Rationale**: Optimizing lower body could regress upper body quality. Different 
constraint regimes (arms don't care about ground contact; legs do).

**Stage 1: Upper Body IK (fixed_base=True)**
```
Input: Human skeleton (scaled)
Targets: Left/right wrist + elbow hints
Constraints: Pelvis frozen at (0, 0, Z_fixed)
Output: Arm joint angles (14 DOFs: shoulders + elbows + wrists)
Waist: Constrained pitch/roll to ±5°, yaw free
```

**Stage 2: Lower Body IK (base Z and rotation free)**
```
Input: Human skeleton (scaled) + Stage 1 arm angles
Targets: Left/right ankle positions
Variables: 
  - Leg joints (12 DOFs)
  - Pelvis Z (height only, XY frozen)
  - Pelvis yaw (optional, if human rotates)
Constraints:
  - Arm joints: FROZEN to Stage 1 values (DofFreezingTask)
  - Foot Z clamped near ground
Output: Full 29-DOF pose + pelvis height
```

**Implementation**: `end_effector_ik_retarget.py` → add `retarget_two_stage()` method

### Step 2: Episode → Motion Converter

Convert teleop episode NPZ files to TWIST2 motion pickle format:

```python
# Input: datasets/teleop_episodes/episode_name/episode_name.npz
data['robot_qpos']  # (N, 36) = [x, y, z, qw, qx, qy, qz, joints_29...]

# Output: motion_data/*.pkl (TWIST2 format)
motion_data = {
    'root_pos': robot_qpos[:, 0:3],           # (N, 3)
    'root_rot': robot_qpos[:, 3:7],           # (N, 4) quaternion
    'dof_pos': robot_qpos[:, 7:36],           # (N, 29)
    'local_body_pos': compute_fk_body_pos(),  # (N, num_bodies, 3) 
    'fps': 30,
    'link_body_list': ['pelvis', 'left_hip_pitch_link', ...]
}
```

**Implementation**: Create `convert_episodes_to_motion.py`

### Step 3: Motion Config YAML

Create YAML config pointing to recorded dataset:

```yaml
# motion_data_configs/teleop_dataset.yaml
root_path: /path/to/motion_data/
motions:
  - file: episode_001.pkl
    weight: 1.0
  - file: episode_002.pkl
    weight: 1.0
```

### Step 4: Train with legged_gym

Use TWIST2's motion imitation training:
- `g1_mimic_distill_config.py` as base config
- Point `motion_file` to teleop dataset YAML
- Key rewards: `tracking_joint_dof`, `tracking_keybody_pos`, `tracking_root_translation_z`
- Domain randomization for sim-to-real

### Key TWIST2 Training Concepts

**Motion Library** (`pose/utils/motion_lib_pkl.py`):
- Loads pickle files with root_pos, root_rot, dof_pos, local_body_pos
- Computes velocities via gradient
- Supports motion curriculum (harder motions sampled more)

**Mimic Environment** (`legged_gym/envs/base/humanoid_mimic.py`):
- Reference State Initialization (RSI): Reset robot to random point in motion
- Tracks reference motion frame-by-frame
- Rewards for matching joint positions, key body positions, root pose

**Reward Structure** (from `g1_mimic_distill_config.py`):
- `tracking_joint_dof`: 2.0 (match joint angles)
- `tracking_keybody_pos`: 2.0 (match hands/feet/elbows/knees/head)
- `tracking_root_translation_z`: 1.0 (match pelvis height)
- `tracking_root_rotation`: 1.0 (match root orientation)
- Penalties: feet_slip, dof_pos_limits, action_rate, etc.

### Files to Create/Modify

| File | Purpose |
|------|---------|
| `end_effector_ik_retarget.py` | Add `retarget_two_stage()` method |
| `convert_episodes_to_motion.py` | NPZ → pickle converter (new) |
| `motion_data_configs/teleop_dataset.yaml` | Training config (new) |
| `stream_ik_teleop.py` | Use two-stage IK for recording |

## Session Notes - Jan 26, 2026

### Episode → Motion Converter (COMPLETED)

Created `convert_episodes_to_motion.py` to convert teleop episode NPZ files to TWIST2 motion pickle format.

**Usage:**
```bash
cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
conda activate gmr

# List available episodes
python convert_episodes_to_motion.py --list

# Convert single episode
python convert_episodes_to_motion.py --episode elbow_track_007

# Convert all episodes + generate YAML config
python convert_episodes_to_motion.py --all --yaml teleop_dataset
```

**Format Conversion:**
```
Episode NPZ (input):
  - robot_qpos: (N, 36) = [x, y, z, qw, qx, qy, qz, 29_joints]
  - Quaternion: scalar-first [qw, qx, qy, qz] (MuJoCo format)

Motion PKL (output):
  - fps: float
  - root_pos: (N, 3)
  - root_rot: (N, 4) = [qx, qy, qz, qw] (scalar-last, TWIST2 format)
  - dof_pos: (N, 29)
  - local_body_pos: (N, 38, 3) - FK body positions relative to pelvis
  - link_body_list: list of 38 body names
```

**Key Implementation Details:**
- Quaternion format conversion: MuJoCo uses [qw,qx,qy,qz], TWIST2 uses [qx,qy,qz,qw]
- Forward kinematics computed via MuJoCo to get `local_body_pos`
- 38 bodies from G1 model (excluding 'world')

**Output:**
- Motion files: `datasets/teleop_motions/*.pkl`
- Training config: `motion_data_configs/teleop_dataset.yaml`
- Total: 20 episodes, ~30,000 frames, ~17 minutes of motion data

### Testing Converted Motions

**Option 1: Kinematic Playback (no physics)**
```bash
cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
conda activate gmr

python replay_motion.py --file ../datasets/teleop_motions/elbow_track_007.pkl --loop
```

**Option 2: Sim2Sim with Trained Policy (with physics)**

Terminal 1 - Motion Server (gmr env):
```bash
conda activate gmr
cd ~/projects/g1-pick-n-place/TWIST2/deploy_real

python server_motion_lib.py \
    --motion_file ../datasets/teleop_motions/elbow_track_007.pkl \
    --robot unitree_g1_with_hands \
    --vis \
    --device cpu
```

Terminal 2 - Sim2Sim (twist2 env):
```bash
conda activate twist2
cd ~/projects/g1-pick-n-place/TWIST2

bash sim2sim.sh
```

### Dependency Fixes

**scipy/numpy compatibility issue:**
- scipy must be imported BEFORE matplotlib (numpy 2.x quirk)
- Fixed import order in `server_motion_lib.py`
- Added notes to `requirements_gmr.txt`

**pose module installation:**
- Installed as editable package: `cd TWIST2/pose && pip install -e .`
- No longer need to set PYTHONPATH manually

### Environment Summary

| Environment | Purpose |
|-------------|---------|
| `gmr` | Motion capture, IK retargeting, motion conversion, motion server |
| `twist2` | RL training, sim2sim, policy inference |

### Next Steps

1. [x] Train new policy on teleop motion dataset (Isaac Lab migration - see below)
2. [ ] Test policy tracking quality on custom motions
3. [ ] Implement two-stage IK for better lower body tracking
4. [ ] Record whole-body walking motions with lower body enabled

---

## Session Notes - Jan 26, 2026 (Isaac Lab Migration)

### Major Milestone: Isaac Lab Training Pipeline Working!

Successfully migrated the motion imitation training from legged_gym to Isaac Lab.

**Key Files Created/Modified:**
| File | Purpose |
|------|---------|
| `isaaclab_envs/g1_motion_mimic_env.py` | Custom ManagerBasedRLEnv with MotionLib integration |
| `isaaclab_envs/g1_motion_mimic_env_cfg.py` | Environment config with rewards, terminations, observations |
| `isaaclab_envs/motion_lib.py` | **Vectorized** motion library (228x speedup!) |
| `isaaclab_envs/motion_mdp.py` | Custom observation/reward/termination functions |
| `scripts/train_isaaclab.py` | Training script with robust error handling |

### Performance Optimization: Vectorized MotionLib

**Problem:** Original `get_motion_state()` had a Python for-loop iterating over each environment:
```python
for i in range(num_envs):  # 6144 iterations!
    motion_id = motion_ids[i].item()
    ...
```

**Solution:** Complete rewrite with vectorized GPU operations:
- Pre-stack all motion data into padded tensors at load time: `(num_motions, max_frames, ...)`
- Use advanced tensor indexing: `self._stacked_dof_pos[motion_ids, frame_0]`
- All interpolation done with batch tensor ops

**Results:**
| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Computation | 700 steps/s | 160,195 steps/s | **228x** |
| Iteration time | 200+ seconds | 0.92 seconds | **217x** |
| ETA (20k iters) | 16+ hours | ~6 hours | **2.7x** |

### Reward Function Fixes

**Problem:** `tracking_joint_dof`, `tracking_joint_vel`, `tracking_keybody_pos` all showing 0.0

**Root Cause:** Using `torch.sum()` instead of `torch.mean()` for squared error:
```python
# Before (broken):
dof_error = torch.sum(torch.square(current_dof - target_dof), dim=1)
# With 37 joints, even small errors sum to ~10, giving exp(-40) ≈ 0

# After (fixed):
dof_error = torch.mean(torch.square(current_dof - target_dof), dim=1)
# Mean error ~0.25, giving exp(-1) ≈ 0.37 (useful gradient!)
```

**Results:** `tracking_joint_dof` now showing 0.0503 (non-zero!)

### Configuration Decisions

**Disabled `motion_tracking_failure` termination:**
- Motion data has absolute world positions from teleoperation
- Robot spawns at random positions → immediate large position error
- Disabled for now; TODO: implement relative motion tracking

**Current Terminations:**
- `time_out`: Episode ends at 10 seconds
- `base_contact`: Robot torso touches ground (falling)

### Training Progress (Iteration 5/20000)

```
Computation: 160,195 steps/s
Iteration time: 0.92s
Mean reward: -5.14
Mean episode length: 39.13 steps
Episode_Reward/tracking_joint_dof: 0.0503  ✓ Non-zero!
Episode_Reward/tracking_root_height: 0.0386
Episode_Reward/tracking_root_orientation: 0.0414
Episode_Termination/base_contact: 139.3333 (~140/4096 envs falling)
```

### Training Command

```bash
cd ~/projects/g1-pick-n-place/TWIST2
conda activate env_isaaclab

python scripts/train_isaaclab.py \
    --task Isaac-Motion-Mimic-G1-v0 \
    --num_envs 4096 \
    --max_iterations 20000 \
    --headless
```

### Known Issues

1. **`tracking_joint_vel: 0.0`** - Velocity errors still too large even with mean
2. **`tracking_keybody_pos: 0.0`** - Same issue, need further investigation
3. **`feet_air_time: 0.0`** - Foot contact detection may need tuning
4. **wandb crashes with Isaac Sim** - Using tensorboard for now

### Files Reference

```
TWIST2/
├── isaaclab_envs/
│   ├── __init__.py              # Registers gym task
│   ├── g1_motion_mimic_env.py   # Environment class
│   ├── g1_motion_mimic_env_cfg.py  # Config
│   ├── motion_lib.py            # Vectorized motion library
│   ├── motion_mdp.py            # Custom MDP functions
│   └── agents/
│       └── rsl_rl_ppo_cfg.py    # PPO config
├── scripts/
│   └── train_isaaclab.py        # Training script
├── logs/
│   └── isaaclab/                # TensorBoard logs
└── motion_data_configs/
    └── teleop_dataset.yaml      # Points to converted motions
```

### TensorBoard Monitoring

```bash
cd ~/projects/g1-pick-n-place/TWIST2
tensorboard --logdir logs/isaaclab/motion_mimic
```

---

## Session Notes - Jan 26, 2026 (Continued)

### Play Script for Policy Visualization

Created `scripts/play_isaaclab.py` to visualize trained policies in Isaac Lab viewer.

**Usage:**
```bash
cd ~/projects/g1-pick-n-place/TWIST2
conda activate env_isaaclab

# Basic playback
python scripts/play_isaaclab.py \
    --checkpoint logs/isaaclab/motion_mimic/model_9500.pt \
    --num_envs 16

# With push disturbances (test robustness)
python scripts/play_isaaclab.py \
    --checkpoint logs/isaaclab/motion_mimic/model_9500.pt \
    --num_envs 16 \
    --push --push_force 100 --push_interval 3
```

**Options:**
- `--checkpoint`: Path to trained model checkpoint
- `--num_envs`: Number of robots to visualize (default: 16)
- `--push`: Enable random push disturbances
- `--push_force`: Push force in Newtons (default: 50)
- `--push_interval`: Seconds between pushes (default: 5)

### Bug Fixes

#### 1. Robot Respawning Issue
Fallen robots weren't resetting. Two fixes applied:

1. **Height-based termination** added to `g1_motion_mimic_env_cfg.py`:
   ```python
   bad_height = DoneTerm(
       func=mdp.root_height_below_minimum,
       params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("robot")},
   )
   ```

2. **Fixed play script** - removed `reset_base = None` which was preventing robot position reset

#### 2. Wide Stance Issue
Robot was spreading legs too far apart (>1.5m). Added feet distance penalty:

```python
# In motion_mdp.py
def feet_distance_penalty(env, asset_cfg, min_dist=0.1, max_dist=0.6):
    """Penalty for feet being too far apart or too close together."""
    # Returns positive penalty if outside [min_dist, max_dist] range

# In g1_motion_mimic_env_cfg.py
feet_distance = RewTerm(
    func=motion_mdp.feet_distance_penalty,
    weight=-5.0,
    params={"min_dist": 0.1, "max_dist": 0.6},  # 10-60cm acceptable
)
```

### Robustness Training (Stage 2)

Added support for curriculum learning with push disturbances.

**Workflow:**
1. **Stage 1**: Train basic motion skills (no pushes)
   ```bash
   python scripts/train_isaaclab.py --num_envs 4096 --max_iterations 200000
   ```

2. **Stage 2**: Fine-tune with push disturbances
   ```bash
   python scripts/train_isaaclab.py \
       --robust \
       --checkpoint logs/isaaclab/motion_mimic/model_200000.pt \
       --max_iterations 250000
   ```

**Robust mode** (`--robust` flag):
- Enables random push disturbances every 8-12 seconds
- Push velocity: ±0.8 m/s in X/Y directions
- Uses `G1MotionMimicEnvCfg_ROBUST` config class

### Updated Files Reference

```
TWIST2/
├── isaaclab_envs/
│   ├── __init__.py                 # Registers gym task
│   ├── g1_motion_mimic_env.py      # Environment class
│   ├── g1_motion_mimic_env_cfg.py  # Config (includes _ROBUST, _PLAY variants)
│   ├── motion_lib.py               # Vectorized motion library (228x speedup)
│   ├── motion_mdp.py               # Custom MDP functions (incl. feet_distance_penalty)
│   └── agents/
│       └── rsl_rl_ppo_cfg.py       # PPO config
├── scripts/
│   ├── train_isaaclab.py           # Training script (supports --robust)
│   └── play_isaaclab.py            # Visualization script (supports --push)
├── logs/
│   └── isaaclab/motion_mimic/      # Checkpoints and TensorBoard logs
└── motion_data_configs/
    └── teleop_dataset.yaml         # Points to converted motions
```

### Training Progress Notes

Current training configuration:
- 6144 parallel environments
- ~160,000 steps/s computation speed
- ~0.9s per iteration
- Using TensorBoard for logging (wandb has compatibility issues)

Key metrics to watch:
- `Episode_Reward/tracking_joint_dof`: Should be non-zero and increasing
- `Episode_Reward/feet_distance`: New penalty for wide stance
- `Episode_Termination/base_contact`: Falls due to contact
- `Episode_Termination/bad_height`: Falls detected by height threshold

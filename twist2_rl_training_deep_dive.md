# TWIST2 RL Training Pipeline: Complete Implementation Deep Dive

## Executive Summary

This document provides a comprehensive analysis of TWIST2's reinforcement learning training pipeline for humanoid motion tracking. The system uses **Isaac Gym** for parallel simulation, **PPO** (Proximal Policy Optimization) for training, and a sophisticated curriculum learning approach with **~19,426 motion clips** from multiple datasets.

**Key Innovation**: A two-level hierarchical controller where:
- **Low-level (System 1)**: RL-trained motion tracking policy (trained offline)
- **High-level (System 2)**: Visuomotor diffusion policy (trained from collected data)

---

## Table of Contents

1. [Training Framework Overview](#training-framework-overview)
2. [Datasets](#datasets)
3. [RL Framework & Physics Engine](#rl-framework--physics-engine)
4. [Training Environment](#training-environment)
5. [Reward System](#reward-system)
6. [Domain Randomization](#domain-randomization)
7. [Observation Space](#observation-space)
8. [Network Architecture](#network-architecture)
9. [Training Curriculum](#training-curriculum)
10. [How to Train on Custom Datasets](#training-on-custom-datasets)

---

## Training Framework Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TWIST2 Training Pipeline                  │
└─────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
    ┌─────▼──────┐                     ┌─────▼──────┐
    │ Low-Level  │                     │ High-Level │
    │   Policy   │                     │   Policy   │
    │ (System 1) │                     │ (System 2) │
    └─────┬──────┘                     └─────┬──────┘
          │                                   │
   Isaac Gym (Sim2Real RL)          Imitation Learning
   Motion Tracking Controller        (Diffusion Policy)
          │                                   │
    PPO Training                       TWIST2 Data
    ~19k Motions                    (100 demos/15min)
          │                                   │
    ┌─────▼──────────────────────────────────▼──────┐
    │              Unitree G1 Robot                  │
    └────────────────────────────────────────────────┘
```

### System 1: Motion Tracking Controller (Low-Level)

**Training Approach**: Sim-to-Real RL using Isaac Gym

**Purpose**: Track reference motion trajectories (from PICO VR or motion datasets)

**Training Details**:
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: Isaac Gym + legged_gym (from ETH Zurich RSL)
- **Simulation**: 4,096 parallel environments
- **Training Time**: 1-2 days on single RTX 4090 (24GB)
- **Iterations**: 30,000+ policy updates
- **Control Frequency**: 50Hz (decimation=10, sim_dt=0.002s)

### System 2: Visuomotor Policy (High-Level)

**Training Approach**: Imitation Learning from demonstrations

**Purpose**: Generate future joint targets from egocentric vision

**Training Details**:
- **Algorithm**: 3D Diffusion Policy (improved version)
- **Data**: TWIST2-collected demonstrations
- **Input**: Egocentric camera images (ZED stereo)
- **Output**: Whole-body joint position sequences

---

## Datasets

### Dataset Breakdown (Total: 19,426 motions)

```
Dataset Source         | Count  | Percentage | Description
-----------------------|--------|------------|---------------------------
TWIST1 (In-house)      | 12,788 | 65.8%      | Previous system recordings
OMOMO                  | 5,841  | 30.1%      | Object manipulation motions
AMASS                  | 724    | 3.7%       | Walking & locomotion
Custom VR (v1_v2_v3)   | 73     | 0.4%       | PICO VR recordings
-----------------------|--------|------------|---------------------------
TOTAL                  | 19,426 | 100%       |
```

### 1. OMOMO Dataset (30.1% - 5,841 motions)

**Source**: Object Motion guided human Motion synthesis

**Citation**: 
```bibtex
@article{Li2023OMOMO,
  title={Object Motion Guided Human Motion Synthesis},
  author={Li, Jingbo and Wu, Jiajun and Liu, C. Karen},
  journal={ACM Transactions on Graphics (TOG)},
  year={2023}
}
```

**Content**:
- Human-object interaction motions
- Tasks: clothesstand manipulation, furniture interaction, object placement
- Retargeted via GMR to Unitree G1
- Format: `OMOMO_g1_GMR/sub1_clothesstand_000.pkl` through `sub1_clothesstand_XXX.pkl`

**Example Motions**:
```yaml
- file: OMOMO_g1_GMR/sub1_clothesstand_000.pkl
  weight: 1.0
  description: general movement
```

### 2. AMASS Dataset (3.7% - 724 motions)

**Source**: Archive of Motion Capture as Surface Shapes

**Citation**:
```bibtex
@inproceedings{AMASS:2019,
  title={AMASS: Archive of Motion Capture as Surface Shapes},
  author={Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
  booktitle={ICCV},
  year={2019}
}
```

**Content**:
- Professional motion capture data
- Activities: walking, turning, martial arts stances, side-stepping
- Retargeted via GMR8 (GMR version 8) to G1
- Subset: ACCAD (Advanced Computing Center for the Arts and Design) database

**Example Motions**:
```yaml
- file: AMASS_g1_GMR8/ACCAD_Male1Walking_c3d_Walk_B10_-_Walk_turn_left_45_stageii.pkl
  weight: 1.0
  description: walking and turning
```

**Key Activities in AMASS subset**:
- Walking (forward, backward, with box)
- Turning (45°, 90°, 135°, 180°)
- Martial arts (stances, blocks, dodges)
- Side-stepping (left/right)
- Advanced movements (skip, crouch, bounce)

### 3. TWIST1 Dataset (65.8% - 12,788 motions)

**Source**: Previous iteration of TWIST system (OptiTrack MoCap)

**Content**:
- In-house recorded motions using OptiTrack motion capture
- Diverse whole-body motions from TWIST development
- Already retargeted and validated
- Format: `twist1_to_twist2/XXXX.pkl`

**Why Include TWIST1**:
1. **Domain alignment**: Same robot (Unitree G1)
2. **Proven quality**: Already validated in TWIST paper (CoRL 2025)
3. **Diversity**: Complements OMOMO and AMASS
4. **Volume**: Provides bulk of training data

### 4. Custom VR Recordings (0.4% - 73 motions)

**Source**: PICO 4 Ultra + Motion Trackers (v1, v2, v3 versions)

**Purpose**: **Critical for domain gap bridging**

**From Paper** (Section III-F, Training Data Pipeline):
> "Similarly as found in TWIST, we find that curating a small set of motions from the 
> teleoperation device is essential to bridge the domain gap. We only collect 73 motions 
> via PICO, as these motions already cover most daily movements like walking, crouching, 
> and manipulation."

**Why Only 73 Motions?**:
- **Domain-specific**: Captured with actual PICO VR system
- **Noise characteristics**: Include real VR tracking noise
- **Essential movements**: Cover key daily activities
- **Efficient**: Small but targeted collection

**Example Motions**:
```yaml
- file: v1_v2_v3_g1/v1_walk_0.pkl
  weight: 1.0
  description: walking motion from PICO VR
```

### Dataset Processing Pipeline

```
Raw Human Motion Data (SMPL-X / BVH / MoCap)
              ↓
┌─────────────────────────────────────────┐
│  GMR (General Motion Retargeting)       │
│  - Non-uniform scaling                  │
│  - Two-stage optimization               │
│  - Pelvis-centric for PICO data         │
└─────────────────┬───────────────────────┘
                  ↓
    Robot Motion Format (.pkl)
    ┌────────────────────────┐
    │ - root_pos [x,y,z]     │
    │ - root_rot [quat]      │
    │ - root_vel [vx,vy,vz]  │
    │ - joint_pos [29 dims]  │
    │ - joint_vel [29 dims]  │
    │ - body_pos [N×3]       │
    └────────┬───────────────┘
             ↓
    Motion Library (MotionLib)
    - Indexed by motion_id
    - Weighted sampling
    - Motion decomposition
    - Smoothing (optional)
```

### Motion Data Format

Each `.pkl` file contains:

```python
motion_data = {
    'root_pos': torch.Tensor,      # (T, 3) - XYZ position
    'root_rot': torch.Tensor,      # (T, 4) - Quaternion [w,x,y,z]
    'root_vel': torch.Tensor,      # (T, 3) - Linear velocity
    'root_ang_vel': torch.Tensor,  # (T, 3) - Angular velocity
    'dof_pos': torch.Tensor,       # (T, 29) - Joint positions
    'dof_vel': torch.Tensor,       # (T, 29) - Joint velocities
    'body_pos': torch.Tensor,      # (T, N_bodies, 3) - Body positions
    'fps': int,                    # Frames per second
    'dt': float,                   # Time step
}
```

### Configuration File Structure

From `twist2_dataset.yaml`:

```yaml
root_path: /path/to/motion_data

motions:
  - file: OMOMO_g1_GMR/sub1_clothesstand_000.pkl
    weight: 1.0
    description: general movement
    
  - file: AMASS_g1_GMR8/ACCAD_Male1Walking_c3d_Walk_B10.pkl
    weight: 1.0
    description: walking and turning
    
  - file: twist1_to_twist2/motion_0001.pkl
    weight: 1.0
    description: previous system data
    
  - file: v1_v2_v3_g1/v1_walk_0.pkl
    weight: 1.0
    description: PICO VR recording
```

**Weight System**:
- All motions have `weight: 1.0` (uniform sampling initially)
- Weights can be adjusted for curriculum learning
- Motion difficulty tracked dynamically during training

---

## RL Framework & Physics Engine

### Isaac Gym

**What is Isaac Gym?**

Isaac Gym is NVIDIA's GPU-accelerated physics simulation framework for reinforcement learning.

**Key Features**:
- **Massive Parallelization**: 1,000s of environments on single GPU
- **Physics Engine**: PhysX (GPU-accelerated)
- **Direct Tensor Access**: No CPU/GPU copies needed
- **PyTorch Integration**: Seamless tensor operations

**Why Isaac Gym for TWIST2?**

1. **Speed**: Train 4,096 parallel environments simultaneously
2. **Scale**: Collect millions of samples per hour
3. **Efficiency**: Direct GPU tensor operations
4. **Proven**: Used in legged robotics (ANYmal, Go1, etc.)

### legged_gym Framework

**Source**: ETH Zurich Robotic Systems Lab (RSL)

**Based on**: 
```bibtex
@software{legged_gym,
  author = {Nikita Rudin and others},
  title = {Legged Gym},
  url = {https://github.com/leggedrobotics/legged_gym},
  year = {2021}
}
```

**Architecture**:

```
legged_gym/
├── envs/                    # Environment definitions
│   ├── base/
│   │   ├── legged_robot.py      # Base robot class
│   │   ├── humanoid.py          # Humanoid extension
│   │   ├── humanoid_mimic.py    # Motion tracking
│   │   └── ...
│   └── g1/
│       ├── g1_mimic.py          # G1-specific config
│       ├── g1_mimic_distill.py  # Student policy
│       └── g1_mimic_future.py   # TWIST2 main config
├── gym_utils/               # Helper utilities
└── scripts/
    └── train.py            # Training entry point
```

### Integration Details

**Isaac Gym in Code** (`humanoid_mimic.py` line 3-4):

```python
from isaacgym.torch_utils import *
from isaacgym import gymtorch
```

**Environment Creation**:

```python
# From train.py
env, _ = task_registry.make_env(name='g1_mimic_future', args=args)
```

**Physics Timestep** (`g1_mimic_config.py`):

```python
class sim:
    dt = 0.002  # 2ms physics timestep = 500Hz
    
class control:
    decimation = 10  # Control at 50Hz (500Hz / 10)
```

**Parallel Environments**:

```python
class env:
    num_envs = 4096  # Number of parallel simulations
```

### Physics Configuration

**Robot Dynamics** (`g1_mimic_config.py` lines 93-159):

```python
class control:
    # PD controller gains
    stiffness = {
        'hip_yaw': 100,
        'hip_roll': 100,
        'hip_pitch': 100,
        'knee': 150,
        'ankle': 40,
        'waist': 150,
        'shoulder': 40,
        'elbow': 40,
        'wrist': 40,
    }  # [N*m/rad]
    
    damping = {
        'hip_yaw': 2,
        'hip_roll': 2,
        'hip_pitch': 2,
        'knee': 4,
        'ankle': 2,
        'waist': 4,
        'shoulder': 5,
        'elbow': 5,
        'wrist': 5,
    }  # [N*m*s/rad]
    
    action_scale = 0.5
    decimation = 10

class asset:
    # Motor inertia (critical for accurate dynamics)
    # shoulder, elbow, ankle joints
    dof_armature = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + \
                   [0.0103] * 3 + [0.003597] * 14
```

**Why Inertia Matters**:
- Accounts for motor rotor inertia
- Critical for sim-to-real transfer
- Calculated from Unitree G1 specifications

---

## Training Environment

### Humanoid Mimic Environment

**Class Hierarchy**:

```
BaseTask (isaacgym)
    ↓
LeggedRobot (legged_gym)
    ↓
Humanoid (adds humanoid-specific features)
    ↓
HumanoidChar (character animation support)
    ↓
HumanoidMimic (motion tracking)
    ↓
G1Mimic (Unitree G1 specific)
    ↓
G1MimicFuture (TWIST2 implementation)
```

### Key Environment Features

#### 1. Motion Library Integration

**From `humanoid_mimic.py` lines 94-99**:

```python
def _load_motions(self):
    self._motion_lib = MotionLib(
        motion_file=self.cfg.motion.motion_file,  # twist2_dataset.yaml
        device=self.device,
        sample_ratio=self.cfg.motion.sample_ratio,  # Subsampling
        motion_decompose=self.cfg.motion.motion_decompose,  # Optional
        motion_smooth=self.cfg.motion.motion_smooth  # Smoothing
    )
```

**Motion Sampling** (`humanoid_mimic.py` lines 126-149):

```python
def _reset_ref_motion(self, env_ids, motion_ids=None):
    n = len(env_ids)
    if motion_ids is None:
        # Error-aware sampling (optional)
        if self.cfg.motion.use_error_aware_sampling:
            motion_ids = self._motion_lib.sample_motions(
                n, 
                motion_difficulty=self.motion_difficulty,
                max_key_body_error=self.max_key_body_error,
                use_error_aware_sampling=True,
                error_sampling_power=5.0,
                error_sampling_threshold=0.15
            )
        else:
            # Uniform sampling with difficulty weighting
            motion_ids = self._motion_lib.sample_motions(
                n, 
                motion_difficulty=self.motion_difficulty
            )
    
    # Random time offset (or start from beginning)
    if self._rand_reset:
        motion_times = self._motion_lib.sample_time(motion_ids)
    else:
        motion_times = torch.zeros(motion_ids.shape)
```

#### 2. Reference Motion Retrieval

**Getting target poses** (`humanoid_mimic.py`):

```python
def _get_mimic_obs(self):
    num_steps = self._tar_motion_steps_priv.shape[0]  # Future frames
    motion_times = self._get_motion_times().unsqueeze(-1)
    
    # Get future reference frames
    obs_motion_times = self._tar_motion_steps_priv * self.dt + motion_times
    
    # Fetch from motion library
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = \
        self._motion_lib.calc_motion_frame(motion_ids, obs_motion_times)
    
    # Apply motion domain randomization (if enabled)
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = \
        self._apply_motion_domain_randomization(...)
    
    return mimic_obs
```

#### 3. Termination Conditions

**From `humanoid_mimic.py` lines 450-505**:

```python
def check_termination(self):
    # 1. Root height termination
    root_too_low = self.root_states[:, 2] < 0.3
    
    # 2. Orientation termination (roll/pitch too large)
    roll_too_large = torch.abs(self.roll) > self.cfg.rewards.termination_roll
    pitch_too_large = torch.abs(self.pitch) > self.cfg.rewards.termination_pitch
    
    # 3. Contact termination (torso hits ground)
    torso_contact = torch.any(
        torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., 
        dim=1
    )
    
    # 4. Pose tracking termination (optional)
    if self._pose_termination:
        pose_err = self._compute_key_body_error()
        pose_too_far = pose_err > self._pose_termination_dist  # 0.7m
    
    # Combine all termination conditions
    self.reset_buf = root_too_low | roll_too_large | pitch_too_large | torso_contact
    
    if self._pose_termination:
        self.reset_buf = self.reset_buf | pose_too_far
```

#### 4. Target Motion Steps (Future Frames)

**Student Policy** (`g1_mimic_config.py`):

```python
tar_motion_steps_priv = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
# 20 future frames at [0.02s, 0.1s, 0.2s, ..., 1.9s]
```

**What This Means**:
- Policy receives reference motion at 20 future timesteps
- Each step is 0.02s apart (50Hz control)
- Total lookahead: 1.9 seconds
- Helps policy anticipate future motion

---

## Reward System

### Reward Structure

**Total Reward**:

```
R_total = Σ(w_i * R_tracking_i) + Σ(w_j * R_regularization_j)
```

Where:
- **Tracking rewards** encourage matching reference motion
- **Regularization rewards** penalize undesirable behaviors

### Tracking Rewards

**From `humanoid_mimic.py` lines 685-853**:

#### 1. Joint Position Tracking (`_reward_tracking_joint_dof`)

```python
def _reward_tracking_joint_dof(self):
    dof_diff = self._ref_dof_pos - self.dof_pos  # Shape: (n_envs, 29)
    dof_err = torch.sum(self._dof_err_w * dof_diff * dof_diff, dim=-1)
    
    pos_scale = 0.15
    return torch.exp(-pos_scale * dof_err)
```

**Weight**: 2.0 (from `g1_mimic_future_config.py`)

**DOF Error Weights** (`g1_mimic_config.py` lines 39-44):
```python
dof_err_w = [
    1.0, 1.0, 1.0, 1.0, 0.1, 0.1,  # Left Leg (lower weight on ankles)
    1.0, 1.0, 1.0, 1.0, 0.1, 0.1,  # Right Leg
    1.0, 1.0, 1.0,                  # Waist (yaw, roll, pitch)
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Left Arm
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Right Arm
]
```

**Why Lower Ankle Weights?**:
- Ankles are compliance joints
- Exact tracking less critical than other joints
- Prevents over-stiff behavior

#### 2. Joint Velocity Tracking (`_reward_tracking_joint_vel`)

```python
def _reward_tracking_joint_vel(self):
    vel_diff = self._ref_dof_vel - self.dof_vel
    vel_err = torch.sum(self._dof_err_w * vel_diff * vel_diff, dim=-1)
    
    vel_scale = 0.01
    return torch.exp(-vel_scale * vel_err)
```

**Weight**: 0.2

#### 3. Root Translation Tracking

**XY Translation** (`_reward_tracking_root_translation_xy`):

```python
def _reward_tracking_root_translation_xy(self):
    root_pos_diff = self._ref_root_pos[:, :2] - self.root_states[:, :2]
    root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
    
    root_pose_scale = 5.0
    return torch.exp(-root_pose_scale * root_pos_err)
```

**Weight**: Not explicitly set (disabled in favor of global tracking)

**Z Translation** (`_reward_tracking_root_translation_z`):

```python
def _reward_tracking_root_translation_z(self):
    root_pos_diff = self._ref_root_pos[:, 2:3] - self.root_states[:, 2:3]
    root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
    
    root_pose_scale = 5.0
    return torch.exp(-root_pose_scale * root_pos_err)
```

**Weight**: 1.0

#### 4. Root Rotation Tracking (`_reward_tracking_root_rotation`)

```python
def _reward_tracking_root_rotation(self):
    root_rot_err = quat_diff_angle(self.root_states[:, 3:7], self._ref_root_rot)
    root_rot_err *= root_rot_err
    
    root_pose_scale = 5.0
    return torch.exp(-root_pose_scale * root_rot_err)
```

**Weight**: 1.0

#### 5. Root Velocity Tracking (`_reward_tracking_root_vel`)

```python
def _reward_tracking_root_vel(self):
    if self.global_obs:
        root_vel_diff = self._ref_root_vel - self.root_states[:, 7:10]
        root_ang_vel_diff = self._ref_root_ang_vel - self.root_states[:, 10:13]
    else:
        # Local frame velocities
        local_ref_root_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_vel)
        root_vel_diff = local_ref_root_vel - self.base_lin_vel
        
        local_ref_root_ang_vel = quat_rotate_inverse(self._ref_root_rot, self._ref_root_ang_vel)
        root_ang_vel_diff = local_ref_root_ang_vel - self.base_ang_vel
    
    root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)
    root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)
    root_vel_scale = 1.0
    
    return torch.exp(-root_vel_scale * (root_vel_err + 0.5 * root_ang_vel_err))
```

**Weights**: 
- Linear velocity: 1.0
- Angular velocity: 1.0

#### 6. Key Body Position Tracking (`_reward_tracking_keybody_pos`)

**Most Important Tracking Reward**

```python
def _reward_tracking_keybody_pos(self):
    # Get current key body positions
    key_body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3]
    key_body_pos = key_body_pos - self.root_states[:, 0:3].unsqueeze(1)
    
    # Convert to local frame (yaw-only)
    base_yaw_quat = quat_from_euler_xyz(0, 0, self.yaw)
    key_body_pos = convert_to_local_root_body_pos(base_yaw_quat, key_body_pos)
    
    # Get reference key body positions
    tar_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :]
    tar_key_body_pos = tar_key_body_pos - self._ref_root_pos.unsqueeze(1)
    
    _, _, ref_yaw = euler_from_quaternion(self._ref_root_rot)
    ref_yaw_quat = quat_from_euler_xyz(0, 0, ref_yaw)
    tar_key_body_pos = convert_to_local_root_body_pos(ref_yaw_quat, tar_key_body_pos)
    
    # Compute error
    key_body_pos_diff = key_body_pos - tar_key_body_pos
    key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)
    key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)
    
    key_body_pos_scale = 10.0
    return torch.exp(-key_body_pos_scale * key_body_pos_err)
```

**Weight**: 2.0

**Key Bodies** (`g1_mimic_future_config.py` line 297):

```python
key_bodies = [
    "left_rubber_hand",      # Left hand end-effector
    "right_rubber_hand",     # Right hand end-effector
    "left_ankle_roll_link",  # Left foot
    "right_ankle_roll_link", # Right foot
    "left_knee_link",        # Left knee
    "right_knee_link",       # Right knee
    "left_elbow_link",       # Left elbow
    "right_elbow_link",      # Right elbow
    "head_mocap"             # Head
]
```

**Why These Bodies?**:
- **End-effectors** (hands, feet): Critical for manipulation/locomotion
- **Mid-segments** (knees, elbows): Ensure proper limb configuration
- **Head**: Maintains overall posture and balance

#### 7. Global Key Body Position Tracking

```python
def _reward_tracking_keybody_pos_global(self):
    key_body_pos = self.rigid_body_states[:, self._key_body_ids, 0:3]
    tar_key_body_pos = self._ref_body_pos[:, self._key_body_ids, :]
    
    key_body_pos_diff = key_body_pos - tar_key_body_pos
    key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)
    key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)
    
    key_body_pos_scale = 10.0
    return torch.exp(-key_body_pos_scale * key_body_pos_err)
```

**Weight**: 2.0

**Difference from local version**:
- Tracks absolute world positions
- Doesn't subtract root position
- Used for global motion tracking tasks

### Regularization Rewards (Penalties)

**From `g1_mimic_config.py` lines 162-179**:

```python
regularization_names = [
    "feet_stumble",           # Prevent foot dragging
    "feet_contact_forces",    # Limit excessive ground forces
    "lin_vel_z",             # Discourage vertical velocity
    "ang_vel_xy",            # Discourage roll/pitch rates
    "orientation",           # Maintain upright orientation
    "dof_pos_limits",        # Stay within joint limits
    "dof_torque_limits",     # Limit motor torques
    "collision",             # Penalize body collisions
    "torque_penalty",        # Minimize overall torque
    "thigh_torque_roll_yaw", # Specific joint penalties
    "thigh_roll_yaw_acc",    # Limit joint accelerations
    "dof_acc",               # General joint acceleration penalty
    "dof_vel",               # Limit joint velocities
    "action_rate",           # Smooth action changes
]
```

#### Key Regularization Details

**1. Feet Stumble** (`_reward_feet_stumble`):

```python
def _reward_feet_stumble(self):
    # Check if horizontal foot contact forces > 4× vertical forces
    rew = torch.any(
        torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
        4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), 
        dim=1
    )
    return rew.float()
```

**Weight**: -1.25

**Purpose**: Penalize foot dragging/stumbling

**2. Feet Contact Forces** (`_reward_feet_contact_forces`):

```python
def _reward_feet_contact_forces(self):
    rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
    rew[rew < 350] = 0  # max_contact_force threshold
    rew[rew > 350] -= 350
    return rew
```

**Weight**: -5e-4

**Purpose**: Penalize excessive ground reaction forces

**3. DOF Position Limits** (`_reward_dof_pos_limits`):

```python
def _reward_dof_pos_limits(self):
    # Penalize being outside joint limits
    out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
    out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)
```

**Weight**: -5.0

**4. DOF Torque Limits** (`_reward_dof_torque_limits`):

```python
def _reward_dof_torque_limits(self):
    out_of_limits = torch.sum(
        (torch.abs(self.torques) / self.torque_limits - 0.95).clip(min=0), 
        dim=1
    )
    return out_of_limits
```

**Weight**: -1.0

**Soft limit**: 95% of maximum torque

**5. Action Rate** (`_reward_action_rate`):

```python
def _reward_action_rate(self):
    # Penalize rapid action changes
    return torch.sum(
        torch.square(self.action_history_buf[:, -1] - self.action_history_buf[:, -2]), 
        dim=1
    )
```

**Weight**: -0.05 (increased from -0.01 for TWIST2)

**Purpose**: Encourage smooth control

### Reward Computation

**From base class**:

```python
def compute_reward(self):
    # Tracking rewards
    rew_tracking_joint_dof = self._reward_tracking_joint_dof()
    rew_tracking_joint_vel = self._reward_tracking_joint_vel()
    rew_tracking_keybody_pos = self._reward_tracking_keybody_pos()
    # ... other tracking rewards
    
    # Regularization penalties
    rew_feet_stumble = self._reward_feet_stumble()
    rew_action_rate = self._reward_action_rate()
    # ... other penalties
    
    # Combine with weights
    total_reward = (
        2.0 * rew_tracking_joint_dof +
        0.2 * rew_tracking_joint_vel +
        2.0 * rew_tracking_keybody_pos +
        # ... other tracking
        -1.25 * rew_feet_stumble +
        -0.05 * rew_action_rate +
        # ... other penalties
    )
    
    # Apply regularization scaling
    total_reward *= self.regularization_scale
    
    return total_reward
```

### Reward Scaling Curriculum

```python
class rewards:
    regularization_scale = 1.0
    regularization_scale_range = [0.8, 2.0]
    regularization_scale_curriculum = False  # Disabled for TWIST2
    regularization_scale_gamma = 0.0001
```

**If enabled**:
- Gradually increases penalty weights during training
- Helps initially learn tracking, then refine naturalness

---

## Domain Randomization

### Why Domain Randomization?

**Sim-to-Real Gap**: Simulations are perfect; reality is messy

**Solution**: Add randomness to simulation → policy learns robustness

### DR Categories

**From `g1_mimic_config.py` lines 242-270**:

```python
class domain_rand:
    domain_rand_general = True  # Master switch
    
    # 1. Gravity Randomization
    randomize_gravity = True
    gravity_rand_interval_s = 4  # Change every 4s
    gravity_range = (-0.1, 0.1)  # ±0.1 m/s² around 9.81
    
    # 2. Friction Randomization
    randomize_friction = True
    friction_range = [0.1, 2.0]  # 0.1× to 2× nominal friction
    
    # 3. Mass Randomization
    randomize_base_mass = True
    added_mass_range = [-3., 3.]  # ±3 kg to base
    
    # 4. Center of Mass Randomization
    randomize_base_com = True
    added_com_range = [-0.05, 0.05]  # ±5cm COM offset
    
    # 5. External Pushes
    push_robots = True
    push_interval_s = 4  # Push every 4s
    max_push_vel_xy = 1.0  # m/s impulse
    
    # 6. End-Effector Pushes
    push_end_effector = True
    push_end_effector_interval_s = 2
    max_push_force_end_effector = 20.0  # Newtons
    
    # 7. Motor Strength Randomization
    randomize_motor = True
    motor_strength_range = [0.8, 1.2]  # 80-120% of nominal
    
    # 8. Action Delay
    action_delay = True
    action_buf_len = 8  # Buffer up to 8 steps of delay
```

### Implementation Details

#### 1. Mass Randomization

```python
def _randomize_base_mass(self, env_ids):
    # Sample from range
    added_mass = torch.rand(len(env_ids), device=self.device) * \
                 (self.cfg.domain_rand.added_mass_range[1] - 
                  self.cfg.domain_rand.added_mass_range[0]) + \
                 self.cfg.domain_rand.added_mass_range[0]
    
    # Apply to base link
    for i, env_id in enumerate(env_ids):
        self.gym.set_actor_mass(
            self.envs[env_id], 
            self.actor_handles[env_id], 
            self.base_mass + added_mass[i]
        )
```

#### 2. Motor Strength Randomization

```python
def _randomize_motor_strength(self, env_ids):
    # Sample per-actuator strength factors
    motor_strength = torch.rand(len(env_ids), 2, device=self.device) * \
                     (self.cfg.domain_rand.motor_strength_range[1] - 
                      self.cfg.domain_rand.motor_strength_range[0]) + \
                     self.cfg.domain_rand.motor_strength_range[0]
    
    # motor_strength[:, 0] = Proportional gain multiplier
    # motor_strength[:, 1] = Derivative gain multiplier
    self.motor_strength[env_ids] = motor_strength
```

**Applied in control loop**:

```python
def _compute_torques(self, actions):
    # PD control with randomized gains
    torques = self.p_gains * self.motor_strength[:, 0:1] * \
              (actions - self.dof_pos) + \
              self.d_gains * self.motor_strength[:, 1:2] * \
              (0 - self.dof_vel)
    return torques
```

#### 3. External Pushes

```python
def _push_robots(self):
    # Check if it's time to push
    if self.common_step_counter % (self.cfg.domain_rand.push_interval_s / self.dt) == 0:
        # Sample push direction
        push_theta = 2 * np.pi * torch.rand(self.num_envs, device=self.device)
        push_vel_x = torch.cos(push_theta) * self.cfg.domain_rand.max_push_vel_xy
        push_vel_y = torch.sin(push_theta) * self.cfg.domain_rand.max_push_vel_xy
        
        # Apply impulse to root
        self.root_states[:, 7] += push_vel_x
        self.root_states[:, 8] += push_vel_y
        
        # Update in simulation
        self.gym.set_actor_root_state_tensor(
            self.sim, 
            gymtorch.unwrap_tensor(self.root_states)
        )
```

### Motion Domain Randomization (NEW in TWIST2)

**Purpose**: Add noise to reference motion → robustness to noisy VR tracking

**From `g1_mimic_future_config.py` lines 84-96**:

```python
class motion:
    # Motion Domain Randomization
    motion_dr_enabled = False  # Can enable for extra robustness
    root_position_noise = [0.01, 0.05]      # ±1-5cm
    root_orientation_noise = [0.1, 0.2]     # ±5.7-11.4° (rad)
    root_velocity_noise = [0.05, 0.1]       # ±0.05-0.1 m/s
    joint_position_noise = [0.05, 0.1]      # ±0.05-0.1 rad
    motion_dr_resampling = True             # New noise each step
```

**Implementation** (`humanoid_mimic.py` lines 545-617):

```python
def _apply_motion_domain_randomization(self, root_pos, root_rot, root_vel, 
                                       root_ang_vel, dof_pos, dof_vel):
    if not self.cfg.motion.motion_dr_enabled:
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel
    
    # Get noise ranges
    pos_noise_range = self.cfg.motion.root_position_noise
    ori_noise_range = self.cfg.motion.root_orientation_noise
    vel_noise_range = self.cfg.motion.root_velocity_noise
    joint_noise_range = self.cfg.motion.joint_position_noise
    
    batch_size = root_pos.shape[0]
    
    if self.cfg.motion.motion_dr_resampling:
        # Sample noise magnitude from uniform distribution
        pos_noise_mag = torch.rand(batch_size, 1, device=self.device) * \
                       (pos_noise_range[1] - pos_noise_range[0]) + pos_noise_range[0]
        
        # Add position noise
        pos_noise = (torch.rand(batch_size, 3, device=self.device) * 2 - 1) * pos_noise_mag
        root_pos_noisy = root_pos + pos_noise
        
        # Add orientation noise (via quaternion multiplication)
        ori_noise_mag = torch.rand(batch_size, 1, device=self.device) * \
                       (ori_noise_range[1] - ori_noise_range[0]) + ori_noise_range[0]
        
        # Generate random rotation axis
        axis_noise = torch.randn(batch_size, 3, device=self.device)
        axis_noise = axis_noise / (torch.norm(axis_noise, dim=1, keepdim=True) + 1e-8)
        
        # Sample angle uniformly
        angle_noise = (torch.rand(batch_size, 1, device=self.device) * 2 - 1) * ori_noise_mag
        
        # Convert axis-angle to quaternion
        half_angle = angle_noise / 2
        sin_half = torch.sin(half_angle)
        cos_half = torch.cos(half_angle)
        
        quat_noise = torch.zeros(batch_size, 4, device=self.device)
        quat_noise[:, :3] = axis_noise * sin_half
        quat_noise[:, 3:4] = cos_half
        
        # Apply rotation
        root_rot_noisy = quat_mul(root_rot, quat_noise)
        
        # Add velocity noise
        vel_noise_mag = torch.rand(batch_size, 1, device=self.device) * \
                       (vel_noise_range[1] - vel_noise_range[0]) + vel_noise_range[0]
        vel_noise = (torch.rand(batch_size, 3, device=self.device) * 2 - 1) * vel_noise_mag
        root_vel_noisy = root_vel + vel_noise
        
        ang_vel_noise = (torch.rand(batch_size, 3, device=self.device) * 2 - 1) * vel_noise_mag
        root_ang_vel_noisy = root_ang_vel + ang_vel_noise
        
        # Add joint noise
        joint_noise_mag = torch.rand(batch_size, 1, device=self.device) * \
                         (joint_noise_range[1] - joint_noise_range[0]) + joint_noise_range[0]
        joint_noise = (torch.rand(batch_size, dof_pos.shape[1], device=self.device) * 2 - 1) * \
                      joint_noise_mag
        dof_pos_noisy = dof_pos + joint_noise
    
    return root_pos_noisy, root_rot_noisy, root_vel_noisy, root_ang_vel_noisy, \
           dof_pos_noisy, dof_vel
```

**Noise Ranges Explained**:

| Parameter | Range | Typical VR Noise |
|-----------|-------|------------------|
| Position | ±1-5cm | PICO headset drift |
| Orientation | ±5.7-11.4° | IMU sensor drift |
| Velocity | ±0.05-0.1 m/s | Finite differencing noise |
| Joints | ±0.05-0.1 rad | Retargeting uncertainty |

---

## Observation Space

### Observation Structure

**Total Size**: Variable based on configuration

**From `g1_mimic_future_config.py` lines 24-35**:

```python
# Single-frame mimic observation
n_mimic_obs_single = 6 + 29  # 35 dims
# - root_pos: 3 (x, y, z)
# - root_ori: 2 (roll, pitch) - yaw excluded
# - root_vel: 3 (vx, vy, vz)
# - root_ang_vel: 1 (yaw rate only)
# - dof_pos: 29 (joint positions)

# Current frame observation (for history)
tar_motion_steps = [0]  # Only current frame
n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single  # 35 dims

# Proprioception
n_proprio = n_mimic_obs + 3 + 2 + 3*num_actions
# - base_ang_vel: 3
# - imu: 2 (roll, pitch)
# - dof_pos: 29
# - dof_vel: 29
# - prev_actions: 29

# Privileged latent (domain randomization parameters)
n_priv_latent = 4 + 1 + 2*num_actions  # 63 dims
# - mass_params: 4 (mass offsets)
# - friction: 1
# - motor_strength: 58 (2 per actuator)

# History
history_len = 10
n_obs_single = n_mimic_obs + n_proprio  # Single timestep

# Future observations (for TWIST2 student)
tar_motion_steps_future = [0]  # Can add [5,10,15,...] for future
n_future_obs_single = 6 + 29
n_future_obs = len(tar_motion_steps_future) * n_future_obs_single

# Total
num_observations = n_obs_single * (history_len + 1) + n_future_obs
```

### Observation Components Breakdown

#### 1. Reference Motion (Mimic Obs)

**Purpose**: Tell policy what motion to track

**From `humanoid_mimic.py` lines 507-543**:

```python
def _get_mimic_obs(self):
    # Get future motion frames
    num_steps = len(self._tar_motion_steps_priv)  # 20 future frames
    motion_times = self._get_motion_times().unsqueeze(-1)
    obs_motion_times = self._tar_motion_steps_priv * self.dt + motion_times
    
    # Fetch from motion library
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = \
        self._motion_lib.calc_motion_frame(motion_ids, obs_motion_times)
    
    # Apply motion DR (if enabled)
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = \
        self._apply_motion_domain_randomization(...)
    
    # Convert to euler angles
    roll, pitch, _ = euler_from_quaternion(root_rot)
    roll = roll.reshape(self.num_envs, num_steps, 1)
    pitch = pitch.reshape(self.num_envs, num_steps, 1)
    
    # Convert to local frame (if not global obs)
    if not self.global_obs:
        root_vel = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
    
    # Reshape for output
    root_pos = root_pos.reshape(self.num_envs, num_steps, 3)
    root_vel = root_vel.reshape(self.num_envs, num_steps, 3)
    root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, 3)
    dof_pos = dof_pos.reshape(self.num_envs, num_steps, 29)
    
    # Concatenate
    mimic_obs_buf = torch.cat((
        root_pos[..., 0:3],          # 3 dims - XYZ position
        roll, pitch,                 # 2 dims - orientation (no yaw)
        root_vel,                    # 3 dims - linear velocity
        root_ang_vel[..., 2:3],     # 1 dim - yaw rate only
        dof_pos,                    # 29 dims - joint positions
    ), dim=-1)
    
    return mimic_obs_buf.reshape(self.num_envs, -1)  # Flatten frames
```

**Why exclude yaw from orientation?**:
- Yaw is rotation around Z-axis (heading direction)
- For locomotion, absolute heading doesn't matter
- Policy should work regardless of facing direction
- Only relative yaw rate matters (angular velocity Z)

#### 2. Proprioception

**Purpose**: Tell policy current robot state

```python
def compute_observations(self):
    imu_obs = torch.stack((self.roll, self.pitch), dim=1)  # 2 dims
    
    mimic_obs = self._get_mimic_obs()  # Reference motion
    
    obs_buf = torch.cat((
        mimic_obs,  # Future reference frames
        self.base_ang_vel * self.obs_scales.ang_vel,  # 3 dims
        imu_obs,  # 2 dims
        self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),  # 29
        self.reindex(self.dof_vel * self.obs_scales.dof_vel),  # 29
        self.reindex(self.action_history_buf[:, -1]),  # 29 (previous action)
    ), dim=-1)
```

**Observation Scaling** (`obs_scales`):

```python
class normalization:
    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
```

**Why scaling?**:
- Neural networks train better with normalized inputs
- Different quantities have different natural scales
- Scaling puts all observations in similar ranges (~[-1, 1])

#### 3. Privileged Latent (Domain Randomization Info)

**Purpose**: Help critic estimate value under DR

```python
if self.cfg.domain_rand.domain_rand_general:
    priv_latent = torch.cat((
        self.mass_params_tensor,        # 4 dims
        self.friction_coeffs_tensor,    # 1 dim
        self.motor_strength[0] - 1,     # 29 dims (P gain offset)
        self.motor_strength[1] - 1,     # 29 dims (D gain offset)
        self.base_lin_vel,              # 3 dims
    ), dim=-1)
else:
    priv_latent = torch.zeros((self.num_envs, self.cfg.env.n_priv_latent))
    priv_latent = torch.cat((priv_latent, self.base_lin_vel), dim=-1)
```

**Why privileged?**:
- Only available in simulation (critic uses it)
- Not available in real world (actor doesn't use it)
- Helps value estimation under domain randomization
- See "Asymmetric Actor-Critic" papers

#### 4. Observation History

**Purpose**: Provide temporal context

```python
if self.cfg.env.history_len > 0:
    self.obs_history_buf = torch.where(
        (self.episode_length_buf <= 1)[:, None, None], 
        torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),  # Reset
        torch.cat([
            self.obs_history_buf[:, 1:],  # Shift left
            obs_buf.unsqueeze(1)          # Add new
        ], dim=1)
    )
```

**History Length**: 10 frames = 0.2 seconds (at 50Hz)

**Why history?**:
- Partial observability: velocities estimated from positions
- Temporal patterns: footstep timing, oscillations
- Smoother control: react to trends, not instantaneous noise

### Final Observation Assembly

```python
self.obs_buf = torch.cat([
    obs_buf,                                  # Current observation
    priv_latent,                             # DR parameters
    self.obs_history_buf.view(self.num_envs, -1)  # History (flattened)
], dim=-1)
```

### Observation Noise

**From `g1_mimic_config.py` lines 272-281**:

```python
class noise:
    add_noise = True
    noise_increasing_steps = 3000  # Curriculum: gradually increase noise
    
    class noise_scales:
        dof_pos = 0.01      # ±0.01 rad
        dof_vel = 0.1       # ±0.1 rad/s
        lin_vel = 0.1       # ±0.1 m/s
        ang_vel = 0.1       # ±0.1 rad/s
        gravity = 0.05      # ±0.05 m/s²
        imu = 0.1           # ±0.1 rad
```

**Application**:

```python
if self.cfg.noise.add_noise and self.headless:
    # Curriculum: scale noise up over training
    noise_scale = min(self.total_env_steps_counter / \
                     (self.cfg.noise.noise_increasing_steps * 24), 1.0)
    obs_buf += (2 * torch.rand_like(obs_buf) - 1) * \
               self.noise_scale_vec * noise_scale
```

---

## Network Architecture

### Actor-Critic Architecture

```
┌─────────────────────────────────────────────┐
│               Observations                   │
│  (history + proprio + privileged + future)   │
└────────────┬────────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
┌─────▼─────┐  ┌───▼─────┐
│   Actor   │  │  Critic  │
│  Policy   │  │  Value   │
└─────┬─────┘  └───┬──────┘
      │            │
      │            │
  [Action]    [Value Est.]
   (29 dim)
```

### Actor Network

**From `g1_mimic_future_config.py` (inherits from PPO config)**:

```python
class policy:
    actor_hidden_dims = [512, 512, 256, 128]
    activation = 'silu'  # Swish activation
    init_noise_std = 0.8
    action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
    # Lower std for waist joints (more precise control needed)
```

**Architecture**:

```
Input: (batch, n_obs)
   ↓
Linear(n_obs, 512) + SiLU
   ↓
Linear(512, 512) + SiLU
   ↓
Linear(512, 256) + SiLU
   ↓
Linear(256, 128) + SiLU
   ↓
Linear(128, 29)  → Mean actions
   ↓
Add Gaussian noise with learned std
   ↓
Output: (batch, 29) actions
```

**Action Representation**:

```python
# Actions are target joint positions (residual from default pose)
actions = mean_actions + torch.randn_like(mean_actions) * action_std

# Applied as:
target_dof_pos = self.default_dof_pos + self.action_scale * actions
```

### Critic Network

```python
class policy:
    critic_hidden_dims = [512, 512, 256, 128]
    activation = 'silu'
```

**Architecture**:

```
Input: (batch, n_obs + n_privileged)
   ↓
Linear(n_total, 512) + SiLU
   ↓
Linear(512, 512) + SiLU
   ↓
Linear(512, 256) + SiLU
   ↓
Linear(256, 128) + SiLU
   ↓
Linear(128, 1)  → Value estimate
   ↓
Output: (batch, 1) state value
```

**Asymmetric Setup**:
- **Actor** sees: observations + history
- **Critic** sees: observations + history + privileged info

**Why?**:
- Critic needs privileged info to estimate value under DR
- Actor must work without it (real world doesn't have DR params)

### SiLU Activation

**SiLU** (Sigmoid Linear Unit) = Swish = x * sigmoid(x)

```python
def silu(x):
    return x * torch.sigmoid(x)
```

**Why SiLU over ReLU?**:
- Smoother gradients
- No "dying ReLU" problem
- Slightly better performance in RL (empirically)

---

## Training Curriculum

### Multi-Level Curriculum Learning

TWIST2 uses **three simultaneous curriculum strategies**:

#### 1. Motion Difficulty Curriculum

**Purpose**: Start with easy motions, gradually add harder ones

**Implementation** (`humanoid_mimic.py`):

```python
# Initialize all motions as "hard" (100)
self.motion_difficulty = 100 * torch.ones((num_motions), device=self.device)

# During training, track success
if episode_success:
    # Make this motion easier (lower difficulty score)
    self.motion_difficulty[motion_id] *= 0.99
else:
    # Make this motion harder (higher difficulty score)
    self.motion_difficulty[motion_id] *= 1.01

# Clamp to reasonable range
self.motion_difficulty = torch.clamp(self.motion_difficulty, 1.0, 200.0)
```

**Sampling**:

```python
# Sample motions inversely proportional to difficulty
prob = 1.0 / self.motion_difficulty
prob = prob / prob.sum()
motion_id = torch.multinomial(prob, num_samples=1)
```

**Effect**:
- Easy motions sampled less often (already learned)
- Hard motions sampled more often (need practice)
- Automatic balancing

**Config** (`g1_mimic_future_config.py`):

```python
class motion:
    motion_curriculum = True
    motion_curriculum_gamma = 0.01  # Speed of difficulty adjustment
```

#### 2. Error-Aware Sampling (Optional)

**Purpose**: Focus on motions where tracking error is highest

**From `g1_mimic_future_config.py`**:

```python
class motion:
    use_error_aware_sampling = False  # Can enable
    error_sampling_power = 5.0
    error_sampling_threshold = 0.15  # 15cm max key body error
```

**Implementation** (`humanoid_mimic.py` lines 129-141):

```python
if self.cfg.motion.use_error_aware_sampling:
    motion_ids = self._motion_lib.sample_motions(
        n, 
        motion_difficulty=self.motion_difficulty,
        max_key_body_error=self.max_key_body_error,  # Tracked per motion
        use_error_aware_sampling=True,
        error_sampling_power=5.0,
        error_sampling_threshold=0.15
    )
```

**Probability Calculation**:

```python
# Normalize error
norm_error = max_key_body_error / error_sampling_threshold

# Apply power (emphasize high errors)
weighted_error = norm_error ** error_sampling_power

# Sample probability
prob = weighted_error / weighted_error.sum()
```

**Effect**:
- Motions with high tracking error sampled more
- Focuses training on hard-to-track motions
- Complementary to difficulty curriculum

#### 3. Regularization Scaling Curriculum

**Purpose**: Initially focus on tracking, then refine naturalness

```python
class rewards:
    regularization_scale = 1.0
    regularization_scale_range = [0.8, 2.0]
    regularization_scale_curriculum = False  # Disabled for TWIST2
    regularization_scale_gamma = 0.0001
```

**If enabled**:

```python
# Start with low regularization (focus on tracking)
regularization_scale = 0.8

# Gradually increase
regularization_scale += regularization_scale_gamma * success_rate

# Clamp to max
regularization_scale = min(regularization_scale, 2.0)

# Apply to penalties
total_reward = tracking_rewards + regularization_scale * penalties
```

#### 4. Noise Curriculum

**From observation noise section**:

```python
noise_scale = min(
    self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24), 
    1.0
)
obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * noise_scale
```

**Effect**:
- Start with clean observations
- Gradually add noise over first 3,000 iterations × 24 steps = 72,000 steps
- Helps initial learning, then builds robustness

---

## Training on Custom Datasets

### Step-by-Step Guide

#### Step 1: Record Your Motion Data

**Option A: Use PICO VR System**

```bash
# 1. Setup PICO hardware
# - PICO 4 Ultra headset
# - 2× ankle trackers
# - 1× waist tracker (optional)

# 2. Install XRoboToolkit
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb

# 3. Start PICO streaming application
# (Follow PICO SDK setup instructions)

# 4. Record motion
python gmr/scripts/record_pico_motion.py \
    --output_dir ./my_motions \
    --motion_name "my_walk" \
    --duration 30  # seconds
```

**Option B: Use Motion Capture System**

```bash
# OptiTrack, Vicon, etc.
# Export as BVH or FBX format
```

**Option C: Use Video + GVHMR**

```bash
# Install GVHMR
git clone https://github.com/zju3dv/GVHMR
cd GVHMR && pip install -e .

# Extract pose from video
python scripts/extract_pose_from_video.py \
    --video my_motion.mp4 \
    --output my_motion.pkl
```

#### Step 2: Retarget to Robot

```bash
# Activate GMR environment
conda activate gmr

# Retarget human motion to G1
python gmr/scripts/smplx_to_robot.py \
    --smplx_file my_motions/my_walk.pkl \
    --robot unitree_g1 \
    --save_path retargeted/my_walk_g1.pkl \
    --visualize  # Optional: see result in MuJoCo
```

**GMR Config** (automatic for PICO data):

```yaml
# For PICO VR data, GMR automatically uses:
pelvis_centric: true
optimize_lower_body_position: true
optimize_upper_body_rotation_only: true
```

#### Step 3: Create Dataset YAML

```yaml
# my_dataset.yaml
root_path: /path/to/retargeted/motions

motions:
  # Your custom motions
  - file: my_walk_g1.pkl
    weight: 1.0
    description: my walking motion
    
  - file: my_crouch_g1.pkl
    weight: 1.0
    description: crouching motion
    
  # Optional: Include pre-existing datasets for diversity
  - file: AMASS_g1_GMR8/ACCAD_Male1Walking_c3d_Walk_B10.pkl
    weight: 0.5  # Lower weight = sampled less
    description: AMASS walking
```

**Weight Guidelines**:
- 1.0 for your custom motions (highest priority)
- 0.5-1.0 for complementary motions
- Lower weights for background diversity

#### Step 4: Modify Training Config

```python
# my_train_config.py
from legged_gym.envs.g1.g1_mimic_future_config import G1MimicStuFutureCfg

class MyTrainCfg(G1MimicStuFutureCfg):
    class motion(G1MimicStuFutureCfg.motion):
        # Point to your dataset
        motion_file = "path/to/my_dataset.yaml"
        
        # Recommended settings for custom data
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        
        # Optional: Enable motion DR for PICO data
        motion_dr_enabled = True  # If using PICO VR
        root_position_noise = [0.01, 0.05]
        root_orientation_noise = [0.1, 0.2]
        root_velocity_noise = [0.05, 0.1]
        joint_position_noise = [0.05, 0.1]
```

#### Step 5: Train

```bash
# Activate training environment
conda activate twist2

# Start training
python legged_gym/scripts/train.py \
    --task g1_mimic_future \
    --config my_train_config \
    --exptid my_experiment \
    --num_envs 4096 \
    --headless
```

**Training Arguments**:

```bash
--task: Environment name (g1_mimic_future for TWIST2)
--config: Your custom config class
--exptid: Experiment name (for logging)
--num_envs: Number of parallel environments (4096 recommended)
--headless: Run without GUI (faster)
--debug: Use fewer envs, enable GUI (for debugging)
```

#### Step 6: Monitor Training

**WandB Dashboard**:

```python
# Training will log to Weights & Biases
# View at: https://wandb.ai/your-entity/twist

# Key metrics to watch:
# - mean_reward: Should increase
# - tracking_joint_dof: Should approach 1.0
# - tracking_keybody_pos: Should approach 1.0
# - episode_length: Should increase
# - motion_difficulty: Should balance out
```

**Local Logs**:

```bash
# Logs saved to:
# legged_gym/logs/my_experiment/

# Tensorboard:
tensorboard --logdir logs/my_experiment
```

#### Step 7: Evaluate

```bash
# Test trained policy
python legged_gym/scripts/play.py \
    --task g1_mimic_future \
    --load_run my_experiment \
    --checkpoint -1  # Latest checkpoint
```

#### Step 8: Export for Deployment

```bash
# Convert to ONNX for fast inference
python legged_gym/scripts/save_onnx.py \
    --task g1_mimic_future \
    --load_run my_experiment \
    --checkpoint -1 \
    --output my_policy.onnx
```

### Best Practices for Custom Datasets

#### 1. Data Quality

**✅ Good motions**:
- Smooth and natural
- Physically feasible
- Clear start/end poses
- 30-60 seconds duration

**❌ Avoid**:
- Jumpy or glitchy retargeting
- Physically impossible poses
- Very short clips (<5 seconds)
- Abrupt transitions

#### 2. Dataset Diversity

**Minimum viable dataset**:
- 20-50 motions covering key behaviors
- Walk forward/backward/sideways
- Turn left/right
- Crouch/stand
- Arm movements (if needed)

**Recommended**:
- 100+ motions for robust policy
- Mix of locomotion + manipulation
- Include "domain-specific" motions (73 PICO clips in TWIST2)

#### 3. Motion Curriculum

```python
# Start training with:
motion_curriculum = True
motion_curriculum_gamma = 0.01  # Faster if dataset is small

# Monitor motion_difficulty values
# All motions should converge to similar difficulty (balanced)
```

#### 4. Reward Tuning

```python
# Start with TWIST2 defaults
# If tracking is poor, increase tracking weights:
class rewards:
    class scales:
        tracking_joint_dof = 3.0  # Increased from 2.0
        tracking_keybody_pos = 3.0  # Increased from 2.0

# If motion is unnatural, increase penalties:
        feet_stumble = -2.0  # Increased from -1.25
        action_rate = -0.1   # Increased from -0.05
```

#### 5. Debugging Tips

**Low tracking performance**:
- Check retargeting quality (visualize in MuJoCo)
- Increase tracking reward weights
- Reduce motion DR noise
- Start with simpler motions

**Unnatural motion**:
- Increase regularization penalties
- Add more reference motions
- Check joint limits and PD gains

**Training instability**:
- Reduce learning rate
- Reduce number of parallel envs
- Check for NaN in observations/rewards

---

## Summary & Key Takeaways

### Training Pipeline Overview

```
1. Dataset Preparation
   ├─ AMASS (724 motions) - Public mocap
   ├─ OMOMO (5,841 motions) - Object manipulation
   ├─ TWIST1 (12,788 motions) - Previous system
   └─ PICO Custom (73 motions) - Domain-specific ⭐

2. GMR Retargeting
   ├─ Non-uniform scaling
   ├─ Two-stage optimization
   └─ Pelvis-centric for VR data

3. RL Training (Isaac Gym)
   ├─ 4,096 parallel environments
   ├─ PPO algorithm
   ├─ 30,000 iterations (~1-2 days on RTX 4090)
   └─ Multi-level curriculum learning

4. Deployment
   ├─ Export to ONNX
   ├─ 50Hz control frequency
   └─ Whole-body motion tracking
```

### Critical Success Factors

1. **High-Quality Retargeting**: GMR is essential for artifact-free motion
2. **Diverse Dataset**: Mix of public + custom data
3. **Domain-Specific Data**: Small set (73 clips) of PICO data bridges gap
4. **Curriculum Learning**: Motion difficulty + noise gradually increase
5. **Domain Randomization**: Physics randomization for sim-to-real transfer
6. **Reward Design**: Balance tracking accuracy vs. natural motion

### Innovations in TWIST2

1. **Motion Domain Randomization**: Add noise to reference motion
2. **Error-Aware Sampling**: Focus on hard-to-track motions
3. **Pelvis-Centric Coordinates**: Robust to VR tracking noise
4. **Future Motion Observations**: 1.9s lookahead for anticipation
5. **Hierarchical Control**: Separate low-level tracking from high-level planning

### For Your Research

**To reproduce TWIST2**:
1. Use Isaac Gym + legged_gym framework
2. Collect ~100 custom motions with PICO VR
3. Retarget with GMR
4. Train with motion curriculum + domain randomization
5. Test thoroughly in simulation before deploying

**To build on TWIST2**:
1. Add your own reward terms for specific tasks
2. Integrate additional sensors (force, vision)
3. Extend to bi-manual manipulation
4. Explore different policy architectures (transformers, etc.)

---

## References

### Papers

```bibtex
@article{ze2025twist2,
  title={TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System},
  author={Ze, Yanjie and Zhao, Siheng and Wang, Weizhuo and others},
  journal={arXiv preprint arXiv:2511.02832},
  year={2025}
}

@article{araujo2025gmr,
  title={Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking},
  author={Araujo, Joao Pedro and Ze, Yanjie and Xu, Pei and Wu, Jiajun and Liu, C. Karen},
  journal={arXiv preprint arXiv:2510.02252},
  year={2025}
}

@article{ze2025twist,
  title={TWIST: Teleoperated Whole-Body Imitation System},
  author={Ze, Yanjie and Chen, Zixuan and others},
  journal={arXiv preprint arXiv:2505.02833},
  year={2025}
}

@inproceedings{AMASS:2019,
  title={AMASS: Archive of Motion Capture as Surface Shapes},
  author={Mahmood, Naureen and Ghorbani, Nima and others},
  booktitle={ICCV},
  year={2019}
}

@article{Li2023OMOMO,
  title={Object Motion Guided Human Motion Synthesis},
  author={Li, Jingbo and Wu, Jiajun and Liu, C. Karen},
  journal={ACM TOG},
  year={2023}
}
```

### Code Resources

- **TWIST2**: https://github.com/amazon-far/TWIST2
- **GMR**: https://github.com/YanjieZe/GMR
- **Isaac Gym**: https://developer.nvidia.com/isaac-gym
- **legged_gym**: https://github.com/leggedrobotics/legged_gym

---

*This document provides implementation-level details for researchers looking to reproduce or build upon TWIST2's RL training pipeline. For questions or clarifications, refer to the original paper and codebase.*

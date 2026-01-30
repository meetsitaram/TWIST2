# Isaac Lab G1 Motion Mimic Observation Specification

## Overview

This document defines the exact 178-dimensional observation vector used by the 
G1MotionMimicEnv in Isaac Lab.

**Source:** `isaaclab_envs/g1_motion_mimic_env_cfg.py` (G1MotionMimicObservations.PolicyCfg)

## Observation Vector Structure (178 dims)

| Index Range | Size | Name | Description | Source Function |
|-------------|------|------|-------------|-----------------|
| 0-2 | 3 | `base_lin_vel` | Base linear velocity (m/s) in body frame | `mdp.base_lin_vel` |
| 3-5 | 3 | `base_ang_vel` | Base angular velocity (rad/s) in body frame | `mdp.base_ang_vel` |
| 6-8 | 3 | `projected_gravity` | Gravity vector projected to body frame | `mdp.projected_gravity` |
| 9-45 | 37 | `joint_pos` | Joint positions relative to default (rad) | `mdp.joint_pos_rel` |
| 46-82 | 37 | `joint_vel` | Joint velocities relative to default (rad/s) | `mdp.joint_vel_rel` |
| 83-119 | 37 | `actions` | Previous action (policy output) | `mdp.last_action` |
| 120-156 | 37 | `target_joint_pos` | Motion target joint positions | `motion_mdp.target_joint_pos` |
| 157-177 | 21 | `target_keybody_pos` | Motion target key body positions (7 bodies × 3) | `motion_mdp.target_keybody_pos_local` |

**Total: 3 + 3 + 3 + 37 + 37 + 37 + 37 + 21 = 178 dimensions**

## Isaac Lab Joint Order (37 joints, alphabetical)

The 37-joint observations/actions follow Isaac Lab's alphabetical ordering:

| Index | Joint Name | Body Part |
|-------|------------|-----------|
| 0 | left_ankle_pitch_joint | Left Leg |
| 1 | left_ankle_roll_joint | Left Leg |
| 2 | left_elbow_pitch_joint | Left Arm |
| 3 | left_elbow_roll_joint | Left Arm |
| 4 | left_five_joint | Left Hand |
| 5 | left_hip_pitch_joint | Left Leg |
| 6 | left_hip_roll_joint | Left Leg |
| 7 | left_hip_yaw_joint | Left Leg |
| 8 | left_knee_joint | Left Leg |
| 9 | left_one_joint | Left Hand |
| 10 | left_shoulder_pitch_joint | Left Arm |
| 11 | left_shoulder_roll_joint | Left Arm |
| 12 | left_shoulder_yaw_joint | Left Arm |
| 13 | left_three_joint | Left Hand |
| 14 | left_zero_joint | Left Hand |
| 15 | right_ankle_pitch_joint | Right Leg |
| 16 | right_ankle_roll_joint | Right Leg |
| 17 | right_elbow_pitch_joint | Right Arm |
| 18 | right_elbow_roll_joint | Right Arm |
| 19 | right_five_joint | Right Hand |
| 20 | right_hip_pitch_joint | Right Leg |
| 21 | right_hip_roll_joint | Right Leg |
| 22 | right_hip_yaw_joint | Right Leg |
| 23 | right_knee_joint | Right Leg |
| 24 | right_one_joint | Right Hand |
| 25 | right_shoulder_pitch_joint | Right Arm |
| 26 | right_shoulder_roll_joint | Right Arm |
| 27 | right_shoulder_yaw_joint | Right Arm |
| 28 | right_three_joint | Right Hand |
| 29 | right_zero_joint | Right Hand |
| 30 | torso_joint | Waist |
| 31-36 | (head/other joints if present) | Head |

**Note:** Actual indices depend on the robot URDF. Run this to verify:
```python
robot = env.scene["robot"]
print("Joint names:", robot.joint_names)
print("Num joints:", robot.num_joints)
```

## Key Body Positions (21 dims = 7 bodies × 3)

Target key body positions in local (robot) frame:

| Index Range | Size | Body Name |
|-------------|------|-----------|
| 0-2 | 3 | left_ankle_roll_link (left foot) |
| 3-5 | 3 | right_ankle_roll_link (right foot) |
| 6-8 | 3 | left_elbow_pitch_link (left elbow) |
| 9-11 | 3 | right_elbow_pitch_link (right elbow) |
| 12-14 | 3 | left_shoulder_pitch_link (left shoulder) |
| 15-17 | 3 | right_shoulder_pitch_link (right shoulder) |
| 18-20 | 3 | torso_link (torso) |

**Source:** `g1_motion_mimic_env_cfg.py` line 266-279

## Observation Noise (Training Only)

During training, noise is added:

| Observation | Noise Type | Min | Max |
|-------------|------------|-----|-----|
| base_lin_vel | Uniform | -0.1 | 0.1 |
| base_ang_vel | Uniform | -0.2 | 0.2 |
| projected_gravity | Uniform | -0.05 | 0.05 |
| joint_pos | Uniform | -0.01 | 0.01 |
| joint_vel | Uniform | -1.5 | 1.5 |
| actions | None | - | - |
| target_* | None | - | - |

## Sim2Real: Constructing Observations from Real Robot

To deploy an Isaac Lab trained policy on the real robot:

```python
def construct_observation(
    # From IMU
    base_lin_vel,      # (3,) - may need estimation from IMU
    base_ang_vel,      # (3,) - from IMU omega
    quat,              # (4,) - from IMU quaternion
    # From motors
    joint_pos,         # (29,) - real robot joint positions (29 DOF)
    joint_vel,         # (29,) - real robot joint velocities
    # From policy state
    last_action,       # (37,) - previous Isaac Lab action
    # From motion target (or zeros if not tracking)
    target_joint_pos,  # (37,) - motion target joints
    target_keybody_pos # (21,) - motion target key bodies
):
    # 1. Compute projected gravity from quaternion
    proj_grav = compute_projected_gravity(quat)
    
    # 2. Map 29-DOF real robot joints to 37-DOF Isaac Lab order
    il_joint_pos = map_real_to_isaaclab_joints(joint_pos)
    il_joint_vel = map_real_to_isaaclab_joints(joint_vel)
    
    # 3. Compute relative joint positions
    joint_pos_rel = il_joint_pos - DEFAULT_JOINT_POS_ISAACLAB
    joint_vel_rel = il_joint_vel  # Already relative (0 = no movement)
    
    # 4. Concatenate observation
    obs = np.concatenate([
        base_lin_vel,      # 0-2
        base_ang_vel,      # 3-5
        proj_grav,         # 6-8
        joint_pos_rel,     # 9-45
        joint_vel_rel,     # 46-82
        last_action,       # 83-119
        target_joint_pos,  # 120-156
        target_keybody_pos # 157-177
    ])
    
    return obs  # Shape: (178,)
```

## MuJoCo to Isaac Lab Joint Mapping

| MuJoCo Index | MuJoCo Name | Isaac Lab Name | Isaac Lab Index |
|--------------|-------------|----------------|-----------------|
| 0 | left_hip_pitch | left_hip_pitch_joint | 5 |
| 1 | left_hip_roll | left_hip_roll_joint | 6 |
| 2 | left_hip_yaw | left_hip_yaw_joint | 7 |
| 3 | left_knee | left_knee_joint | 8 |
| 4 | left_ankle_pitch | left_ankle_pitch_joint | 0 |
| 5 | left_ankle_roll | left_ankle_roll_joint | 1 |
| 6-11 | right leg | right_*_joint | 15, 16, 20, 21, 22, 23 |
| 12 | waist_yaw | torso_joint | 30 |
| 13 | waist_roll | (no equivalent) | - |
| 14 | waist_pitch | (no equivalent) | - |
| 15-21 | left arm | left_shoulder/elbow_* | 10, 11, 12, 2, 3, -, - |
| 22-28 | right arm | right_shoulder/elbow_* | 25, 26, 27, 17, 18, -, - |

**Note:** MuJoCo has 29 joints, Isaac Lab has 37. Extra Isaac Lab joints (hands, head) should be zeroed.

## Real Robot (29 DOF) to Isaac Lab (37 DOF) Mapping

For sim2real, map the real 29-DOF robot to Isaac Lab's 37-DOF space:

```python
# Real robot joint order (from g1.yaml):
# 0-5: left leg, 6-11: right leg, 12-14: waist, 15-21: left arm, 22-28: right arm

REAL_TO_ISAACLAB = {
    # Left leg
    0: 5,   # left_hip_pitch
    1: 6,   # left_hip_roll
    2: 7,   # left_hip_yaw
    3: 8,   # left_knee
    4: 0,   # left_ankle_pitch
    5: 1,   # left_ankle_roll
    # Right leg
    6: 20,  # right_hip_pitch
    7: 21,  # right_hip_roll
    8: 22,  # right_hip_yaw
    9: 23,  # right_knee
    10: 15, # right_ankle_pitch
    11: 16, # right_ankle_roll
    # Waist
    12: 30, # torso_joint (waist_yaw only)
    # Left arm
    15: 10, # left_shoulder_pitch
    16: 11, # left_shoulder_roll
    17: 12, # left_shoulder_yaw
    18: 2,  # left_elbow_pitch
    19: 3,  # left_elbow_roll
    # Right arm
    22: 25, # right_shoulder_pitch
    23: 26, # right_shoulder_roll
    24: 27, # right_shoulder_yaw
    25: 17, # right_elbow_pitch
    26: 18, # right_elbow_roll
}
# Note: Indices 13,14 (waist_roll/pitch), 20,21,27,28 (wrists) have no Isaac Lab equivalent
# Note: Isaac Lab hand joints (4,9,13,14,19,24,28,29) should be zeroed
```

## Action Space (37 dims)

Policy outputs 37-dimensional action in Isaac Lab joint order. These are position deltas from default pose.

For real robot deployment, map back to 29-DOF:
```python
ISAACLAB_TO_REAL = {v: k for k, v in REAL_TO_ISAACLAB.items()}
```

## References

- Environment config: `isaaclab_envs/g1_motion_mimic_env_cfg.py`
- Motion MDP functions: `isaaclab_envs/motion_mdp.py`
- Motion library: `isaaclab_envs/motion_lib.py`
- Real robot config: `deploy_real/robot_control/configs/g1.yaml`

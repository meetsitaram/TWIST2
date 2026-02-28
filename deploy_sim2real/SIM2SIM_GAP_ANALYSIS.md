https://github.com/meetsitaram/TWIST2/tree/isaaclab# Sim2Sim Gap Analysis: Isaac Lab vs MuJoCo

**Last Updated**: 2024-02-04  
**Validation Script**: `deploy_sim2real/validate_sim2sim_isaaclab_to_mujoco.py`

## Overview

This document details the differences between Isaac Lab (training) and MuJoCo (sim2sim validation) configurations for the Unitree G1 robot in the TWIST2 project.

## Quick Summary

| Feature | Isaac Lab | MuJoCo | Status |
|---------|-----------|--------|--------|
| **Total Joints** | 37 DOF | 29 DOF | ⚠️ Different models |
| **Mapped Joints** | 23 | 23 | ✅ Match |
| **PD Gains** | See below | Matched | ✅ Match |
| **Action Scale** | 0.5 | 0.5 | ✅ Match |
| **Control Freq** | 50Hz | 50Hz | ✅ Match |
| **Physics Freq** | 200Hz | 1000Hz | ⚠️ MuJoCo more accurate |
| **Init Height** | 0.74m | 0.74m | ✅ Match |
| **Default Pose** | See below | Aligned | ✅ Match |

## Joint Mapping

### Matched Joints (23/29)

| MuJoCo Joint | Isaac Lab Joint |
|--------------|-----------------|
| `left_hip_pitch_joint` | `left_hip_pitch_joint` |
| `left_hip_roll_joint` | `left_hip_roll_joint` |
| `left_hip_yaw_joint` | `left_hip_yaw_joint` |
| `left_knee_joint` | `left_knee_joint` |
| `left_ankle_pitch_joint` | `left_ankle_pitch_joint` |
| `left_ankle_roll_joint` | `left_ankle_roll_joint` |
| `right_hip_pitch_joint` | `right_hip_pitch_joint` |
| `right_hip_roll_joint` | `right_hip_roll_joint` |
| `right_hip_yaw_joint` | `right_hip_yaw_joint` |
| `right_knee_joint` | `right_knee_joint` |
| `right_ankle_pitch_joint` | `right_ankle_pitch_joint` |
| `right_ankle_roll_joint` | `right_ankle_roll_joint` |
| `waist_yaw_joint` | `torso_joint` |
| `left_shoulder_pitch_joint` | `left_shoulder_pitch_joint` |
| `left_shoulder_roll_joint` | `left_shoulder_roll_joint` |
| `left_shoulder_yaw_joint` | `left_shoulder_yaw_joint` |
| `left_elbow_joint` | `left_elbow_pitch_joint` |
| `left_wrist_roll_joint` | `left_elbow_roll_joint` |
| `right_shoulder_pitch_joint` | `right_shoulder_pitch_joint` |
| `right_shoulder_roll_joint` | `right_shoulder_roll_joint` |
| `right_shoulder_yaw_joint` | `right_shoulder_yaw_joint` |
| `right_elbow_joint` | `right_elbow_pitch_joint` |
| `right_wrist_roll_joint` | `right_elbow_roll_joint` |

### Unmapped Joints (6 - MuJoCo only)

These joints exist in the MuJoCo model but have **no Isaac Lab equivalent**:

| MuJoCo Joint | Note |
|--------------|------|
| `waist_roll_joint` | G1 only has 1 waist DOF in Isaac Lab |
| `waist_pitch_joint` | G1 only has 1 waist DOF in Isaac Lab |
| `left_wrist_pitch_joint` | Isaac Lab G1 has no wrist joints |
| `left_wrist_yaw_joint` | Isaac Lab G1 has no wrist joints |
| `right_wrist_pitch_joint` | Isaac Lab G1 has no wrist joints |
| `right_wrist_yaw_joint` | Isaac Lab G1 has no wrist joints |

**For real robot deployment**: Keep these joints at 0.0 as the policy doesn't control them.

## PD Gains

| Joint Group | Stiffness (Kp) | Damping (Kd) | Torque Limit |
|-------------|----------------|--------------|--------------|
| Hip Pitch | 200.0 | 5.0 | 300 Nm |
| Hip Roll | 150.0 | 5.0 | 300 Nm |
| Hip Yaw | 150.0 | 5.0 | 300 Nm |
| Knee | 200.0 | 5.0 | 300 Nm |
| Ankle | 20.0 | 2.0 | 20 Nm |
| Torso | 200.0 | 5.0 | 300 Nm |
| Arms | 40.0 | 10.0 | 300 Nm |

## Default Pose (Aligned)

| Joint | Value (rad) | Degrees |
|-------|-------------|---------|
| Hip Pitch | -0.20 | -11.5° |
| Hip Roll | 0.0 | 0° |
| Hip Yaw | 0.0 | 0° |
| Knee | 0.42 | 24.1° |
| Ankle Pitch | -0.23 | -13.2° |
| Ankle Roll | 0.0 | 0° |
| Shoulder Pitch | 0.35 | 20.1° |
| Shoulder Roll (L) | 0.16 | 9.2° |
| Shoulder Roll (R) | -0.16 | -9.2° |
| Shoulder Yaw | 0.0 | 0° |
| Elbow | 0.87 | 49.8° |
| Wrist Roll | 0.0 | 0° |

## Physics Settings

| Setting | Isaac Lab | MuJoCo | Notes |
|---------|-----------|--------|-------|
| Physics dt | 0.005s (200Hz) | 0.001s (1000Hz) | MuJoCo is 5x finer |
| Control dt | 0.02s (50Hz) | 0.02s (50Hz) | **Matches** |
| Decimation | 4 | 20 | Different but same control freq |
| Solver | PhysX GPU | PGS | Different physics engines |
| Gravity | -9.81 m/s² | -9.81 m/s² | **Matches** |

## Running Validation

```bash
cd deploy_sim2real
python validate_sim2sim_isaaclab_to_mujoco.py         # Basic report
python validate_sim2sim_isaaclab_to_mujoco.py -v      # Verbose with full mapping
```

## Files Modified

- `deploy_sim2real/sim2sim_mujoco.py` - Aligned default poses and init height
- `deploy_sim2real/validate_sim2sim_isaaclab_to_mujoco.py` - Created validation script
- `robot_config.py` - Single source of truth for joint mappings

## Real Robot Deployment Checklist

1. ✅ PD gains match Isaac Lab training config
2. ✅ Default pose matches Isaac Lab init_state
3. ✅ Control frequency is 50Hz
4. ✅ Action scale is 0.5
5. ⚠️ Lock unmapped joints (waist_roll/pitch, wrist_pitch/yaw) at 0.0
6. ⚠️ Verify joint limits match real robot datasheet


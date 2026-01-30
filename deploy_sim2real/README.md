# G1 Whole-Body Teleop Policy - Sim2Real Deployment

This package contains everything needed to deploy a trained TWIST2 policy to a Unitree G1 robot.

## Installation

```bash
# Create conda environment
conda create -n deploy_sim2real python=3.10 -y
conda activate deploy_sim2real

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Analyze a checkpoint
python get_policy_info.py --checkpoint policy_stage4_82000.pt

# 2. Export model to ONNX (if you have a .pt checkpoint)
python export_onnx.py --checkpoint policy_stage4_82000.pt --output policy.onnx

# 3. Test sim2sim in MuJoCo (uses included G1 model and meshes)
python sim2sim_mujoco.py --model policy_stage4_82000.onnx --duration 10

# 4. Run inference example
python inference_example.py --model policy_stage4_82000.onnx
```

## Directory Structure

```
deploy_sim2real/
├── assets/
│   └── g1/
│       ├── g1_sim2sim_29dof.xml    # MuJoCo model
│       └── meshes/                  # Robot mesh files (STL)
├── export_onnx.py                   # Export PyTorch → ONNX
├── get_policy_info.py               # Analyze checkpoint
├── inference_example.py             # Minimal inference example
├── sim2sim_mujoco.py                # Test policy in MuJoCo
├── g1_robot_config.py               # Joint mapping utilities
├── requirements.txt                 # Python dependencies
├── policy_stage4_82000.pt           # Example PyTorch checkpoint
└── policy_stage4_82000.onnx         # Example ONNX model
```

## Model Architecture

### Network Structure
- **Type**: MLP Actor-Critic (PPO)
- **Actor Hidden Layers**: [512, 256, 128]
- **Activation**: ELU
- **Output**: Joint position targets (delta from default pose)

### Input/Output Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| **Observation** | `(batch, 195)` | See breakdown below |
| **Action** | `(batch, 37)` | Joint position deltas |

### Observation Space Breakdown (195 dims)

| Component | Dims | Description |
|-----------|------|-------------|
| `base_lin_vel` | 3 | Base linear velocity (body frame) |
| `base_ang_vel` | 3 | Base angular velocity (body frame) |
| `projected_gravity` | 3 | Gravity vector in body frame |
| `joint_pos` | 37 | Current joint positions (relative to default) |
| `joint_vel` | 37 | Current joint velocities |
| `last_actions` | 37 | Previous action output |
| `target_joint_pos` | 29 | Target joint positions from teleop (MuJoCo order) |
| `target_keybody_pos` | 12 | Target key body positions (4 bodies × 3D) |
| **Total** | 195 | |

### Action Space (37 dims)

Actions are **joint position deltas** that get added to default positions:

```python
target_position = default_position + (action * action_scale)
```

Where `action_scale = 0.5` (configured in environment).

## Joint Mapping: MuJoCo (29) vs Isaac Lab (37)

### The Problem
- **MuJoCo model**: 29 DOFs (used for teleop input)
- **Isaac Lab model**: 37 DOFs (used for policy training)
- **Different joint names** for same joints (e.g., `left_elbow_joint` vs `left_elbow_pitch_joint`)
- **Some joints don't exist** in one system (e.g., `waist_roll`, `waist_pitch`, `wrist_pitch`, `wrist_yaw`)

### Joint Mapping Table

| MuJoCo Index | MuJoCo Name | Isaac Lab Index | Isaac Lab Name | Notes |
|--------------|-------------|-----------------|----------------|-------|
| 0 | left_hip_pitch_joint | 0 | left_hip_pitch_joint | Same |
| 1 | left_hip_roll_joint | 1 | left_hip_roll_joint | Same |
| 2 | left_hip_yaw_joint | 2 | left_hip_yaw_joint | Same |
| 3 | left_knee_joint | 3 | left_knee_joint | Same |
| 4 | left_ankle_pitch_joint | 4 | left_ankle_pitch_joint | Same |
| 5 | left_ankle_roll_joint | 5 | left_ankle_roll_joint | Same |
| 6 | right_hip_pitch_joint | 6 | right_hip_pitch_joint | Same |
| 7 | right_hip_roll_joint | 7 | right_hip_roll_joint | Same |
| 8 | right_hip_yaw_joint | 8 | right_hip_yaw_joint | Same |
| 9 | right_knee_joint | 9 | right_knee_joint | Same |
| 10 | right_ankle_pitch_joint | 10 | right_ankle_pitch_joint | Same |
| 11 | right_ankle_roll_joint | 11 | right_ankle_roll_joint | Same |
| 12 | waist_yaw_joint | 12 | torso_joint | Name differs |
| 13 | waist_roll_joint | - | (none) | **NO EQUIVALENT** |
| 14 | waist_pitch_joint | - | (none) | **NO EQUIVALENT** |
| 15 | left_shoulder_pitch_joint | 13 | left_shoulder_pitch_joint | Index differs |
| 16 | left_shoulder_roll_joint | 14 | left_shoulder_roll_joint | Index differs |
| 17 | left_shoulder_yaw_joint | 15 | left_shoulder_yaw_joint | Index differs |
| 18 | left_elbow_joint | 16 | left_elbow_pitch_joint | Name differs |
| 19 | left_wrist_roll_joint | 17 | left_elbow_roll_joint | Name differs |
| 20 | left_wrist_pitch_joint | - | (none) | **NO EQUIVALENT** |
| 21 | left_wrist_yaw_joint | - | (none) | **NO EQUIVALENT** |
| 22 | right_shoulder_pitch_joint | 27 | right_shoulder_pitch_joint | Index differs |
| 23 | right_shoulder_roll_joint | 28 | right_shoulder_roll_joint | Index differs |
| 24 | right_shoulder_yaw_joint | 29 | right_shoulder_yaw_joint | Index differs |
| 25 | right_elbow_joint | 30 | right_elbow_pitch_joint | Name differs |
| 26 | right_wrist_roll_joint | 31 | right_elbow_roll_joint | Name differs |
| 27 | right_wrist_pitch_joint | - | (none) | **NO EQUIVALENT** |
| 28 | right_wrist_yaw_joint | - | (none) | **NO EQUIVALENT** |

### Using the Mapping

```python
from g1_robot_config import G1RobotConfig

# Teleop gives you 29 DOF positions
teleop_dof_mujoco = get_teleop_targets()  # shape (29,)

# Convert to Isaac Lab order for policy observation
isaaclab_joint_names = get_robot_joint_names()  # from your robot wrapper
teleop_dof_isaaclab = G1RobotConfig.remap_mujoco_to_isaaclab_numpy(
    teleop_dof_mujoco, 
    isaaclab_joint_names
)  # shape (37,), unmapped joints are 0

# Policy outputs 37 DOF actions
action = policy.forward(observation)  # shape (37,)

# Convert back to MuJoCo order if needed for your robot
action_mujoco = G1RobotConfig.remap_isaaclab_to_mujoco_numpy(
    action,
    isaaclab_joint_names
)  # shape (29,)
```

## Forward Pass (Inference)

```python
import torch
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("policy.onnx")

def forward_pass(observation: np.ndarray) -> np.ndarray:
    """
    Run policy inference.
    
    Args:
        observation: Shape (195,) or (batch, 195)
        
    Returns:
        action: Shape (37,) or (batch, 37) - joint position deltas
    """
    if observation.ndim == 1:
        observation = observation.reshape(1, -1)
    
    outputs = session.run(None, {"observation": observation.astype(np.float32)})
    action = outputs[0]
    
    return action.squeeze()
```

## Building Observations

```python
def build_observation(
    base_lin_vel: np.ndarray,      # (3,) body frame
    base_ang_vel: np.ndarray,      # (3,) body frame
    projected_gravity: np.ndarray,  # (3,) gravity in body frame
    joint_pos: np.ndarray,          # (37,) current positions - default
    joint_vel: np.ndarray,          # (37,) current velocities
    last_action: np.ndarray,        # (37,) previous action
    target_joint_pos: np.ndarray,   # (29,) teleop targets (MuJoCo order)
    target_keybody_pos: np.ndarray, # (12,) key body targets (local frame)
) -> np.ndarray:
    """Build observation vector for policy."""
    
    obs = np.concatenate([
        base_lin_vel,       # 3
        base_ang_vel,       # 3
        projected_gravity,  # 3
        joint_pos,          # 37
        joint_vel,          # 37
        last_action,        # 37
        target_joint_pos,   # 29 (MuJoCo order - remapped internally)
        target_keybody_pos, # 12 (4 key bodies × 3D)
    ])
    
    return obs  # (195,)
```

## Key Bodies for Tracking

The policy tracks 4 key body positions:
1. `left_ankle_roll_link` (left foot)
2. `right_ankle_roll_link` (right foot)  
3. `left_elbow_link` (left elbow)
4. `right_elbow_link` (right elbow)

## Default Joint Positions (Isaac Lab)

```python
DEFAULT_JOINT_POS = {
    # Legs - slightly bent for stability
    ".*_hip_pitch_joint": 0.0,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    ".*_knee_joint": 0.4,       # Slightly bent
    ".*_ankle_pitch_joint": -0.2,
    ".*_ankle_roll_joint": 0.0,
    # Torso
    "torso_joint": 0.0,
    # Arms - natural pose
    ".*_shoulder_pitch_joint": 0.35,
    ".*_shoulder_roll_joint": 0.16,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_pitch_joint": 0.52,
    ".*_elbow_roll_joint": 0.0,
}
```

## Files in This Package

| File | Description |
|------|-------------|
| `export_onnx.py` | Export PyTorch checkpoint to ONNX |
| `sim2sim_mujoco.py` | Test policy in MuJoCo simulation |
| `inference_example.py` | Minimal inference example |
| `g1_robot_config.py` | Joint mapping utilities |
| `policy.onnx` | Exported ONNX model (after running export) |
| `policy.pt` | PyTorch checkpoint (copy latest here) |

## Training Details

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: RSL-RL + Isaac Lab
- **Training Stages**:
  1. Stage 1: Basic balance (10k iters)
  2. Stage 2: Movement (10k iters)
  3. Stage 3: Upper body tracking (40k iters)
  4. Stage 4: Robust with push forces (40k iters)
- **Total**: ~100k iterations with curriculum

## Contact

For questions about the policy or deployment, contact the TWIST2 team.

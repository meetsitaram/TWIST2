# Isaac Lab Migration Plan

## Overview

**Goal:** Migrate TWIST2's motion imitation training from Isaac Gym → Isaac Lab to support RTX 5090.

**Why:** Isaac Gym is deprecated and doesn't support RTX 5090 (Blackwell architecture). Isaac Lab is the successor with modern GPU support.

---

## GREAT NEWS: Existing Assets Found!

Your local setup already has most components ready:

### 1. Isaac Lab G1 Locomotion (Ready to use!)
```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/
├── flat_env_cfg.py      # Flat terrain G1 locomotion
├── rough_env_cfg.py     # Rough terrain G1 locomotion
└── agents/
    └── rsl_rl_ppo_cfg.py  # RSL-RL config (same as TWIST2!)
```

### 2. Unitree Sim Isaac Lab (G1 29-DOF Wholebody)
```
unitree_sim_isaaclab/
├── tasks/common_config/robot_configs.py  # G1RobotPresets with wholebody
├── tasks/common_observations/g1_29dof_state.py  # Observations
└── robots/unitree.py  # G129_CFG_WITH_*_WHOLEBODY configs
```

### 3. Isaac Lab Mimic Extension
```
IsaacLab/source/isaaclab_mimic/  # Imitation learning framework
```

---

## Simplified Migration (1-2 days instead of 5!)

Since G1 locomotion exists, we only need to:

1. **Extend G1 locomotion env** with motion tracking
2. **Port MotionLib** for motion data loading
3. **Add motion tracking rewards** (replace velocity tracking)
4. **Test with teleop motions**

---

## Simplified Implementation Plan

### Step 1: Test Existing G1 Locomotion (30 min)

```bash
cd ~/projects/g1-pick-n-place/IsaacLab
conda activate isaaclab

# Test G1 velocity tracking (should work on RTX 5090!)
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 \
    --num_envs 1024 \
    --headless
```

If this works, your RTX 5090 + Isaac Lab setup is confirmed!

### Step 2: Create Motion Imitation Env (2-4 hours)

Create a new env that extends G1 locomotion with motion tracking:

```python
# File: isaaclab_envs/g1_motion_mimic_env_cfg.py

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab.utils import configclass

@configclass  
class G1MotionMimicEnvCfg(G1FlatEnvCfg):
    """G1 Motion Imitation Environment.
    
    Extends velocity locomotion with motion tracking from pkl files.
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # Add motion file path
        self.motion_file = "motion_data_configs/teleop_dataset.yaml"
        
        # Replace velocity tracking rewards with motion tracking
        self.rewards.track_lin_vel_xy_exp = None  # Remove velocity tracking
        self.rewards.track_ang_vel_z_exp = None
        
        # Add motion tracking rewards (custom)
        # These will be added via RewardManager
```

### Step 3: Port MotionLib (1-2 hours)

The existing `pose/utils/motion_lib_pkl.py` mostly works. Just update device handling:

```python
# Copy and adapt
cp TWIST2/pose/pose/utils/motion_lib_pkl.py isaaclab_envs/motion_lib.py

# Key changes:
# - Update device handling for Isaac Lab
# - Ensure tensor shapes match new env
```

### Step 4: Add Motion Tracking Rewards (2-4 hours)

```python
# File: isaaclab_envs/motion_rewards.py

def tracking_joint_dof(env, asset_cfg, motion_lib):
    """Track target joint positions from motion."""
    target_dof = motion_lib.get_dof_pos(env.episode_time)
    current_dof = env.scene["robot"].data.joint_pos
    error = torch.sum(torch.abs(current_dof - target_dof), dim=1)
    return torch.exp(-error * 0.5)

def tracking_keybody_pos(env, asset_cfg, motion_lib, key_bodies):
    """Track key body positions (hands, feet, etc.)."""
    target_pos = motion_lib.get_keybody_pos(env.episode_time)
    current_pos = env.scene["robot"].data.body_pos_w[:, key_bodies]
    error = torch.sum(torch.norm(current_pos - target_pos, dim=-1), dim=1)
    return torch.exp(-error * 2.0)
```

### Step 5: Register and Train (30 min)

```bash
# Register the new task
# Then train!
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Motion-Mimic-G1-v0 \
    --num_envs 4096 \
    --headless
```

---

## Phase 1: Environment Setup

### 1.1 Install Isaac Lab

```bash
# Create new conda environment with Python 3.11
conda create -n isaaclab python=3.11 -y
conda activate isaaclab

# Option A: Pip install (simpler)
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics \
    isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app \
    --extra-index-url https://pypi.nvidia.com

pip install isaaclab isaaclab_tasks --extra-index-url https://pypi.nvidia.com

# Option B: From source (if you have it cloned)
cd ~/IsaacLab
./isaaclab.sh --install  # or python -m pip install -e .
```

### 1.2 Verify RTX 5090 Works

```bash
conda activate isaaclab

# Test Isaac Lab import
python -c "import omni.isaac.lab; print('Isaac Lab OK')"

# Test GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
```

---

## Phase 2: Architecture Mapping

| Isaac Gym Concept | Isaac Lab Equivalent |
|-------------------|---------------------|
| `VecEnv` subclass | `DirectRLEnv` or `ManagerBasedRLEnv` |
| YAML config | `@configclass` Python decorators |
| `gymapi` physics | `omni.isaac.core` + PhysX |
| URDF/MJCF robot | USD format (preferred) |
| `substeps` param | `dt` + `decimation` |
| `rl_games` / `rsl_rl` | `rsl_rl` / `skrl` / `rl_games` (all supported) |

---

## Phase 3: Component Porting

### 3.1 Robot Model (G1)

**Current TWIST2:**
```
TWIST2/assets/g1/g1_mocap_29dof.xml  (MuJoCo format)
```

**Isaac Lab approach:**
1. Check if G1 USD already exists in Isaac Lab
2. If not, convert URDF → USD using Isaac Sim
3. Create `ArticulationCfg` for the robot

```python
# Example ArticulationCfg
from omni.isaac.lab.assets import ArticulationCfg

G1_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="path/to/g1.usd",
        # ...
    ),
    # joint configs, actuators, etc.
)
```

### 3.2 Motion Library

**Current:** `TWIST2/pose/pose/utils/motion_lib_pkl.py`

**Migration:** Can largely keep the same code - it's just data loading. May need:
- Device adjustments (`cuda:0` handling)
- Path updates

```python
# MotionLib is standalone, just update device handling
class MotionLib:
    def __init__(self, motion_file, device="cuda:0"):
        # ... existing code works
```

### 3.3 Environment

**Current:** `TWIST2/legged_gym/legged_gym/envs/base/humanoid_mimic.py`

**Isaac Lab structure:**

```python
# isaaclab_envs/humanoid_mimic_env.py

from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from dataclasses import MISSING
from omni.isaac.lab.utils import configclass

@configclass
class HumanoidMimicEnvCfg(DirectRLEnvCfg):
    # Scene
    scene: SceneCfg = MISSING
    
    # Robot
    robot_cfg: ArticulationCfg = MISSING
    
    # Motion
    motion_file: str = MISSING
    
    # Rewards (as class attributes)
    tracking_joint_weight: float = 2.0
    tracking_keybody_weight: float = 3.0
    # ...

class HumanoidMimicEnv(DirectRLEnv):
    cfg: HumanoidMimicEnvCfg
    
    def __init__(self, cfg, render_mode=None):
        super().__init__(cfg, render_mode)
        self._motion_lib = MotionLib(cfg.motion_file, device=self.device)
        
    def _setup_scene(self):
        # Create robot articulation
        self.robot = Articulation(self.cfg.robot_cfg)
        # ...
        
    def _get_observations(self):
        # Similar to existing code
        pass
        
    def _get_rewards(self):
        # Port existing reward functions
        pass
        
    def _get_dones(self):
        # Port termination conditions
        pass
        
    def _reset_idx(self, env_ids):
        # Reset to motion frames
        pass
```

### 3.4 Reward Functions

Port these from `humanoid_mimic.py`:

```python
def _reward_tracking_joint_dof(self):
    # Same logic, updated tensor access
    joint_err = torch.sum(torch.abs(self.dof_pos - self.target_dof_pos), dim=1)
    return torch.exp(-joint_err * 0.5)

def _reward_tracking_keybody_pos(self):
    # Same logic
    keybody_err = torch.sum(torch.norm(self.keybody_pos - self.target_keybody_pos, dim=-1), dim=1)
    return torch.exp(-keybody_err * 2.0)
```

### 3.5 Training Script

```python
# scripts/train_isaaclab.py

from omni.isaac.lab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

# Import your env
from isaaclab_envs.humanoid_mimic_env import HumanoidMimicEnv, HumanoidMimicEnvCfg

# Create env
env_cfg = HumanoidMimicEnvCfg()
env = HumanoidMimicEnv(cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

# Use RSL-RL (same as TWIST2!)
from rsl_rl.runners import OnPolicyRunner
runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
runner.learn(num_learning_iterations=20000)
```

---

## Phase 4: Directory Structure

```
TWIST2/
├── isaaclab_envs/              # NEW - Isaac Lab environments
│   ├── __init__.py
│   ├── humanoid_mimic_cfg.py   # Config classes
│   ├── humanoid_mimic_env.py   # Environment
│   └── motion_lib.py           # Ported from pose/
│
├── isaaclab_assets/            # NEW - Robot assets
│   └── g1/
│       └── g1_29dof.usd        # Converted or from Isaac Lab
│
├── scripts/
│   └── train_isaaclab.py       # NEW - Training entry point
│
├── legged_gym/                 # KEEP - Original Isaac Gym code
├── pose/                       # KEEP - Motion utilities
├── rsl_rl/                     # KEEP - Works with both!
└── ...
```

---

## Phase 5: Step-by-Step Implementation

### Step 1: Check existing Isaac Lab assets
```bash
# In your IsaacLab clone
find . -name "*unitree*" -o -name "*g1*" | head -20
find . -name "*humanoid*" | head -20
find . -name "*locomotion*" | head -20
```

### Step 2: Run an existing locomotion example
```bash
cd ~/IsaacLab
python source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0
```

### Step 3: Create minimal HumanoidMimic env
- Start with existing humanoid/locomotion task
- Add MotionLib integration
- Add motion tracking rewards

### Step 4: Test with teleop motions
```bash
python scripts/train_isaaclab.py --motion_file motion_data_configs/teleop_dataset.yaml
```

---

## Key Resources

1. **Isaac Lab Docs:** https://isaac-sim.github.io/IsaacLab/
2. **Migration Guide:** https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html
3. **Locomotion Examples:** `IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/locomotion/`
4. **RSL-RL Integration:** https://isaac-sim.github.io/IsaacLab/main/source/tutorials/05_controllers/run_rsl_rl.html

---

## Quick Win: Use Existing Humanoid Task

If Isaac Lab has a humanoid locomotion task, you might only need to:

1. Add MotionLib loading
2. Replace velocity tracking rewards with motion tracking rewards
3. Use your teleop motion files

This could reduce migration to **1-2 days** instead of a full rewrite.

---

## Next Steps for Fresh Chat

1. Tell the assistant your IsaacLab path (e.g., `~/IsaacLab`)
2. Ask to explore what G1/humanoid assets already exist
3. Find the closest existing task to build from
4. Start porting MotionLib and rewards

---

## Commands to Run First

```bash
# 1. Find your Isaac Lab location
ls ~/IsaacLab 2>/dev/null || ls ~/projects/IsaacLab 2>/dev/null

# 2. Check for Unitree/G1 robots
find ~/IsaacLab -type f -name "*.usd" | xargs -I{} basename {} | sort -u | grep -i "unitree\|g1"

# 3. Check existing tasks
ls ~/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/

# 4. Check for humanoid examples
find ~/IsaacLab -type f -name "*.py" | xargs grep -l "humanoid" | head -10
```

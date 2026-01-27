# Multi-Stage Curriculum Training Plan for G1 Humanoid

## Overview

Progressive training approach to build a robust humanoid that can:
1. Stand and walk stably
2. Resist external disturbances (pushes)
3. Perform precise upper-body teleoperation for manipulation tasks

This approach is validated by recent research (ALMI, Mobile-TeleVision, H2O) and is particularly suited for the Unitree G1 robot.

## Training Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Base Locomotion                     │
│           Stand/Walk with motion tracking rewards               │
│                     (Tonight's training)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: Robustness                          │
│           Resume + Enable push disturbances                     │
│                   (Use --robust flag)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 3: Upper Body Teleop                      │
│        Resume + End-effector position/orientation tracking      │
│              (New rewards + manipulation demos)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT                                   │
│     Lower body: velocity commands (joystick/planner)            │
│     Upper body: real-time teleop tracking                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Base Locomotion (Current)

**Goal:** Robot learns to stand, maintain balance, and track basic motion patterns.

**Config:** `G1MotionMimicEnvCfg` (default)

**Rewards:**
- `tracking_joint_dof` - Track target joint positions
- `tracking_joint_vel` - Track target joint velocities
- `tracking_keybody_pos` - Track key body positions (feet, elbows, shoulders)
- `tracking_root_height` - Maintain target pelvis height
- `tracking_root_pos_xy` - Track horizontal position
- `tracking_root_orientation` - Stay upright
- Regularization: action rate, torques, joint limits, etc.

**Command:**
```bash
cd /home/stickbot/projects/g1-pick-n-place/TWIST2

python scripts/train_isaaclab.py \
    --motion_file motion_data_configs/stand_and_walk.yaml \
    --num_envs 4096 \
    --max_iterations 50000 \
    --headless
```

**Expected Outcome:**
- Robot stands stably
- Tracks motion data (leg movements, arm positions)
- Episode length increases over training
- `tracking_joint_dof` reward increases

**Checkpoint:** `logs/isaaclab/motion_mimic/model_XXXXX.pt`

---

## Stage 2: Robustness Training

**Goal:** Robot resists external disturbances (pushes) while maintaining balance.

**Config:** `G1MotionMimicEnvCfg_ROBUST` (already defined!)

**Additional Features:**
- Random push disturbances every 8-12 seconds
- Velocity range: ±0.8 m/s in X and Y

**Command:**
```bash
cd /home/stickbot/projects/g1-pick-n-place/TWIST2

python scripts/train_isaaclab.py \
    --motion_file motion_data_configs/stand_and_walk.yaml \
    --num_envs 4096 \
    --max_iterations 20000 \
    --robust \
    --resume logs/isaaclab/motion_mimic/model_STAGE1.pt
```

**Expected Outcome:**
- Robot recovers from pushes
- More stable stance
- Slightly lower tracking accuracy (tradeoff for robustness)

**Checkpoint:** `logs/isaaclab/motion_mimic/model_STAGE2.pt`

---

## Stage 3: Upper Body End-Effector Tracking

**Goal:** Precise upper body control for manipulation while lower body maintains balance.

### 3.1 New Reward Functions Needed

Add to `isaaclab_envs/motion_mdp.py`:

```python
def tracking_end_effector_pos(
    env: ManagerBasedRLEnv,
    ee_bodies: list[str],
    std: float = 0.05,
) -> torch.Tensor:
    """Reward for tracking end-effector positions (hands).
    
    Precise position tracking for manipulation tasks.
    Uses tighter std than keybody tracking for precision.
    
    Args:
        env: The environment instance.
        ee_bodies: List of end-effector body names (e.g., wrist links)
        std: Standard deviation for exponential kernel (meters).
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    target_state = get_target_state(env)
    target_keybody = target_state["keybody_pos"]  # (num_envs, num_bodies, 3)
    
    robot = env.scene["robot"]
    
    # Get end-effector body indices
    ee_ids = []
    for body_name in ee_bodies:
        try:
            found_ids = robot.find_bodies(body_name)[0]
            if found_ids:
                ee_ids.extend(found_ids)
        except ValueError:
            pass
    
    if not ee_ids:
        return torch.zeros(env.num_envs, device=env.device)
    
    current_pos = robot.data.body_pos_w[:, ee_ids, :]  # (num_envs, num_ee, 3)
    
    # Match with corresponding target positions (assuming same order)
    # For hands, these would be indices 2,3 in keybody (elbows) or need separate tracking
    # TODO: Add proper hand position targets to motion data
    
    # Compute position error
    num_compare = min(current_pos.shape[1], 2)  # Left and right hand
    target_hands = target_keybody[:, 2:4, :]  # Assuming indices 2,3 are hand-related
    
    pos_error = torch.norm(current_pos[:, :num_compare] - target_hands[:, :num_compare], dim=-1)
    mean_error = torch.mean(pos_error, dim=1)
    
    return torch.exp(-mean_error / std)


def tracking_end_effector_rot(
    env: ManagerBasedRLEnv,
    ee_bodies: list[str],
    std: float = 0.3,
) -> torch.Tensor:
    """Reward for tracking end-effector orientations (hands).
    
    Important for manipulation tasks where hand orientation matters.
    
    Args:
        env: The environment instance.
        ee_bodies: List of end-effector body names
        std: Standard deviation for exponential kernel.
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    # TODO: Add end-effector orientation targets to motion data
    # For now, return placeholder
    # Need to:
    # 1. Store hand quaternions in motion data during recording
    # 2. Compare current hand orientation vs target orientation
    
    robot = env.scene["robot"]
    
    # Get end-effector body indices
    ee_ids = []
    for body_name in ee_bodies:
        try:
            found_ids = robot.find_bodies(body_name)[0]
            if found_ids:
                ee_ids.extend(found_ids)
        except ValueError:
            pass
    
    if not ee_ids:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Placeholder: reward for keeping hands level (manipulation-ready)
    current_quat = robot.data.body_quat_w[:, ee_ids, :]  # (num_envs, num_ee, 4)
    
    # Simple check: penalize if hands are tilted too much
    # Quaternion w component close to 1 = upright
    # This is a placeholder - replace with actual target tracking
    upright_reward = torch.mean(torch.abs(current_quat[:, :, 0]), dim=1)  # w component
    
    return upright_reward
```

### 3.2 New Config for Stage 3

Add to `isaaclab_envs/g1_motion_mimic_env_cfg.py`:

```python
@configclass
class G1MotionMimicEnvCfg_TELEOP(G1MotionMimicEnvCfg_ROBUST):
    """Stage 3: Upper body teleop precision training.
    
    Builds on robust base, adds precise end-effector tracking.
    
    Usage:
        python scripts/train_isaaclab.py --teleop_precision \
            --resume logs/.../model_STAGE2.pt
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # Increase upper body tracking weights
        # End-effector position tracking (hands)
        self.rewards.tracking_ee_pos = RewTerm(
            func=motion_mdp.tracking_end_effector_pos,
            weight=5.0,  # High weight for precision
            params={
                "ee_bodies": [
                    "left_wrist_yaw_link",   # Left hand
                    "right_wrist_yaw_link",  # Right hand
                ],
                "std": 0.03,  # Tight tolerance (3cm)
            }
        )
        
        # End-effector orientation tracking
        self.rewards.tracking_ee_rot = RewTerm(
            func=motion_mdp.tracking_end_effector_rot,
            weight=3.0,
            params={
                "ee_bodies": [
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                "std": 0.2,
            }
        )
        
        # Reduce lower body motion tracking (focus on stability)
        # Keep balance but don't enforce specific leg poses
        self.rewards.tracking_joint_dof.weight = 1.0  # Reduced from 2.0
        
        # Increase keybody tracking for elbows (arm positioning)
        self.rewards.tracking_keybody_pos.weight = 4.0  # Increased from 3.0
```

### 3.3 Training Command

```bash
cd /home/stickbot/projects/g1-pick-n-place/TWIST2

python scripts/train_isaaclab.py \
    --motion_file motion_data_configs/manipulation_demos.yaml \
    --num_envs 4096 \
    --max_iterations 30000 \
    --teleop_precision \
    --resume logs/isaaclab/motion_mimic/model_STAGE2.pt
```

### 3.4 Recording Manipulation Demos

Record demos focused on upper body manipulation:

```bash
cd /home/stickbot/projects/g1-pick-n-place/TWIST2/deploy_real

# Record reaching, grasping, placing motions
python stream_ik_teleop.py --record --name reach_left_001 --duration 20
python stream_ik_teleop.py --record --name reach_right_001 --duration 20
python stream_ik_teleop.py --record --name pick_place_001 --duration 30
python stream_ik_teleop.py --record --name bimanual_001 --duration 30

# Convert to motion format
python convert_episodes_to_motion.py --all --yaml manipulation_demos
```

---

## Stage 3 Data Requirements

### Current Motion Data Format
```python
{
    'fps': 30.0,
    'root_pos': (N, 3),      # Pelvis position
    'root_rot': (N, 4),      # Pelvis quaternion
    'dof_pos': (N, 29),      # Joint angles
    'local_body_pos': (N, 38, 3),  # Body positions relative to root
}
```

### Enhanced Motion Data (for Stage 3)
```python
{
    # Existing fields...
    
    # NEW: End-effector data for precise tracking
    'left_hand_pos': (N, 3),     # Left wrist world position
    'left_hand_rot': (N, 4),     # Left wrist quaternion
    'right_hand_pos': (N, 3),    # Right wrist world position
    'right_hand_rot': (N, 4),    # Right wrist quaternion
    
    # Optional: elbow positions for arm chain
    'left_elbow_pos': (N, 3),
    'right_elbow_pos': (N, 3),
}
```

**TODO:** Update `convert_episodes_to_motion.py` to extract and save hand positions/orientations.

---

## Deployment Architecture

After training all stages, the deployment uses the trained policy with teleop overlay:

```
┌─────────────────────────────────────────────────────────────────┐
│                     REAL-TIME DEPLOYMENT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │  Joystick/  │───▶│   Velocity   │───▶│   Lower Body    │    │
│  │   Planner   │    │   Commands   │    │   (from Policy) │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│                                                  │              │
│  ┌─────────────┐    ┌──────────────┐    ┌───────▼───────┐      │
│  │   Camera    │───▶│  IK Teleop   │───▶│   Blend with  │      │
│  │   Input     │    │  Publisher   │    │   Upper Body  │      │
│  └─────────────┘    └──────────────┘    └───────┬───────┘      │
│                                                  │              │
│                                         ┌───────▼───────┐      │
│                                         │  Final Joint  │      │
│                                         │   Commands    │      │
│                                         └───────┬───────┘      │
│                                                  │              │
│                                         ┌───────▼───────┐      │
│                                         │   Real Robot  │      │
│                                         │   (G1)        │      │
│                                         └───────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Existing code supports this:**
- `play_isaaclab_teleop.py` - Policy + teleop blend
- `isaac_lab_teleop_publisher.py` - Camera → Redis

---

## Checklist for Tomorrow

### Morning: Evaluate Stage 1
```bash
# Check training progress
tail -100 training_overnight.log

# Test the trained model
python scripts/play_isaaclab_teleop.py \
    --checkpoint logs/isaaclab/motion_mimic/model_XXXXX.pt \
    --teleop none \
    --num_envs 16
```

### If Stage 1 looks good: Start Stage 2
```bash
python scripts/train_isaaclab.py \
    --motion_file motion_data_configs/stand_and_walk.yaml \
    --num_envs 4096 \
    --max_iterations 20000 \
    --robust \
    --resume logs/isaaclab/motion_mimic/model_STAGE1.pt \
    --headless
```

### Prepare Stage 3
- [ ] Add `tracking_end_effector_pos` to `motion_mdp.py`
- [ ] Add `tracking_end_effector_rot` to `motion_mdp.py`
- [ ] Add `G1MotionMimicEnvCfg_TELEOP` config
- [ ] Add `--teleop_precision` flag to `train_isaaclab.py`
- [ ] Update `convert_episodes_to_motion.py` for hand poses
- [ ] Record manipulation demos

### Test Stage 3
```bash
python scripts/train_isaaclab.py \
    --motion_file motion_data_configs/manipulation_demos.yaml \
    --num_envs 4096 \
    --max_iterations 30000 \
    --teleop_precision \
    --resume logs/isaaclab/motion_mimic/model_STAGE2.pt
```

---

## References

1. **ALMI** (NeurIPS 2025) - Adversarial Locomotion and Motion Imitation
   - Separate upper/lower body adversarial learning
   - Validated on Unitree H1

2. **Mobile-TeleVision** (ICRA 2025) - Decoupled humanoid control
   - Upper body: IK + motion retargeting
   - Lower body: RL locomotion
   - Predictive Motion Priors (PMP)

3. **H2O** (IROS 2024) - Human to Humanoid whole-body teleoperation
   - RGB camera input to whole-body control
   - Sim-to-real transfer

4. **Gait-Conditioned Multi-Phase Curriculum** (UCL)
   - Progressive stand → walk → run training
   - Validated on Unitree G1

---

## Notes

- Stage times are estimates; adjust based on convergence
- Monitor `tracking_*` rewards to gauge progress
- If robot falls too much, reduce disturbance strength in Stage 2
- Stage 3 requires good manipulation demo data - quality matters!

# Isaac Lab Reference Configs

These files are copies of the original Isaac Lab configuration files for reference.
They document the default settings used by the G1 robot in Isaac Lab.

**Source**: `/home/sitaram/projects/g1-pick-n-place/IsaacLab/source/`

## Files

| File | Original Path | Description |
|------|---------------|-------------|
| `isaaclab_unitree_robots.py` | `isaaclab_assets/robots/unitree.py` | G1_CFG, G1_MINIMAL_CFG with PD gains, default poses, actuator configs |
| `isaaclab_g1_flat_env_cfg.py` | `isaaclab_tasks/.../g1/flat_env_cfg.py` | G1FlatEnvCfg - flat terrain locomotion |
| `isaaclab_g1_rough_env_cfg.py` | `isaaclab_tasks/.../g1/rough_env_cfg.py` | G1RoughEnvCfg - rough terrain locomotion |
| `isaaclab_velocity_env_cfg.py` | `isaaclab_tasks/.../velocity_env_cfg.py` | Base velocity env config (physics dt, decimation, actions) |

## Key Values (from isaaclab_unitree_robots.py)

### G1_CFG Default Pose
```python
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.74),  # Height: 0.74m
    joint_pos={
        ".*_hip_pitch_joint": -0.20,
        ".*_knee_joint": 0.42,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_pitch_joint": 0.87,
        "left_shoulder_roll_joint": 0.16,
        "right_shoulder_roll_joint": -0.16,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_pitch_joint": 0.35,
    },
)
```

### PD Gains
| Joint Group | Stiffness | Damping | Effort Limit |
|-------------|-----------|---------|--------------|
| Legs (hip, knee, torso) | 150-200 | 5.0 | 300 |
| Feet (ankle) | 20.0 | 2.0 | 20 |
| Arms | 40.0 | 10.0 | 300 |

### Physics Settings (from velocity_env_cfg.py)
- `sim.dt = 0.005` (200Hz physics)
- `decimation = 4` (50Hz control)
- `action_scale = 0.5`

## Usage

These are READ-ONLY reference files. Do not modify them.
Your custom configs should extend the base classes and override settings as needed.

See `../g1_motion_mimic_env_cfg.py` for how TWIST2 extends these configs.


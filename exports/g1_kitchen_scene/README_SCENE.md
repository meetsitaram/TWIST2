# G1 Kitchen Scene — Portable Package

A self-contained kitchen environment for the Unitree G1 robot in NVIDIA Isaac Sim.

## What's Inside

```
g1_kitchen_scene/
├── scene.usd              ← Main scene (all paths relative)
├── load_scene.py           ← Ready-to-run Isaac Lab launch script
├── kitchen/
│   ├── kitchen_splat.usdz  ← Photorealistic Gaussian splat volume (NuRec)
│   └── kitchen_room_mesh.usd ← Collision mesh (invisible, physics only)
├── fixtures/
│   ├── fridge_articulated.usd       ← Fridge with door + drawers
│   ├── dishwasher_articulated.usd   ← Dishwasher with door + racks
│   └── microwave_counter_articulated.usd ← Microwave with door + turntable
├── configs/
│   ├── scene_params.json    ← Kitchen scale, camera, G1 spawn position
│   └── fixture_placements.json ← Fixture positions and rotations
└── README_SCENE.md          ← This file
```

## Quick Start

### Option 1: Run the included launch script

```bash
# From Isaac Lab
isaaclab -p path/to/g1_kitchen_scene/load_scene.py

# Without robot (scene only)
isaaclab -p path/to/g1_kitchen_scene/load_scene.py --no-robot
```

### Option 2: Load scene.usd in your own code

```python
from isaaclab.sim.spawners.from_files import UsdFileCfg

# Point to scene.usd — all internal paths are relative and resolve automatically
SCENE_DIR = "/path/to/g1_kitchen_scene"
scene_cfg = UsdFileCfg(usd_path=f"{SCENE_DIR}/scene.usd")
scene_cfg.func("/World/Kitchen", scene_cfg, translation=(0.0, 0.0, 0.0))
```

### Option 3: Load in Isaac Sim GUI

1. Open Isaac Sim
2. File → Open → select `scene.usd`
3. All fixtures, lighting, and physics load automatically

## Scene Details

| Property | Value |
|----------|-------|
| Kitchen size | ~10ft × 10ft (3.05m × 3.05m) |
| Scale factor | 1.47× applied to room shell |
| Floor Z (world) | -1.352m |
| Up axis | Z |
| Units | Meters |

### Fixtures (Articulated — doors/drawers work)

| Fixture | Joints | Key Interactions |
|---------|--------|-----------------|
| **Fridge** | 1 revolute + 2 prismatic | Door opens, freezer drawer slides, bottom drawer slides |
| **Dishwasher** | 1 revolute + 4 prismatic | Door opens, 2 racks slide out, 2 button slides |
| **Microwave** | 1 revolute + 3 mixed | Door opens, turntable rotates, buttons |

### Physics

- All fixtures have `kinematic=True` root bodies (anchored in place)
- Doors and drawers respond to applied forces/joint commands
- Room shell has collision mesh for robot/object interaction
- Ground plane has collision enabled

## Integration with Robot Training

### Loading in your Isaac Lab task:

```python
import os
from isaaclab.sim.spawners.from_files import UsdFileCfg

# 1. Set the scene path
KITCHEN_SCENE = os.path.expanduser("~/path/to/g1_kitchen_scene/scene.usd")

# 2. Load the kitchen as a background environment
kitchen_cfg = UsdFileCfg(usd_path=KITCHEN_SCENE)
kitchen_cfg.func("/World/Kitchen", kitchen_cfg, translation=(0.0, 0.0, 0.0))

# 3. Add your robot separately
from isaaclab_assets import G1_CFG
robot_cfg = G1_CFG.replace(prim_path="/World/G1")
robot_cfg.init_state.pos = (0.0, 0.0, -0.612)  # Kitchen floor level
robot = Articulation(robot_cfg)
```

### Interacting with fixture joints (open fridge door, etc.):

```python
from isaaclab.assets import Articulation

# Load a fixture as a separate articulation for joint control
from isaaclab.sim.spawners.from_files import UsdFileCfg

# The fixtures are already in the scene, but to control their joints
# you need to wrap them as Articulation objects:
fridge = Articulation(
    cfg=ArticulationCfg(
        prim_path="/World/Kitchen/World/Fixtures/fridge",
        # ... configure as needed
    )
)
```

## Requirements

- **NVIDIA Isaac Sim 5.0+** (for NuRec Gaussian splat rendering)
- **Isaac Lab** (for `sim_utils`, `Articulation`, etc.)
- **isaaclab_assets** (for G1 robot model — optional if bringing your own robot)

## Generated From

This scene was exported from the [g1-kitchen-sim](https://github.com/your-org/g1-kitchen-sim) pipeline.
Source commit: `2547423`
Export date: 2026-02-26 22:39

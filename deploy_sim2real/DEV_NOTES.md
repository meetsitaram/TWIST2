# Kitchen Teleop - Dev Notes

Development notes and known issues for `play_kitchen_teleop.py` and related teleop pipeline.

## Issues & Fixes (Feb 2026)

### 1. Robot Right Hand Flapping During Teleop

**Symptom**: Robot's right hand flaps erratically even when no teleop data is actively being published.

**Root Causes**:
- **Stale Redis data**: Publisher wrote to Redis without a TTL, leaving stale joint targets. The consumer kept reading and applying this old data, causing the robot to fight the policy.
- **NaN handling regression**: A change to fill missing landmarks with a fixed `default_skeleton` (T-pose proportions) instead of `last_valid_skeleton` created inconsistent skeletons and jerky IK results.

**Fixes** (`isaac_lab_teleop_publisher.py`):
- Added `px=300` (300ms TTL) on Redis `set()` so published data auto-expires if the publisher stops.
- Reverted NaN fill logic to use `last_valid_skeleton` for non-required landmarks.

**Fixes** (`play_kitchen_teleop.py`):
- Added `redis_client.delete()` on startup to flush stale data from previous sessions.
- `get_target_dof()` returns `None` when Redis key is absent (expired), triggering `clear_teleop_targets()` instead of holding stale positions.

### 2. Frequent Resets / Death Spiral

**Symptom**: Robot resets multiple times per second, even when not falling.

**Root Causes**:
- **Episode timeout**: Default `episode_length_s = 10.0` caused resets every 10 seconds.
- **`base_contact` termination**: Triggered on any torso/pelvis contact (force > 0.5N), which fires constantly with full collision meshes near furniture.
- **Reset death spiral**: Calling `respawn_robot()` on natural dones would reset teleop targets, but next frame stale Redis data gets re-applied, robot becomes unstable, triggering another reset.

**Fixes**:
- Set `episode_length_s = 600.0` (10 min) for interactive demo.
- Disabled `base_contact` termination (`env_cfg.terminations.base_contact = None`).
- Added height-based fall detection (`root_height_below_minimum = 0.65m`) — catches falls early before mesh entanglement.
- Added drift-based respawn (0.3m threshold) to catch the robot sliding away.
- Natural dones now respawn at the *current* active preset, not the default.

### 3. Physics Instability with Full Collision Meshes

**Symptom**: Robot collapses into physically impossible suspended poses when interacting with furniture (fridge door). Limbs get entangled with scene geometry.

**Root Cause**: Switching to `G1_CFG` (full collision meshes) with `enabled_self_collisions = True` causes complex mesh interpenetration with scene objects.

**Current Config** (`g1_motion_mimic_env_cfg.py`):
```python
self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
```
Full collision meshes for furniture interaction, but self-collisions disabled to avoid entanglement. This is a trade-off: the robot can push/touch objects but its own limbs won't collide with each other.

### 4. Keyboard Spawn Keys Not Working

**Symptom**: Number keys 1-5 had no effect.

**Root Cause**: `carb.input.KeyboardInput` names number keys as `KEY_1`, `KEY_2`, etc. — not bare `"1"`, `"2"`.

**Fix**: Changed `event.input.name == str(n)` to `event.input.name == f"KEY_{n}"`.

### 5. Kitchen Fixture/Object Reset Not Working

**Symptom**: Fridge doors, pots, and mugs stayed in disturbed positions after switching spawn locations.

**Root Causes**:
- Kitchen fixtures are loaded from USD, NOT registered as Isaac Lab scene assets — `scene.rigid_objects` and `scene.articulations` are empty (except robot).
- USD xformOp writes don't sync with the PhysX simulation state.
- PhysX `RigidBodyView.set_transforms()` requires a GPU `indices` tensor (device `cuda:0`), not CPU.

**Fix**: `KitchenObjectResetter` class traverses the USD stage at startup, finds all prims with `UsdPhysics.RigidBodyAPI` under `/Kitchen/`, creates PhysX `rigid_body_view` for each via `physics_sim_view`, captures initial transforms, and restores them on location switch using the same PhysX tensor API that powers `robot.write_root_pose_to_sim()`.

## Architecture Notes

### Spawn Presets
Defined in `SPAWN_PRESETS` dict — `(name, x, y, yaw_degrees)` in kitchen world frame. Camera angles in `CAMERA_PRESETS` are captured from the Isaac Sim viewport using the **C** key at runtime.

### Camera Capture Workflow
1. Spawn at a location (press 1-4)
2. Orbit the viewport camera to the desired angle
3. Press **C** — prints exact `eye` and `target` coordinates to terminal
4. Copy into `CAMERA_PRESETS` dict

### Teleop Data Flow
```
Cameras -> MediaPipe -> Triangulation -> IK Retargeting -> Redis (px=300ms TTL)
                                                               |
Isaac Lab env <- Policy + Teleop blend <- RedisTeleopClient <--+
```

When publisher stops: Redis key expires -> `get_target_dof()` returns `None` -> `clear_teleop_targets()` -> policy-only control.

### Scene Object Reset
Kitchen fixtures (fridge, dishwasher, microwave doors, drawers) and loose objects (pots, mugs, kettle) are tracked via `KitchenObjectResetter`:
- Discovers all rigid bodies under `/Kitchen/` at startup (excluding RoomShell)
- Saves initial PhysX transforms
- Restores on location switch (number keys or auto-loop), NOT on fall/drift resets

The fixtures' root joints fail with "cannot create a joint between static bodies" — they are NOT PhysX articulations. Door bodies are individual rigid bodies constrained by joints. Reset works by restoring their rigid body transforms directly.

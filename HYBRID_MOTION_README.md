# Hybrid Motion: PKL Body + Camera Arms

This feature combines pre-recorded PKL motion files (for stable locomotion) with real-time camera-based arm tracking. This allows you to control the robot's arms with your own movements while the legs and torso follow recorded motions.

## Overview

- **PKL Motion**: Controls legs, torso, and waist (stable, pre-recorded locomotion)
- **Camera Tracking**: Controls arms in real-time (your movements via multi-camera setup)

This approach bypasses the need for full-body camera tracking and leg joint mapping.

## Requirements

1. **gmr conda environment** (Python 3.10 with MediaPipe)
2. **Multi-camera setup** with calibration (see `CALIBRATION_QUICKSTART.md`)
3. **Redis server** running
4. **sim2sim.sh** running in another terminal (the low-level RL controller)

## Quick Start

### 1. Start the Low-Level Controller (Terminal 1)

```bash
conda activate twist2
cd TWIST2
bash sim2sim.sh
```

### 2. Run Hybrid Motion (Terminal 2)

```bash
conda activate gmr
cd TWIST2
bash run_hybrid_motion.sh 001
```

The number `001` selects the motion file. Available motions are in `assets/example_motions/`:
- `001` - Walk motion 1
- `002` - Walk motion 2
- `003` - Walk motion 3
- etc.

### 3. Follow the Prompts

1. **Phase 1: Camera Validation** - Stand visible to cameras, wait for arm detection confirmation
2. **Phase 2: Hybrid Motion** - Robot follows PKL motion while your arms control robot arms

## Command Options

```bash
bash run_hybrid_motion.sh <motion_id> [options]

# Examples:
bash run_hybrid_motion.sh 001              # Motion 001 with default settings
bash run_hybrid_motion.sh 003 --no-loop    # Motion 003, play once then stop
bash run_hybrid_motion.sh 002 --no-display # No camera preview windows
```

Full options (in `multicam_with_motion.py`):
- `--motion-file`: Path to PKL file (auto-set by shell script)
- `--camera-ids`: Comma-separated camera IDs (default: 4,6,2)
- `--calibration`: Path to calibration.toml
- `--startup-delay`: Countdown before motion starts (default: 5 seconds)
- `--target-fps`: Motion playback rate (default: 50 Hz)
- `--no-loop`: Don't loop the motion
- `--no-display`: Disable camera preview windows
- `--device`: cpu or cuda (default: cuda with auto-fallback)

## Preview Motions

To preview motions before running hybrid mode:

```bash
conda activate gmr
bash try_motion.sh 001  # Preview motion 001
```

## Troubleshooting

### "No valid pose detected"
- Make sure you're standing visible to all cameras
- Check that `multicam_to_twist2.py` works first (camera-only mode)
- Ensure cameras are calibrated (`calibration/calibration.toml` exists)

### Robot collapses immediately
- Make sure `sim2sim.sh` is running first
- Wait for Phase 1 camera validation to complete

### Arms not following
- Check the status display for "Arm Detection Rate"
- Should show > 0% when tracking is working
- "Arm Source: CAMERA" means camera arms are active

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Hybrid Motion Script                 │
│                  (multicam_with_motion.py)              │
│                                                         │
│  ┌─────────────────┐    ┌─────────────────────────────┐│
│  │   PKL Motion    │    │   Multi-Camera Tracking     ││
│  │   (MotionLib)   │    │   (MultiCamPoseStreamer)    ││
│  │                 │    │                             ││
│  │ Legs: [6-17]    │    │ Left Arm: [21-27]           ││
│  │ Waist: [18-20]  │    │ Right Arm: [28-34]          ││
│  │ Root: [0-5]     │    │                             ││
│  └────────┬────────┘    └────────────┬────────────────┘│
│           │                          │                  │
│           └──────────┬───────────────┘                  │
│                      ▼                                  │
│            ┌─────────────────┐                          │
│            │  Merge Motion   │                          │
│            │  (35-dim obs)   │                          │
│            └────────┬────────┘                          │
│                     │                                   │
└─────────────────────┼───────────────────────────────────┘
                      │ Redis
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Low-Level RL Controller                    │
│                   (sim2sim.sh)                          │
│                                                         │
│  Reads mimic_obs from Redis, executes RL policy         │
│  to achieve target pose in MuJoCo simulation            │
└─────────────────────────────────────────────────────────┘
```

## Related Files

- `deploy_real/multicam_with_motion.py` - Main hybrid motion script
- `run_hybrid_motion.sh` - Convenience launcher
- `deploy_real/multicam_to_twist2.py` - Camera-only mode (no PKL)
- `deploy_real/multicam_pose_streamer.py` - Multi-camera capture & triangulation
- `deploy_real/mediapipe_to_g1_direct.py` - MediaPipe to G1 joint conversion
- `assets/example_motions/*.pkl` - Pre-recorded motion files

## See Also

- `TWIST2_ARCHITECTURE.md` - Overall system architecture
- `CALIBRATION_QUICKSTART.md` - Camera calibration guide
- `MULTI_CAMERA_QUICKSTART.md` - Multi-camera setup guide

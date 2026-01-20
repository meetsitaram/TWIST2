# Multi-Camera Teleoperation

Control the G1 robot using real-time pose tracking from multiple cameras. Your full body movements are captured via MediaPipe and triangulated into 3D, then mapped to robot joint angles.

## Overview

This is a camera-based alternative to VR teleoperation. It uses:
- **Multiple USB cameras** (3+ recommended) for 3D pose estimation
- **MediaPipe** for 2D pose detection per camera
- **Triangulation** (DLT) to reconstruct 3D skeleton
- **Direct joint mapping** to convert skeleton to G1 robot joints

## Requirements

1. **gmr conda environment** (Python 3.10 with MediaPipe)
2. **Multiple USB cameras** (tested with 3 cameras)
3. **Camera calibration** completed (see `CALIBRATION_QUICKSTART.md`)
4. **Redis server** running
5. **sim2sim.sh** running (the low-level RL controller)

## Quick Start

### 1. Start Redis (if not already running)

```bash
redis-server --daemonize yes
```

### 2. Start the Low-Level Controller (Terminal 1)

```bash
conda activate twist2
cd TWIST2
bash sim2sim.sh
```

### 3. Run Multi-Camera Teleop (Terminal 2)

```bash
conda activate gmr
cd TWIST2/deploy_real
python multicam_to_twist2.py
```

### 4. Stand in View of Cameras

- Position yourself so all cameras can see you
- Stand with feet shoulder-width apart, arms at sides
- The robot will mirror your movements

## Command Options

```bash
python multicam_to_twist2.py [options]

# Examples:
python multicam_to_twist2.py                          # Default settings
python multicam_to_twist2.py --camera-ids 0,1,2       # Use cameras 0, 1, 2
python multicam_to_twist2.py --no-display             # No camera preview
python multicam_to_twist2.py --startup-delay 10       # 10 second countdown
```

Full options:
- `--camera-ids`: Comma-separated camera IDs (default: 4,6,2)
- `--calibration`: Path to calibration.toml (default: ../calibration/calibration.toml)
- `--width`: Camera resolution width (default: 1280)
- `--height`: Camera resolution height (default: 720)
- `--redis-host`: Redis server host (default: localhost)
- `--startup-delay`: Countdown before tracking starts (default: 5)
- `--no-display`: Disable camera preview windows

## Status Display

While running, you'll see a status table:

```
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric       ┃ Value       ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Frame Count  │ 150         │
│ FPS          │ 29.9 Hz     │
│ Tracking     │ ✓ Active    │
│ Redis        │ ✓ Connected │
│ Cameras      │ 3 active    │
│ Reproj Error │ 181.0px     │
│ Frame Sync   │ 8ms         │
└──────────────┴─────────────┘
```

- **Tracking: ✓ Active** - Pose detected and streaming
- **Reproj Error** - 3D reconstruction quality (lower is better, <200px is OK)
- **Frame Sync** - Time difference between camera frames

## Finding Your Camera IDs

```bash
conda activate gmr
cd TWIST2/deploy_real
python list_cameras.py
```

This will show available cameras and their IDs.

## Troubleshooting

### "No skeleton detected"
- Make sure you're visible to at least 2 cameras
- Check lighting conditions
- Verify cameras are working: `python list_cameras.py`

### High reprojection error (>300px)
- Recalibrate cameras: `bash calibrate.sh`
- Make sure cameras haven't moved since calibration
- Check that ChArUco board was detected in all cameras during calibration

### Robot not moving
- Verify `sim2sim.sh` is running
- Check Redis connection: `redis-cli ping` should return PONG
- Look for error messages in the sim2sim terminal

### Low FPS (<20 Hz)
- Close other GPU-intensive applications
- Try lower resolution: `--width 640 --height 480`
- Check CPU usage

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Multi-Camera Teleop Pipeline               │
│                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │Camera 1 │  │Camera 2 │  │Camera 3 │  USB Cameras    │
│  └────┬────┘  └────┬────┘  └────┬────┘                 │
│       │            │            │                       │
│       ▼            ▼            ▼                       │
│  ┌─────────────────────────────────────┐               │
│  │     MediaPipe Pose Detection        │               │
│  │     (2D landmarks per camera)       │               │
│  └──────────────────┬──────────────────┘               │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────┐               │
│  │     DLT Triangulation               │               │
│  │     (3D skeleton from 2D views)     │               │
│  └──────────────────┬──────────────────┘               │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────┐               │
│  │     MediaPipe → G1 Joint Mapping    │               │
│  │     (mediapipe_to_g1_direct.py)     │               │
│  └──────────────────┬──────────────────┘               │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────┐               │
│  │     mimic_obs (35-dim vector)       │               │
│  │     [root, legs, waist, arms]       │               │
│  └──────────────────┬──────────────────┘               │
│                     │                                   │
└─────────────────────┼───────────────────────────────────┘
                      │ Redis
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Low-Level RL Controller                    │
│                   (sim2sim.sh)                          │
└─────────────────────────────────────────────────────────┘
```

## Related Files

- `deploy_real/multicam_to_twist2.py` - Main teleop script
- `deploy_real/multicam_pose_streamer.py` - Camera capture & triangulation
- `deploy_real/mediapipe_to_g1_direct.py` - Skeleton to joint conversion
- `calibration/calibration.toml` - Camera calibration data

## Comparison with Hybrid Motion

| Feature | multicam_to_twist2.py | multicam_with_motion.py |
|---------|----------------------|------------------------|
| Arm control | Camera tracking | Camera tracking |
| Leg control | Camera tracking | PKL motion file |
| Body control | Camera tracking | PKL motion file |
| Use case | Full-body teleop | Stable locomotion + arm control |
| Difficulty | Requires full visibility | Only arms need visibility |

For stable walking while controlling arms, use `run_hybrid_motion.sh` instead.

## See Also

- `HYBRID_MOTION_README.md` - Hybrid motion (PKL body + camera arms)
- `CALIBRATION_QUICKSTART.md` - Camera calibration guide
- `TWIST2_ARCHITECTURE.md` - Overall system architecture

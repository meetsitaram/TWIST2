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
2. **Multiple USB cameras** (3+ cameras recommended, minimum 2)
3. **Camera calibration** completed (see Setup section below)
4. **Redis server** running
5. **sim2sim.sh** running (the low-level RL controller)

## Setup (First Time)

If this is your first time or you've moved your cameras, complete these steps:

### Step 1: Identify and Configure Cameras

Run the interactive camera setup tool to identify which camera is left/center/right:

```bash
conda activate gmr
cd deploy_real/utils
python setup_cameras.py
```

This will:
1. Detect all connected cameras
2. Capture a test image from each camera
3. Open the folder so you can view the images
4. Ask you to assign positions (left, center, right)
5. Generate `camera_config.yaml` with the correct camera IDs

### Step 2: Print ChArUco Calibration Board

You need a ChArUco board for camera calibration:

```bash
cd deploy_real/utils
python generate_4page_large_charuco.py
```

- Print the 4 pages and assemble them into a large board
- Mount it on a rigid surface (cardboard/foam board)
- Measure the actual printed square size and update `calibration/camera_config.yaml` if needed

### Step 3: Calibrate Cameras

With the ChArUco board ready, run the calibration:

```bash
conda activate gmr
cd deploy_real
python calibrate_cameras.py
```

During calibration:
- Wave the ChArUco board slowly in front of all cameras
- Cover different depths (close and far)
- Rotate the board at various angles
- The script will auto-stop after collecting enough diverse frames (~60 per camera)
- Results saved to `calibration/calibration.toml`

### Step 4: Test Calibration

Verify everything works:

```bash
cd deploy_real
python test_camera_only.py
```

Look for:
- ✅ "Arms detected" messages
- ✅ Reprojection error < 200px
- ✅ Detection rate > 80%

## Quick Start

### 1. Start Redis (if not already running)

```bash
redis-server --daemonize yes
```

### 2. Start the Low-Level Controller (Terminal 1)

```bash
conda activate twist2
bash sim2sim.sh
```

### 3. Run Multi-Camera Teleop (Terminal 2)

```bash
conda activate gmr
cd deploy_real
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
cd deploy_real
python list_cameras.py
```

For full setup (identify positions + generate config), see **Step 1** in the Setup section above.

## Troubleshooting

### "No skeleton detected"
- Make sure you're visible to at least 2 cameras
- Check lighting conditions
- Verify cameras are working: `python list_cameras.py`
- Test camera-only mode: `python deploy_real/test_camera_only.py`

### High reprojection error (>300px)
- **Camera moved?** If cameras have been moved, you must recalibrate:
  ```bash
  # Step 1: Update camera config (if camera IDs changed)
  python deploy_real/utils/setup_cameras.py
  
  # Step 2: Recalibrate
  python deploy_real/calibrate_cameras.py
  ```
- Make sure cameras haven't moved since calibration
- Check that ChArUco board was detected in all cameras during calibration
- Verify calibration file exists: `calibration/calibration.toml`

### Robot not moving
- Verify `sim2sim.sh` is running
- Check Redis connection: `redis-cli ping` should return PONG
- Look for error messages in the sim2sim terminal

### Low FPS (<20 Hz)
- Close other GPU-intensive applications
- Try lower resolution: `--width 640 --height 480`
- Check CPU usage

### Camera IDs changed after reconnecting
USB camera IDs can change when cameras are reconnected. Run:
```bash
python deploy_real/utils/setup_cameras.py
```
Then recalibrate if camera positions changed.

## Motion Recording

Record your movements to PKL files for training or playback.

### Record Motion

```bash
conda activate gmr
cd deploy_real

# Default: 10s countdown, 30s recording
python record_motion.py

# Custom duration
python record_motion.py --duration 60    # 60 second recording

# Custom countdown
python record_motion.py --countdown 15   # 15 seconds to get ready

# Specify output file
python record_motion.py --output ../recordings/walking_01.pkl
```

**Recording workflow:**
1. Run the script
2. 10 second countdown - walk to your position
3. 30 second recording - perform your motion
4. Auto-save and exit

**Options:**
- `--duration`: Recording length in seconds (default: 30)
- `--countdown`: Countdown before recording (default: 10)
- `--output`: Output file path (default: auto-generated with timestamp)
- `--trim-start`: Seconds to trim from start (default: 1.0)
- `--trim-end`: Seconds to trim from end (default: 1.0)
- `--no-viz`: Disable MuJoCo visualization

### Replay Motion

```bash
cd deploy_real

# Play a recording
python replay_motion.py --file ../recordings/motion_20260121_231731.pkl

# Loop playback
python replay_motion.py --file ../recordings/motion_20260121_231731.pkl --loop

# Slow motion (0.5x speed)
python replay_motion.py --file ../recordings/motion_20260121_231731.pkl --speed 0.5

# Just print info (no visualization)
python replay_motion.py --file ../recordings/motion_20260121_231731.pkl --no-viz
```

**Options:**
- `--file`, `-f`: Path to PKL motion file (required)
- `--speed`, `-s`: Playback speed (default: 1.0)
- `--loop`, `-l`: Loop playback
- `--no-viz`: Print file info only, no visualization

### Recordings Location

Recordings are saved to `TWIST2/recordings/` with auto-generated timestamps:
```
recordings/
  motion_20260121_231731.pkl
  motion_20260121_232045.pkl
  ...
```

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

### Main Scripts
- `deploy_real/multicam_to_twist2.py` - Main teleop script
- `deploy_real/multicam_pose_streamer.py` - Camera capture & triangulation
- `deploy_real/mediapipe_to_g1_direct.py` - Skeleton to joint conversion

### Recording Scripts
- `deploy_real/record_motion.py` - Record motion to PKL files
- `deploy_real/replay_motion.py` - Replay recorded motions

### Setup & Calibration
- `deploy_real/utils/setup_cameras.py` - Interactive camera setup tool
- `deploy_real/utils/capture_camera_views.py` - Quick camera snapshot tool
- `deploy_real/calibrate_cameras.py` - Camera calibration script
- `deploy_real/utils/generate_4page_large_charuco.py` - Generate calibration board
- `deploy_real/list_cameras.py` - List available camera IDs
- `deploy_real/test_camera_only.py` - Test camera tracking

### Configuration Files
- `calibration/camera_config.yaml` - Camera IDs and settings
- `calibration/calibration.toml` - Camera calibration data (intrinsic/extrinsic)

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

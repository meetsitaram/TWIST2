":# Multi-Camera Calibration Guide

This guide covers the camera calibration pipeline for FreeMoCap integration with TWIST2.

## Prerequisites

```bash
conda activate gmr
cd TWIST2
```

## 1. Setup Cameras

### Detect and Configure Cameras
Interactive tool to detect cameras and generate config:

```bash
python deploy_real/utils/setup_cameras.py
```

This will:
- Detect all connected cameras
- Let you assign positions (left, center, right)
- Generate `calibration/camera_config.yaml`

### Capture Test Views
Capture a single frame from each camera to verify setup:

```bash
python deploy_real/utils/capture_camera_views.py
```

Images are saved to `calibration/camera_views/`.

## 2. Generate Charuco Board

Generate a printable 4-page Charuco board for calibration:

```bash
python deploy_real/utils/generate_4page_large_charuco.py
```

This creates 4 PNG files in `calibration/` that you print on a4 and tape together.

**Board options:**
- 4x6 at 70mm (biggest squares, 15 corners) - recommended
- 6x8 at 55mm or 60mm (more corners, smaller squares)

## 3. Configure Cameras

Edit `calibration/camera_config.yaml` to match your setup:

```yaml
cameras:
  ids: [4, 6, 2]  # Your camera device IDs
  resolution: [1280, 720]

charuco:
  squares_x: 4          # Must match your printed board
  squares_y: 6
  square_size_mm: 60    # Measure your actual printed squares!
  marker_size_mm: 45    # ~75% of square size
```

## 4. Run Calibration

### Standard Mode
```bash
python deploy_real/calibrate_cameras.py --recapture
```

### Sync Mode (Recommended)
Better for stereo calibration - captures when board is stable:

```bash
python deploy_real/calibrate_cameras.py --recapture --sync --target-frames 70
```

**Tips for good calibration:**
- Move board to all parts of each camera's view
- Vary distance: some poses close, some far
- Tilt board at 30-45° angles (not just flat facing camera)
- In sync mode, hold board still briefly at each position

### Re-run with Saved Frames
```bash
python deploy_real/calibrate_cameras.py
# Answer 'y' to use saved data
```

## 5. Test Triangulation

Verify calibration quality with live triangulation:

```bash
python deploy_real/test_triangulation.py
```

**Quality indicators:**
- **GOOD** (green): < 10px error (~2cm at 1m distance)
- **FAIR** (yellow): 10-20px error (~2-4cm at 1m)
- **POOR** (red): > 20px error

## Output Files

After calibration, you'll have:
- `calibration/calibration.toml` - Camera intrinsics and extrinsics
- `calibration/captured_corners.pkl` - Saved detection data (for re-running)

## Troubleshooting

### "initIntrinsicParams2D" error
The board wasn't seen from enough varied angles. The script will:
1. Try selecting a diverse subset of frames
2. Try with an initial camera matrix guess
3. Show recommendations if all attempts fail

**Fix:** Tilt board more aggressively, vary distance more.

### High stereo reprojection error
Frames weren't synchronized well between cameras.

**Fix:** Use `--sync` mode and hold board very still at each position.

### Camera not detected
Check device IDs with:
```bash
ls /dev/video*
```
Update `camera_config.yaml` with correct IDs.

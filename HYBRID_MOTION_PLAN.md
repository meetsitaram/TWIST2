# Hybrid Motion Plan: PKL + Camera Arms

This document captures the implementation plan for combining pre-recorded PKL motion (legs/torso) with real-time camera arm tracking.

---

## Goal

Overlay real-time arm movements from multi-camera tracking onto pre-recorded locomotion from PKL files. This allows:
- Stable walking/standing from tested PKL motions
- Real-time arm control without needing to tune leg joint mappings
- Faster iteration on arm tracking quality

---

## Architecture

```
┌──────────────────┐          ┌──────────────────┐
│   PKL Motion     │          │  Camera Tracking │
│  (MotionLib)     │          │ (MultiCamPose)   │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         │ Root + Legs + Waist         │ Arms only
         │ (indices 0-20)              │ (indices 21-34)
         │                             │
         └──────────┬──────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  Merged mimic   │
          │  (35 dims)      │
          └────────┬────────┘
                   │
                   ▼
             ┌───────────┐
             │   Redis   │
             └─────┬─────┘
                   │
                   ▼
          ┌─────────────────┐
          │  sim2sim.sh     │
          │  (RL Controller)│
          └─────────────────┘
```

---

## mimic_obs Index Mapping (35 dims)

| Index | Content | Source |
|-------|---------|--------|
| 0-5 | Root state (vel_x, vel_y, z, roll, pitch, yaw_vel) | PKL |
| 6-11 | Left leg (6 joints) | PKL |
| 12-17 | Right leg (6 joints) | PKL |
| 18-20 | Waist (3 joints) | PKL |
| **21-27** | **Left arm** (shoulder×3, elbow, wrist×3) | **Camera** |
| **28-34** | **Right arm** (shoulder×3, elbow, wrist×3) | **Camera** |

---

## Implementation: `multicam_with_motion.py`

### Location
`TWIST2/deploy_real/multicam_with_motion.py`

### Key Components

```python
#!/usr/bin/env python3
"""
Hybrid motion: PKL locomotion + real-time camera arm tracking.

Usage:
    # Terminal 1: Start robot simulation
    conda activate twist2
    bash sim2sim.sh
    
    # Terminal 2: Start hybrid streaming
    conda activate gmr
    python multicam_with_motion.py --motion_file ../assets/example_motions/0807_yanjie_walk_001.pkl
"""

import argparse
import time
import numpy as np
import json
import redis
import torch

from multicam_pose_streamer import MultiCamPoseStreamer
from pose.utils.motion_lib_pkl import MotionLib
from data_utils.params import DEFAULT_MIMIC_OBS

# Arm indices in mimic_obs
LEFT_ARM_START = 21
LEFT_ARM_END = 28   # exclusive
RIGHT_ARM_START = 28
RIGHT_ARM_END = 35  # exclusive

class HybridMotionStreamer:
    def __init__(self, motion_file, camera_ids, calibration_file, device="cpu"):
        # Load PKL motion
        self.motion_lib = MotionLib(motion_file, device=device)
        self.device = device
        
        # Initialize camera streamer
        self.camera_streamer = MultiCamPoseStreamer(
            camera_ids=camera_ids,
            calibration_file=calibration_file,
            enable_display=True
        )
        
        # Motion playback state
        self.t_step = 0
        self.control_dt = 0.02  # 50 Hz
        motion_id = torch.tensor([0], device=device, dtype=torch.long)
        self.motion_length = self.motion_lib.get_motion_length(motion_id)
        self.num_steps = int(self.motion_length / self.control_dt)
        
        # Config
        self.loop_motion = True
        self.blend_arms = False  # Future: smooth blend between PKL and camera
        
    def get_pkl_mimic_obs(self):
        """Get current frame from PKL motion."""
        motion_times = torch.tensor([self.t_step * self.control_dt], device=self.device)
        motion_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        
        # Handle looping
        motion_length = self.motion_lib.get_motion_length(motion_ids)
        if motion_times >= motion_length:
            if self.loop_motion:
                self.t_step = 0
                motion_times = torch.tensor([0.0], device=self.device)
            else:
                return None
        
        # Get motion frame (simplified - full impl uses build_mimic_obs)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _, _, _, _ = \
            self.motion_lib.calc_motion_frame(motion_ids, motion_times)
        
        # Build mimic_obs (would need full implementation from server_motion_lib.py)
        # This is a placeholder - actual impl needs euler conversion, velocity transform, etc.
        mimic_obs = build_mimic_obs_from_motion(...)  # TODO
        
        return mimic_obs
    
    def get_camera_arm_angles(self):
        """Get arm joint angles from camera tracking."""
        mimic_obs = self.camera_streamer.get_mimic_obs()
        if mimic_obs is None:
            return None, None
        
        left_arm = mimic_obs[LEFT_ARM_START:LEFT_ARM_END]
        right_arm = mimic_obs[RIGHT_ARM_START:RIGHT_ARM_END]
        return left_arm, right_arm
    
    def merge_motion(self, pkl_obs, camera_left_arm, camera_right_arm):
        """Merge PKL body with camera arms."""
        merged = pkl_obs.copy()
        
        if camera_left_arm is not None:
            merged[LEFT_ARM_START:LEFT_ARM_END] = camera_left_arm
        if camera_right_arm is not None:
            merged[RIGHT_ARM_START:RIGHT_ARM_END] = camera_right_arm
        
        return merged
    
    def run(self, redis_host="localhost"):
        redis_client = redis.Redis(host=redis_host, port=6379)
        self.camera_streamer.start()
        
        try:
            while self.camera_streamer.is_running:
                t0 = time.time()
                
                # Get PKL motion for body
                pkl_obs = self.get_pkl_mimic_obs()
                if pkl_obs is None:
                    break
                
                # Get camera arms
                left_arm, right_arm = self.get_camera_arm_angles()
                
                # Merge (fallback to PKL arms if camera fails)
                merged_obs = self.merge_motion(pkl_obs, left_arm, right_arm)
                
                # Send to Redis
                redis_client.set(
                    "action_body_unitree_g1_with_hands",
                    json.dumps(merged_obs.tolist())
                )
                
                # Advance PKL playback
                self.t_step += 1
                
                # Maintain timing
                elapsed = time.time() - t0
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
                    
        finally:
            self.camera_streamer.stop()
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PKL playback | Loop continuously | Keep robot moving while testing arms |
| Camera fallback | Use PKL arm values | Graceful degradation on tracking loss |
| Arm indices | Hard-coded 21-34 | Match mimic_obs format exactly |
| Blending | None initially | Add smooth blending later if jittery |

---

## Configuration Options (Future)

```yaml
# hybrid_config.yaml
motion:
  file: assets/example_motions/0807_yanjie_walk_001.pkl
  loop: true
  speed: 1.0

override:
  left_arm: true   # Use camera for left arm
  right_arm: true  # Use camera for right arm
  waist: false     # Keep PKL waist
  
blending:
  enabled: false
  factor: 0.3      # 0 = full PKL, 1 = full camera
```

---

## Usage (Once Implemented)

```bash
# Terminal 1: Start robot simulation
cd ~/projects/g1-pick-n-place/TWIST2
conda activate twist2
bash sim2sim.sh

# Terminal 2: Start hybrid motion
cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
conda activate gmr
python multicam_with_motion.py \
    --motion_file ../assets/example_motions/0807_yanjie_walk_001.pkl \
    --camera-ids 4,6,2
```

---

## Dependencies

- `multicam_pose_streamer.py` — camera tracking (already working)
- `mediapipe_to_g1_direct.py` — skeleton → joint angles (already working)
- `motion_lib_pkl.py` — PKL motion loading
- `build_mimic_obs()` from `server_motion_lib.py` — motion → mimic_obs conversion

---

## Next Steps (After CUDA Fixed)

1. [ ] Implement `multicam_with_motion.py` based on this plan
2. [ ] Test with walking motion + arm overlay
3. [ ] Add blending option if transitions are jerky
4. [ ] Add motion selection UI/config
5. [ ] Document usage in FREEMOCAP_MULTICAM_INTEGRATION_PLAN.md

---

## Related Files

- `FREEMOCAP_MULTICAM_INTEGRATION_PLAN.md` — overall project status
- `TWIST2_ARCHITECTURE.md` — system architecture
- `server_motion_lib.py` — reference for PKL playback
- `multicam_to_twist2.py` — reference for camera streaming

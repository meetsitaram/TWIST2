#!/usr/bin/env python3
"""
Live IK Teleoperation - Stream human pose to robot using End-Effector IK.

Real-time teleoperation using:
- Multi-camera 3D skeleton capture (MediaPipe + triangulation)
- End-effector IK retargeting (hands and feet)
- MuJoCo passive viewer (kinematics only, no physics)

Usage:
    cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
    conda activate gmr
    python stream_ik_teleop.py
    
Press 'Q' or Ctrl+C to quit.
"""

import argparse
import time
import numpy as np
import os
import sys
import mujoco
import toml
from pathlib import Path

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from deploy_real.multicam_pose_streamer import MultiCamPoseStreamer
from deploy_real.end_effector_ik_retarget import EndEffectorIKRetargeter, ROBOT_MODEL_PATH

# Default calibration file
DEFAULT_CALIBRATION = Path(project_root) / "calibration" / "calibration.toml"


class IKTeleopStreamer:
    """Live IK teleoperation with MuJoCo visualization."""
    
    def __init__(
        self,
        camera_ids: list,
        calibration_file: str,
        target_fps: int = 30,
        verbose: bool = False,
    ):
        self.target_fps = target_fps
        self.verbose = verbose
        self.running = False
        
        # Initialize camera streamer
        print(f"[IK Teleop] Initializing cameras: {camera_ids}")
        self.camera_streamer = MultiCamPoseStreamer(
            camera_ids=camera_ids,
            calibration_file=calibration_file,
            target_fps=target_fps,
            enable_display=True,  # Show camera feed
        )
        
        # Initialize IK retargeter
        print(f"[IK Teleop] Initializing IK retargeter...")
        self.retargeter = EndEffectorIKRetargeter(
            verbose=False,
            max_iterations=30,  # Fewer iterations for real-time
        )
        
        # Initialize MuJoCo model for visualization
        print(f"[IK Teleop] Loading robot model: {ROBOT_MODEL_PATH}")
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        
        # Set initial pose
        self._set_default_pose()
        
        # Stats
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
    def _set_default_pose(self):
        """Set robot to default standing pose."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.75  # Base height
        self.data.qpos[3] = 1.0   # Quaternion w (upright)
        mujoco.mj_forward(self.model, self.data)
        
    def _apply_joint_angles(self, result: dict):
        """Apply IK result to MuJoCo data."""
        if not result.get('valid', False):
            return
            
        qpos = result.get('qpos')
        if qpos is not None:
            # Copy joint angles (index 7+)
            self.data.qpos[7:] = qpos[7:]
            
            # Copy Z height (for crouching/standing) but keep X, Y fixed
            # qpos[0:2] = X, Y position - keep fixed
            # qpos[2] = Z height - copy from IK for crouching
            # qpos[3:7] = base orientation quaternion - keep fixed
            self.data.qpos[2] = qpos[2]
            
            mujoco.mj_forward(self.model, self.data)
            
    def _wait_for_start(self, viewer):
        """Wait for ENTER key press in terminal, then countdown before starting."""
        
        print("\n" + "="*60)
        print("IK TELEOPERATION - KINEMATICS MODE")
        print("="*60)
        print("Controls:")
        print("  - Press ENTER in terminal to start (10s countdown)")
        print("  - Press Ctrl+C to quit")
        print("  - Close MuJoCo viewer to quit")
        print("="*60 + "\n")
        
        input("[IK Teleop] Press ENTER to start countdown...")
        
        if not viewer.is_running():
            return False
        
        # Countdown
        countdown_seconds = 10
        print(f"\n[IK Teleop] Starting in {countdown_seconds} seconds - GET INTO POSITION!")
        
        for remaining in range(countdown_seconds, 0, -1):
            print(f"  {remaining}...")
            time.sleep(1)
            viewer.sync()
            
            if not viewer.is_running():
                return False
        
        print("\n[IK Teleop] GO!\n")
        return True
    
    def run(self):
        """Main loop: stream poses and visualize."""
        from mujoco.viewer import launch_passive
        
        # Start camera streaming (has its own display window)
        self.camera_streamer.start()
        self.running = True
        
        # Launch MuJoCo passive viewer
        viewer = launch_passive(
            self.model, 
            self.data, 
            show_left_ui=False, 
            show_right_ui=True
        )
        
        # Wait for enter and countdown
        if not self._wait_for_start(viewer):
            self.camera_streamer.stop()
            viewer.close()
            return
        
        try:
            while viewer.is_running() and self.running:
                loop_start = time.time()
                
                # Get latest 3D skeleton from cameras
                skeleton_3d, reproj_error = self.camera_streamer.get_3d_skeleton()
                
                if skeleton_3d is not None:
                    # Run IK retargeting
                    # fixed_base=False allows pelvis height to adjust for crouching/standing
                    result = self.retargeter.retarget(
                        skeleton_3d, 
                        reset_to_default=False,  # Keep previous pose for smoothness
                        fixed_base=False,  # Allow dynamic pelvis height
                    )
                    
                    if result.get('valid', False):
                        self._apply_joint_angles(result)
                        
                        if self.verbose:
                            print(f"[IK] error={result['error']:.4f}, "
                                  f"iters={result['iterations']}")
                
                # Update viewer
                viewer.sync()
                
                # Update FPS counter
                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (now - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = now
                    print(f"[IK Teleop] FPS: {self.fps:.1f}")
                
                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / self.target_fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n[IK Teleop] Interrupted by user")
        finally:
            self.running = False
            self.camera_streamer.stop()
            viewer.close()
            print("[IK Teleop] Stopped")


def main():
    parser = argparse.ArgumentParser(description="Live IK Teleoperation")
    parser.add_argument("--cameras", "-c", type=int, nargs="+", default=None,
                       help="Camera IDs (default: read from calibration file)")
    parser.add_argument("--calibration", type=str, 
                       default=str(DEFAULT_CALIBRATION),
                       help="Path to calibration file")
    parser.add_argument("--fps", type=int, default=30,
                       help="Target FPS (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    args = parser.parse_args()
    
    # Check calibration file exists
    if not os.path.exists(args.calibration):
        print(f"Error: Calibration file not found: {args.calibration}")
        print("Run calibration first or specify --calibration path")
        return
    
    # Read camera IDs from calibration file if not specified
    camera_ids = args.cameras
    if camera_ids is None:
        calib_data = toml.load(args.calibration)
        camera_ids = calib_data.get('metadata', {}).get('cameras', [0, 1, 2])
        print(f"[IK Teleop] Using cameras from calibration: {camera_ids}")
    
    streamer = IKTeleopStreamer(
        camera_ids=camera_ids,
        calibration_file=args.calibration,
        target_fps=args.fps,
        verbose=args.verbose,
    )
    
    streamer.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Motion Recording Script - Capture human motion to PKL format for training.

This script captures full-body motion from multi-camera setup and saves to PKL format
compatible with TWIST2's MotionLib. Uses kinematic-only MuJoCo visualization (no physics,
no falling) so you can record any motion.

Features:
- Full locomotion capture (walking, moving around)
- Kinematic visualization (see robot pose without physics simulation)
- Saves to PKL format for RL training
- Press 'R' to start/stop recording
- Press 'Q' or Ctrl+C to quit

Usage:
    cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
    conda activate gmr
    python record_motion.py --output ../recordings/my_motion.pkl
    
Note: Lower body joint mappings are not yet calibrated - expect some inaccuracies.
"""

import argparse
import time
import numpy as np
import pickle
import os
import sys
import yaml
import mujoco
from datetime import datetime

from rich import print
from rich.console import Console

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from deploy_real.multicam_pose_streamer import MultiCamPoseStreamer
from deploy_real.mediapipe_to_g1_direct import MediaPipeToG1Direct

console = Console()


class MotionRecorder:
    """Records human motion from cameras to PKL format with kinematic visualization."""
    
    def __init__(
        self,
        camera_ids: list,
        calibration_file: str,
        output_file: str,
        target_fps: int = 30,
        enable_visualization: bool = True,
        human_height: float = 1.7,
        countdown_seconds: int = 10,
        record_duration: float = 30.0,
        trim_start: float = 0.0,
        trim_end: float = 0.0
    ):
        self.output_file = output_file
        self.target_fps = target_fps
        self.enable_visualization = enable_visualization
        self.human_height = human_height
        self.countdown_seconds = countdown_seconds
        self.record_duration = record_duration
        self.trim_start = trim_start  # Seconds to trim from start
        self.trim_end = trim_end      # Seconds to trim from end
        
        # Recording state
        self.recorded_frames = []
        self.frame_count = 0
        
        # Initialize camera streamer
        print(f"[cyan]Initializing cameras: {camera_ids}[/cyan]")
        self.camera_streamer = MultiCamPoseStreamer(
            camera_ids=camera_ids,
            calibration_file=calibration_file,
            resolution=(1280, 720),
            enable_display=True  # Show camera views with skeleton overlay
        )
        print(f"[green]✓[/green] Camera streamer initialized")
        
        # Initialize skeleton to joint converter
        self.joint_converter = MediaPipeToG1Direct(
            robot_height=0.8  # Robot's pelvis height
        )
        # Override arms_only to False for full body recording
        self.joint_converter.arms_only = False
        print(f"[green]✓[/green] Joint converter initialized (full body mode, human height: {human_height}m)")
        
        # Initialize MuJoCo for kinematic visualization (no physics)
        if enable_visualization:
            self._init_mujoco()
        
        # Initial root position tracking
        self.initial_pelvis_pos = None
        
    def _init_mujoco(self):
        """Initialize MuJoCo model for kinematic-only visualization."""
        xml_file = os.path.join(project_root, "assets/g1/g1_mocap_29dof.xml")
        
        if not os.path.exists(xml_file):
            print(f"[yellow]Warning: MuJoCo model not found at {xml_file}[/yellow]")
            print("[yellow]Visualization disabled[/yellow]")
            self.enable_visualization = False
            return
        
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        
        # Launch passive viewer (kinematic only - no physics simulation)
        from mujoco.viewer import launch_passive
        self.viewer = launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -15
        
        # Default standing pose
        self.default_qpos = np.array([
            0, 0, 0.793,  # base position
            1, 0, 0, 0,   # base quaternion (w, x, y, z)
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
            0.0, 0.0, 0.0,  # torso (3)
            0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,  # left arm (7)
            0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,  # right arm (7)
        ])
        self.data.qpos[:] = self.default_qpos
        mujoco.mj_forward(self.model, self.data)
        
        print(f"[green]✓[/green] MuJoCo kinematic viewer initialized")
    
    def _skeleton_to_pkl_frame(self, skeleton: np.ndarray) -> dict:
        """
        Convert 3D skeleton to PKL frame format.
        
        Returns:
            dict with 'root_pos', 'root_rot', 'dof_pos' for this frame
        """
        # Key landmarks (MediaPipe indices)
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        
        left_hip = skeleton[LEFT_HIP]
        right_hip = skeleton[RIGHT_HIP]
        left_shoulder = skeleton[LEFT_SHOULDER]
        right_shoulder = skeleton[RIGHT_SHOULDER]
        
        # --- Root position (pelvis center) ---
        pelvis = (left_hip + right_hip) / 2.0
        
        # Set initial position as origin (for relative motion)
        if self.initial_pelvis_pos is None:
            self.initial_pelvis_pos = pelvis.copy()
            self.initial_pelvis_pos[2] = 0  # Keep Z absolute
        
        # Root position relative to start, but keep absolute height
        root_pos = pelvis - self.initial_pelvis_pos
        root_pos[2] = pelvis[2]  # Absolute height
        
        # Scale to robot dimensions (human ~1.7m, robot ~0.8m pelvis height when standing)
        scale = 0.793 / self.human_height  # Robot pelvis height / human height
        root_pos = root_pos * scale
        root_pos[2] = max(0.4, min(1.0, root_pos[2] * scale + 0.793))  # Clamp height
        
        # --- Root orientation (quaternion) ---
        # Build rotation matrix from torso frame
        x_axis = right_shoulder - left_shoulder  # Left to right
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
        
        hip_center = (left_hip + right_hip) / 2.0
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        y_axis = shoulder_center - hip_center  # Up
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
        
        z_axis = np.cross(x_axis, y_axis)  # Forward
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
        
        # Re-orthogonalize
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
        
        # Rotation matrix to quaternion
        R = np.column_stack([x_axis, y_axis, z_axis])
        root_rot = self._rotation_matrix_to_quaternion(R)  # Returns xyzw format
        
        # --- Joint angles (29 DOF) ---
        mimic_obs = self.joint_converter.skeleton_to_mimic_obs(skeleton)
        if mimic_obs is None:
            return None
        
        dof_pos = mimic_obs[6:]  # Skip root state, get 29 joint angles
        
        return {
            'root_pos': root_pos.astype(np.float32),
            'root_rot': root_rot.astype(np.float32),  # xyzw format
            'dof_pos': dof_pos.astype(np.float32)
        }
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion (xyzw format)."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        # Normalize
        q = np.array([x, y, z, w])
        q = q / (np.linalg.norm(q) + 1e-8)
        return q
    
    def _update_visualization(self, frame_data: dict):
        """Update MuJoCo visualization with current pose (kinematic only)."""
        if not self.enable_visualization or not hasattr(self, 'viewer'):
            return
        
        if not self.viewer.is_running():
            return
        
        # Set root position
        self.data.qpos[:3] = frame_data['root_pos']
        
        # Set root rotation (convert xyzw to wxyz for MuJoCo)
        xyzw = frame_data['root_rot']
        self.data.qpos[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]  # wxyz
        
        # Set joint positions
        self.data.qpos[7:] = frame_data['dof_pos']
        
        # Forward kinematics only (no physics!)
        mujoco.mj_forward(self.model, self.data)
        
        # Update camera to follow robot
        try:
            pelvis_id = self.model.body("pelvis").id
            pelvis_pos = self.data.xpos[pelvis_id]
            self.viewer.cam.lookat = pelvis_pos
        except:
            pass
        
        self.viewer.sync()
    
    def save_recording(self):
        """Save recorded frames to PKL file with auto-trim."""
        if not self.recorded_frames:
            print("[yellow]No frames to save![/yellow]")
            return False
        
        # Calculate trim amounts in frames
        trim_start_frames = int(self.trim_start * self.target_fps)
        trim_end_frames = int(self.trim_end * self.target_fps)
        
        total_frames = len(self.recorded_frames)
        
        # Apply trimming
        if trim_start_frames + trim_end_frames >= total_frames:
            print(f"[red]Error: Trim amount ({self.trim_start + self.trim_end}s) exceeds recording length![/red]")
            return False
        
        start_idx = trim_start_frames
        end_idx = total_frames - trim_end_frames if trim_end_frames > 0 else total_frames
        
        trimmed_frames = self.recorded_frames[start_idx:end_idx]
        
        if not trimmed_frames:
            print("[yellow]No frames left after trimming![/yellow]")
            return False
        
        # Report trimming
        if trim_start_frames > 0 or trim_end_frames > 0:
            print(f"[dim]Trimmed: {trim_start_frames} frames from start, {trim_end_frames} from end[/dim]")
            print(f"[dim]Original: {total_frames} frames → Trimmed: {len(trimmed_frames)} frames[/dim]")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Stack all frames
        root_pos = np.array([f['root_pos'] for f in trimmed_frames])
        root_rot = np.array([f['root_rot'] for f in trimmed_frames])
        dof_pos = np.array([f['dof_pos'] for f in trimmed_frames])
        
        motion_data = {
            'fps': self.target_fps,
            'root_pos': root_pos,
            'root_rot': root_rot,
            'dof_pos': dof_pos,
            'local_body_pos': None,
            'link_body_list': None,
        }
        
        with open(self.output_file, 'wb') as f:
            pickle.dump(motion_data, f)
        
        duration = len(trimmed_frames) / self.target_fps
        print(f"[green]✓ Saved {len(trimmed_frames)} frames ({duration:.1f}s) to {self.output_file}[/green]")
        return True
    
    def run(self):
        """Main recording loop - simple timed recording."""
        print()
        print("[bold green]╔═══════════════════════════════════════════════════╗[/bold green]")
        print("[bold green]║       Motion Recording - Camera to PKL            ║[/bold green]")
        print("[bold green]╚═══════════════════════════════════════════════════╝[/bold green]")
        print()
        print("[yellow]Note: Lower body joint mappings are not yet calibrated.[/yellow]")
        print("[yellow]      Expect some inaccuracies in leg movements.[/yellow]")
        print()
        print(f"[dim]Output: {self.output_file}[/dim]")
        print(f"[dim]Target FPS: {self.target_fps}[/dim]")
        print(f"[dim]Countdown: {self.countdown_seconds}s[/dim]")
        print(f"[dim]Recording duration: {self.record_duration}s[/dim]")
        if self.trim_start > 0 or self.trim_end > 0:
            print(f"[dim]Auto-trim: {self.trim_start}s from start, {self.trim_end}s from end[/dim]")
        print()
        
        # Start camera streamer
        self.camera_streamer.start()
        
        # Give cameras time to warm up
        print("[dim]Waiting for cameras to warm up...[/dim]")
        time.sleep(1.0)
        
        target_dt = 1.0 / self.target_fps
        
        try:
            # === PHASE 1: COUNTDOWN ===
            print()
            print("[bold yellow]╔════════════════════════════════════════════════╗[/bold yellow]")
            print("[bold yellow]║     GET INTO POSITION - COUNTDOWN STARTING     ║[/bold yellow]")
            print("[bold yellow]╚════════════════════════════════════════════════╝[/bold yellow]")
            print()
            
            for i in range(self.countdown_seconds, 0, -1):
                print(f"[bold yellow]  ⏱  {i} seconds...[/bold yellow]")
                time.sleep(1.0)
            
            # === PHASE 2: RECORDING ===
            print()
            print("[bold green]╔════════════════════════════════════════════════╗[/bold green]")
            print("[bold green]║          🔴 RECORDING - GO!                    ║[/bold green]")
            print("[bold green]╚════════════════════════════════════════════════╝[/bold green]")
            print()
            
            self.recorded_frames = []
            self.frame_count = 0
            self.initial_pelvis_pos = None
            
            recording_start_time = time.time()
            last_status_time = time.time()
            success_count = 0
            total_count = 0
            
            while True:
                t_start = time.time()
                
                # Check if recording time is up
                elapsed_recording = time.time() - recording_start_time
                if elapsed_recording >= self.record_duration:
                    print()
                    print("[bold yellow]⏹ Recording complete![/bold yellow]")
                    break
                
                # Check if MuJoCo viewer was closed
                if self.enable_visualization and hasattr(self, 'viewer'):
                    if not self.viewer.is_running():
                        print("\n[yellow]Viewer closed - stopping early[/yellow]")
                        break
                
                # Get 3D skeleton from cameras
                skeleton, reproj_error = self.camera_streamer.get_3d_skeleton()
                total_count += 1
                
                if skeleton is not None:
                    success_count += 1
                    
                    # Convert to PKL frame format
                    frame_data = self._skeleton_to_pkl_frame(skeleton)
                    
                    if frame_data is not None:
                        # Update visualization
                        self._update_visualization(frame_data)
                        
                        # Record frame
                        self.recorded_frames.append(frame_data)
                        self.frame_count += 1
                
                # Print status every second
                if time.time() - last_status_time > 1.0:
                    rate = (success_count / total_count * 100) if total_count > 0 else 0
                    reproj_str = f"{reproj_error:.1f}px" if reproj_error else "N/A"
                    remaining = self.record_duration - elapsed_recording
                    
                    print(f"\r[red]🔴 REC[/red] {elapsed_recording:.0f}s / {self.record_duration:.0f}s | Track: {rate:.0f}% | Frames: {self.frame_count}     ", end="")
                    
                    success_count = 0
                    total_count = 0
                    last_status_time = time.time()
                
                # Maintain frame rate
                elapsed = time.time() - t_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
            
            # === PHASE 3: SAVE ===
            print()
            self.save_recording()
        
        except KeyboardInterrupt:
            print("\n[yellow]Interrupted by user[/yellow]")
            if self.recorded_frames:
                self.save_recording()
        
        finally:
            # Cleanup
            self.camera_streamer.stop()
            if self.enable_visualization and hasattr(self, 'viewer'):
                self.viewer.close()
            print("[green]Done![/green]")


def main():
    parser = argparse.ArgumentParser(
        description='Record human motion from cameras to PKL format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: 10s countdown, 30s recording
    python record_motion.py
    
    # Record to specific file
    python record_motion.py --output ../recordings/walking_01.pkl
    
    # Shorter recording (15 seconds)
    python record_motion.py --duration 15
    
    # Longer countdown (15 seconds to get into position)
    python record_motion.py --countdown 15
    
    # Without MuJoCo visualization
    python record_motion.py --no-viz
"""
    )
    
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output PKL file path (default: auto-generated with timestamp)')
    parser.add_argument('--camera-ids', type=str, default=None,
                       help='Comma-separated camera IDs (default: read from camera_config.yaml)')
    parser.add_argument('--calibration', type=str, default=None,
                       help='Path to calibration.toml')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target recording FPS (default: 30)')
    parser.add_argument('--human-height', type=float, default=1.7,
                       help='Your height in meters (default: 1.7)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable MuJoCo visualization')
    parser.add_argument('--countdown', type=int, default=10,
                       help='Countdown seconds before recording starts (default: 10)')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Recording duration in seconds (default: 30)')
    parser.add_argument('--trim-start', type=float, default=1.0,
                       help='Seconds to auto-trim from start of recording (default: 1.0)')
    parser.add_argument('--trim-end', type=float, default=1.0,
                       help='Seconds to auto-trim from end of recording (default: 1.0)')
    
    args = parser.parse_args()
    
    # Load camera IDs from config if not provided
    if args.camera_ids is None:
        config_path = os.path.join(project_root, 'calibration/camera_config.yaml')
        if not os.path.exists(config_path):
            print(f"[red]✗ Error: Camera config not found at {config_path}[/red]")
            print("[yellow]Run 'python deploy_real/utils/setup_cameras.py' first![/yellow]")
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        camera_ids = config.get('camera_ids', [])
        print(f"[dim]Loaded camera IDs from config: {camera_ids}[/dim]")
    else:
        camera_ids = [int(x) for x in args.camera_ids.split(',')]
    
    # Set calibration file path
    if args.calibration is None:
        calibration_file = os.path.join(project_root, 'calibration/calibration.toml')
    else:
        calibration_file = args.calibration
    
    if not os.path.exists(calibration_file):
        print(f"[red]✗ Error: Calibration file not found at {calibration_file}[/red]")
        print("[yellow]Run 'python deploy_real/calibrate_cameras.py' first![/yellow]")
        return
    
    # Set output file path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recordings_dir = os.path.join(project_root, 'recordings')
        os.makedirs(recordings_dir, exist_ok=True)
        output_file = os.path.join(recordings_dir, f'motion_{timestamp}.pkl')
    else:
        output_file = args.output
        if not output_file.endswith('.pkl'):
            output_file += '.pkl'
    
    # Create recorder and run
    recorder = MotionRecorder(
        camera_ids=camera_ids,
        calibration_file=calibration_file,
        output_file=output_file,
        target_fps=args.fps,
        enable_visualization=not args.no_viz,
        human_height=args.human_height,
        countdown_seconds=args.countdown,
        record_duration=args.duration,
        trim_start=args.trim_start,
        trim_end=args.trim_end
    )
    
    recorder.run()


if __name__ == '__main__':
    main()

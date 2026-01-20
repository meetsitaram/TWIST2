#!/usr/bin/env python3
"""
Hybrid Motion: PKL locomotion + real-time camera arm tracking.

This script combines pre-recorded motion from PKL files (for stable lower body)
with real-time arm tracking from multi-camera pose estimation.

Usage:
    # Terminal 1: Start robot simulation
    cd ~/projects/g1-pick-n-place/TWIST2
    conda activate twist2
    bash sim2sim.sh
    
    # Terminal 2: Start hybrid motion streaming
    cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
    conda activate gmr
    python multicam_with_motion.py --motion_file ../assets/example_motions/0807_yanjie_walk_001.pkl
"""

import argparse
import time
import numpy as np
import json
import redis
import torch
import os
import sys

from rich import print
from rich.console import Console
from rich.table import Table
from rich.live import Live

# Add pose module to path
script_dir = os.path.dirname(os.path.abspath(__file__))
pose_path = os.path.join(os.path.dirname(script_dir), "pose")
if pose_path not in sys.path:
    sys.path.insert(0, pose_path)

from pose.utils.motion_lib_pkl import MotionLib
from multicam_pose_streamer import MultiCamPoseStreamer
from data_utils.params import DEFAULT_MIMIC_OBS

console = Console()


def create_status_table(frame_count, fps, tracking_success, redis_connected, 
                        num_cameras, reproj_error, time_spread_ms, motion_progress, arm_rate,
                        arms_warmed_up=False, using_camera_arms=False):
    """Create a nice status table for display"""
    table = Table(title="Hybrid Motion: PKL Body + Camera Arms")
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Frame Count", str(frame_count))
    table.add_row("FPS", f"{fps:.1f} Hz")
    table.add_row("Motion Progress", f"{motion_progress:.1f}%")
    
    # Arm status with warmup info
    if using_camera_arms:
        arm_str = "[green]✓ Using YOUR arms[/green]"
    elif arms_warmed_up:
        arm_str = "[yellow]⚠ Tracking lost - using PKL arms[/yellow]"
    else:
        arm_str = "[yellow]⏳ Warming up... (using PKL arms)[/yellow]"
    table.add_row("Arm Source", arm_str)
    
    table.add_row("Arm Detection Rate", f"{arm_rate:.0f}%")
    table.add_row("Redis", "✓ Connected" if redis_connected else "✗ Disconnected")
    table.add_row("Cameras", f"{num_cameras} active")
    
    # Reprojection error with color
    if reproj_error < 10:
        error_str = f"[green]{reproj_error:.1f}px[/green]"
    elif reproj_error < 30:
        error_str = f"[yellow]{reproj_error:.1f}px[/yellow]"
    else:
        error_str = f"[red]{reproj_error:.1f}px[/red]"
    table.add_row("Reproj Error", error_str)
    
    # Sync status
    if time_spread_ms < 30:
        sync_str = f"[green]{time_spread_ms:.0f}ms[/green]"
    elif time_spread_ms < 50:
        sync_str = f"[yellow]{time_spread_ms:.0f}ms[/yellow]"
    else:
        sync_str = f"[red]{time_spread_ms:.0f}ms[/red]"
    table.add_row("Frame Sync", sync_str)
    
    return table


# =============================================================================
# Inline torch utilities (to avoid scipy import from rot_utils)
# =============================================================================

def euler_from_quaternion_torch(quat_angle, scalar_first=True):
    """Convert quaternion to euler angles (roll, pitch, yaw) in radians."""
    if scalar_first:
        quat_angle = quat_angle[..., [1, 2, 3, 0]]
    else:
        quat_angle = quat_angle[..., [0, 1, 2, 3]]
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

def quat_rotate_inverse_torch(q, v, scalar_first=True):
    """Rotate vector v by inverse of quaternion q."""
    if scalar_first:
        q = q[..., [1, 2, 3, 0]]
    else:
        q = q[..., [0, 1, 2, 3]]
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

# =============================================================================
# mimic_obs indices
# =============================================================================

# mimic_obs format (35 dims):
# [0-5]: root state (vel_x, vel_y, z, roll, pitch, yaw_vel)
# [6-11]: left leg (6 joints)
# [12-17]: right leg (6 joints)  
# [18-20]: waist (3 joints)
# [21-27]: left arm (7 joints: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
# [28-34]: right arm (7 joints)

LEFT_ARM_START = 21
LEFT_ARM_END = 28   # exclusive
RIGHT_ARM_START = 28
RIGHT_ARM_END = 35  # exclusive

# =============================================================================
# Motion building (adapted from server_motion_lib.py)
# =============================================================================

def build_mimic_obs_from_motion(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    device: str = "cpu"
) -> np.ndarray:
    """
    Build mimic_obs from motion library at given time step.
    Returns numpy array of shape (35,).
    """
    device_torch = torch.device(device)
    
    # Single step (current frame only)
    tar_motion_steps = torch.tensor([1], device=device_torch, dtype=torch.int)
    
    # Build times
    motion_times = torch.tensor([t_step * control_dt], device=device_torch).unsqueeze(-1)
    obs_motion_times = tar_motion_steps * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()
    
    # Single motion in the .pkl
    motion_ids = torch.zeros(len(tar_motion_steps), dtype=torch.int, device=device_torch)
    
    # Retrieve motion frames
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, _, _ = \
        motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

    # Convert to euler (roll, pitch, yaw)
    roll, pitch, yaw = euler_from_quaternion_torch(root_rot, scalar_first=False)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)

    # Transform velocities to root frame
    root_vel_local = quat_rotate_inverse_torch(root_rot, root_vel, scalar_first=False).reshape(1, -1, 3)
    root_ang_vel_local = quat_rotate_inverse_torch(root_rot, root_ang_vel, scalar_first=False).reshape(1, -1, 3)

    root_pos = root_pos.reshape(1, -1, 3)
    dof_pos = dof_pos.reshape(1, -1, dof_pos.shape[-1])
    
    # Build mimic_obs: root_vel_xy + root_pos_z + roll_pitch + yaw_ang_vel + dof_pos
    mimic_obs_buf = torch.cat((
        root_vel_local[..., :2],      # 2 dims (xy velocity)
        root_pos[..., 2:3],           # 1 dim (z position)
        roll, pitch,                   # 2 dims (roll/pitch orientation)
        root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
        dof_pos,                       # 29 dims (joint angles)
    ), dim=-1)[:, :]
    
    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
    return mimic_obs_buf.detach().cpu().numpy().squeeze()

# =============================================================================
# Hybrid Motion Streamer
# =============================================================================

class HybridMotionStreamer:
    """
    Combines PKL motion playback with real-time camera arm tracking.
    """
    
    def __init__(
        self,
        motion_file: str,
        camera_ids: list,
        calibration_file: str,
        device: str = "cpu",
        loop_motion: bool = True,
        enable_display: bool = True
    ):
        self.device = device
        self.loop_motion = loop_motion
        self.control_dt = 0.02  # 50 Hz
        
        # Initialize camera streamer FIRST (before PyTorch/CUDA to avoid GPU conflicts)
        print(f"[cyan]Initializing cameras: {camera_ids}[/cyan]")
        self.camera_streamer = MultiCamPoseStreamer(
            camera_ids=camera_ids,
            calibration_file=calibration_file,
            resolution=(1280, 720),  # Match multicam_to_twist2.py default
            enable_display=enable_display
        )
        print(f"[green]✓[/green] Camera streamer initialized")
        
        # NOW do CUDA test (after EGL/MediaPipe init)
        if device == "cuda":
            try:
                test_tensor = torch.randn(10, device="cuda")
                del test_tensor
                print(f"[green]✓[/green] Using CUDA for motion")
            except Exception as e:
                print(f"[yellow]CUDA failed ({e}), using CPU[/yellow]")
                device = "cpu"
        if device == "cpu":
            print(f"[dim]Using CPU for motion loading[/dim]")
        self.device = device  # Update in case fallback occurred
        
        # Load PKL motion (after camera init to avoid CUDA/EGL conflicts)
        print(f"[cyan]Loading motion: {motion_file}[/cyan]")
        self.motion_lib = MotionLib(motion_file, device=self.device)
        
        motion_id = torch.tensor([0], device=self.device, dtype=torch.long)
        self.motion_length = float(self.motion_lib.get_motion_length(motion_id))
        self.num_steps = int(self.motion_length / self.control_dt)
        print(f"[green]✓[/green] Motion loaded: {self.motion_length:.2f}s, {self.num_steps} steps")
        
        # State
        self.t_step = 0
        self.is_running = False
        self.last_failure_reason = None
        self.last_reproj_error = 0.0
        self.last_time_spread = 0.0
        
        # Stats
        self.frame_count = 0
        self.arm_success_count = 0
        
    def get_pkl_mimic_obs(self) -> np.ndarray:
        """Get current frame from PKL motion."""
        # Handle looping
        if self.t_step >= self.num_steps:
            if self.loop_motion:
                self.t_step = 0
                print("[dim]Motion looped[/dim]")
            else:
                return None
        
        mimic_obs = build_mimic_obs_from_motion(
            self.motion_lib,
            self.t_step,
            self.control_dt,
            self.device
        )
        
        return mimic_obs
    
    def get_camera_arms(self) -> tuple:
        """
        Get arm joint angles from camera tracking.
        Returns (left_arm, right_arm) or (None, None) if tracking failed.
        
        Matches multicam_to_twist2.py approach: just call get_mimic_obs() directly.
        """
        # Get mimic_obs directly (same as multicam_to_twist2.py)
        mimic_obs = self.camera_streamer.get_mimic_obs()
        
        # Get quality metrics (for display)
        skeleton_3d, reproj_error = self.camera_streamer.get_3d_skeleton()
        self.last_reproj_error = reproj_error if reproj_error is not None else float('nan')
        self.last_time_spread = getattr(self.camera_streamer, '_last_time_spread', 0) * 1000
        
        if mimic_obs is None:
            self.last_failure_reason = "No valid pose detected"
            return None, None
        
        self.last_failure_reason = None  # Success
        left_arm = mimic_obs[LEFT_ARM_START:LEFT_ARM_END].copy()
        right_arm = mimic_obs[RIGHT_ARM_START:RIGHT_ARM_END].copy()
        return left_arm, right_arm
    
    def merge_motion(
        self,
        pkl_obs: np.ndarray,
        camera_left_arm: np.ndarray,
        camera_right_arm: np.ndarray
    ) -> np.ndarray:
        """
        Merge PKL body motion with camera arm tracking.
        Uses PKL for indices 0-20, camera for indices 21-34.
        Falls back to PKL arms if camera tracking fails.
        """
        merged = pkl_obs.copy()
        
        if camera_left_arm is not None:
            merged[LEFT_ARM_START:LEFT_ARM_END] = camera_left_arm
        
        if camera_right_arm is not None:
            merged[RIGHT_ARM_START:RIGHT_ARM_END] = camera_right_arm
        
        return merged
    
    def run(self, redis_host: str = "localhost", target_fps: int = 50, startup_delay: int = 5):
        """Main loop: merge PKL motion with camera arms and stream to Redis."""
        
        # Connect to Redis
        try:
            redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=False)
            redis_client.ping()
            print(f"[green]✓[/green] Connected to Redis at {redis_host}:6379")
        except Exception as e:
            print(f"[red]✗ Failed to connect to Redis: {e}[/red]")
            return
        
        # Start camera streamer
        self.camera_streamer.start()
        self.is_running = True
        
        # =====================================================================
        # PHASE 1: Camera validation - ensure arm tracking works before hybrid
        # =====================================================================
        print()
        print("[bold cyan]╔═══════════════════════════════════════════════════════╗[/bold cyan]")
        print("[bold cyan]║  Phase 1: Camera Validation                           ║[/bold cyan]")
        print("[bold cyan]╚═══════════════════════════════════════════════════════╝[/bold cyan]")
        print()
        print("[yellow]Stand visible to all cameras with arms at your sides.[/yellow]")
        print("[dim]Waiting for stable arm tracking before starting hybrid motion...[/dim]")
        print()
        
        VALIDATION_REQUIRED_FRAMES = 30  # Need 30 consecutive good frames (~1s at 30fps)
        consecutive_good = 0
        validation_start = time.time()
        validation_timeout = 60  # Max 60 seconds to get validation
        last_print_time = 0
        
        while consecutive_good < VALIDATION_REQUIRED_FRAMES:
            # Check timeout
            if time.time() - validation_start > validation_timeout:
                print("\n[red]✗ Camera validation timed out after 60 seconds[/red]")
                print("[yellow]Proceeding anyway - arm tracking may not work correctly[/yellow]")
                break
            
            # Try to get arm data
            left_arm, right_arm = self.get_camera_arms()
            
            if left_arm is not None:
                consecutive_good += 1
            else:
                consecutive_good = 0
            
            # Print status every 0.5 seconds
            if time.time() - last_print_time >= 0.5:
                skeleton_3d, reproj_error = self.camera_streamer.get_3d_skeleton()
                reproj_str = f"{reproj_error:.1f}px" if reproj_error and np.isfinite(reproj_error) else "N/A"
                
                if left_arm is not None:
                    progress = consecutive_good / VALIDATION_REQUIRED_FRAMES * 100
                    print(f"[green]✓ Arms detected![/green] Progress: {progress:.0f}% ({consecutive_good}/{VALIDATION_REQUIRED_FRAMES}) | Reproj: {reproj_str}", end='\r')
                else:
                    reason = self.last_failure_reason or "Unknown"
                    print(f"[yellow]⚠ Waiting for arms...[/yellow] Reason: {reason} | Reproj: {reproj_str}     ", end='\r')
                last_print_time = time.time()
            
            time.sleep(0.033)  # ~30 Hz check rate
        
        if consecutive_good >= VALIDATION_REQUIRED_FRAMES:
            print()
            print("[bold green]✓ Camera validation successful! Arm tracking confirmed.[/bold green]")
        print()
        
        # =====================================================================
        # PHASE 2: Hybrid motion with PKL body + camera arms
        # =====================================================================
        print("[bold green]╔═══════════════════════════════════════════════════════╗[/bold green]")
        print("[bold green]║  Phase 2: Hybrid Motion (PKL Body + Camera Arms)      ║[/bold green]")
        print("[bold green]╚═══════════════════════════════════════════════════════╝[/bold green]")
        print()
        
        # Startup delay countdown - keep sending PKL motion to robot!
        if startup_delay > 0:
            print("[bold yellow]╔════════════════════════════════════════════════╗[/bold yellow]")
            print("[bold yellow]║  GET READY! Robot motion starting soon...      ║[/bold yellow]")
            print("[bold yellow]╚════════════════════════════════════════════════╝[/bold yellow]")
            print()
            print("[cyan]Keep standing with:[/cyan]")
            print("  • Arms visible to cameras")
            print("  • Body upright")
            print()
            
            # Continuously send PKL frames during countdown to keep robot stable
            print("[dim]Streaming PKL motion during countdown...[/dim]")
            countdown_fps = 50
            countdown_dt = 1.0 / countdown_fps
            countdown_start = time.time()
            
            while time.time() - countdown_start < startup_delay:
                # Get and send PKL pose
                pkl_obs = self.get_pkl_mimic_obs()
                if pkl_obs is not None:
                    redis_client.set("action_body_unitree_g1_with_hands", json.dumps(pkl_obs.tolist()))
                    redis_client.set("action_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
                    redis_client.set("action_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
                    redis_client.set("action_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
                
                # Advance PKL motion
                self.t_step += 1
                if self.t_step >= self.num_steps and self.loop_motion:
                    self.t_step = 0
                
                # Print countdown
                remaining = startup_delay - (time.time() - countdown_start)
                if int(remaining) != int(remaining + countdown_dt):
                    print(f"[bold green]Starting in {int(remaining)+1} second{'s' if int(remaining) > 0 else ''}...[/bold green]", end='\r')
                
                time.sleep(countdown_dt)
            
            print()
            print("[bold green]✓ GO! Move your arms - robot will follow![/bold green]")
            print()
        
        print("[bold yellow]PKL motion controls body, YOUR ARMS control robot arms![/bold yellow]")
        print("[dim]Press Ctrl+C to stop (or 'q' in display window)[/dim]")
        print()
        
        target_dt = 1.0 / target_fps
        last_status_time = time.time()
        total_frames = 0
        total_arm_success = 0
        redis_connected = True
        
        # Arm warmup - don't use camera arms until we have stable tracking
        ARM_WARMUP_FRAMES = 30  # Need 30 consecutive good frames (~0.6s at 50fps)
        consecutive_good_frames = 0
        arms_warmed_up = False
        
        try:
            while self.is_running and self.camera_streamer.is_running:
                t0 = time.time()
                
                # Get PKL motion for body (this is ALWAYS used for legs/torso)
                pkl_obs = self.get_pkl_mimic_obs()
                if pkl_obs is None:
                    print("[yellow]Motion finished[/yellow]")
                    break
                
                # Get camera arms
                left_arm, right_arm = self.get_camera_arms()
                arm_tracking_success = left_arm is not None
                
                # Track consecutive good frames for warmup
                if arm_tracking_success:
                    consecutive_good_frames += 1
                    self.arm_success_count += 1
                    total_arm_success += 1
                else:
                    consecutive_good_frames = 0
                
                # Check if arms are warmed up
                if not arms_warmed_up and consecutive_good_frames >= ARM_WARMUP_FRAMES:
                    arms_warmed_up = True
                    print("[bold green]✓ Camera arm tracking warmed up - now following your arms![/bold green]")
                
                # Quality metrics are updated in get_camera_arms() above
                
                # Only use camera arms if warmed up AND currently tracking
                # Otherwise use PKL arms (safe fallback)
                if arms_warmed_up and arm_tracking_success:
                    merged_obs = self.merge_motion(pkl_obs, left_arm, right_arm)
                else:
                    # Use full PKL motion including arms (safe)
                    merged_obs = pkl_obs.copy()
                
                # Send to Redis
                try:
                    redis_client.set(
                        "action_body_unitree_g1_with_hands",
                        json.dumps(merged_obs.tolist())
                    )
                    redis_client.set(
                        "action_hand_left_unitree_g1_with_hands",
                        json.dumps(np.zeros(7).tolist())
                    )
                    redis_client.set(
                        "action_hand_right_unitree_g1_with_hands",
                        json.dumps(np.zeros(7).tolist())
                    )
                    redis_client.set(
                        "action_neck_unitree_g1_with_hands",
                        json.dumps(np.zeros(2).tolist())
                    )
                    redis_connected = True
                except Exception as e:
                    redis_connected = False
                    print(f"[red]Redis error: {e}[/red]")
                
                # Advance PKL playback
                self.t_step += 1
                self.frame_count += 1
                total_frames += 1
                
                # Track if we're actually using camera arms this frame
                using_camera_arms = arms_warmed_up and arm_tracking_success
                
                # Print detailed status every second
                if time.time() - last_status_time >= 1.0:
                    motion_progress = (self.t_step % self.num_steps) / self.num_steps * 100
                    arm_rate = total_arm_success / total_frames * 100 if total_frames > 0 else 0
                    fps = self.frame_count / (time.time() - last_status_time + 0.001)
                    
                    console.clear()
                    table = create_status_table(
                        total_frames, fps, arm_tracking_success, redis_connected,
                        len(self.camera_streamer.camera_ids), self.last_reproj_error, 
                        self.last_time_spread, motion_progress, arm_rate,
                        arms_warmed_up=arms_warmed_up, using_camera_arms=using_camera_arms
                    )
                    console.print(table)
                    
                    # Show sample data with source indication
                    arm_source = "CAMERA" if using_camera_arms else "PKL"
                    print(f"\n[dim]Arm source: {arm_source}[/dim]")
                    print(f"[dim]Arm values (L): [{merged_obs[21]:.2f}, {merged_obs[22]:.2f}, {merged_obs[23]:.2f}, {merged_obs[24]:.2f}][/dim]")
                    print(f"[dim]Arm values (R): [{merged_obs[28]:.2f}, {merged_obs[29]:.2f}, {merged_obs[30]:.2f}, {merged_obs[31]:.2f}][/dim]")
                    
                    # Show failure reason if tracking isn't working
                    if self.last_failure_reason and not using_camera_arms:
                        print(f"[yellow]⚠ Camera fail: {self.last_failure_reason}[/yellow]")
                    
                    self.frame_count = 0
                    last_status_time = time.time()
                
                # Maintain timing
                elapsed = time.time() - t0
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
        
        except KeyboardInterrupt:
            print("\n[yellow]Stopped by user[/yellow]")
        
        finally:
            self.is_running = False
            self.camera_streamer.stop()
            
            # Print summary
            print("\n[bold green]═══ Session Summary ═══[/bold green]")
            print(f"Total frames: {total_frames}")
            print(f"Arm tracking success: {total_arm_success} ({total_arm_success/total_frames*100:.1f}%)" if total_frames > 0 else "")
            print(f"Final reproj error: {self.last_reproj_error:.1f}px" if np.isfinite(self.last_reproj_error) else f"Final reproj error: N/A")
            print("\n[green]Done![/green]")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hybrid motion: PKL body + camera arms')
    parser.add_argument('--motion-file', type=str, required=True,
                       help='Path to PKL motion file')
    parser.add_argument('--camera-ids', type=str, default='4,6,2',
                       help='Comma-separated camera IDs (default: 4,6,2)')
    parser.add_argument('--calibration', type=str, 
                       default='../calibration/calibration.toml',
                       help='Path to calibration.toml file')
    parser.add_argument('--redis-host', type=str, default='localhost',
                       help='Redis server host (default: localhost)')
    parser.add_argument('--target-fps', type=int, default=50,
                       help='Target frame rate (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for motion loading: cpu or cuda (default: cuda)')
    parser.add_argument('--no-loop', action='store_true',
                       help='Play motion once instead of looping')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable camera display')
    parser.add_argument('--startup-delay', type=int, default=5,
                       help='Countdown delay in seconds before tracking starts (default: 5)')
    
    args = parser.parse_args()
    
    # Parse camera IDs
    camera_ids = [int(x) for x in args.camera_ids.split(',')]
    
    # Don't initialize CUDA here - let HybridMotionStreamer do it AFTER camera init
    # to avoid GPU context conflicts between PyTorch and MediaPipe/EGL
    device = args.device
    
    # Create and run streamer (camera init happens FIRST, then motion/CUDA)
    streamer = HybridMotionStreamer(
        motion_file=args.motion_file,
        camera_ids=camera_ids,
        calibration_file=args.calibration,
        device=device,
        loop_motion=not args.no_loop,
        enable_display=not args.no_display
    )
    
    streamer.run(
        redis_host=args.redis_host,
        target_fps=args.target_fps,
        startup_delay=args.startup_delay
    )


if __name__ == '__main__':
    main()

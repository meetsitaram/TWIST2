#!/usr/bin/env python3
"""
Multi-Camera 3D Pose to TWIST2 Real-Time Streaming

This script captures 3D body movements from multiple calibrated cameras via 
triangulation and streams them to TWIST2's low-level controller via Redis.

Key advantages over single-camera:
- True 3D skeleton in metric space (no depth approximation)
- More accurate joint positions from multiple viewpoints
- Robust to partial occlusions

Usage:
    # Terminal 1: Start robot simulation
    cd ~/projects/g1-pick-n-place/TWIST2
    conda activate twist2
    bash sim2sim.sh
    
    # Terminal 2: Start multi-camera streaming
    cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
    conda activate gmr
    python multicam_to_twist2.py
"""

import argparse
import time
import numpy as np
import json
import redis
import sys
import os

from rich import print
from rich.console import Console
from rich.live import Live
from rich.table import Table

# Import our multi-camera streamer
from multicam_pose_streamer import MultiCamPoseStreamer
from data_utils.params import DEFAULT_MIMIC_OBS

console = Console()


def create_status_table(frame_count, fps, tracking_success, redis_connected, 
                        num_cameras, reproj_error, time_spread_ms):
    """Create a nice status table for display"""
    table = Table(title="Multi-Camera 3D → TWIST2 Streaming")
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Frame Count", str(frame_count))
    table.add_row("FPS", f"{fps:.1f} Hz")
    table.add_row("Tracking", "✓ Active" if tracking_success else "✗ Lost")
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


def main(args):
    print("[bold green]╔═══════════════════════════════════════════════════╗[/bold green]")
    print("[bold green]║  Multi-Camera 3D → TWIST2 Real-Time Teleoperation ║[/bold green]")
    print("[bold green]╚═══════════════════════════════════════════════════╝[/bold green]")
    print()
    
    # Connect to Redis
    try:
        redis_client = redis.Redis(host=args.redis_host, port=6379, decode_responses=False)
        redis_client.ping()
        print(f"[green]✓[/green] Connected to Redis at {args.redis_host}:6379")
    except Exception as e:
        print(f"[red]✗ Failed to connect to Redis: {e}[/red]")
        print("[yellow]Make sure Redis is running: sudo systemctl start redis-server[/yellow]")
        return
    
    # Parse camera IDs
    camera_ids = [int(x) for x in args.camera_ids.split(',')]
    print(f"[cyan]Initializing multi-camera capture with cameras {camera_ids}...[/cyan]")
    
    # Initialize multi-camera streamer
    try:
        streamer = MultiCamPoseStreamer(
            camera_ids=camera_ids,
            calibration_file=args.calibration,
            resolution=(args.width, args.height),
            enable_display=args.display
        )
        streamer.start()
        print(f"[green]✓[/green] Multi-camera streamer initialized ({len(camera_ids)} cameras)")
    except Exception as e:
        print(f"[red]✗ Failed to initialize streamer: {e}[/red]")
        import traceback
        traceback.print_exc()
        return
    
    # Get default safe pose
    default_pose = DEFAULT_MIMIC_OBS["unitree_g1_with_hands"]
    print(f"[green]✓[/green] Using direct MediaPipe → G1 joint conversion")
    
    print()
    
    # Startup delay countdown
    if args.startup_delay > 0:
        print("[bold yellow]╔════════════════════════════════════════════════╗[/bold yellow]")
        print("[bold yellow]║  GET READY! Match the robot's standing pose   ║[/bold yellow]")
        print("[bold yellow]╚════════════════════════════════════════════════╝[/bold yellow]")
        print()
        print("[cyan]Stand with:[/cyan]")
        print("  • Feet shoulder-width apart")
        print("  • Arms relaxed at your sides")
        print("  • Body upright, visible from all cameras")
        print()
        
        # Send safe standing pose to robot during countdown
        print("[dim]Setting robot to safe standing pose...[/dim]")
        redis_client.set("action_body_unitree_g1_with_hands", json.dumps(default_pose.tolist()))
        redis_client.set("action_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        redis_client.set("action_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        redis_client.set("action_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
        
        for i in range(args.startup_delay, 0, -1):
            print(f"[bold green]Starting in {i} second{'s' if i > 1 else ''}...[/bold green]", end='\r')
            time.sleep(1)
        print()
        print("[bold green]✓ GO! Robot will now follow your movements![/bold green]")
        print()
    
    print("[bold yellow]Stand in view of all cameras and move![/bold yellow]")
    print("[dim]Press Ctrl+C to stop (or 'q' in display window)[/dim]")
    print()
    
    # Stats
    frame_count = 0
    success_count = 0
    last_fps_time = time.time()
    fps = 0.0
    last_reproj_error = 0.0
    last_time_spread = 0.0
    
    try:
        while streamer.is_running:
            loop_start = time.time()
            
            # Get mimic_obs directly from streamer (uses new direct G1 conversion)
            mimic_obs = streamer.get_mimic_obs()
            
            # Get quality metrics
            skeleton_3d, reproj_error = streamer.get_3d_skeleton()
            last_reproj_error = reproj_error
            last_time_spread = streamer._last_time_spread * 1000  # ms
            
            frame_count += 1
            redis_connected = False
            tracking_success = mimic_obs is not None
            
            if tracking_success:
                success_count += 1
                
                # Send to Redis
                try:
                    redis_client.set(
                        "action_body_unitree_g1_with_hands",
                        json.dumps(mimic_obs.tolist())
                    )
                    
                    # Send hand data (zeros for now - could add hand triangulation)
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
                    
                    # Print detailed info every second
                    if frame_count % args.target_fps == 0:
                        console.clear()
                        table = create_status_table(
                            frame_count, fps, tracking_success, redis_connected,
                            len(camera_ids), last_reproj_error, last_time_spread
                        )
                        console.print(table)
                        
                        # Show sample data
                        print(f"\n[dim]Sample mimic_obs (first 10): {mimic_obs[:10]}[/dim]")
                        print(f"[dim]Root height: {mimic_obs[2]:.3f}m[/dim]")
                
                except Exception as e:
                    print(f"[red]Error converting/sending data: {e}[/red]")
            
            else:
                # No tracking - send default standing pose to keep robot stable
                try:
                    redis_client.set("action_body_unitree_g1_with_hands", json.dumps(default_pose.tolist()))
                    redis_client.set("action_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
                    redis_client.set("action_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
                    redis_client.set("action_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
                except:
                    pass
                
                if frame_count % 30 == 0:
                    print("[yellow]⚠ No skeleton detected - stand visible to at least 2 cameras![/yellow]")
            
            # Calculate FPS
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30.0 / (current_time - last_fps_time)
                last_fps_time = current_time
            
            # Maintain target frame rate
            elapsed = time.time() - loop_start
            target_dt = 1.0 / args.target_fps
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
    
    except KeyboardInterrupt:
        print("\n[yellow]Stopped by user[/yellow]")
    
    except Exception as e:
        print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n[cyan]Cleaning up...[/cyan]")
        streamer.stop()
        
        # Print summary
        print("\n[bold green]═══ Session Summary ═══[/bold green]")
        print(f"Total frames: {frame_count}")
        print(f"Successful tracking: {success_count}")
        print(f"Success rate: {100 * success_count / frame_count if frame_count > 0 else 0:.1f}%")
        print(f"Average FPS: {fps:.1f} Hz")
        print(f"Final reproj error: {last_reproj_error:.1f}px")
        print("\n[green]Done![/green]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream multi-camera 3D pose to TWIST2')
    parser.add_argument('--camera-ids', type=str, default='4,6,2',
                       help='Comma-separated camera IDs (default: 4,6,2)')
    parser.add_argument('--calibration', type=str, 
                       default='../calibration/calibration.toml',
                       help='Path to calibration.toml file')
    parser.add_argument('--width', type=int, default=1280,
                       help='Camera resolution width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Camera resolution height (default: 720)')
    parser.add_argument('--redis-host', type=str, default='localhost',
                       help='Redis server host (default: localhost)')
    parser.add_argument('--target-fps', type=int, default=30,
                       help='Target frame rate (default: 30)')
    parser.add_argument('--human-height', type=float, default=1.7,
                       help='Your height in meters (default: 1.7)')
    parser.add_argument('--display', action='store_true', default=True,
                       help='Show camera views (default: True)')
    parser.add_argument('--no-display', dest='display', action='store_false',
                       help='Disable display for headless operation')
    parser.add_argument('--startup-delay', type=int, default=5,
                       help='Countdown delay in seconds before tracking starts (default: 5)')
    parser.add_argument('--joint-config', type=str, default='../joint_mapping_config.json',
                       help='Path to joint calibration config file')
    
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
"""
Replay recorded motion from PKL file.

This script loads a motion PKL file (recorded by record_motion.py) and plays it
back in MuJoCo with kinematic visualization (no physics).

Usage:
    cd ~/projects/g1-pick-n-place/TWIST2/deploy_real
    python replay_motion.py --file ../recordings/motion_20260121_231731.pkl
    
    # Loop playback
    python replay_motion.py --file ../recordings/motion_20260121_231731.pkl --loop
    
    # Slower playback
    python replay_motion.py --file ../recordings/motion_20260121_231731.pkl --speed 0.5
"""

import argparse
import pickle
import time
import numpy as np
import os
import sys

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Replay recorded motion from PKL file',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--file', '-f', type=str, required=True,
                       help='Path to PKL motion file')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                       help='Playback speed (default: 1.0)')
    parser.add_argument('--loop', '-l', action='store_true',
                       help='Loop playback')
    parser.add_argument('--no-viz', action='store_true',
                       help='Print data only, no MuJoCo visualization')
    
    args = parser.parse_args()
    
    # Load motion file
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return
    
    print(f"Loading: {args.file}")
    with open(args.file, 'rb') as f:
        motion_data = pickle.load(f)
    
    # Print info
    fps = motion_data.get('fps', 30)
    root_pos = motion_data.get('root_pos')
    root_rot = motion_data.get('root_rot')
    dof_pos = motion_data.get('dof_pos')
    
    if root_pos is None or dof_pos is None:
        print("Error: Invalid motion file - missing root_pos or dof_pos")
        return
    
    num_frames = len(root_pos)
    duration = num_frames / fps
    
    print(f"\nMotion Info:")
    print(f"  Frames: {num_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Playback speed: {args.speed}x")
    print(f"  Loop: {args.loop}")
    print()
    
    # Print position range
    print(f"  Root position range:")
    print(f"    X: {root_pos[:, 0].min():.3f} to {root_pos[:, 0].max():.3f} m")
    print(f"    Y: {root_pos[:, 1].min():.3f} to {root_pos[:, 1].max():.3f} m")
    print(f"    Z: {root_pos[:, 2].min():.3f} to {root_pos[:, 2].max():.3f} m")
    print()
    
    if args.no_viz:
        print("Skipping visualization (--no-viz)")
        return
    
    # Load MuJoCo
    try:
        import mujoco
        from mujoco.viewer import launch_passive
    except ImportError:
        print("Error: MuJoCo not available. Install with: pip install mujoco")
        return
    
    xml_file = os.path.join(project_root, "assets/g1/g1_mocap_29dof.xml")
    if not os.path.exists(xml_file):
        print(f"Error: MuJoCo model not found: {xml_file}")
        return
    
    print(f"Loading MuJoCo model: {xml_file}")
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)
    
    # Launch viewer
    viewer = launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -15
    
    print("\nPlaying motion... (close viewer window to stop)")
    print()
    
    frame_dt = 1.0 / fps / args.speed
    frame_idx = 0
    
    try:
        while viewer.is_running():
            t_start = time.time()
            
            # Get current frame data
            pos = root_pos[frame_idx]
            rot = root_rot[frame_idx]  # xyzw format
            joints = dof_pos[frame_idx]
            
            # Apply to MuJoCo
            data.qpos[:3] = pos
            data.qpos[3:7] = [rot[3], rot[0], rot[1], rot[2]]  # Convert xyzw to wxyz
            data.qpos[7:7+len(joints)] = joints
            
            # Forward kinematics (no physics)
            mujoco.mj_forward(model, data)
            
            # Update camera to follow robot
            try:
                pelvis_id = model.body("pelvis").id
                pelvis_pos = data.xpos[pelvis_id]
                viewer.cam.lookat = pelvis_pos
            except:
                pass
            
            viewer.sync()
            
            # Print progress
            current_time = frame_idx / fps
            print(f"\rFrame {frame_idx+1}/{num_frames} | Time: {current_time:.1f}s / {duration:.1f}s", end="")
            
            # Next frame
            frame_idx += 1
            if frame_idx >= num_frames:
                if args.loop:
                    frame_idx = 0
                    print("\n[Looping...]")
                else:
                    print("\n\nPlayback complete!")
                    break
            
            # Maintain playback speed
            elapsed = time.time() - t_start
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        viewer.close()
        print("Done!")


if __name__ == '__main__':
    main()

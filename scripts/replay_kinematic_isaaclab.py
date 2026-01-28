#!/usr/bin/env python3
"""
Replay PKL motion files in Isaac Lab (kinematic mode).

This script loads a PKL motion file and directly sets robot joint positions
(no physics, no policy) to verify the joint mapping and visualize motions.

Usage:
    cd ~/projects/g1-pick-n-place/TWIST2
    conda activate env_isaaclab
    python scripts/replay_kinematic_isaaclab.py \
        --motion_file datasets/teleop_motions/elbow_track_007.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
import pickle
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWIST2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TWIST2_ROOT)

##############################################################################
# ISAAC LAB APP LAUNCHER
##############################################################################

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Kinematic teleop joint test")
parser.add_argument("--motion_file", type=str, required=True,
                    help="PKL motion file to visualize")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of robots to show")
parser.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

##############################################################################
# IMPORTS
##############################################################################

import torch
import numpy as np
import gymnasium as gym

import isaaclab_envs
from isaaclab_envs.g1_motion_mimic_env_cfg import G1MotionMimicEnvCfg

# Import centralized robot configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWIST2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TWIST2_ROOT)
from robot_config import G1RobotConfig

# Use centralized config
MOTION_DOF_COUNT = G1RobotConfig.MUJOCO_NUM_JOINTS
MUJOCO_JOINT_ORDER = G1RobotConfig.MUJOCO_JOINT_ORDER


def build_joint_mapping(joint_names: list) -> dict:
    """Build a mapping from MuJoCo motion indices to Isaac Lab joint indices.
    
    Uses centralized G1RobotConfig for consistent mapping.
    """
    return G1RobotConfig.build_mujoco_to_isaaclab_mapping(joint_names)


def load_motion(motion_file: str) -> dict:
    """Load motion data from PKL or NPZ file."""
    if motion_file.endswith('.npz'):
        npz = np.load(motion_file, allow_pickle=True)
        data = {k: npz[k] for k in npz.files}
        # Extract fps from metadata
        if '_meta_fps' in data:
            data['fps'] = float(data['_meta_fps'][0])
            del data['_meta_fps']
    else:
        with open(motion_file, 'rb') as f:
            data = pickle.load(f)
    
    print(f"[Motion] Loaded: {motion_file}")
    print(f"[Motion] Keys: {list(data.keys())}")
    print(f"[Motion] FPS: {data.get('fps', 30)}")
    print(f"[Motion] Frames: {len(data['dof_pos'])}")
    print(f"[Motion] DOF shape: {data['dof_pos'].shape}")
    
    if 'root_pos' in data:
        print(f"[Motion] Root pos shape: {data['root_pos'].shape}")
    if 'root_rot' in data:
        print(f"[Motion] Root rot shape: {data['root_rot'].shape}")
    
    return data


def main():
    print("=" * 60)
    print("  Kinematic Teleop Joint Test")
    print("=" * 60)
    
    # Load motion
    motion_path = args.motion_file
    if not os.path.isabs(motion_path):
        motion_path = os.path.join(TWIST2_ROOT, motion_path)
    
    motion_data = load_motion(motion_path)
    dof_pos = motion_data['dof_pos']  # (num_frames, 29)
    root_pos = motion_data.get('root_pos', None)  # (num_frames, 3)
    root_rot = motion_data.get('root_rot', None)  # (num_frames, 4)
    fps = motion_data.get('fps', 30.0)
    num_frames = len(dof_pos)
    
    print(f"\n[Motion] Duration: {num_frames / fps:.1f}s at {fps} FPS")
    print(f"[Motion] Playback speed: {args.speed}x")
    
    # Print joint value ranges for debugging
    print("\n[Motion] Joint value ranges (radians):")
    for i in range(min(MOTION_DOF_COUNT, dof_pos.shape[1])):
        min_val = dof_pos[:, i].min()
        max_val = dof_pos[:, i].max()
        mean_val = dof_pos[:, i].mean()
        print(f"  Joint {i:2d}: min={min_val:7.3f}, max={max_val:7.3f}, mean={mean_val:7.3f}")
    
    # Create environment
    print("\n[Env] Creating Isaac Lab environment...")
    env_cfg = G1MotionMimicEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    # Use the specified motion file directly (single file or YAML)
    # For kinematic replay, we just need a valid motion file to initialize
    env_cfg.motion_file = motion_path
    
    # Disable all events
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.events.add_base_mass = None
    
    env = gym.make("Isaac-Motion-Mimic-G1-v0", cfg=env_cfg)
    
    # Get robot reference
    isaac_env = env.unwrapped
    robot = isaac_env.scene["robot"]
    
    # Get joint info
    num_joints = robot.num_joints
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    joint_names = robot.joint_names
    
    print(f"\n[Robot] Total joints: {num_joints}")
    print(f"[Robot] Motion DOFs: {MOTION_DOF_COUNT}")
    print(f"[Robot] Extra joints (fingers): {num_joints - MOTION_DOF_COUNT}")
    
    # Build joint mapping using centralized config
    joint_mapping = build_joint_mapping(joint_names)
    
    print("\n[Robot] Joint mapping (MuJoCo motion -> Isaac Lab):")
    print("  MJ_Idx | MuJoCo Name                    | IL_Idx | Isaac Lab Name")
    print("  -------|--------------------------------|--------|------------------")
    for mj_idx in range(MOTION_DOF_COUNT):
        mj_name = MUJOCO_JOINT_ORDER[mj_idx]
        il_idx = joint_mapping.get(mj_idx)
        if il_idx is not None:
            il_name = joint_names[il_idx]
            print(f"  {mj_idx:5d} | {mj_name:30s} | {il_idx:6d} | {il_name}")
        else:
            print(f"  {mj_idx:5d} | {mj_name:30s} |  (NONE) | NO EQUIVALENT")
    
    # Count mapped vs unmapped
    mapped_count = sum(1 for v in joint_mapping.values() if v is not None)
    print(f"\n[Mapping] {mapped_count}/{MOTION_DOF_COUNT} joints mapped, {MOTION_DOF_COUNT - mapped_count} unmapped")
    
    print("\n[Robot] Default joint positions:")
    for i in range(min(MOTION_DOF_COUNT, len(default_pos))):
        print(f"  Joint {i:2d}: {default_pos[i]:7.3f} rad")
    
    # Reset environment
    obs, _ = env.reset()
    
    print("\n" + "-" * 60)
    print("Starting kinematic playback (no physics)...")
    print("Press Ctrl+C to stop")
    print("-" * 60 + "\n")
    
    # Debug: Print first frame joint values
    print("\n[Debug] First frame joint positions:")
    first_frame_dof = dof_pos[0]
    for i in range(min(MOTION_DOF_COUNT, len(first_frame_dof))):
        print(f"  Joint {i:2d}: {first_frame_dof[i]:8.4f} rad ({np.degrees(first_frame_dof[i]):8.2f} deg)")
    
    # Playback loop
    frame = 0
    start_time = time.time()
    last_print_frame = -1
    
    try:
        while simulation_app.is_running():
            # Get current frame based on time
            elapsed = (time.time() - start_time) * args.speed
            frame = int(elapsed * fps) % num_frames
            
            # Get joint positions for this frame
            frame_dof = dof_pos[frame]  # (29,) in MuJoCo order
            
            # Create full joint position tensor starting from default
            full_joint_pos = torch.tensor(
                default_pos, dtype=torch.float32, device=isaac_env.device
            ).unsqueeze(0).expand(args.num_envs, -1).clone()
            
            # Map motion joints to Isaac Lab joints using the mapping
            for mj_idx, il_idx in joint_mapping.items():
                if il_idx is not None and mj_idx < len(frame_dof):
                    full_joint_pos[:, il_idx] = float(frame_dof[mj_idx])
            
            # Use motion root position if available, else fixed height
            env_origins = isaac_env.scene.env_origins
            
            if root_pos is not None:
                # Use actual motion root height (important for crouching/walking)
                motion_root = torch.tensor(
                    root_pos[frame], dtype=torch.float32, device=isaac_env.device
                ).unsqueeze(0).expand(args.num_envs, -1).clone()
                # Keep XY from env origin, use Z from motion
                motion_root[:, 0] = env_origins[:, 0]  # X from env
                motion_root[:, 1] = env_origins[:, 1]  # Y from env
                # Apply small offset to put feet on ground
                # MuJoCo feet at ~0m, Isaac Lab feet at ~0.07m for same root
                # Lower by 0.02m to put ankles at ~0.05m (ground contact)
                motion_root[:, 2] -= 0.02
                actual_root_pos = motion_root
            else:
                # Fallback: fixed height
                actual_root_pos = env_origins.clone()
                actual_root_pos[:, 2] = 0.78
            
            # Use motion root orientation if available
            if root_rot is not None:
                # Motion uses [qx, qy, qz, qw], Isaac Lab uses [w, x, y, z]
                quat_xyzw = root_rot[frame]  # scalar-last
                actual_root_quat = torch.tensor(
                    [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],  # Convert to [w,x,y,z]
                    dtype=torch.float32, device=isaac_env.device
                ).unsqueeze(0).expand(args.num_envs, -1)
            else:
                # Fallback: identity quaternion
                actual_root_quat = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0],
                    dtype=torch.float32, device=isaac_env.device
                ).unsqueeze(0).expand(args.num_envs, -1)
            
            # Step simulation first (for rendering pipeline)
            isaac_env.sim.step(render=False)
            
            # AFTER physics step, override with our kinematic pose
            # This ensures our values aren't overwritten by physics
            root_pose = torch.cat([actual_root_pos, actual_root_quat], dim=-1)
            robot.write_root_pose_to_sim(root_pose)
            
            # Also set root velocity to zero to prevent drift
            zero_vel = torch.zeros(args.num_envs, 6, device=isaac_env.device)
            robot.write_root_velocity_to_sim(zero_vel)
            
            # Write joint positions
            robot.write_joint_state_to_sim(
                full_joint_pos,
                torch.zeros_like(full_joint_pos)  # Zero velocity
            )
            
            # Render the frame
            isaac_env.sim.render()
            
            # Update robot data
            robot.update(isaac_env.sim.cfg.dt)
            
            # Debug print - verify joints and foot positions
            if frame != last_print_frame and frame % 30 == 0:
                last_print_frame = frame
                progress = frame / num_frames * 100
                
                # Read back actual positions
                actual_joints = robot.data.joint_pos[0].cpu().numpy()
                body_pos = robot.data.body_pos_w[0].cpu().numpy()  # (num_bodies, 3)
                root_z = actual_root_pos[0, 2].item()
                
                # Find ankle body indices
                body_names = robot.body_names
                left_ankle_z = None
                right_ankle_z = None
                for i, name in enumerate(body_names):
                    if name == "left_ankle_roll_link":
                        left_ankle_z = body_pos[i, 2]
                    elif name == "right_ankle_roll_link":
                        right_ankle_z = body_pos[i, 2]
                
                if left_ankle_z is not None and right_ankle_z is not None:
                    print(f"\r[Frame {frame:4d}] Root Z: {root_z:.3f}m, "
                          f"L_ankle Z: {left_ankle_z:.3f}m, "
                          f"R_ankle Z: {right_ankle_z:.3f}m", end="", flush=True)
                else:
                    print(f"\r[Frame {frame:4d}] Root Z: {root_z:.3f}m", end="", flush=True)
            
            # Small delay
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n[Kinematic] Stopped by user")
    
    env.close()
    simulation_app.close()
    print("\n[Kinematic] Done!")


if __name__ == "__main__":
    main()

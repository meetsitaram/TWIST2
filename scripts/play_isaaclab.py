#!/usr/bin/env python3
# Copyright (c) 2025
# Play/evaluate script for G1 Motion Imitation using Isaac Lab
#
# Usage:
#   cd TWIST2
#   python scripts/play_isaaclab.py --checkpoint logs/isaaclab/motion_mimic/model_5500.pt
#   python scripts/play_isaaclab.py --checkpoint logs/isaaclab/motion_mimic/model_5500.pt --num_envs 16

"""Play/evaluate G1 motion imitation policy using Isaac Lab."""

from __future__ import annotations

import argparse
import os
import sys

# Add TWIST2 root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWIST2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TWIST2_ROOT)

##############################################################################
# ISAAC LAB APP LAUNCHER (must be before other Isaac imports)
##############################################################################

from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Play G1 motion imitation policy")

# Checkpoint
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint file (.pt)")

# Task arguments
parser.add_argument("--task", type=str, default="Isaac-Motion-Mimic-G1-v0",
                    help="Task name")
parser.add_argument("--motion_file", type=str, 
                    default="motion_data_configs/teleop_dataset.yaml",
                    help="Motion dataset YAML file (relative to TWIST2 root)")

# Environment arguments
parser.add_argument("--num_envs", type=int, default=16,
                    help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=1,
                    help="Random seed")

# Disturbance options
parser.add_argument("--push", action="store_true",
                    help="Enable random push disturbances")
parser.add_argument("--push_force", type=float, default=50.0,
                    help="Max push force in Newtons (default: 50)")
parser.add_argument("--push_interval", type=float, default=5.0,
                    help="Interval between pushes in seconds (default: 5)")

# AppLauncher args (adds --headless, --video, etc.)
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args = parser.parse_args()

# Launch Isaac Sim (with rendering by default)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

##############################################################################
# IMPORTS (after AppLauncher)
##############################################################################

import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import our custom environment (registers gym tasks)
import isaaclab_envs
from isaaclab_envs.g1_motion_mimic_env_cfg import G1MotionMimicEnvCfg

# RSL-RL imports
from rsl_rl.modules import ActorCritic


def main():
    """Main play function."""
    
    print(f"[Play] Task: {args.task}")
    print(f"[Play] Checkpoint: {args.checkpoint}")
    print(f"[Play] Num envs: {args.num_envs}")
    
    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(TWIST2_ROOT, checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        print(f"[Play] ERROR: Checkpoint not found: {checkpoint_path}")
        simulation_app.close()
        return
    
    # Create environment config
    env_cfg = G1MotionMimicEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.motion_file = os.path.join(TWIST2_ROOT, args.motion_file)
    
    # Configure disturbances
    if args.push:
        # Enable random push disturbances
        from isaaclab.managers import EventTermCfg, SceneEntityCfg
        from isaaclab.envs.mdp import events as mdp_events
        
        print(f"[Play] Push disturbances ENABLED: force={args.push_force}N, interval={args.push_interval}s")
        
        # Configure push event - applies random impulse to robot base
        env_cfg.events.base_external_force_torque = EventTermCfg(
            func=mdp_events.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(args.push_interval * 0.8, args.push_interval * 1.2),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    "x": (-args.push_force / 50, args.push_force / 50),  # ~1 m/s per 50N
                    "y": (-args.push_force / 50, args.push_force / 50),
                    "yaw": (-args.push_force / 100, args.push_force / 100),  # rad/s
                },
            },
        )
    else:
        # Disable external forces for cleaner visualization
        env_cfg.events.base_external_force_torque = None
        print("[Play] Push disturbances DISABLED (use --push to enable)")
    
    # Create environment
    env = gym.make(args.task, cfg=env_cfg)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # Get observation and action dimensions from the wrapper
    obs_dim = env.num_obs
    act_dim = env.num_actions
    
    print(f"[Play] Observation dim: {obs_dim}")
    print(f"[Play] Action dim: {act_dim}")
    
    # Load checkpoint
    print(f"[Play] Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    
    # Create actor-critic with same architecture as training
    actor_critic = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=act_dim,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    ).to("cuda:0")
    
    # Load weights
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()
    
    print(f"[Play] Model loaded successfully!")
    print(f"[Play] Starting playback... Press Ctrl+C to stop.")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Get access to the underlying Isaac Lab env for debugging
    isaac_env = env.unwrapped
    
    # Play loop
    step = 0
    total_resets = 0
    try:
        while simulation_app.is_running():
            # Get action from policy (deterministic)
            with torch.no_grad():
                actions = actor_critic.act_inference(obs)
            
            # Step environment (RslRlVecEnvWrapper returns 4 values)
            obs, rewards, dones, infos = env.step(actions)
            
            # The wrapper should auto-reset, but let's check for done envs
            num_dones = dones.sum().item()
            if num_dones > 0:
                total_resets += num_dones
            
            step += 1
            
            # Debug: Print robot heights and termination info every 50 steps
            if step % 50 == 0:
                # Get robot root positions (Z = height)
                robot = isaac_env.scene["robot"]
                root_pos = robot.data.root_pos_w  # (num_envs, 3)
                heights = root_pos[:, 2]  # Z coordinate
                
                # Get termination info (with error handling)
                try:
                    base_contact_term = isaac_env.termination_manager.get_term("base_contact")
                    base_contact_count = base_contact_term.sum().item()
                except (KeyError, AttributeError):
                    base_contact_count = "N/A"
                
                try:
                    height_term = isaac_env.termination_manager.get_term("bad_height")
                    height_term_count = height_term.sum().item()
                except (KeyError, AttributeError):
                    height_term_count = "N/A"
                
                min_h = heights.min().item()
                max_h = heights.max().item()
                mean_h = heights.mean().item()
                
                # List active termination terms
                active_terms = isaac_env.termination_manager.active_terms
                
                print(f"[DEBUG] Step {step}: Heights min={min_h:.3f} max={max_h:.3f} mean={mean_h:.3f}")
                print(f"[DEBUG]   Active terms: {active_terms}")
                print(f"[DEBUG]   dones={dones.sum().item()}, base_contact={base_contact_count}, bad_height={height_term_count}")
                
                # Print individual env heights if any are low
                low_mask = heights < 0.5
                if low_mask.any():
                    low_envs = torch.where(low_mask)[0].tolist()
                    low_heights = heights[low_mask].tolist()
                    print(f"[DEBUG]   LOW envs: {low_envs} heights: {[f'{h:.2f}' for h in low_heights]}")
            
            # Print stats periodically
            if step % 100 == 0:
                mean_reward = rewards.mean().item()
                print(f"[Play] Step {step}, Mean reward: {mean_reward:.3f}, Total resets: {int(total_resets)}")
    
    except KeyboardInterrupt:
        print("\n[Play] Stopped by user.")
    
    # Cleanup
    env.close()
    simulation_app.close()
    
    print("[Play] Done!")


if __name__ == "__main__":
    main()

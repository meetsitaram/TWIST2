#!/usr/bin/env python3
# Copyright (c) 2025
# Training script for G1 Motion Imitation using Isaac Lab
#
# Usage:
#   cd TWIST2
#   python scripts/train_isaaclab.py --task Isaac-Motion-Mimic-G1-v0 --num_envs 4096 --headless

"""Train G1 motion imitation policy using Isaac Lab and RSL-RL."""

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
parser = argparse.ArgumentParser(description="Train G1 motion imitation policy")

# Task arguments
parser.add_argument("--task", type=str, default="Isaac-Motion-Mimic-G1-v0",
                    help="Task name")
parser.add_argument("--motion_file", type=str, 
                    default="motion_data_configs/teleop_dataset.yaml",
                    help="Motion dataset YAML file (relative to TWIST2 root)")

# Training arguments
parser.add_argument("--num_envs", type=int, default=4096,
                    help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=20000,
                    help="Maximum training iterations")
parser.add_argument("--seed", type=int, default=1,
                    help="Random seed")

# Logging arguments
parser.add_argument("--run_name", type=str, default="motion_mimic",
                    help="Run name for logging")
parser.add_argument("--log_dir", type=str, default="logs/isaaclab",
                    help="Log directory (relative to TWIST2 root)")
parser.add_argument("--logger", type=str, default="wandb", choices=["tensorboard", "wandb", "neptune"],
                    help="Logger to use (default: wandb)")
parser.add_argument("--wandb_project", type=str, default="twist2-isaaclab",
                    help="Wandb project name")

# Checkpoint
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint to resume from")

# AppLauncher args (adds --headless, --video, etc.)
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

##############################################################################
# IMPORTS (after AppLauncher)
##############################################################################

# Now import Isaac Lab modules
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import our custom environment (registers gym tasks)
import isaaclab_envs
from isaaclab_envs.g1_motion_mimic_env_cfg import G1MotionMimicEnvCfg
from isaaclab_envs.agents.rsl_rl_ppo_cfg import G1MotionMimicPPORunnerCfg

# RSL-RL imports
from rsl_rl.runners import OnPolicyRunner


def main():
    """Main training function."""
    
    print(f"[Train] Task: {args.task}")
    print(f"[Train] Motion file: {args.motion_file}")
    print(f"[Train] Num envs: {args.num_envs}")
    print(f"[Train] Max iterations: {args.max_iterations}")
    print(f"[Train] Device: cuda:0")
    
    # Create environment config
    env_cfg = G1MotionMimicEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.motion_file = os.path.join(TWIST2_ROOT, args.motion_file)
    
    # Create environment (G1MotionMimicEnv auto-initializes MotionLib)
    env = gym.make(args.task, cfg=env_cfg)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # Create PPO runner config
    agent_cfg = G1MotionMimicPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.experiment_name = args.run_name
    agent_cfg.logger = args.logger
    agent_cfg.wandb_project = args.wandb_project
    
    if args.checkpoint is not None:
        agent_cfg.resume = True
    
    print(f"[Train] Logger: {args.logger}")
    if args.logger == "wandb":
        print(f"[Train] Wandb project: {args.wandb_project}")
    
    # Create log directory
    log_dir = os.path.join(TWIST2_ROOT, args.log_dir, args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create runner
    runner = OnPolicyRunner(
        env=env,
        train_cfg=agent_cfg.to_dict(),
        log_dir=log_dir,
        device="cuda:0",
    )
    
    # Resume from checkpoint if specified
    if args.checkpoint is not None:
        print(f"[Train] Resuming from: {args.checkpoint}")
        runner.load(args.checkpoint)
    
    # Train
    print(f"[Train] Starting training for {args.max_iterations} iterations...")
    runner.learn(num_learning_iterations=args.max_iterations)
    
    # Cleanup
    env.close()
    simulation_app.close()
    
    print("[Train] Training complete!")


if __name__ == "__main__":
    main()

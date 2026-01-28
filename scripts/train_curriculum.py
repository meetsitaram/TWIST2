#!/usr/bin/env python3
"""
Curriculum Training Script

Multi-stage training pipeline for G1 motion imitation:

Stage 1: Baby Steps (stand without falling, basic balance)
Stage 2: Baby Steps + Big Steps (walking and movement)
Stage 3: Upper Body (arm control in stable environment)
Stage 4: Robust Upper Body (push forces + upper body motions)

This curriculum order trains upper body control FIRST in a stable environment,
then adds robustness training with push disturbances so the robot learns to
maintain arm control under perturbations.

Usage:
    # Quick test run (100 iterations per stage)
    python scripts/train_curriculum.py --iterations 100
    
    # Full training run
    python scripts/train_curriculum.py --full
    
    # Resume from a specific stage
    python scripts/train_curriculum.py --start_stage 3 --checkpoint logs/curriculum/.../model.pt
    
    # Custom iterations per stage (4 values)
    python scripts/train_curriculum.py --iterations 5000 5000 10000 5000
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
TWIST2_ROOT = SCRIPT_DIR.parent
MOTIONS_DIR = TWIST2_ROOT / "datasets" / "teleop_motions"
CONFIGS_DIR = TWIST2_ROOT / "motion_data_configs"
LOGS_DIR = TWIST2_ROOT / "logs" / "curriculum"

# Isaac Lab environment Python interpreter
# The curriculum script can be run from any environment, but training requires Isaac Lab
ISAACLAB_PYTHON = Path.home() / "miniconda3" / "envs" / "env_isaaclab" / "bin" / "python"


def create_motion_config(name: str, motion_dirs: list, output_path: Path):
    """Create a YAML config file combining motion directories."""
    import yaml
    
    motions = []
    for motion_dir in motion_dirs:
        dir_path = MOTIONS_DIR / motion_dir
        if not dir_path.exists():
            print(f"Warning: Motion directory not found: {dir_path}")
            continue
        
        for pkl_file in sorted(dir_path.glob("*.pkl")):
            rel_path = f"datasets/teleop_motions/{motion_dir}/{pkl_file.name}"
            motions.append({
                'file': rel_path,
                'weight': 1.0,
            })
    
    if not motions:
        raise ValueError(f"No motion files found for {motion_dirs}")
    
    config = {
        'name': name,
        'root_path': '..',  # Go up from motion_data_configs/ to TWIST2 root
        'motions': motions,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created motion config: {output_path} ({len(motions)} motions)")
    return output_path


def get_latest_checkpoint(stage_dir: Path) -> str:
    """Find the latest checkpoint in a stage directory."""
    if not stage_dir.exists():
        return None
    
    checkpoints = list(stage_dir.glob("model_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by iteration number
    def get_iter(p):
        try:
            return int(p.stem.split("_")[1])
        except:
            return 0
    
    latest = max(checkpoints, key=get_iter)
    return str(latest)


def run_training_stage(
    stage_name: str,
    motion_config: str,
    output_dir: Path,
    max_iterations: int,
    checkpoint: str = None,
    robust: bool = False,
    robust_level: str = None,  # "medium" or "hard"
    stage3: bool = False,
    num_envs: int = 4096,
    extra_args: list = None,
):
    """Run a single training stage."""
    
    print("\n" + "=" * 70)
    print(f"  STAGE: {stage_name}")
    print("=" * 70)
    print(f"  Motion config: {motion_config}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Max iterations: {max_iterations}")
    if checkpoint:
        print(f"  Checkpoint:    {checkpoint}")
    if robust:
        level = robust_level or "easy"
        print(f"  Robust mode:   {level}")
    if stage3:
        print(f"  Stage 3 mode:  enabled (upper body tracking)")
    print("=" * 70 + "\n")
    
    # Build command - use Isaac Lab Python interpreter
    if not ISAACLAB_PYTHON.exists():
        print(f"ERROR: Isaac Lab Python not found at {ISAACLAB_PYTHON}")
        print("Please ensure env_isaaclab conda environment is installed.")
        return False
    
    cmd = [
        str(ISAACLAB_PYTHON),
        str(SCRIPT_DIR / "train_isaaclab.py"),
        "--motion_file", motion_config,
        "--max_iterations", str(max_iterations),
        "--num_envs", str(num_envs),
        "--run_name", stage_name,
        "--log_dir", str(output_dir.parent),
        "--headless",
    ]
    
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    
    if robust:
        if robust_level == "medium":
            cmd.append("--robust_medium")
        elif robust_level == "hard":
            cmd.append("--robust_hard")
        else:
            cmd.append("--robust")
    
    if stage3:
        cmd.append("--stage3")
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}\n")
    
    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, cwd=str(TWIST2_ROOT))
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\nERROR: Stage {stage_name} failed with return code {result.returncode}")
        return False
    
    print(f"\nStage {stage_name} completed in {elapsed/60:.1f} minutes")
    
    # Find the output checkpoint
    checkpoint_path = get_latest_checkpoint(output_dir)
    if checkpoint_path:
        print(f"Checkpoint saved: {checkpoint_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Training for G1 Motion Imitation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--test", action="store_true",
                       help="Quick test run (500 iterations per stage)")
    parser.add_argument("--full", action="store_true",
                       help="Full training run (default iterations)")
    parser.add_argument("--iterations", nargs="+", type=int,
                       help="Custom iterations for each stage (4 values)")
    parser.add_argument("--start_stage", type=int, default=1,
                       help="Start from this stage (1-4)")
    parser.add_argument("--checkpoint", type=str,
                       help="Initial checkpoint for start_stage")
    parser.add_argument("--num_envs", type=int, default=4096,
                       help="Number of parallel environments")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print commands without running")
    
    args = parser.parse_args()
    
    # Determine iterations per stage
    # Stage 1: stand, Stage 2: move, Stage 3: upper body, Stage 4: robust upper
    if args.test:
        iterations = [500, 500, 500, 500]  # ~15 min each
    elif args.full:
        iterations = [5000, 5000, 10000, 5000]  # Full training (upper body needs more)
    elif args.iterations:
        if len(args.iterations) == 1:
            iterations = [args.iterations[0]] * 4
        elif len(args.iterations) == 4:
            iterations = args.iterations
        else:
            print("Error: --iterations requires 1 or 4 values")
            return 1
    else:
        iterations = [500, 500, 500, 500]  # Default to test
    
    print("\n" + "=" * 70)
    print("  CURRICULUM TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Start stage:   {args.start_stage}")
    print(f"  Iterations:    Stage1={iterations[0]}, Stage2={iterations[1]}, Stage3={iterations[2]}, Stage4={iterations[3]}")
    print(f"  Num envs:      {args.num_envs}")
    print("=" * 70)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {run_dir}\n")
    
    # Stage definitions (5-stage curriculum)
    # 1. Standing - basic balance
    # 2. Walking - movement
    # 3. Upper body - arm control in stable environment
    # 4. Robust upper body - push forces + upper body motions
    stages = [
        {
            'name': 'stage1_stand',
            'motion_dirs': ['baby_steps'],
            'robust': False,
            'stage3': False,
            'description': 'Basic standing balance',
        },
        {
            'name': 'stage2_move',
            'motion_dirs': ['baby_steps', 'big_steps'],
            'robust': False,
            'stage3': False,
            'description': 'Walking and movement',
        },
        {
            'name': 'stage3_upper',
            'motion_dirs': ['stage3_upper_body'],
            'robust': False,
            'stage3': True,  # Use upper body tracking rewards
            'description': 'Upper body control (stable environment)',
        },
        {
            'name': 'stage4_robust_upper',
            'motion_dirs': ['stage3_upper_body'],
            'robust': True,
            'robust_level': None,  # easy pushes
            'stage3': True,  # Keep upper body tracking rewards
            'description': 'Robust upper body (with push forces)',
        },
    ]
    
    # Create motion configs
    print("Creating motion configs...")
    for stage in stages:
        config_path = CONFIGS_DIR / f"curriculum_{stage['name']}.yaml"
        try:
            create_motion_config(stage['name'], stage['motion_dirs'], config_path)
            stage['motion_config'] = str(config_path.relative_to(TWIST2_ROOT))
        except ValueError as e:
            print(f"Error creating config for {stage['name']}: {e}")
            return 1
    
    if args.dry_run:
        print("\n[DRY RUN] Would run the following stages:")
        for i, stage in enumerate(stages, 1):
            if i < args.start_stage:
                continue
            print(f"\n  Stage {i}: {stage['name']}")
            print(f"    Motion: {stage['motion_config']}")
            print(f"    Iterations: {iterations[i-1]}")
        return 0
    
    # Run stages
    checkpoint = args.checkpoint
    
    for i, stage in enumerate(stages, 1):
        if i < args.start_stage:
            continue
        
        stage_dir = run_dir / stage['name']
        
        success = run_training_stage(
            stage_name=stage['name'],
            motion_config=stage['motion_config'],
            output_dir=stage_dir,
            max_iterations=iterations[i-1],
            checkpoint=checkpoint,
            robust=stage.get('robust', False),
            robust_level=stage.get('robust_level'),
            stage3=stage.get('stage3', False),
            num_envs=args.num_envs,
        )
        
        if not success:
            print(f"\nTraining failed at stage {i}. Stopping curriculum.")
            return 1
        
        # Get checkpoint for next stage
        checkpoint = get_latest_checkpoint(stage_dir)
        if not checkpoint and i < len(stages):
            print(f"\nWarning: No checkpoint found for stage {i}. Next stage may fail.")
    
    print("\n" + "=" * 70)
    print("  CURRICULUM TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nAll stages completed successfully.")
    print(f"Final model: {checkpoint}")
    print(f"Logs: {run_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

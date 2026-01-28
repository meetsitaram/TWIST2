#!/usr/bin/env python3
# Copyright (c) 2025-2026
# Teleop Overlay Deployment for G1 Motion Imitation using Isaac Lab
#
# This script deploys a trained RL policy with teleop overlay:
# - Lower body (legs): Controlled by trained policy for balance/locomotion
# - Upper body (arms + optionally waist): Controlled by teleop targets
#
# Usage:
#   # Play with live Redis teleop (from camera streaming)
#   python scripts/play_isaaclab_teleop.py \
#       --checkpoint logs/isaaclab/motion_mimic/model_17500.pt \
#       --teleop redis
#
#   # FASTEST: Use MuJoCo viewer (headless Isaac Lab + lightweight MuJoCo viz):
#   python scripts/play_isaaclab_teleop.py \
#       --checkpoint logs/isaaclab/motion_mimic/model_17500.pt \
#       --teleop redis --headless --mujoco_viz
#
#   # FASTER RENDERING (use performance mode for better FPS):
#   python scripts/play_isaaclab_teleop.py \
#       --checkpoint logs/isaaclab/motion_mimic/model_17500.pt \
#       --teleop redis --rendering_mode performance
#
#   # Play with PKL motion overlay
#   python scripts/play_isaaclab_teleop.py \
#       --checkpoint logs/isaaclab/motion_mimic/model_17500.pt \
#       --teleop pkl --motion_file datasets/teleop_motions/elbow_track_007.pkl
#
#   # Policy only (no teleop overlay, for comparison)
#   python scripts/play_isaaclab_teleop.py \
#       --checkpoint logs/isaaclab/motion_mimic/model_17500.pt \
#       --teleop none

"""Teleop overlay deployment for G1 motion imitation policy."""

from __future__ import annotations

import argparse
import os
import sys
import pickle
import time

# Add TWIST2 root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWIST2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TWIST2_ROOT)

##############################################################################
# ISAAC LAB APP LAUNCHER (must be before other Isaac imports)
##############################################################################

from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Play G1 policy with teleop overlay")

# Checkpoint
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint file (.pt)")

# Task arguments
parser.add_argument("--task", type=str, default="Isaac-Motion-Mimic-G1-v0",
                    help="Task name")
parser.add_argument("--env_motion_file", type=str, 
                    default="motion_data_configs/teleop_dataset.yaml",
                    help="Motion dataset YAML for environment (for policy observations)")

# Environment arguments
parser.add_argument("--num_envs", type=int, default=4,
                    help="Number of parallel environments (default: 4 for teleop)")
parser.add_argument("--seed", type=int, default=1,
                    help="Random seed")

# Teleop mode
parser.add_argument("--teleop", type=str, default="none",
                    choices=["none", "pkl", "redis"],
                    help="Teleop source: 'none' (policy only), 'pkl' (recorded motion), 'redis' (live camera)")

# PKL motion file (for --teleop pkl)
parser.add_argument("--motion_file", type=str, default=None,
                    help="PKL motion file for upper body teleop overlay")
parser.add_argument("--motion_loop", action="store_true",
                    help="Loop motion file when it ends")

# Redis settings (for --teleop redis)
parser.add_argument("--redis_host", type=str, default="localhost",
                    help="Redis server host")
parser.add_argument("--redis_port", type=int, default=6379,
                    help="Redis server port")
parser.add_argument("--redis_key", type=str, default="teleop:mimic_obs",
                    help="Redis key for teleop targets")

# Overlay settings
parser.add_argument("--blend_waist", action="store_true",
                    help="Also control waist joints from teleop (default: policy controls waist)")
parser.add_argument("--blend_alpha", type=float, default=1.0,
                    help="Blend factor for teleop: 0=policy only, 1=teleop only (default: 1.0)")
parser.add_argument("--direct_override", action="store_true",
                    help="Directly override upper body joint ACTIONS instead of injecting targets. "
                         "This bypasses policy learning and directly sets joint positions.")

# Debug
parser.add_argument("--debug_timing", action="store_true",
                    help="Print timing breakdown to identify lag sources")
parser.add_argument("--no_realtime", action="store_true",
                    help="Disable real-time throttling (run as fast as possible)")
parser.add_argument("--publish_state", action="store_true",
                    help="Publish robot state to Redis for external viewer (use with mujoco_state_viewer.py)")

# Real-time optimization settings
parser.add_argument("--fast_render", action="store_true",
                    help="Optimize for real-time: skip render frames, reduce solver iterations")
parser.add_argument("--render_interval", type=int, default=None,
                    help="Render every N physics steps (default: 2, use higher for speed)")
parser.add_argument("--physics_dt", type=float, default=None,
                    help="Physics timestep in seconds (default: 0.0167 = 60Hz)")

# AppLauncher args (adds --headless, etc.)
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
import numpy as np
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import our custom environment (registers gym tasks)
import isaaclab_envs
from isaaclab_envs.g1_motion_mimic_env_cfg import G1MotionMimicEnvCfg

# RSL-RL imports
from rsl_rl.modules import ActorCritic

# Redis for state publishing (to external MuJoCo viewer)
if args.publish_state:
    import redis
    import json
    state_redis = redis.Redis(host='localhost', port=6379)
    REDIS_STATE_KEY = "isaaclab:robot_state"
    print("[Play] Will publish robot state to Redis for external viewer")


##############################################################################
# JOINT INDEX DEFINITIONS - Using centralized G1RobotConfig
##############################################################################

# Import centralized robot configuration
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWIST2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TWIST2_ROOT)
from robot_config import G1RobotConfig

# Convenience aliases from centralized config
MOTION_DOF_COUNT = G1RobotConfig.MUJOCO_NUM_JOINTS
MUJOCO_UPPER_BODY_INDICES = G1RobotConfig.MUJOCO_UPPER_BODY_INDICES
MUJOCO_WAIST_INDICES = G1RobotConfig.MUJOCO_WAIST_INDICES


def build_joint_mapping(joint_names: list) -> dict:
    """Build mapping from MuJoCo motion indices to Isaac Lab joint indices.
    
    Uses centralized G1RobotConfig for consistent mapping.
    """
    return G1RobotConfig.build_mujoco_to_isaaclab_mapping(joint_names)


##############################################################################
# MOTION FILE READER
##############################################################################

class MotionFileReader:
    """Read joint targets from PKL motion file."""
    
    def __init__(self, motion_file: str, loop: bool = False):
        self.motion_file = motion_file
        self.loop = loop
        
        # Load motion data
        with open(motion_file, 'rb') as f:
            self.motion_data = pickle.load(f)
        
        # Extract DOF positions: (num_frames, 29)
        self.dof_pos = self.motion_data['dof_pos']
        self.num_frames = len(self.dof_pos)
        self.fps = self.motion_data.get('fps', 30.0)
        
        self.current_frame = 0
        self.start_time = None
        
        print(f"[MotionReader] Loaded {motion_file}")
        print(f"[MotionReader] Frames: {self.num_frames}, FPS: {self.fps}, Duration: {self.num_frames/self.fps:.1f}s")
    
    def reset(self):
        """Reset to start of motion."""
        self.current_frame = 0
        self.start_time = time.time()
    
    def get_target_dof(self) -> np.ndarray | None:
        """Get current target DOF positions."""
        if self.start_time is None:
            self.start_time = time.time()
        
        # Compute frame from elapsed time
        elapsed = time.time() - self.start_time
        frame = int(elapsed * self.fps)
        
        if frame >= self.num_frames:
            if self.loop:
                self.reset()
                frame = 0
            else:
                return None
        
        self.current_frame = frame
        return self.dof_pos[frame]
    
    @property
    def progress(self) -> float:
        """Current progress through motion (0-1)."""
        return self.current_frame / self.num_frames


##############################################################################
# REDIS TELEOP CLIENT
##############################################################################

class RedisTeleopClient:
    """Receive teleop targets from Redis (from camera streaming)."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, key: str = "teleop:mimic_obs"):
        self.host = host
        self.port = port
        self.key = key
        self.redis_client = None
        self.last_dof = None
        self._debug_count = 0
        
        self._connect()
    
    def _connect(self):
        """Connect to Redis server."""
        try:
            import redis
            print(f"[RedisTeleop] Connecting to Redis at {self.host}:{self.port}...")
            self.redis_client = redis.Redis(host=self.host, port=self.port, decode_responses=False)
            self.redis_client.ping()
            print(f"[RedisTeleop] Connected! Listening on key: '{self.key}'")
            
            # Check if key already has data
            data = self.redis_client.get(self.key)
            if data:
                print(f"[RedisTeleop] Found existing data in key ({len(data)} bytes)")
            else:
                print(f"[RedisTeleop] No data in key yet - waiting for publisher...")
        except ImportError:
            print("[RedisTeleop] ERROR: redis-py not installed!")
            print("[RedisTeleop] Install with: pip install redis")
            self.redis_client = None
        except Exception as e:
            print(f"[RedisTeleop] ERROR: Could not connect to Redis: {e}")
            self.redis_client = None
    
    def get_target_dof(self) -> np.ndarray | None:
        """Get current target DOF positions from Redis."""
        if self.redis_client is None:
            return self.last_dof
        
        try:
            data = self.redis_client.get(self.key)
            if data is None:
                if self._debug_count % 100 == 0:
                    print(f"[RedisTeleop] No data in key '{self.key}'")
                self._debug_count += 1
                return self.last_dof
            
            # Deserialize numpy array
            arr = np.frombuffer(data, dtype=np.float32)
            
            # Debug: print on first successful read
            if self.last_dof is None:
                print(f"[RedisTeleop] First data received! Array len={len(arr)}")
            
            # Handle different formats
            if len(arr) == 29:
                # Direct joint positions
                self.last_dof = arr.copy()
            elif len(arr) == 35:
                # mimic_obs format: [root_state(6), joint_pos(29)]
                self.last_dof = arr[6:35].copy()
            elif len(arr) >= 29:
                # Assume first 29 are joint positions
                self.last_dof = arr[:29].copy()
            else:
                print(f"[RedisTeleop] WARNING: Unexpected array length: {len(arr)}")
                return self.last_dof
            
            return self.last_dof
            
        except Exception as e:
            print(f"[RedisTeleop] Error reading from Redis: {e}")
            return self.last_dof


##############################################################################
# MAIN
##############################################################################

def main():
    """Main play function with teleop overlay."""
    
    print("=" * 60)
    print("  G1 Teleop Overlay Deployment")
    print("=" * 60)
    print(f"\n[Play] Task: {args.task}")
    print(f"[Play] Num envs: {args.num_envs}")
    print(f"[Play] Mode: {args.teleop.upper()}")
    
    # Validate checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(TWIST2_ROOT, checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        print(f"[Play] ERROR: Checkpoint not found: {checkpoint_path}")
        simulation_app.close()
        return
    
    print(f"[Play] Checkpoint: {checkpoint_path}")
    
    # Initialize teleop source
    teleop_source = None
    
    if args.teleop == "pkl":
        if args.motion_file is None:
            print("[Play] ERROR: --motion_file required for --teleop pkl")
            simulation_app.close()
            return
        
        motion_path = args.motion_file
        if not os.path.isabs(motion_path):
            motion_path = os.path.join(TWIST2_ROOT, motion_path)
        
        if not os.path.exists(motion_path):
            print(f"[Play] ERROR: Motion file not found: {motion_path}")
            simulation_app.close()
            return
        
        teleop_source = MotionFileReader(motion_path, loop=args.motion_loop)
        print(f"[Play] Upper body from: {motion_path}")
        
    elif args.teleop == "redis":
        teleop_source = RedisTeleopClient(
            host=args.redis_host,
            port=args.redis_port,
            key=args.redis_key
        )
        print(f"[Play] Upper body from: Redis {args.redis_host}:{args.redis_port}")
    
    # Print overlay settings
    has_teleop = teleop_source is not None
    if has_teleop:
        print(f"[Play] Blend alpha: {args.blend_alpha}")
        print(f"[Play] Blend waist: {args.blend_waist}")
        waist_mode = "teleop" if args.blend_waist else "policy"
        print(f"[Play] Lower body: policy | Waist: {waist_mode} | Upper body: teleop")
    else:
        print("[Play] Full policy control (no teleop overlay)")
    
    print("-" * 60)
    
    # Create environment config
    env_cfg = G1MotionMimicEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.motion_file = os.path.join(TWIST2_ROOT, args.env_motion_file)
    
    # Configure viewport camera to follow robot
    env_cfg.viewer.eye = (3.0, 3.0, 2.0)  # Camera position offset
    env_cfg.viewer.lookat = (0.0, 0.0, 0.8)  # Look at robot torso height
    env_cfg.viewer.origin_type = "asset_root"  # Follow robot root
    env_cfg.viewer.asset_name = "robot"  # Track the robot asset
    print("[Play] Camera tracking: following robot")
    
    # Real-time optimization settings
    if args.fast_render:
        # Skip render frames for speed (render every 4th physics step)
        env_cfg.sim.render_interval = 4
        # Use faster physics settings
        env_cfg.sim.dt = 1.0 / 60.0  # 60Hz physics
        env_cfg.decimation = 1  # Action every physics step
        # Reduce solver iterations for speed (may reduce accuracy)
        env_cfg.sim.physx.num_position_iterations = 4  # Default is usually 4-8
        env_cfg.sim.physx.num_velocity_iterations = 0  # Default is 0-1
        print("[Play] Fast render mode: render_interval=4, dt=1/60, decimation=1")
    
    # Override render interval if specified
    if args.render_interval is not None:
        env_cfg.sim.render_interval = args.render_interval
        print(f"[Play] Render interval: {args.render_interval}")
    
    # Override physics dt if specified
    if args.physics_dt is not None:
        env_cfg.sim.dt = args.physics_dt
        print(f"[Play] Physics dt: {args.physics_dt}s ({1.0/args.physics_dt:.0f}Hz)")
    
    # Disable all push/disturbance events for clean teleop visualization
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.events.add_base_mass = None
    print("[Play] Push disturbances DISABLED")
    
    # Create environment
    env = gym.make(args.task, cfg=env_cfg)
    
    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # Get dimensions from wrapper
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
    print("-" * 60)
    
    # Get default joint positions from robot for reference
    isaac_env = env.unwrapped
    robot = isaac_env.scene["robot"]
    default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy()
    joint_names = robot.joint_names
    
    # Build joint mapping from MuJoCo motion indices to Isaac Lab indices
    joint_mapping = build_joint_mapping(joint_names)
    mapped_count = sum(1 for v in joint_mapping.values() if v is not None)
    print(f"[Play] Robot joints: {len(joint_names)} | Motion joints: {MOTION_DOF_COUNT} | Mapped: {mapped_count}")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Reset teleop source
    if teleop_source is not None and hasattr(teleop_source, 'reset'):
        teleop_source.reset()
    
    # Play loop
    step = 0
    total_resets = 0
    teleop_active = False
    start_time = time.time()
    
    # Real-time control: target 50 Hz (20ms per step)
    # This matches the environment step size (0.02s)
    target_dt = 0.02  # 50 Hz
    
    # Timing tracking
    step_times = []
    policy_times = []
    redis_times = []
    
    try:
        while simulation_app.is_running():
            loop_start = time.time()
            
            # Timing
            t0 = time.time()
            
            # Get action from policy (deterministic)
            with torch.no_grad():
                policy_actions = actor_critic.act_inference(obs)
            t_policy = time.time() - t0
            
            # Inject teleop targets into observation space (proper approach)
            t1 = time.time()
            t_redis = 0
            if teleop_source is not None:
                teleop_dof = teleop_source.get_target_dof()
                t_redis = time.time() - t1
                
                if teleop_dof is not None:
                    if not teleop_active:
                        print(f"[Play] Teleop active! Injecting {len(teleop_dof)} DOFs (MuJoCo order)")
                        # Print joint mapping for debugging using centralized config
                        print(f"[Play] Joint mapping (MuJoCo -> Isaac Lab):")
                        for mj_idx in [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]:
                            il_idx = joint_mapping.get(mj_idx)
                            mj_name = G1RobotConfig.MUJOCO_JOINT_ORDER[mj_idx] if mj_idx < len(G1RobotConfig.MUJOCO_JOINT_ORDER) else "?"
                            print(f"  MJ[{mj_idx}] -> IL[{il_idx}] ({mj_name})")
                    teleop_active = True
                    
                    # Debug: print teleop values periodically (MuJoCo indices)
                    if step % 100 == 0:
                        l_shoulder = teleop_dof[15] if len(teleop_dof) > 15 else 0
                        l_elbow = teleop_dof[18] if len(teleop_dof) > 18 else 0
                        r_shoulder = teleop_dof[22] if len(teleop_dof) > 22 else 0
                        r_elbow = teleop_dof[25] if len(teleop_dof) > 25 else 0
                        print(f"\r[Teleop] L_sh={l_shoulder:.2f} L_el={l_elbow:.2f} R_sh={r_shoulder:.2f} R_el={r_elbow:.2f}", end="")
                    
                    # NOTE: teleop_dof is in MuJoCo order (29 DOFs)
                    # This matches the motion library format that the policy was trained on
                    teleop_tensor = torch.tensor(teleop_dof, dtype=torch.float32, device="cuda:0")
                    
                    # Get underlying environment (unwrap RSL-RL wrapper)
                    base_env = env.unwrapped
                    
                    if args.direct_override:
                        # DIRECT OVERRIDE MODE: Bypass policy for upper body joints
                        # Directly set the action values for upper body joints to move them
                        # to the teleop target positions.
                        #
                        # Action scaling from velocity_env_cfg.py:
                        #   joint_pos = default_pos + action * scale  (where scale=0.5)
                        # Therefore:
                        #   action = (target_pos - default_pos) / scale
                        
                        ACTION_SCALE = 0.5  # From velocity_env_cfg.py JointPositionActionCfg
                        
                        # Remap teleop from MuJoCo to Isaac Lab order
                        teleop_il = base_env._remap_mujoco_to_isaaclab(teleop_tensor.unsqueeze(0))
                        
                        # Get default joint positions
                        default_pos = robot.data.default_joint_pos  # (num_envs, num_joints)
                        
                        # Get upper body indices from env
                        upper_body_indices = base_env._upper_body_indices
                        
                        # Override policy actions for upper body joints
                        for il_idx in upper_body_indices:
                            if il_idx < teleop_il.shape[1] and il_idx < policy_actions.shape[1]:
                                # Direct position control: action = (target - default) / scale
                                target_pos = teleop_il[0, il_idx]
                                default_val = default_pos[0, il_idx]
                                action_val = (target_pos - default_val) / ACTION_SCALE
                                # Broadcast to all envs
                                policy_actions[:, il_idx] = action_val
                        
                        # Debug: Print first few joints every 100 steps
                        if step % 100 == 0:
                            # Show a few key joints
                            debug_joints = [(15, "L_sh_pitch"), (16, "L_sh_roll"), (18, "L_elbow")]
                            debug_str = " [DIRECT:"
                            for mj_idx, name in debug_joints:
                                il_idx_check = joint_mapping.get(mj_idx)
                                if il_idx_check is not None and il_idx_check < teleop_il.shape[1]:
                                    tgt = teleop_il[0, il_idx_check].item()
                                    dfl = default_pos[0, il_idx_check].item()
                                    act = (tgt - dfl) / ACTION_SCALE
                                    debug_str += f" {name}:tgt={tgt:.2f},dfl={dfl:.2f},act={act:.2f}"
                            debug_str += "]"
                            print(debug_str, end="")
                    else:
                        # OBSERVATION INJECTION MODE: Policy sees teleop targets and learns to track
                        # Inject teleop targets - upper body only so policy maintains lower body balance
                        upper_body_only = not args.blend_waist  # If waist blending, include waist too
                        base_env.set_teleop_targets(teleop_tensor, upper_body_only=upper_body_only)
                
                else:
                    # Teleop source ended or unavailable
                    if teleop_active:
                        print("\n[Play] Teleop source ended, clearing targets")
                        base_env = env.unwrapped
                        base_env.clear_teleop_targets()
                        teleop_active = False
            
            # Step environment
            t2 = time.time()
            obs, rewards, dones, infos = env.step(policy_actions)
            t_step = time.time() - t2
            
            # Track timing stats
            step_times.append(t_step * 1000)
            policy_times.append(t_policy * 1000)
            redis_times.append(t_redis * 1000)
            
            # Publish state to Redis for external viewer
            if args.publish_state:
                il_joint_pos = robot.data.joint_pos[0].cpu().numpy()
                root_pos = robot.data.root_pos_w[0].cpu().numpy()
                root_quat = robot.data.root_quat_w[0].cpu().numpy()
                
                state = {
                    'joint_pos': il_joint_pos.tolist(),
                    'joint_names': joint_names,  # Isaac Lab joint order
                    'root_pos': root_pos.tolist(),
                    'root_quat': root_quat.tolist(),
                    'step': step,
                    'time': time.time()
                }
                state_redis.set(REDIS_STATE_KEY, json.dumps(state))
            
            # Track resets
            num_dones = dones.sum().item()
            if num_dones > 0:
                total_resets += num_dones
                if teleop_source is not None and hasattr(teleop_source, 'reset'):
                    teleop_source.reset()
            
            step += 1
            
            # Print status periodically
            if step % 50 == 0:
                mean_reward = rewards.mean().item()
                
                # Get robot heights
                root_pos = robot.data.root_pos_w
                heights = root_pos[:, 2]
                mean_height = heights.mean().item()
                
                # Teleop progress
                progress_str = ""
                if teleop_source is not None and hasattr(teleop_source, 'progress'):
                    progress_str = f" | Motion: {teleop_source.progress*100:.0f}%"
                elif teleop_source is not None:
                    progress_str = f" | Teleop: {'active' if teleop_active else 'waiting'}"
                
                # Debug timing
                timing_str = ""
                if args.debug_timing:
                    timing_str = f" | Policy:{t_policy*1000:.0f}ms Redis:{t_redis*1000:.0f}ms Step:{t_step*1000:.0f}ms"
                
                print(f"\r[Play] Step {step} | Reward: {mean_reward:.3f} | "
                      f"Height: {mean_height:.2f}m | Resets: {int(total_resets)}{progress_str}{timing_str}", 
                      end="", flush=True)
            
            # Real-time throttling: wait if we're ahead of schedule
            if not args.no_realtime:
                loop_elapsed = time.time() - loop_start
                if loop_elapsed < target_dt:
                    time.sleep(target_dt - loop_elapsed)
    
    except KeyboardInterrupt:
        print("\n\n[Play] Stopped by user.")
    
    # Print summary stats
    elapsed = time.time() - start_time
    if step > 0 and step_times:
        avg_step = sum(step_times) / len(step_times)
        avg_policy = sum(policy_times) / len(policy_times)
        avg_redis = sum(redis_times) / len(redis_times)
        actual_fps = step / elapsed
        target_fps = 1.0 / target_dt
        
        print("\n" + "=" * 60)
        print("  Final Statistics")
        print("=" * 60)
        print(f"  Total steps:     {step}")
        print(f"  Duration:        {elapsed:.1f}s")
        print(f"  Target FPS:      {target_fps:.0f}")
        print(f"  Actual FPS:      {actual_fps:.1f}")
        print(f"  Real-time:       {'disabled (--no_realtime)' if args.no_realtime else 'enabled'}")
        print(f"  Total resets:    {int(total_resets)}")
        print("=" * 60)
        print("  Timing Breakdown (avg per step):")
        print(f"    Policy:        {avg_policy:.1f}ms")
        print(f"    Redis:         {avg_redis:.1f}ms")
        print(f"    Sim step:      {avg_step:.1f}ms")
        print(f"    Compute total: {avg_policy + avg_redis + avg_step:.1f}ms")
        print("=" * 60)
        if avg_step > target_dt * 1000:
            print(f"  ⚠️  Sim step ({avg_step:.0f}ms) exceeds target ({target_dt*1000:.0f}ms)")
            print("      Try --headless to reduce rendering overhead")
        elif actual_fps >= target_fps * 0.95:
            print("  ✓ Running at real-time!")
        print()
    
    # Cleanup
    env.close()
    simulation_app.close()
    
    print("[Play] Done!")


if __name__ == "__main__":
    main()

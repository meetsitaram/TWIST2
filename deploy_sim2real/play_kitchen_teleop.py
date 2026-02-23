#!/usr/bin/env python3
"""
G1 Kitchen Teleop - Run trained policy in a kitchen environment.

Loads the kitchen scene from exports/g1_kitchen_scene/ and runs the G1 robot
with teleop overlay (upper body from motion file or live Redis stream,
lower body controlled by the trained RL policy for balance).

Usage:
    # PKL motion in kitchen (from TWIST2 root)
    python deploy_sim2real/play_kitchen_teleop.py \
        --teleop pkl --motion_file deploy_sim2real/sample_motions/pick-place-2.pkl \
        --motion_loop

    # Live Redis teleop in kitchen
    python deploy_sim2real/play_kitchen_teleop.py --teleop redis

    # Policy only (no teleop overlay)
    python deploy_sim2real/play_kitchen_teleop.py --teleop none
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWIST2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TWIST2_ROOT)

KITCHEN_DIR = os.path.join(TWIST2_ROOT, "exports", "g1_kitchen_scene")
SCENE_USD = os.path.join(KITCHEN_DIR, "scene.usd")
SCENE_PARAMS_FILE = os.path.join(KITCHEN_DIR, "configs", "scene_params.json")

##############################################################################
# ISAAC LAB APP LAUNCHER (must be before other Isaac imports)
##############################################################################

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Kitchen Teleop")

parser.add_argument("--checkpoint", type=str,
                    default=os.path.join(SCRIPT_DIR, "policy_stage4_121999.pt"),
                    help="Path to checkpoint file (.pt)")
parser.add_argument("--task", type=str, default="Isaac-Motion-Mimic-G1-v0",
                    help="Task name")
parser.add_argument("--env_motion_file", type=str,
                    default="motion_data_configs/teleop_dataset.yaml",
                    help="Motion dataset YAML for environment (for policy observations)")

# Teleop mode
parser.add_argument("--teleop", type=str, default="none",
                    choices=["none", "pkl", "redis"],
                    help="Teleop source: 'none' (policy only), 'pkl' (recorded motion), 'redis' (live)")
parser.add_argument("--motion_file", type=str, default=None,
                    help="PKL motion file for upper body teleop overlay")
parser.add_argument("--motion_loop", action="store_true",
                    help="Loop motion file when it ends")

# Redis settings
parser.add_argument("--redis_host", type=str, default="localhost")
parser.add_argument("--redis_port", type=int, default=6379)
parser.add_argument("--redis_key", type=str, default="teleop:mimic_obs")

# Overlay settings
parser.add_argument("--blend_waist", action="store_true",
                    help="Also control waist joints from teleop")
parser.add_argument("--blend_alpha", type=float, default=1.0,
                    help="Blend factor: 0=policy only, 1=teleop only")
parser.add_argument("--direct_override", action="store_true",
                    help="Directly override upper body ACTIONS instead of injecting targets")
parser.add_argument("--no_realtime", action="store_true",
                    help="Disable real-time throttling")
parser.add_argument("--debug_timing", action="store_true",
                    help="Print timing breakdown")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Default to quality rendering for kitchen scene (better reflections/materials)
if not hasattr(args, "rendering_mode") or args.rendering_mode is None:
    args.rendering_mode = "quality"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

##############################################################################
# IMPORTS (after AppLauncher)
##############################################################################

import torch
import numpy as np
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_envs  # noqa: F401 - registers gym tasks
from isaaclab_envs.g1_motion_mimic_env_cfg import G1MotionMimicEnvCfg

from rsl_rl.modules import ActorCritic

from robot_config import G1RobotConfig

MOTION_DOF_COUNT = G1RobotConfig.MUJOCO_NUM_JOINTS
MUJOCO_UPPER_BODY_INDICES = G1RobotConfig.MUJOCO_UPPER_BODY_INDICES
MUJOCO_WAIST_INDICES = G1RobotConfig.MUJOCO_WAIST_INDICES


def build_joint_mapping(joint_names: list) -> dict:
    """Build mapping from MuJoCo motion indices to Isaac Lab joint indices."""
    return G1RobotConfig.build_mujoco_to_isaaclab_mapping(joint_names)


##############################################################################
# SCENE MATERIALS
##############################################################################
# KITCHEN SCENE CONFIG
##############################################################################

def load_scene_params() -> dict:
    """Load kitchen scene parameters from JSON config."""
    with open(SCENE_PARAMS_FILE) as f:
        return json.load(f)


def apply_kitchen_scene_config(env_cfg: G1MotionMimicEnvCfg, params: dict):
    """Modify environment config to use kitchen scene.

    The scene.usd has the RoomShell scaled 1.47x with floor at ~Z=-1.352 and
    ceiling at ~Z=+1.35. Loaded at origin, the terrain collision plane at Z=0
    sits inside the kitchen, and the robot (pelvis at ~0.74m) stands naturally
    inside the room. The scene.usd also contains its own visual ground plane
    at Z=0 which aligns with the terrain.
    """
    kitchen_scale = params.get("kitchen_scale", {}).get("value", 1.47)

    # Single environment - one kitchen
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 8.0

    # Kitchen at origin -- its floor is below the terrain plane, ceiling above,
    # so the robot ends up standing inside the kitchen naturally.
    env_cfg.scene.kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        spawn=UsdFileCfg(usd_path=SCENE_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Fixture positions from scene.usd (for spawn presets)
    # fridge: (-0.6858, 1.5097, 0.8271), front face ~Y=1.15
    # dishwasher: (-1.3518, -0.3494, 0.3543), rotated 88.6 deg
    # microwave: (0.576, 1.4405, 0.9953)

    # Robot spawn -- position in front of fridge, facing it (+Y direction)
    import math
    robot_x, robot_y = -0.69, 0.75
    yaw = math.pi / 2  # face +Y (toward fridge / back wall)
    default_z = env_cfg.scene.robot.init_state.pos[2]
    env_cfg.scene.robot.init_state.pos = (robot_x, robot_y, default_z)
    env_cfg.scene.robot.init_state.rot = (
        math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
    )
    print(f"[Kitchen] Robot spawn: ({robot_x}, {robot_y}) facing fridge (yaw={math.degrees(yaw):.0f} deg)")

    # Indoor lighting (replace outdoor sky)
    env_cfg.scene.sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.85)),
    )
    env_cfg.scene.distant_light = AssetBaseCfg(
        prim_path="/World/distantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=600.0, color=(1.0, 1.0, 0.95), angle=0.53
        ),
    )

    # Free camera positioned for kitchen overview
    cam = params.get("camera", {})
    env_cfg.viewer.eye = tuple(cam.get("eye", [2.5, 2.5, 1.0]))
    env_cfg.viewer.lookat = tuple(cam.get("target", [0.0, 0.0, -0.3]))

    # Constrain resets to stay near spawn point with same orientation
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        },
    }

    # Disable disturbances
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.events.add_base_mass = None

    print(f"[Kitchen] Scene USD: {SCENE_USD}")
    print(f"[Kitchen] Scale: {kitchen_scale}")
    print(f"[Kitchen] Camera eye: {env_cfg.viewer.eye}, target: {env_cfg.viewer.lookat}")


##############################################################################
# MOTION FILE READER
##############################################################################

class MotionFileReader:
    """Read joint targets from PKL motion file."""

    def __init__(self, motion_file: str, loop: bool = False):
        self.motion_file = motion_file
        self.loop = loop

        with open(motion_file, 'rb') as f:
            self.motion_data = pickle.load(f)

        self.dof_pos = self.motion_data['dof_pos']
        self.num_frames = len(self.dof_pos)
        self.fps = self.motion_data.get('fps', 30.0)
        self.current_frame = 0
        self.start_time = None

        print(f"[MotionReader] Loaded {motion_file}")
        print(f"[MotionReader] Frames: {self.num_frames}, FPS: {self.fps}, "
              f"Duration: {self.num_frames/self.fps:.1f}s")

    def reset(self):
        self.current_frame = 0
        self.start_time = time.time()

    def get_target_dof(self) -> np.ndarray | None:
        if self.start_time is None:
            self.start_time = time.time()
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
        return self.current_frame / self.num_frames


##############################################################################
# REDIS TELEOP CLIENT
##############################################################################

class RedisTeleopClient:
    """Receive teleop targets from Redis (from camera streaming)."""

    def __init__(self, host: str = "localhost", port: int = 6379,
                 key: str = "teleop:mimic_obs"):
        self.host = host
        self.port = port
        self.key = key
        self.redis_client = None
        self.last_dof = None
        self._debug_count = 0
        self._connect()

    def _connect(self):
        try:
            import redis
            print(f"[RedisTeleop] Connecting to {self.host}:{self.port}...")
            self.redis_client = redis.Redis(
                host=self.host, port=self.port, decode_responses=False
            )
            self.redis_client.ping()
            print(f"[RedisTeleop] Connected! Key: '{self.key}'")
            data = self.redis_client.get(self.key)
            if data:
                print(f"[RedisTeleop] Found existing data ({len(data)} bytes)")
            else:
                print(f"[RedisTeleop] No data yet - waiting for publisher...")
        except ImportError:
            print("[RedisTeleop] ERROR: pip install redis")
            self.redis_client = None
        except Exception as e:
            print(f"[RedisTeleop] ERROR: {e}")
            self.redis_client = None

    def get_target_dof(self) -> np.ndarray | None:
        if self.redis_client is None:
            return self.last_dof
        try:
            data = self.redis_client.get(self.key)
            if data is None:
                if self._debug_count % 100 == 0:
                    print(f"[RedisTeleop] No data in '{self.key}'")
                self._debug_count += 1
                return self.last_dof
            arr = np.frombuffer(data, dtype=np.float32)
            if self.last_dof is None:
                print(f"[RedisTeleop] First data! len={len(arr)}")
            if len(arr) == 29:
                self.last_dof = arr.copy()
            elif len(arr) == 35:
                self.last_dof = arr[6:35].copy()
            elif len(arr) >= 29:
                self.last_dof = arr[:29].copy()
            else:
                print(f"[RedisTeleop] WARNING: unexpected len={len(arr)}")
                return self.last_dof
            return self.last_dof
        except Exception as e:
            print(f"[RedisTeleop] Error: {e}")
            return self.last_dof


##############################################################################
# MAIN
##############################################################################

def main():
    print("=" * 60)
    print("  G1 Kitchen Teleop")
    print("=" * 60)
    print(f"\n[Play] Mode: {args.teleop.upper()}")

    # Load kitchen scene params
    scene_params = load_scene_params()

    # Validate checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(TWIST2_ROOT, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        print(f"[Play] ERROR: Checkpoint not found: {checkpoint_path}")
        simulation_app.close()
        return
    print(f"[Play] Checkpoint: {checkpoint_path}")

    # Validate kitchen assets
    if not os.path.exists(SCENE_USD):
        print(f"[Play] ERROR: Kitchen USD not found: {SCENE_USD}")
        simulation_app.close()
        return

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
            host=args.redis_host, port=args.redis_port, key=args.redis_key
        )
        print(f"[Play] Upper body from: Redis {args.redis_host}:{args.redis_port}")

    has_teleop = teleop_source is not None
    if has_teleop:
        print(f"[Play] Blend alpha: {args.blend_alpha}")
        waist_mode = "teleop" if args.blend_waist else "policy"
        print(f"[Play] Lower body: policy | Waist: {waist_mode} | Upper body: teleop")
    else:
        print("[Play] Full policy control (no teleop overlay)")
    print("-" * 60)

    # Create environment config with kitchen scene
    env_cfg = G1MotionMimicEnvCfg()
    env_cfg.motion_file = os.path.join(TWIST2_ROOT, args.env_motion_file)
    apply_kitchen_scene_config(env_cfg, scene_params)

    # Create environment
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    obs_dim = env.num_obs
    act_dim = env.num_actions
    print(f"[Play] Observation dim: {obs_dim}, Action dim: {act_dim}")

    # Load checkpoint
    print("[Play] Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    actor_critic = ActorCritic(
        num_actor_obs=obs_dim, num_critic_obs=obs_dim, num_actions=act_dim,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation="elu", init_noise_std=1.0,
    ).to("cuda:0")
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()
    print("[Play] Model loaded!")

    # Robot reference data
    isaac_env = env.unwrapped
    robot = isaac_env.scene["robot"]
    joint_names = robot.joint_names
    joint_mapping = build_joint_mapping(joint_names)
    mapped_count = sum(1 for v in joint_mapping.values() if v is not None)
    print(f"[Play] Robot joints: {len(joint_names)} | Mapped: {mapped_count}")
    print(f"[Play] Starting... Press Ctrl+C to stop.")
    print("-" * 60)

    obs, _ = env.reset()
    if teleop_source is not None and hasattr(teleop_source, 'reset'):
        teleop_source.reset()

    step = 0
    total_resets = 0
    teleop_active = False
    start_time = time.time()
    target_dt = 0.02  # 50 Hz

    step_times = []
    policy_times = []
    teleop_times = []

    try:
        while simulation_app.is_running():
            loop_start = time.time()

            t0 = time.time()
            with torch.no_grad():
                policy_actions = actor_critic.act_inference(obs)
            t_policy = time.time() - t0

            t1 = time.time()
            t_teleop = 0
            if teleop_source is not None:
                teleop_dof = teleop_source.get_target_dof()
                t_teleop = time.time() - t1

                if teleop_dof is not None:
                    if not teleop_active:
                        print(f"[Play] Teleop active! {len(teleop_dof)} DOFs")
                        for mj_idx in [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]:
                            il_idx = joint_mapping.get(mj_idx)
                            mj_name = G1RobotConfig.MUJOCO_JOINT_ORDER[mj_idx]
                            print(f"  MJ[{mj_idx}] -> IL[{il_idx}] ({mj_name})")
                    teleop_active = True

                    teleop_tensor = torch.tensor(
                        teleop_dof, dtype=torch.float32, device="cuda:0"
                    )
                    base_env = env.unwrapped

                    if args.direct_override:
                        ACTION_SCALE = 0.5
                        teleop_il = base_env._remap_mujoco_to_isaaclab(
                            teleop_tensor.unsqueeze(0)
                        )
                        default_pos = robot.data.default_joint_pos
                        for il_idx in base_env._upper_body_indices:
                            if il_idx < teleop_il.shape[1] and il_idx < policy_actions.shape[1]:
                                target_pos = teleop_il[0, il_idx]
                                default_val = default_pos[0, il_idx]
                                policy_actions[:, il_idx] = (
                                    (target_pos - default_val) / ACTION_SCALE
                                )
                    else:
                        upper_body_only = not args.blend_waist
                        base_env.set_teleop_targets(
                            teleop_tensor, upper_body_only=upper_body_only
                        )

                else:
                    if teleop_active:
                        print("\n[Play] Teleop ended, clearing targets")
                        env.unwrapped.clear_teleop_targets()
                        teleop_active = False

            # Step
            t2 = time.time()
            obs, rewards, dones, infos = env.step(policy_actions)
            t_step = time.time() - t2

            step_times.append(t_step * 1000)
            policy_times.append(t_policy * 1000)
            teleop_times.append(t_teleop * 1000)

            num_dones = dones.sum().item()
            if num_dones > 0:
                total_resets += num_dones
                if teleop_source is not None and hasattr(teleop_source, 'reset'):
                    teleop_source.reset()

            step += 1

            # Status every 50 steps
            if step % 50 == 0:
                mean_reward = rewards.mean().item()
                root_pos = robot.data.root_pos_w
                mean_height = root_pos[:, 2].mean().item()

                progress_str = ""
                if teleop_source is not None and hasattr(teleop_source, 'progress'):
                    progress_str = f" | Motion: {teleop_source.progress*100:.0f}%"

                timing_str = ""
                if args.debug_timing:
                    timing_str = (f" | Policy:{t_policy*1000:.0f}ms "
                                  f"Teleop:{t_teleop*1000:.0f}ms "
                                  f"Step:{t_step*1000:.0f}ms")

                print(f"\r[Play] Step {step} | Reward: {mean_reward:.3f} | "
                      f"Height: {mean_height:.2f}m | Resets: {int(total_resets)}"
                      f"{progress_str}{timing_str}", end="", flush=True)

            # Real-time throttle
            if not args.no_realtime:
                loop_elapsed = time.time() - loop_start
                if loop_elapsed < target_dt:
                    time.sleep(target_dt - loop_elapsed)

    except KeyboardInterrupt:
        print("\n\n[Play] Stopped by user.")

    # Summary
    elapsed = time.time() - start_time
    if step > 0 and step_times:
        avg_step = sum(step_times) / len(step_times)
        avg_policy = sum(policy_times) / len(policy_times)
        actual_fps = step / elapsed

        print("\n" + "=" * 60)
        print("  Final Statistics")
        print("=" * 60)
        print(f"  Total steps:     {step}")
        print(f"  Duration:        {elapsed:.1f}s")
        print(f"  Actual FPS:      {actual_fps:.1f}")
        print(f"  Total resets:    {int(total_resets)}")
        print(f"  Avg policy:      {avg_policy:.1f}ms")
        print(f"  Avg sim step:    {avg_step:.1f}ms")
        print("=" * 60)

    env.close()
    simulation_app.close()
    print("[Play] Done!")


if __name__ == "__main__":
    main()

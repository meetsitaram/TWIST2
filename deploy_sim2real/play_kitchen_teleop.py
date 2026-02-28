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
parser.add_argument("--auto_loop", action="store_true",
                    help="Start with auto-loop enabled (cycle spawn presets)")
parser.add_argument("--loop_interval", type=float, default=30.0,
                    help="Seconds per spawn preset when auto-looping (default: 30)")
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

import math
import torch
import numpy as np
import gymnasium as gym

import carb.input
import omni.appwindow

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
# SPAWN PRESETS — named locations inside the kitchen
##############################################################################

# (name, x, y, yaw_degrees)
# Coordinates are in the kitchen world frame (origin = center of kitchen).
# Yaw 0 = facing +X, 90 = facing +Y (toward back wall), 180 = facing -X, etc.
SPAWN_PRESETS = {
    1: ("fridge",      -0.85,  0.60,   75),  # offset left of fridge, angled toward door
    2: ("stove",        0.95,  0.20,    0),  # in front of stove, facing right toward it
    3: ("dishwasher",  -0.80, -0.10,  180),  # facing dishwasher (pulled back)
    4: ("microwave",    0.58,  0.75,   90),  # in front of microwave
}

# Per-preset camera positions captured from the Isaac Sim viewport.
CAMERA_PRESETS = {
    1: {"eye": (-1.332, -0.509, 1.617), "target": (-0.556, 1.247, 1.058)},  # fridge
    2: {"eye": (0.593, 1.069, 1.792),  "target": (1.551, -0.262, 0.648)},   # stove
    3: {"eye": (-0.251, -1.541, 1.611), "target": (-1.182, -0.073, 0.622)}, # dishwasher
    4: {"eye": (-0.346, -0.379, 1.805), "target": (0.788, 0.962, 0.848)},   # microwave
}


_kitchen_resetter = None


class KitchenObjectResetter:
    """Capture and restore initial state of kitchen fixtures/objects via PhysX tensor API.

    Uses the same physics_sim_view that Isaac Lab uses to reset the robot,
    so transforms and joint states are written directly into PhysX.
    """

    def __init__(self, env):
        import sys
        from pxr import UsdPhysics
        self._env = env
        sim = env.unwrapped.sim
        psv = sim.physics_sim_view
        stage = sim.stage
        kitchen = "/World/envs/env_0/Kitchen"

        self._rigid_body_data = {}

        # Find ALL rigid bodies under the entire Kitchen hierarchy
        # (fixture doors, loose objects, etc.) — skip the RoomShell.
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(kitchen):
                continue
            if "/RoomShell" in path:
                continue
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue

            prim_name = prim.GetName()
            # Use full path as key to avoid name collisions
            key = path.replace(kitchen + "/", "")
            try:
                view = psv.create_rigid_body_view(path)
                count = view.count
                if count == 0:
                    continue
                indices = torch.arange(count, dtype=torch.int32, device="cuda:0")
                self._rigid_body_data[key] = {
                    "view": view,
                    "indices": indices,
                    "transforms": view.get_transforms().clone(),
                    "velocities": view.get_velocities().clone(),
                }
            except Exception as e:
                print(f"[Kitchen] Could not track {key}: {e}", flush=True)

        print(f"\n[Kitchen] Object resetter ready — {len(self._rigid_body_data)} rigid bodies tracked:", flush=True)
        for key in sorted(self._rigid_body_data.keys()):
            print(f"  - {key}", flush=True)
        sys.stdout.flush()

    def reset(self):
        """Restore all tracked rigid bodies to their initial PhysX state."""
        for key, data in self._rigid_body_data.items():
            try:
                idx = data["indices"]
                data["view"].set_transforms(data["transforms"], idx)
                data["view"].set_velocities(data["velocities"], idx)
            except Exception as e:
                print(f"[Kitchen] Failed to reset {key}: {e}")


def respawn_robot(env, robot, preset_key: int, reset_objects: bool = False):
    """Reset the environment and place the robot at the given spawn preset."""
    name, x, y, yaw_deg = SPAWN_PRESETS[preset_key]
    yaw = math.radians(yaw_deg)
    default_z = robot.data.default_root_state[0, 2].item()

    pos = torch.tensor([[x, y, default_z]], device=robot.device)
    quat = torch.tensor(
        [[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
        device=robot.device,
    )

    obs, _ = env.reset()
    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))

    if reset_objects and _kitchen_resetter is not None:
        _kitchen_resetter.reset()
        print("[Kitchen] Scene objects reset to defaults")

    cam = CAMERA_PRESETS.get(preset_key)
    if cam is not None:
        env.unwrapped.sim.set_camera_view(eye=cam["eye"], target=cam["target"])

    print(f"\n[Kitchen] Spawned at: {name} ({x:.2f}, {y:.2f}, yaw={yaw_deg} deg)")
    return obs


##############################################################################
# KEYBOARD HANDLER — number keys 1-5 select spawn presets
##############################################################################

class SpawnKeyboardHandler:
    """Listen for number key presses via Omniverse carb.input."""

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._pending_spawn: int | None = None
        self._capture_camera: bool = False
        self._toggle_loop: bool = False
        self._kb_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_key
        )

    def _on_key(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "C":
                self._capture_camera = True
            if event.input.name == "L":
                self._toggle_loop = True
            for n in range(1, len(SPAWN_PRESETS) + 1):
                if event.input.name == f"KEY_{n}":
                    self._pending_spawn = n
        return True

    def poll(self) -> int | None:
        """Return pending spawn preset key (1-5) or None."""
        val = self._pending_spawn
        self._pending_spawn = None
        return val

    def close(self):
        self._input.unsubscribe_to_keyboard_events(
            self._keyboard, self._kb_sub
        )


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

    # Robot spawn -- default position in front of fridge, facing it (+Y direction)
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

    # Long episode for interactive demo (no auto-reset timeout).
    # Resets only happen on keyboard spawn (1-5) or actual falls.
    env_cfg.episode_length_s = 600.0

    # Constrain resets to stay near spawn point with same orientation
    env_cfg.events.reset_base.params = {
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        },
    }

    # Replace contact-based termination with height-based fall detection.
    # Contact termination fires on ANY touch (fridge, counter) which is
    # expected in a kitchen. Height check only fires on actual falls.
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    import isaaclab.envs.mdp as base_mdp

    env_cfg.terminations.base_contact = None
    env_cfg.terminations.fallen_over = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.65},
    )

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
            # Flush stale data from previous sessions
            deleted = self.redis_client.delete(self.key, f"{self.key}:ik_error")
            if deleted:
                print(f"[RedisTeleop] Flushed {deleted} stale key(s) — waiting for fresh publisher data...")
            else:
                print(f"[RedisTeleop] No stale data — waiting for publisher...")
        except ImportError:
            print("[RedisTeleop] ERROR: pip install redis")
            self.redis_client = None
        except Exception as e:
            print(f"[RedisTeleop] ERROR: {e}")
            self.redis_client = None

    def get_target_dof(self) -> np.ndarray | None:
        """Return fresh teleop DOFs, or None if no publisher is active.

        Returns None (not stale data) when the Redis key is absent,
        which happens when the publisher isn't running or its TTL expired.
        This lets the consumer call clear_teleop_targets() and hand full
        control back to the policy.
        """
        if self.redis_client is None:
            return None
        try:
            data = self.redis_client.get(self.key)
            if data is None:
                if self.last_dof is not None:
                    print("\n[RedisTeleop] Publisher gone — clearing teleop")
                    self.last_dof = None
                return None
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

    # Kitchen object resetter — captures initial state of fixtures & objects
    global _kitchen_resetter
    _kitchen_resetter = KitchenObjectResetter(env)

    # Keyboard handler for spawn presets
    kb_handler = SpawnKeyboardHandler()
    active_preset = 1  # default spawn location

    auto_loop = args.auto_loop
    LOOP_INTERVAL = args.loop_interval

    def print_controls(current_preset, looping=False):
        print()
        print("=" * 60)
        print("  Keyboard Controls")
        print("=" * 60)
        for key, (name, x, y, yaw) in SPAWN_PRESETS.items():
            marker = " <-- active" if key == current_preset else ""
            print(f"    {key} = {name:12s} ({x:+.2f}, {y:+.2f}, yaw={yaw:3d} deg){marker}")
        loop_state = " [ON]" if looping else ""
        print(f"    L = Toggle auto-loop (cycle every {LOOP_INTERVAL:.0f}s){loop_state}")
        print("    C = Capture current camera position")
        print("    Ctrl+C = Quit")
        print("=" * 60)
        print()

    print_controls(active_preset, auto_loop)

    obs, _ = env.reset()
    if teleop_source is not None and hasattr(teleop_source, 'reset'):
        teleop_source.reset()

    step = 0
    total_resets = 0
    teleop_active = False
    start_time = time.time()
    target_dt = 0.02  # 50 Hz
    loop_timer = time.time()
    preset_keys = sorted(SPAWN_PRESETS.keys())

    step_times = []
    policy_times = []
    teleop_times = []

    try:
        while simulation_app.is_running():
            loop_start = time.time()

            # Capture current viewport camera position on "C" key
            if kb_handler._capture_camera:
                kb_handler._capture_camera = False
                from pxr import UsdGeom
                stage = env.unwrapped.sim.stage
                cam_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
                if cam_prim.IsValid():
                    xform = UsdGeom.Xformable(cam_prim)
                    world_tf = xform.ComputeLocalToWorldTransform(0)
                    pos = world_tf.ExtractTranslation()
                    # Target = position + forward direction (camera looks down -Z in local frame)
                    fwd = world_tf.TransformDir((0, 0, -1))
                    tgt = pos + fwd * 2.0
                    print(f"\n{'='*60}")
                    print(f"  Current Camera (copy into CAMERA_PRESETS)")
                    print(f"  Preset {active_preset} ({SPAWN_PRESETS[active_preset][0]}):")
                    print(f'    "eye":    ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}),')
                    print(f'    "target": ({tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}),')
                    print(f"{'='*60}")

            # Toggle auto-loop on "L" key
            if kb_handler._toggle_loop:
                kb_handler._toggle_loop = False
                auto_loop = not auto_loop
                loop_timer = time.time()
                print_controls(active_preset, auto_loop)

            # Auto-loop: cycle to next preset every LOOP_INTERVAL seconds
            if auto_loop and (time.time() - loop_timer) >= LOOP_INTERVAL:
                loop_timer = time.time()
                idx = preset_keys.index(active_preset)
                active_preset = preset_keys[(idx + 1) % len(preset_keys)]
                obs = respawn_robot(env, robot, active_preset, reset_objects=True)
                total_resets += 1
                teleop_active = False
                print_controls(active_preset, auto_loop)
                continue

            # Check for keyboard spawn request
            spawn_key = kb_handler.poll()
            if spawn_key is not None:
                active_preset = spawn_key
                obs = respawn_robot(env, robot, active_preset, reset_objects=True)
                total_resets += 1
                teleop_active = False
                print_controls(active_preset, auto_loop)
                continue

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

            # Natural dones (fall, timeout) — respawn at the current
            # active preset so the robot stays at the same location.
            num_dones = dones.sum().item()
            if num_dones > 0:
                total_resets += num_dones
                obs = respawn_robot(env, robot, active_preset)

            # Drift check — respawn if the robot wanders too far from
            # its spawn location (e.g. sliding on furniture, pushed away).
            DRIFT_THRESHOLD = 0.3  # metres
            _, sx, sy, _ = SPAWN_PRESETS[active_preset]
            rpos = robot.data.root_pos_w[0]
            drift = math.sqrt((rpos[0].item() - sx) ** 2 + (rpos[1].item() - sy) ** 2)
            if drift > DRIFT_THRESHOLD:
                print(f"\n[Kitchen] Drift {drift:.2f}m > {DRIFT_THRESHOLD}m — respawning")
                total_resets += 1
                obs = respawn_robot(env, robot, active_preset)

            step += 1

            # Status every 50 steps
            if step % 50 == 0:
                mean_reward = rewards.mean().item()
                root_pos = robot.data.root_pos_w
                mean_height = root_pos[:, 2].mean().item()
                preset_name = SPAWN_PRESETS[active_preset][0]

                progress_str = ""
                if teleop_source is not None and hasattr(teleop_source, 'progress'):
                    progress_str = f" | Motion: {teleop_source.progress*100:.0f}%"

                timing_str = ""
                if args.debug_timing:
                    timing_str = (f" | Policy:{t_policy*1000:.0f}ms "
                                  f"Teleop:{t_teleop*1000:.0f}ms "
                                  f"Step:{t_step*1000:.0f}ms")

                print(f"\r[Play] Step {step} | {preset_name} | Reward: {mean_reward:.3f} | "
                      f"Height: {mean_height:.2f}m | Resets: {int(total_resets)}"
                      f"{progress_str}{timing_str}", end="", flush=True)

            # Real-time throttle
            if not args.no_realtime:
                loop_elapsed = time.time() - loop_start
                if loop_elapsed < target_dt:
                    time.sleep(target_dt - loop_elapsed)

    except KeyboardInterrupt:
        print("\n\n[Play] Stopped by user.")
    finally:
        kb_handler.close()

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

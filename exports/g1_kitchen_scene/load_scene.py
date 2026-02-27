#!/usr/bin/env python3
"""
Isaac Lab launch script for G1 Kitchen Scene (portable).

This script loads the kitchen scene and spawns the G1 robot.
It works from any directory — all asset paths are relative to this file.

Usage:
    # From Isaac Lab:
    isaaclab -p {script_path} load_scene.py

    # Or directly (if isaaclab is on PATH):
    python load_scene.py
"""

import json
import os
import sys

from isaaclab.app import AppLauncher

parser = __import__("argparse").ArgumentParser(description="G1 Kitchen Sim")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--no-robot", action="store_true", help="Load scene without G1 robot")
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.spawners.from_files import UsdFileCfg  # noqa: E402

# All paths relative to THIS script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_USD = os.path.join(SCRIPT_DIR, "scene.usd")
PARAMS_FILE = os.path.join(SCRIPT_DIR, "configs", "scene_params.json")

# Load scene params
with open(PARAMS_FILE) as f:
    PARAMS = json.load(f)

KITCHEN_SCALE = PARAMS.get("kitchen_scale", {}).get("value", 1.0)
FLOOR_Z = PARAMS.get("kitchen_floor_z", {}).get("value", -0.92)
CAM = PARAMS.get("camera", {})
G1 = PARAMS.get("g1_spawn", {})
G1_POS = list(G1.get("position", [0.0, 0.0, 0.0]))
G1_POS[2] = FLOOR_Z * KITCHEN_SCALE + G1.get("pelvis_offset_z", 0.74)


def build_scene():
    """Load kitchen environment and optionally spawn G1 robot."""

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg(
        size=tuple(PARAMS.get("ground_plane_size", {}).get("value", [4.0, 4.0]))
    )
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    # Kitchen scene (all fixtures included via USD references)
    kitchen_cfg = UsdFileCfg(usd_path=SCENE_USD)
    kitchen_cfg.func("/World/Kitchen", kitchen_cfg, translation=(0.0, 0.0, 0.0))

    # Lighting
    dome_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.85))
    dome_cfg.func("/World/DomeLight", dome_cfg)
    dist_cfg = sim_utils.DistantLightCfg(intensity=600.0, color=(1.0, 1.0, 0.95), angle=0.53)
    dist_cfg.func("/World/DistantLight", dist_cfg)

    if args.no_robot:
        print("[OK] Scene loaded (no robot)")
        return None

    # G1 Robot
    try:
        from isaaclab_assets import G1_CFG
        robot_cfg = G1_CFG.replace(prim_path="/World/G1")
        robot_cfg.init_state.pos = tuple(G1_POS)
        robot = Articulation(robot_cfg)
        print(f"[OK] G1 robot spawned at {G1_POS}")
        return robot
    except ImportError:
        print("[WARN] isaaclab_assets.G1_CFG not found — scene loaded without robot.")
        print("  Install: cd $ISAACLAB_PATH && isaaclab -e isaaclab_assets")
        return None


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, render_interval=1)
    sim = SimulationContext(sim_cfg)

    cam_eye = CAM.get("eye", [1.0, 1.0, 0.0])
    cam_target = CAM.get("target", [0.0, 0.0, -0.3])
    cam_speed = CAM.get("move_speed", 0.01)
    sim.set_camera_view(eye=cam_eye, target=cam_target)

    robot = build_scene()

    # Camera speed for small kitchen
    try:
        import carb.settings
        s = carb.settings.get_settings()
        for p in [
            "/persistent/app/viewport/camMoveVelocity",
            "/persistent/app/viewport/camVelocity",
            "/persistent/exts/omni.kit.viewport.window/cameraSpeedMult",
            "/exts/omni.kit.manipulator.camera/flySpeed",
            "/persistent/exts/omni.kit.manipulator.camera/flySpeed",
        ]:
            s.set(p, cam_speed)
        s.set("/app/viewport/grid/enabled", False)
    except Exception:
        pass

    sim.reset()
    print()
    print("=" * 50)
    print("  G1 Kitchen Simulation Running")
    print(f"  Scene: {SCENE_USD}")
    print("  Press Ctrl+C to exit")
    print("=" * 50)

    while simulation_app.is_running():
        if robot is not None:
            robot.write_data_to_sim()
        sim.step()
        if robot is not None:
            robot.update(sim_cfg.dt)
            robot.set_joint_position_target(robot.data.default_joint_pos.clone())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        simulation_app.close()

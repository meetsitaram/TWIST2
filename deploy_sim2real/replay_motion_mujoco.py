#!/usr/bin/env python3
"""
Replay Motion in MuJoCo (no policy, direct PD control to motion targets).

This script directly replays motion from pkl files using PD control,
without any policy inference. Useful for verifying motion data is correct.

Usage:
    python replay_motion_mujoco.py --motion sample_motions/open-doors.pkl
"""

import argparse
import os
import pickle
import sys
import time
from types import ModuleType
import numpy as np

# Patch sys.modules to handle numpy version differences
class _FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)

if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = _FakeModule('numpy._core', np.core if hasattr(np, 'core') else np)
if 'numpy._core.multiarray' not in sys.modules:
    sys.modules['numpy._core.multiarray'] = _FakeModule('numpy._core.multiarray', getattr(np.core, 'multiarray', None))

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("Error: mujoco not installed. Install with: pip install mujoco")
    sys.exit(1)


# MuJoCo default positions (29 DOF) - standing pose
MUJOCO_DEFAULT_POS = np.zeros(29, dtype=np.float32)
MUJOCO_DEFAULT_POS[3] = 0.4    # left_knee
MUJOCO_DEFAULT_POS[4] = -0.2   # left_ankle_pitch
MUJOCO_DEFAULT_POS[9] = 0.4    # right_knee
MUJOCO_DEFAULT_POS[10] = -0.2  # right_ankle_pitch
MUJOCO_DEFAULT_POS[15] = 0.35  # left_shoulder_pitch
MUJOCO_DEFAULT_POS[16] = 0.16  # left_shoulder_roll
MUJOCO_DEFAULT_POS[18] = 0.52  # left_elbow
MUJOCO_DEFAULT_POS[22] = 0.35  # right_shoulder_pitch
MUJOCO_DEFAULT_POS[23] = 0.16  # right_shoulder_roll
MUJOCO_DEFAULT_POS[25] = 0.52  # right_elbow


class MotionReplayer:
    """Replay motion directly with PD control (no policy)."""
    
    def __init__(self, mujoco_model_path: str, motion_file: str):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        
        # Physics settings
        self.model.opt.timestep = 0.001  # 1kHz
        self.control_dt = 0.02  # 50Hz control
        self.steps_per_control = int(self.control_dt / self.model.opt.timestep)
        
        # Load motion
        print(f"Loading motion from {motion_file}...")
        with open(motion_file, 'rb') as f:
            motion = pickle.load(f)
        
        self.fps = motion.get("fps", 50.0)
        self.motion_dt = 1.0 / self.fps
        self.dof_pos = motion["dof_pos"].astype(np.float32)
        self.root_pos = motion["root_pos"].astype(np.float32)
        self.root_rot = motion["root_rot"].astype(np.float32)  # [x, y, z, w]
        self.num_frames = self.dof_pos.shape[0]
        self.duration = self.num_frames / self.fps
        
        print(f"  Frames: {self.num_frames}, Duration: {self.duration:.2f}s, FPS: {self.fps}")
        print(f"  DOF shape: {self.dof_pos.shape}")
        
        # Check if upper-body only motion
        leg_joints = self.dof_pos[:, :12]
        self.is_upper_body_only = np.allclose(leg_joints, 0, atol=0.01)
        if self.is_upper_body_only:
            print("  Detected: UPPER-BODY ONLY motion (legs will use default pose)")
        
        # PD gains (same as real robot)
        self.kp = np.array([
            100, 100, 100, 150, 40, 40,   # left leg
            100, 100, 100, 150, 40, 40,   # right leg
            150, 150, 150,                 # waist
            40, 40, 40, 40, 4.0, 4.0, 4.0, # left arm
            40, 40, 40, 40, 4.0, 4.0, 4.0, # right arm
        ], dtype=np.float32)
        
        self.kd = np.array([
            2, 2, 2, 4, 2, 2,
            2, 2, 2, 4, 2, 2,
            4, 4, 4,
            5, 5, 5, 5, 0.2, 0.2, 0.2,
            5, 5, 5, 5, 0.2, 0.2, 0.2,
        ], dtype=np.float32)
        
        self.torque_limits = np.array([
            100, 100, 100, 150, 40, 40,
            100, 100, 100, 150, 40, 40,
            150, 150, 150,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
        ], dtype=np.float32)
        
        self.motion_time = 0.0
    
    def get_motion_target(self, t: float) -> np.ndarray:
        """Get target joint positions at time t."""
        t = t % self.duration
        
        frame_f = t * self.fps
        frame_0 = int(frame_f)
        frame_1 = min(frame_0 + 1, self.num_frames - 1)
        blend = frame_f - frame_0
        frame_0 = max(0, min(frame_0, self.num_frames - 1))
        
        # Interpolate
        target = (1 - blend) * self.dof_pos[frame_0] + blend * self.dof_pos[frame_1]
        
        # For upper-body only motions, use default leg positions
        if self.is_upper_body_only:
            target[:12] = MUJOCO_DEFAULT_POS[:12]
        
        return target
    
    def reset(self):
        """Reset simulation."""
        mujoco.mj_resetData(self.model, self.data)
        
        self.motion_time = 0.0
        
        # Initialize with default standing pose
        self.data.qpos[0:3] = [0, 0, 1.0]
        self.data.qpos[3:7] = [1, 0, 0, 0]  # wxyz
        
        # Set initial joint positions
        init_target = self.get_motion_target(0.0)
        self.data.qpos[7:7+29] = init_target
        
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        
        print(f"Reset: height={self.data.qpos[2]:.3f}")
    
    def run(self, duration: float = None, speed: float = 1.0, kinematic: bool = True):
        """Run motion replay with viewer.
        
        Args:
            duration: Replay duration in seconds
            speed: Playback speed multiplier  
            kinematic: If True, directly set joint positions (no physics)
        """
        if duration is None:
            duration = self.duration * 2  # 2 loops by default
        
        mode = "KINEMATIC" if kinematic else "PHYSICS"
        print(f"\nReplaying motion for {duration:.1f}s at {speed}x speed ({mode} mode)...")
        print("Press Ctrl+C or close viewer to stop\n")
        
        self.reset()
        
        # For kinematic mode, disable gravity
        if kinematic:
            self.model.opt.gravity[:] = 0
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            frame_count = 0
            display_dt = 1.0 / 60.0  # 60 FPS display
            
            while viewer.is_running() and (time.time() - start_time) < duration:
                t_start = time.time()
                
                # Get current motion target
                target = self.get_motion_target(self.motion_time)
                
                if kinematic:
                    # KINEMATIC: Directly set joint positions (no physics)
                    self.data.qpos[7:7+29] = target
                    self.data.qvel[:] = 0
                    mujoco.mj_forward(self.model, self.data)
                else:
                    # PHYSICS: Use PD control
                    joint_pos = self.data.qpos[7:7+29]
                    joint_vel = self.data.qvel[6:6+29]
                    
                    pos_error = target - joint_pos
                    torque = pos_error * self.kp - joint_vel * self.kd
                    torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                    
                    self.data.ctrl[:29] = torque
                    
                    # Step physics multiple times per display frame
                    for _ in range(self.steps_per_control):
                        mujoco.mj_step(self.model, self.data)
                
                # Advance motion time
                self.motion_time += display_dt * speed
                frame_count += 1
                
                # Print progress periodically
                if frame_count % 60 == 0:
                    progress = (self.motion_time % self.duration) / self.duration * 100
                    print(f"Time: {self.motion_time:.1f}s ({progress:.0f}%) | "
                          f"L_shoulder={target[15]:.2f}, R_shoulder={target[22]:.2f}, "
                          f"L_elbow={target[18]:.2f}, R_elbow={target[25]:.2f}")
                
                # Sync viewer
                viewer.sync()
                
                # Real-time pacing
                elapsed = time.time() - t_start
                sleep_time = display_dt / speed - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        print("\nReplay complete!")


def find_g1_model():
    """Find G1 MuJoCo model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(script_dir, "assets/g1/g1_sim2sim_29dof.xml"),
        os.path.join(script_dir, "../assets/g1/g1_sim2sim_29dof.xml"),
    ]
    for path in paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def find_motion_file(motion_path: str) -> str:
    """Find motion file."""
    if os.path.exists(motion_path):
        return os.path.abspath(motion_path)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(script_dir, motion_path),
        os.path.join(script_dir, "sample_motions", motion_path),
        os.path.join(script_dir, "../datasets/teleop_motions/stage3_upper_body", motion_path),
    ]
    for path in paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def main():
    parser = argparse.ArgumentParser(description="Replay motion in MuJoCo (no policy)")
    parser.add_argument("--motion", type=str, required=True,
                        help="Path to motion pkl file")
    parser.add_argument("--mujoco_model", type=str, default=None,
                        help="Path to MuJoCo XML model")
    parser.add_argument("--duration", type=float, default=None,
                        help="Replay duration (default: 2x motion length)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier")
    parser.add_argument("--physics", action="store_true",
                        help="Use physics simulation with PD control (default: kinematic)")
    
    args = parser.parse_args()
    
    # Find model
    mujoco_model = args.mujoco_model or find_g1_model()
    if not mujoco_model or not os.path.exists(mujoco_model):
        print("Error: Could not find MuJoCo model")
        sys.exit(1)
    
    # Find motion
    motion_file = find_motion_file(args.motion)
    if not motion_file:
        print(f"Error: Motion file not found: {args.motion}")
        sys.exit(1)
    
    print(f"MuJoCo model: {mujoco_model}")
    print(f"Motion file: {motion_file}")
    
    replayer = MotionReplayer(mujoco_model, motion_file)
    replayer.run(duration=args.duration, speed=args.speed, kinematic=not args.physics)


if __name__ == "__main__":
    main()

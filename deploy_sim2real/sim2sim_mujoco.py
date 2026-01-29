#!/usr/bin/env python3
"""
Sim2Sim: Test policy in MuJoCo simulation.

This script loads the trained ONNX policy and runs it in MuJoCo
to verify behavior before deploying to real robot.

Usage:
    python sim2sim_mujoco.py --model policy.onnx --mujoco_model path/to/g1.xml
    python sim2sim_mujoco.py --model policy.onnx  # Uses default G1 model
"""

import argparse
import os
import sys
import time
import numpy as np

try:
    import mujoco
    import mujoco.viewer
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("Warning: mujoco not installed. Install with: pip install mujoco")

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime")

from g1_robot_config import G1RobotConfig


class MuJoCoG1Sim:
    """MuJoCo simulation for G1 robot."""
    
    def __init__(self, model_path: str):
        """
        Initialize MuJoCo simulation.
        
        Args:
            model_path: Path to MuJoCo XML model file
        """
        if not HAS_MUJOCO:
            raise RuntimeError("mujoco is required. Install with: pip install mujoco")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Get joint info
        self.n_joints = self.model.nq
        self.n_actuators = self.model.nu
        
        print(f"MuJoCo model loaded:")
        print(f"  DOFs: {self.n_joints}")
        print(f"  Actuators: {self.n_actuators}")
        
        # Get joint names
        self.joint_names = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_names.append(name)
        
        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.control_dt = 0.02  # 50Hz control
        self.steps_per_control = int(self.control_dt / self.dt)
    
    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
    
    def get_state(self):
        """Get current robot state."""
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'base_pos': self.data.qpos[:3].copy(),
            'base_quat': self.data.qpos[3:7].copy(),
            'joint_pos': self.data.qpos[7:].copy() if self.n_joints > 7 else np.array([]),
            'joint_vel': self.data.qvel[6:].copy() if self.n_joints > 7 else np.array([]),
        }
    
    def get_base_velocity(self):
        """Get base linear and angular velocity in body frame."""
        # World frame velocities
        lin_vel_world = self.data.qvel[:3]
        ang_vel_world = self.data.qvel[3:6]
        
        # Rotate to body frame using quaternion
        quat = self.data.qpos[3:7]
        
        # Simple rotation (approximate)
        # For accurate rotation, use proper quaternion math
        return lin_vel_world, ang_vel_world
    
    def get_projected_gravity(self):
        """Get gravity vector in body frame."""
        # World gravity
        gravity_world = np.array([0, 0, -9.81])
        
        # Rotate to body frame (simplified)
        # For accurate rotation, use quaternion
        quat = self.data.qpos[3:7]
        
        # Approximate: just return world gravity direction
        return gravity_world / np.linalg.norm(gravity_world)
    
    def set_joint_positions(self, positions: np.ndarray):
        """Set joint position targets (for position control)."""
        if len(positions) <= self.n_actuators:
            self.data.ctrl[:len(positions)] = positions
    
    def step(self, n_steps: int = 1):
        """Step simulation."""
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)


class PolicyRunner:
    """Run policy in MuJoCo simulation."""
    
    def __init__(self, policy_path: str, mujoco_model_path: str):
        """
        Initialize policy runner.
        
        Args:
            policy_path: Path to ONNX policy file
            mujoco_model_path: Path to MuJoCo XML model file
        """
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime is required")
        
        # Load policy
        self.session = ort.InferenceSession(policy_path)
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        self.obs_dim = input_info.shape[1]
        self.act_dim = output_info.shape[1]
        
        print(f"Policy loaded: obs={self.obs_dim}, act={self.act_dim}")
        
        # Load simulation
        self.sim = MuJoCoG1Sim(mujoco_model_path)
        
        # State
        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        
        # Default positions
        self.default_pos = np.zeros(29, dtype=np.float32)
        self.default_pos[3] = 0.4   # left_knee
        self.default_pos[9] = 0.4   # right_knee
        self.default_pos[4] = -0.2  # left_ankle_pitch
        self.default_pos[10] = -0.2 # right_ankle_pitch
        
        # Action scale
        self.action_scale = 0.5
    
    def build_observation(self, target_joints_mujoco: np.ndarray = None):
        """Build observation from current simulation state."""
        
        state = self.sim.get_state()
        lin_vel, ang_vel = self.sim.get_base_velocity()
        gravity = self.sim.get_projected_gravity()
        
        # Current joint state (MuJoCo 29 DOF)
        joint_pos = state['joint_pos'][:29] if len(state['joint_pos']) >= 29 else \
                    np.pad(state['joint_pos'], (0, 29 - len(state['joint_pos'])))
        joint_vel = state['joint_vel'][:29] if len(state['joint_vel']) >= 29 else \
                    np.pad(state['joint_vel'], (0, 29 - len(state['joint_vel'])))
        
        # Target from teleop (or default if none)
        if target_joints_mujoco is None:
            target_joints_mujoco = self.default_pos
        
        # Key body positions (placeholder - would need forward kinematics)
        keybody_pos = np.zeros(12, dtype=np.float32)
        
        # Build observation (may need adjustment based on actual obs structure)
        # This is a simplified version - actual structure depends on training config
        obs_parts = [
            lin_vel.astype(np.float32),      # 3
            ang_vel.astype(np.float32),       # 3
            gravity.astype(np.float32),       # 3
        ]
        
        # Pad to match expected observation dimension
        obs = np.concatenate(obs_parts)
        
        # Pad to full observation size
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        
        return obs[:self.obs_dim].astype(np.float32)
    
    def forward(self, observation: np.ndarray) -> np.ndarray:
        """Run policy forward pass."""
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        outputs = self.session.run(
            None,
            {'observation': observation.astype(np.float32)}
        )
        
        action = outputs[0].squeeze()
        self.last_action = action.copy()
        
        return action
    
    def action_to_mujoco_positions(self, action: np.ndarray) -> np.ndarray:
        """Convert policy action to MuJoCo joint positions."""
        
        # Action is in Isaac Lab order (37 DOF)
        # Need to convert to MuJoCo order (29 DOF)
        
        # For now, just use first 29 elements (simplified)
        # Real implementation should use G1RobotConfig.remap_isaaclab_to_mujoco
        
        joint_targets = self.default_pos + action[:29] * self.action_scale
        
        return joint_targets
    
    def run_interactive(self, duration: float = 30.0):
        """Run interactive simulation with viewer."""
        
        print("\nStarting interactive simulation...")
        print("Press Ctrl+C to stop\n")
        
        self.sim.reset()
        
        with mujoco.viewer.launch_passive(self.sim.model, self.sim.data) as viewer:
            start_time = time.time()
            step_count = 0
            
            while viewer.is_running() and (time.time() - start_time) < duration:
                # Build observation
                obs = self.build_observation()
                
                # Run policy
                action = self.forward(obs)
                
                # Convert to joint positions
                joint_targets = self.action_to_mujoco_positions(action)
                
                # Apply control
                self.sim.set_joint_positions(joint_targets)
                
                # Step simulation
                self.sim.step(self.sim.steps_per_control)
                
                # Update viewer
                viewer.sync()
                
                step_count += 1
                
                # Print status
                if step_count % 50 == 0:
                    state = self.sim.get_state()
                    print(f"Step {step_count}: height={state['base_pos'][2]:.3f}m")
        
        print(f"\nSimulation completed: {step_count} steps")


def find_g1_model():
    """Try to find G1 MuJoCo model in common locations."""
    
    search_paths = [
        # Relative to this script
        "../mujoco_models/g1.xml",
        "../mujoco_models/g1/g1.xml",
        "../../mujoco_models/g1.xml",
        # Absolute paths
        os.path.expanduser("~/models/g1/g1.xml"),
        "/opt/unitree/g1/g1.xml",
    ]
    
    for path in search_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim: Test policy in MuJoCo")
    parser.add_argument("--model", type=str, default="policy.onnx",
                        help="Path to ONNX policy model")
    parser.add_argument("--mujoco_model", type=str, default=None,
                        help="Path to MuJoCo XML model")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Simulation duration in seconds")
    
    args = parser.parse_args()
    
    # Check for required packages
    if not HAS_MUJOCO:
        print("Error: mujoco is required. Install with: pip install mujoco")
        sys.exit(1)
    
    if not HAS_ONNX:
        print("Error: onnxruntime is required. Install with: pip install onnxruntime")
        sys.exit(1)
    
    # Find MuJoCo model
    if args.mujoco_model is None:
        args.mujoco_model = find_g1_model()
        if args.mujoco_model is None:
            print("Error: Could not find G1 MuJoCo model.")
            print("Please specify with --mujoco_model path/to/g1.xml")
            sys.exit(1)
    
    if not os.path.exists(args.mujoco_model):
        print(f"Error: MuJoCo model not found: {args.mujoco_model}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Policy model not found: {args.model}")
        print("Export model first with: python export_onnx.py --checkpoint path/to/model.pt")
        sys.exit(1)
    
    # Run simulation
    runner = PolicyRunner(args.model, args.mujoco_model)
    runner.run_interactive(args.duration)


if __name__ == "__main__":
    main()

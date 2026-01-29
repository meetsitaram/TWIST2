#!/usr/bin/env python3
"""
Minimal inference example for G1 whole-body teleop policy.

This script demonstrates how to:
1. Load the ONNX model
2. Build observations
3. Run inference
4. Interpret actions

Usage:
    python inference_example.py --model policy.onnx
"""

import argparse
import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime")

from g1_robot_config import G1RobotConfig


class G1TeleopPolicy:
    """Wrapper for running G1 whole-body teleop policy inference."""
    
    # Observation dimensions
    OBS_DIM = 195
    ACT_DIM = 37
    
    # Observation breakdown
    OBS_BASE_LIN_VEL = 3
    OBS_BASE_ANG_VEL = 3
    OBS_PROJECTED_GRAVITY = 3
    OBS_JOINT_POS = 37
    OBS_JOINT_VEL = 37
    OBS_LAST_ACTION = 37
    OBS_TARGET_JOINT_POS = 29  # MuJoCo order
    OBS_TARGET_KEYBODY_POS = 12  # 4 bodies x 3D
    # Total: 3 + 3 + 3 + 37 + 37 + 37 + 29 + 12 = 161 (adjust based on actual)
    
    # Action scale (from training config)
    ACTION_SCALE = 0.5
    
    def __init__(self, model_path: str):
        """
        Initialize policy.
        
        Args:
            model_path: Path to ONNX model file
        """
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime is required. Install with: pip install onnxruntime")
        
        self.session = ort.InferenceSession(model_path)
        
        # Get actual input/output shapes from model
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        
        self.obs_dim = input_info.shape[1] if len(input_info.shape) > 1 else self.OBS_DIM
        self.act_dim = output_info.shape[1] if len(output_info.shape) > 1 else self.ACT_DIM
        
        print(f"Loaded policy: obs_dim={self.obs_dim}, act_dim={self.act_dim}")
        
        # Initialize last action buffer
        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        
        # Default joint positions (Isaac Lab order)
        self.default_joint_pos = self._get_default_positions()
    
    def _get_default_positions(self) -> np.ndarray:
        """Get default joint positions."""
        # Simplified defaults - adjust based on your robot
        defaults = np.zeros(self.act_dim, dtype=np.float32)
        
        # Knees slightly bent
        defaults[3] = 0.4   # left_knee
        defaults[9] = 0.4   # right_knee
        
        # Ankles pitched
        defaults[4] = -0.2  # left_ankle_pitch
        defaults[10] = -0.2 # right_ankle_pitch
        
        # Shoulders
        defaults[13] = 0.35  # left_shoulder_pitch
        defaults[14] = 0.16  # left_shoulder_roll
        defaults[17] = 0.52  # left_elbow
        
        # Right arm (indices may vary)
        # defaults[...] = ...
        
        return defaults
    
    def build_observation(
        self,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        target_joint_pos_mujoco: np.ndarray,
        target_keybody_pos: np.ndarray,
        isaaclab_joint_names: list = None,
    ) -> np.ndarray:
        """
        Build observation vector.
        
        Args:
            base_lin_vel: (3,) base linear velocity in body frame
            base_ang_vel: (3,) base angular velocity in body frame
            projected_gravity: (3,) gravity vector in body frame
            joint_pos: (37,) current joint positions (relative to default)
            joint_vel: (37,) current joint velocities
            target_joint_pos_mujoco: (29,) teleop target positions in MuJoCo order
            target_keybody_pos: (12,) target key body positions (local frame)
            isaaclab_joint_names: List of Isaac Lab joint names (for remapping)
            
        Returns:
            observation: (obs_dim,) observation vector
        """
        obs = np.concatenate([
            base_lin_vel.astype(np.float32),
            base_ang_vel.astype(np.float32),
            projected_gravity.astype(np.float32),
            joint_pos.astype(np.float32),
            joint_vel.astype(np.float32),
            self.last_action,
            target_joint_pos_mujoco.astype(np.float32),
            target_keybody_pos.astype(np.float32),
        ])
        
        return obs
    
    def forward(self, observation: np.ndarray) -> np.ndarray:
        """
        Run policy inference.
        
        Args:
            observation: (obs_dim,) or (batch, obs_dim) observation
            
        Returns:
            action: (act_dim,) or (batch, act_dim) action (joint position deltas)
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        outputs = self.session.run(
            None, 
            {'observation': observation.astype(np.float32)}
        )
        
        action = outputs[0].squeeze()
        
        # Update last action buffer
        self.last_action = action.copy()
        
        return action
    
    def action_to_joint_positions(self, action: np.ndarray) -> np.ndarray:
        """
        Convert action (deltas) to absolute joint positions.
        
        Args:
            action: (act_dim,) action from policy
            
        Returns:
            joint_positions: (act_dim,) absolute joint positions
        """
        return self.default_joint_pos + (action * self.ACTION_SCALE)
    
    def reset(self):
        """Reset internal state."""
        self.last_action = np.zeros(self.act_dim, dtype=np.float32)


def demo_inference():
    """Demonstrate inference with dummy data."""
    
    print("\n" + "="*60)
    print("G1 Whole-Body Teleop Policy - Inference Demo")
    print("="*60 + "\n")
    
    # Create policy (will fail without model file, but shows structure)
    print("Observation Structure:")
    print("-" * 40)
    print(f"  base_lin_vel:         3 dims")
    print(f"  base_ang_vel:         3 dims")
    print(f"  projected_gravity:    3 dims")
    print(f"  joint_pos:           37 dims")
    print(f"  joint_vel:           37 dims")
    print(f"  last_action:         37 dims")
    print(f"  target_joint_pos:    29 dims (MuJoCo order)")
    print(f"  target_keybody_pos:  12 dims")
    print(f"  --------------------------------")
    print(f"  TOTAL:              161+ dims\n")
    
    print("Action Structure:")
    print("-" * 40)
    print(f"  action:              37 dims (joint position deltas)")
    print(f"  target = default + action * 0.5\n")
    
    print("Joint Mapping (MuJoCo 29 -> Isaac Lab 37):")
    print("-" * 40)
    for i, name in enumerate(G1RobotConfig.MUJOCO_JOINT_ORDER[:10]):
        il_name = G1RobotConfig.MUJOCO_TO_ISAACLAB_NAMES.get(name, "N/A")
        print(f"  [{i:2d}] {name:30s} -> {il_name}")
    print("  ... (see g1_robot_config.py for full mapping)")


def main():
    parser = argparse.ArgumentParser(description="G1 policy inference example")
    parser.add_argument("--model", type=str, default="policy.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo without model file")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_inference()
        return
    
    # Load and run policy
    try:
        policy = G1TeleopPolicy(args.model)
        
        # Create dummy observation
        obs = np.random.randn(policy.obs_dim).astype(np.float32)
        
        # Run inference
        action = policy.forward(obs)
        
        print(f"\nInference successful!")
        print(f"  Input shape:  {obs.shape}")
        print(f"  Output shape: {action.shape}")
        print(f"  Action sample: {action[:5]}...")
        
        # Convert to joint positions
        joint_pos = policy.action_to_joint_positions(action)
        print(f"  Joint pos sample: {joint_pos[:5]}...")
        
    except FileNotFoundError:
        print(f"Model not found: {args.model}")
        print("Run with --demo to see structure, or export model first:")
        print("  python export_onnx.py --checkpoint path/to/model.pt")


if __name__ == "__main__":
    main()

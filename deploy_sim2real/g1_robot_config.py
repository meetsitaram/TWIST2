"""
G1 Robot Configuration - Single Source of Truth for Sim2Real

This module provides a centralized configuration for the Unitree G1 robot
that handles joint mappings between different representations:
- MuJoCo (29 DOFs) - used for teleop input
- Isaac Lab (37 DOFs) - used for policy training
- Real Robot (varies by SDK)

Usage:
    from g1_robot_config import G1RobotConfig
    
    # Get MuJoCo joint order
    mj_joints = G1RobotConfig.MUJOCO_JOINT_ORDER
    
    # Convert DOFs from MuJoCo to Isaac Lab order
    il_dof = G1RobotConfig.remap_mujoco_to_isaaclab(mj_dof, il_joint_names)
    
    # Get upper body indices for either system
    mj_upper = G1RobotConfig.MUJOCO_UPPER_BODY_INDICES
    il_upper = G1RobotConfig.get_isaaclab_upper_body_indices(il_joint_names)
"""

import numpy as np
from typing import Dict, List, Optional, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class G1RobotConfig:
    """Centralized configuration for Unitree G1 robot joint mappings.
    
    MuJoCo 29-DOF Model Joint Order:
        0-5:   Left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
        6-11:  Right leg (same order)
        12-14: Waist (yaw, roll, pitch)
        15-21: Left arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
        22-28: Right arm (same order)
    
    Isaac Lab G1 Model (37 DOFs):
        Different joint ordering, different naming for some joints.
        No wrist pitch/yaw, only 1 waist DOF (torso_joint).
    """
    
    # =========================================================================
    # SIMULATION CONFIGURATION
    # =========================================================================
    
    # Initial robot height
    # With hip_pitch=0 (neutral stance), robot is taller than MuJoCo keyframe
    # Adjusted to have feet touch ground without sinking
    INIT_HEIGHT = 0.82  # meters
    
    # Control parameters (from Isaac Lab velocity_env_cfg)
    ACTION_SCALE = 0.5  # action = delta_pos, target = default + action * scale
    CONTROL_DT = 0.02   # 50Hz control frequency
    
    # Observation scaling - Isaac Lab uses RAW values (no scaling)
    # This differs from legged_gym which used:
    #   ang_vel_scale = 0.25, dof_vel_scale = 0.05
    # In Isaac Lab, observations have noise added but no scaling factors
    OBS_SCALES = {
        "lin_vel": 1.0,      # No scaling
        "ang_vel": 1.0,      # No scaling (legged_gym used 0.25)
        "joint_pos": 1.0,    # No scaling
        "joint_vel": 1.0,    # No scaling (legged_gym used 0.05)
    }
    
    # Key bodies for observation (from isaaclab_envs/g1_motion_mimic_env_cfg.py)
    # These are used for target_keybody_pos observation (7 bodies × 3 = 21 dims)
    KEY_BODIES = [
        "left_ankle_roll_link", "right_ankle_roll_link",  # Feet
        "left_elbow_pitch_link", "right_elbow_pitch_link",  # Elbows
        "left_shoulder_pitch_link", "right_shoulder_pitch_link",  # Shoulders
        "torso_link",  # Torso
    ]
    
    # =========================================================================
    # MUJOCO JOINT CONFIGURATION (29 DOFs)
    # =========================================================================
    
    MUJOCO_JOINT_ORDER = [
        # Left leg (0-5)
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        # Right leg (6-11)
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        # Waist (12-14)
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        # Left arm (15-21)
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        # Right arm (22-28)
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    
    MUJOCO_NUM_JOINTS = 29
    
    # MuJoCo joint index ranges
    MUJOCO_LEFT_LEG_INDICES = list(range(0, 6))
    MUJOCO_RIGHT_LEG_INDICES = list(range(6, 12))
    MUJOCO_WAIST_INDICES = list(range(12, 15))
    MUJOCO_LEFT_ARM_INDICES = list(range(15, 22))
    MUJOCO_RIGHT_ARM_INDICES = list(range(22, 29))
    MUJOCO_UPPER_BODY_INDICES = list(range(15, 29))  # Both arms
    MUJOCO_LOWER_BODY_INDICES = list(range(0, 12))   # Both legs
    
    # =========================================================================
    # ISAAC LAB JOINT CONFIGURATION (37 DOFs typical)
    # =========================================================================
    
    # Typical Isaac Lab G1 joint order (may vary by URDF version)
    ISAACLAB_JOINT_ORDER_TYPICAL = [
        # Left leg (0-5)
        "left_hip_pitch_joint",
        "left_hip_roll_joint", 
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        # Right leg (6-11)
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint", 
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        # Torso (12)
        "torso_joint",
        # Left arm (13-17)
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint",
        # Left hand joints (18-26) - varies
        # Right arm (27-31)
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
        # Right hand joints (32-36) - varies
    ]
    
    # =========================================================================
    # MUJOCO TO ISAAC LAB JOINT NAME MAPPING
    # =========================================================================
    
    # Maps MuJoCo joint names to Isaac Lab joint names
    # None means the joint doesn't exist in Isaac Lab
    MUJOCO_TO_ISAACLAB_NAMES = {
        # Left leg - same names
        "left_hip_pitch_joint": "left_hip_pitch_joint",
        "left_hip_roll_joint": "left_hip_roll_joint",
        "left_hip_yaw_joint": "left_hip_yaw_joint",
        "left_knee_joint": "left_knee_joint",
        "left_ankle_pitch_joint": "left_ankle_pitch_joint",
        "left_ankle_roll_joint": "left_ankle_roll_joint",
        # Right leg - same names
        "right_hip_pitch_joint": "right_hip_pitch_joint",
        "right_hip_roll_joint": "right_hip_roll_joint",
        "right_hip_yaw_joint": "right_hip_yaw_joint",
        "right_knee_joint": "right_knee_joint",
        "right_ankle_pitch_joint": "right_ankle_pitch_joint",
        "right_ankle_roll_joint": "right_ankle_roll_joint",
        # Waist - only yaw exists in Isaac Lab (as torso_joint)
        "waist_yaw_joint": "torso_joint",
        "waist_roll_joint": None,  # NO EQUIVALENT
        "waist_pitch_joint": None,  # NO EQUIVALENT
        # Left arm - elbow and wrist have different names
        "left_shoulder_pitch_joint": "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint": "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint": "left_shoulder_yaw_joint",
        "left_elbow_joint": "left_elbow_pitch_joint",
        "left_wrist_roll_joint": "left_elbow_roll_joint",
        "left_wrist_pitch_joint": None,  # NO EQUIVALENT
        "left_wrist_yaw_joint": None,  # NO EQUIVALENT
        # Right arm - elbow and wrist have different names
        "right_shoulder_pitch_joint": "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint": "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint": "right_shoulder_yaw_joint",
        "right_elbow_joint": "right_elbow_pitch_joint",
        "right_wrist_roll_joint": "right_elbow_roll_joint",
        "right_wrist_pitch_joint": None,  # NO EQUIVALENT
        "right_wrist_yaw_joint": None,  # NO EQUIVALENT
    }
    
    # Reverse mapping for action conversion
    ISAACLAB_TO_MUJOCO_NAMES = {
        v: k for k, v in MUJOCO_TO_ISAACLAB_NAMES.items() if v is not None
    }
    
    # =========================================================================
    # DEFAULT JOINT POSITIONS
    # =========================================================================
    
    # Default positions for Isaac Lab (used as action reference)
    # These define the "zero action" pose that actions are relative to
    # Legs: neutral upright stance (hip_pitch=0 for feet under body)
    # Arms: from MuJoCo keyframe (elbows bent, hands forward)
    ISAACLAB_DEFAULT_POSITIONS = {
        # Legs - neutral upright stance
        "left_hip_pitch_joint": 0.0,   # Neutral (not leaning forward)
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.4,
        "left_ankle_pitch_joint": -0.2,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": 0.0,  # Neutral (not leaning forward)
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.4,
        "right_ankle_pitch_joint": -0.2,
        "right_ankle_roll_joint": 0.0,
        # Waist
        "torso_joint": 0.0,
        # Arms - relaxed position
        "left_shoulder_pitch_joint": 0.35,
        "left_shoulder_roll_joint": 0.16,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_pitch_joint": 0.52,
        "left_elbow_roll_joint": 0.0,
        "right_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": 0.16,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_pitch_joint": 0.52,
        "right_elbow_roll_joint": 0.0,
    }
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    @classmethod
    def get_mujoco_index(cls, joint_name: str) -> Optional[int]:
        """Get MuJoCo index for a joint name."""
        try:
            return cls.MUJOCO_JOINT_ORDER.index(joint_name)
        except ValueError:
            return None
    
    @classmethod
    def build_mujoco_to_isaaclab_mapping(cls, isaaclab_joint_names: List[str]) -> Dict[int, Optional[int]]:
        """Build index mapping from MuJoCo to Isaac Lab.
        
        Args:
            isaaclab_joint_names: List of joint names from Isaac Lab robot
            
        Returns:
            Dict mapping MuJoCo index -> Isaac Lab index (or None if no equivalent)
        """
        # Build name -> index lookup for Isaac Lab
        il_name_to_idx = {name: idx for idx, name in enumerate(isaaclab_joint_names)}
        
        mapping = {}
        for mj_idx, mj_name in enumerate(cls.MUJOCO_JOINT_ORDER):
            il_name = cls.MUJOCO_TO_ISAACLAB_NAMES.get(mj_name)
            if il_name is not None and il_name in il_name_to_idx:
                mapping[mj_idx] = il_name_to_idx[il_name]
            else:
                mapping[mj_idx] = None
        
        return mapping
    
    @classmethod
    def build_isaaclab_to_mujoco_mapping(cls, isaaclab_joint_names: List[str]) -> Dict[int, Optional[int]]:
        """Build index mapping from Isaac Lab to MuJoCo.
        
        Args:
            isaaclab_joint_names: List of joint names from Isaac Lab robot
            
        Returns:
            Dict mapping Isaac Lab index -> MuJoCo index (or None if no equivalent)
        """
        mj_to_il = cls.build_mujoco_to_isaaclab_mapping(isaaclab_joint_names)
        
        # Invert the mapping
        il_to_mj = {}
        for mj_idx, il_idx in mj_to_il.items():
            if il_idx is not None:
                il_to_mj[il_idx] = mj_idx
        
        # Fill in None for unmapped Isaac Lab indices
        for il_idx in range(len(isaaclab_joint_names)):
            if il_idx not in il_to_mj:
                il_to_mj[il_idx] = None
        
        return il_to_mj
    
    @classmethod
    def get_isaaclab_upper_body_indices(cls, isaaclab_joint_names: List[str]) -> List[int]:
        """Get upper body joint indices in Isaac Lab order."""
        upper_body_patterns = [
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
            "elbow_pitch", "elbow_roll",
        ]
        
        indices = []
        for idx, name in enumerate(isaaclab_joint_names):
            if any(pattern in name for pattern in upper_body_patterns):
                indices.append(idx)
        
        return sorted(indices)
    
    @classmethod
    def get_isaaclab_lower_body_indices(cls, isaaclab_joint_names: List[str]) -> List[int]:
        """Get lower body joint indices in Isaac Lab order."""
        lower_body_patterns = [
            "hip_pitch", "hip_roll", "hip_yaw",
            "knee", "ankle_pitch", "ankle_roll",
        ]
        
        indices = []
        for idx, name in enumerate(isaaclab_joint_names):
            if any(pattern in name for pattern in lower_body_patterns):
                indices.append(idx)
        
        return sorted(indices)
    
    @classmethod
    def remap_mujoco_to_isaaclab_numpy(
        cls,
        mujoco_dof: np.ndarray,
        isaaclab_joint_names: List[str],
        default_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Remap DOF positions from MuJoCo order to Isaac Lab order.
        
        Args:
            mujoco_dof: Joint positions in MuJoCo order, shape (29,) or (N, 29)
            isaaclab_joint_names: List of Isaac Lab joint names
            default_positions: Default positions for unmapped joints (optional)
            
        Returns:
            Joint positions in Isaac Lab order
        """
        mapping = cls.build_mujoco_to_isaaclab_mapping(isaaclab_joint_names)
        num_il_joints = len(isaaclab_joint_names)
        
        # Handle 1D or 2D input
        if mujoco_dof.ndim == 1:
            mujoco_dof = mujoco_dof.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = mujoco_dof.shape[0]
        
        # Initialize with defaults or zeros
        if default_positions is not None:
            isaaclab_dof = np.tile(default_positions, (batch_size, 1))
        else:
            isaaclab_dof = np.zeros((batch_size, num_il_joints), dtype=mujoco_dof.dtype)
        
        # Apply mapping
        for mj_idx, il_idx in mapping.items():
            if il_idx is not None and mj_idx < mujoco_dof.shape[1]:
                isaaclab_dof[:, il_idx] = mujoco_dof[:, mj_idx]
        
        if squeeze_output:
            isaaclab_dof = isaaclab_dof.squeeze(0)
        
        return isaaclab_dof
    
    @classmethod
    def remap_isaaclab_to_mujoco_numpy(
        cls,
        isaaclab_dof: np.ndarray,
        isaaclab_joint_names: List[str],
        default_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Remap DOF positions from Isaac Lab order to MuJoCo order.
        
        Args:
            isaaclab_dof: Joint positions in Isaac Lab order
            isaaclab_joint_names: List of Isaac Lab joint names
            default_positions: Default positions for unmapped MuJoCo joints
            
        Returns:
            Joint positions in MuJoCo order (29,) or (N, 29)
        """
        mapping = cls.build_mujoco_to_isaaclab_mapping(isaaclab_joint_names)
        
        # Handle 1D or 2D input
        if isaaclab_dof.ndim == 1:
            isaaclab_dof = isaaclab_dof.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = isaaclab_dof.shape[0]
        
        # Initialize with defaults or zeros
        if default_positions is not None:
            mujoco_dof = np.tile(default_positions, (batch_size, 1))
        else:
            mujoco_dof = np.zeros((batch_size, cls.MUJOCO_NUM_JOINTS), dtype=isaaclab_dof.dtype)
        
        # Apply reverse mapping
        for mj_idx, il_idx in mapping.items():
            if il_idx is not None and il_idx < isaaclab_dof.shape[1]:
                mujoco_dof[:, mj_idx] = isaaclab_dof[:, il_idx]
        
        if squeeze_output:
            mujoco_dof = mujoco_dof.squeeze(0)
        
        return mujoco_dof
    
    @classmethod
    def print_mapping(cls, isaaclab_joint_names: List[str]):
        """Print the full joint mapping for debugging."""
        mapping = cls.build_mujoco_to_isaaclab_mapping(isaaclab_joint_names)
        
        print("\nG1 Robot Joint Mapping (MuJoCo -> Isaac Lab):")
        print("=" * 70)
        print(f"{'MJ_Idx':<8} {'MuJoCo Name':<30} {'IL_Idx':<8} {'Isaac Lab Name':<30}")
        print("-" * 70)
        
        for mj_idx, mj_name in enumerate(cls.MUJOCO_JOINT_ORDER):
            il_idx = mapping.get(mj_idx)
            if il_idx is not None:
                il_name = isaaclab_joint_names[il_idx]
                print(f"{mj_idx:<8} {mj_name:<30} {il_idx:<8} {il_name:<30}")
            else:
                print(f"{mj_idx:<8} {mj_name:<30} {'(NONE)':<8} {'NO EQUIVALENT':<30}")
        
        mapped_count = sum(1 for v in mapping.values() if v is not None)
        print("-" * 70)
        print(f"Mapped: {mapped_count}/{cls.MUJOCO_NUM_JOINTS} joints")
        print("=" * 70)


# Convenience aliases
remap_mj_to_il = G1RobotConfig.remap_mujoco_to_isaaclab_numpy
remap_il_to_mj = G1RobotConfig.remap_isaaclab_to_mujoco_numpy


if __name__ == "__main__":
    # Print mapping with typical Isaac Lab joint order
    G1RobotConfig.print_mapping(G1RobotConfig.ISAACLAB_JOINT_ORDER_TYPICAL[:18])

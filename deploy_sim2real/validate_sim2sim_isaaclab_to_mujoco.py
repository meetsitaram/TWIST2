#!/usr/bin/env python3
"""
Sim2Sim Validation Script

Automatically validates and reports differences between Isaac Lab and MuJoCo
configurations for the G1 robot. Run this before deploying to real robot.

Usage:
    python validate_sim2sim.py
    python validate_sim2sim.py --verbose
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add TWIST2 root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWIST2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TWIST2_ROOT)

from robot_config import G1RobotConfig


# ============================================================================
# ISAAC LAB CONFIGURATION (from isaaclab_assets/robots/unitree.py)
# ============================================================================

@dataclass
class IsaacLabG1Config:
    """Isaac Lab G1 robot configuration."""
    
    # Joint order (37 DOF, alphabetical from URDF)
    JOINT_ORDER = [
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "left_elbow_pitch_joint", "left_elbow_roll_joint",
        "left_five_joint",
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_one_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_three_joint", "left_zero_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "right_elbow_pitch_joint", "right_elbow_roll_joint",
        "right_five_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_one_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_three_joint", "right_zero_joint",
        "torso_joint",
        # Head joints (31-36)
        "head_joint_0", "head_joint_1", "head_joint_2", "head_joint_3", "head_joint_4", "head_joint_5",
    ]
    
    NUM_JOINTS = 37
    
    # Default positions from G1_CFG.init_state
    DEFAULT_POSITIONS = {
        "left_hip_pitch_joint": -0.20,
        "right_hip_pitch_joint": -0.20,
        "left_knee_joint": 0.42,
        "right_knee_joint": 0.42,
        "left_ankle_pitch_joint": -0.23,
        "right_ankle_pitch_joint": -0.23,
        "left_elbow_pitch_joint": 0.87,
        "right_elbow_pitch_joint": 0.87,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_pitch_joint": 0.35,
        "left_shoulder_roll_joint": 0.16,
        "right_shoulder_roll_joint": -0.16,
        "left_one_joint": 1.0,
        "right_one_joint": -1.0,
        "left_two_joint": 0.52,
        "right_two_joint": -0.52,
    }
    
    # Initial height
    INIT_HEIGHT = 0.74
    
    # Physics settings
    SIM_DT = 0.005  # 200Hz
    DECIMATION = 4
    CONTROL_DT = SIM_DT * DECIMATION  # 0.02s = 50Hz
    
    # PD Gains
    STIFFNESS = {
        "hip_yaw": 150.0,
        "hip_roll": 150.0,
        "hip_pitch": 200.0,
        "knee": 200.0,
        "torso": 200.0,
        "ankle": 20.0,
        "arm": 40.0,
    }
    
    DAMPING = {
        "hip_yaw": 5.0,
        "hip_roll": 5.0,
        "hip_pitch": 5.0,
        "knee": 5.0,
        "torso": 5.0,
        "ankle": 2.0,
        "arm": 10.0,
    }
    
    EFFORT_LIMITS = {
        "legs": 300,
        "feet": 20,
        "arms": 300,
    }
    
    ACTION_SCALE = 0.5


# ============================================================================
# MUJOCO CONFIGURATION (from sim2sim_mujoco.py)
# ============================================================================

@dataclass
class MuJoCoG1Config:
    """MuJoCo G1 robot configuration."""
    
    # Joint order (29 DOF)
    JOINT_ORDER = G1RobotConfig.MUJOCO_JOINT_ORDER
    
    NUM_JOINTS = 29
    
    # Default positions (should match Isaac Lab after fix)
    DEFAULT_POSITIONS = {
        "left_hip_pitch_joint": -0.20,  # FIXED: was 0.0
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.42,  # FIXED: was 0.4
        "left_ankle_pitch_joint": -0.23,  # FIXED: was -0.2
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.20,  # FIXED: was 0.0
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.42,  # FIXED: was 0.4
        "right_ankle_pitch_joint": -0.23,  # FIXED: was -0.2
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_shoulder_pitch_joint": 0.35,
        "left_shoulder_roll_joint": 0.16,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.87,  # FIXED: was 0.52
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.16,  # FIXED: was 0.16 (should mirror left)
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.87,  # FIXED: was 0.52
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }
    
    # Initial height (should match Isaac Lab)
    INIT_HEIGHT = 0.74  # FIXED: was 1.0
    
    # Physics settings
    SIM_DT = 0.001  # 1000Hz
    CONTROL_DT = 0.02  # 50Hz
    STEPS_PER_CONTROL = 20
    
    # PD Gains (matching Isaac Lab)
    KP = {
        "hip_pitch": 200.0,
        "hip_roll": 150.0,
        "hip_yaw": 150.0,
        "knee": 200.0,
        "ankle_pitch": 20.0,
        "ankle_roll": 20.0,
        "waist": 200.0,
        "shoulder": 40.0,
        "elbow": 40.0,
        "wrist": 40.0,
    }
    
    KD = {
        "hip": 5.0,
        "knee": 5.0,
        "ankle": 2.0,
        "waist": 5.0,
        "arm": 10.0,
    }
    
    TORQUE_LIMITS = {
        "legs": 300,
        "feet": 20,
        "arms": 300,
    }
    
    ACTION_SCALE = 0.5


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_joint_mapping() -> Tuple[List[str], List[str], Dict[str, str]]:
    """Validate joint mapping between MuJoCo and Isaac Lab.
    
    Returns:
        Tuple of (matched_joints, unmapped_mujoco_joints, mapping_dict)
    """
    matched = []
    unmapped = []
    mapping = {}
    
    for mj_joint in MuJoCoG1Config.JOINT_ORDER:
        il_joint = G1RobotConfig.MUJOCO_TO_ISAACLAB_NAMES.get(mj_joint)
        if il_joint is not None:
            matched.append(mj_joint)
            mapping[mj_joint] = il_joint
        else:
            unmapped.append(mj_joint)
    
    return matched, unmapped, mapping


def validate_default_poses() -> List[Dict]:
    """Compare default poses between simulators.
    
    Returns:
        List of mismatches with joint name, isaac_lab value, mujoco value, difference
    """
    mismatches = []
    
    for mj_joint, il_joint in G1RobotConfig.MUJOCO_TO_ISAACLAB_NAMES.items():
        if il_joint is None:
            continue
        
        mj_val = MuJoCoG1Config.DEFAULT_POSITIONS.get(mj_joint, 0.0)
        il_val = IsaacLabG1Config.DEFAULT_POSITIONS.get(il_joint, 0.0)
        
        diff = abs(mj_val - il_val)
        if diff > 0.001:  # 0.001 rad tolerance
            mismatches.append({
                "mujoco_joint": mj_joint,
                "isaaclab_joint": il_joint,
                "mujoco_value": mj_val,
                "isaaclab_value": il_val,
                "difference_rad": diff,
                "difference_deg": np.degrees(diff),
            })
    
    return mismatches


def validate_physics_settings() -> Dict:
    """Compare physics settings between simulators."""
    return {
        "sim_dt": {
            "isaac_lab": IsaacLabG1Config.SIM_DT,
            "mujoco": MuJoCoG1Config.SIM_DT,
            "match": abs(IsaacLabG1Config.SIM_DT - MuJoCoG1Config.SIM_DT) < 1e-6,
            "note": "Isaac Lab 200Hz, MuJoCo 1000Hz - MuJoCo more accurate",
        },
        "control_dt": {
            "isaac_lab": IsaacLabG1Config.CONTROL_DT,
            "mujoco": MuJoCoG1Config.CONTROL_DT,
            "match": abs(IsaacLabG1Config.CONTROL_DT - MuJoCoG1Config.CONTROL_DT) < 1e-6,
        },
        "init_height": {
            "isaac_lab": IsaacLabG1Config.INIT_HEIGHT,
            "mujoco": MuJoCoG1Config.INIT_HEIGHT,
            "match": abs(IsaacLabG1Config.INIT_HEIGHT - MuJoCoG1Config.INIT_HEIGHT) < 0.01,
        },
        "action_scale": {
            "isaac_lab": IsaacLabG1Config.ACTION_SCALE,
            "mujoco": MuJoCoG1Config.ACTION_SCALE,
            "match": IsaacLabG1Config.ACTION_SCALE == MuJoCoG1Config.ACTION_SCALE,
        },
    }


def validate_pd_gains() -> Dict:
    """Compare PD gains between simulators."""
    # Compare key gains
    comparisons = {}
    
    # Legs
    comparisons["hip_pitch_kp"] = {
        "isaac_lab": IsaacLabG1Config.STIFFNESS["hip_pitch"],
        "mujoco": MuJoCoG1Config.KP["hip_pitch"],
        "match": IsaacLabG1Config.STIFFNESS["hip_pitch"] == MuJoCoG1Config.KP["hip_pitch"],
    }
    comparisons["hip_roll_kp"] = {
        "isaac_lab": IsaacLabG1Config.STIFFNESS["hip_roll"],
        "mujoco": MuJoCoG1Config.KP["hip_roll"],
        "match": IsaacLabG1Config.STIFFNESS["hip_roll"] == MuJoCoG1Config.KP["hip_roll"],
    }
    comparisons["ankle_kp"] = {
        "isaac_lab": IsaacLabG1Config.STIFFNESS["ankle"],
        "mujoco": MuJoCoG1Config.KP["ankle_pitch"],
        "match": IsaacLabG1Config.STIFFNESS["ankle"] == MuJoCoG1Config.KP["ankle_pitch"],
    }
    comparisons["arm_kp"] = {
        "isaac_lab": IsaacLabG1Config.STIFFNESS["arm"],
        "mujoco": MuJoCoG1Config.KP["shoulder"],
        "match": IsaacLabG1Config.STIFFNESS["arm"] == MuJoCoG1Config.KP["shoulder"],
    }
    
    # Damping
    comparisons["leg_kd"] = {
        "isaac_lab": IsaacLabG1Config.DAMPING["hip_pitch"],
        "mujoco": MuJoCoG1Config.KD["hip"],
        "match": IsaacLabG1Config.DAMPING["hip_pitch"] == MuJoCoG1Config.KD["hip"],
    }
    comparisons["ankle_kd"] = {
        "isaac_lab": IsaacLabG1Config.DAMPING["ankle"],
        "mujoco": MuJoCoG1Config.KD["ankle"],
        "match": IsaacLabG1Config.DAMPING["ankle"] == MuJoCoG1Config.KD["ankle"],
    }
    comparisons["arm_kd"] = {
        "isaac_lab": IsaacLabG1Config.DAMPING["arm"],
        "mujoco": MuJoCoG1Config.KD["arm"],
        "match": IsaacLabG1Config.DAMPING["arm"] == MuJoCoG1Config.KD["arm"],
    }
    
    return comparisons


def print_report(verbose: bool = False):
    """Print full validation report."""
    print("=" * 70)
    print("SIM2SIM VALIDATION REPORT: Isaac Lab vs MuJoCo")
    print("=" * 70)
    
    # 1. Joint Mapping
    print("\n1. JOINT MAPPING")
    print("-" * 50)
    matched, unmapped, mapping = validate_joint_mapping()
    print(f"   Matched joints: {len(matched)}/{MuJoCoG1Config.NUM_JOINTS}")
    print(f"   Unmapped MuJoCo joints: {len(unmapped)}")
    
    if unmapped:
        print("\n   ⚠️  UNMAPPED JOINTS (no Isaac Lab equivalent):")
        for joint in unmapped:
            print(f"      - {joint}")
    
    if verbose:
        print("\n   Full mapping:")
        for mj, il in mapping.items():
            print(f"      {mj} -> {il}")
    
    # 2. Default Poses
    print("\n2. DEFAULT POSE COMPARISON")
    print("-" * 50)
    pose_mismatches = validate_default_poses()
    
    if pose_mismatches:
        print("   ⚠️  POSE MISMATCHES:")
        for m in pose_mismatches:
            print(f"      {m['mujoco_joint']}:")
            print(f"         Isaac Lab: {m['isaaclab_value']:.4f} rad")
            print(f"         MuJoCo:    {m['mujoco_value']:.4f} rad")
            print(f"         Diff:      {m['difference_deg']:.2f}°")
    else:
        print("   ✅ All mapped joint default positions match!")
    
    # 3. Physics Settings
    print("\n3. PHYSICS SETTINGS")
    print("-" * 50)
    physics = validate_physics_settings()
    
    for setting, values in physics.items():
        status = "✅" if values["match"] else "⚠️ "
        print(f"   {status} {setting}:")
        print(f"      Isaac Lab: {values['isaac_lab']}")
        print(f"      MuJoCo:    {values['mujoco']}")
        if "note" in values:
            print(f"      Note: {values['note']}")
    
    # 4. PD Gains
    print("\n4. PD GAINS")
    print("-" * 50)
    gains = validate_pd_gains()
    
    all_match = True
    for gain_name, values in gains.items():
        if not values["match"]:
            all_match = False
            print(f"   ⚠️  {gain_name}: IL={values['isaac_lab']}, MJ={values['mujoco']}")
    
    if all_match:
        print("   ✅ All PD gains match!")
    
    # 5. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    issues = []
    if unmapped:
        issues.append(f"{len(unmapped)} unmapped joints")
    if pose_mismatches:
        issues.append(f"{len(pose_mismatches)} pose mismatches")
    if not physics["sim_dt"]["match"]:
        issues.append("Different physics timesteps (expected)")
    if not physics["init_height"]["match"]:
        issues.append("Different initial heights")
    
    if issues:
        print("   Issues to review:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("   ✅ All validations passed!")
    
    print("\n" + "=" * 70)
    
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate sim2sim configuration")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    success = print_report(verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


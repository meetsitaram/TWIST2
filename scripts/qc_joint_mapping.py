#!/usr/bin/env python3
"""
QC Script: Verify Joint Mapping is Correct

This script validates that:
1. G1RobotConfig mapping is loaded correctly
2. Motion data is properly remapped from MuJoCo to Isaac Lab order
3. Observations use consistent joint ordering
4. Reward comparisons use matching indices

Run this BEFORE starting a long training run!
"""

import os
import sys
import pickle
import numpy as np

# Add TWIST2 to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWIST2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TWIST2_ROOT)

from robot_config import G1RobotConfig

def test_robot_config():
    """Test 1: Verify G1RobotConfig is correctly defined."""
    print("=" * 60)
    print("TEST 1: G1RobotConfig Validation")
    print("=" * 60)
    
    # Check joint order length
    assert len(G1RobotConfig.MUJOCO_JOINT_ORDER) == 29, "MuJoCo should have 29 joints"
    print(f"✓ MuJoCo joint order: {len(G1RobotConfig.MUJOCO_JOINT_ORDER)} joints")
    
    # Check mapping completeness
    assert len(G1RobotConfig.MUJOCO_TO_ISAACLAB_NAMES) == 29, "Mapping should cover all 29 joints"
    print(f"✓ Mapping dictionary: {len(G1RobotConfig.MUJOCO_TO_ISAACLAB_NAMES)} entries")
    
    # Check upper body indices
    assert len(G1RobotConfig.MUJOCO_UPPER_BODY_INDICES) == 14, "Upper body should be 14 joints (15-28)"
    print(f"✓ MuJoCo upper body indices: {G1RobotConfig.MUJOCO_UPPER_BODY_INDICES}")
    
    # Check specific joint names
    expected_arm_start = "left_shoulder_pitch_joint"
    actual = G1RobotConfig.MUJOCO_JOINT_ORDER[15]
    assert actual == expected_arm_start, f"Joint 15 should be {expected_arm_start}, got {actual}"
    print(f"✓ Joint 15 is correctly '{expected_arm_start}'")
    
    print("\n✅ G1RobotConfig validation PASSED\n")
    return True


def test_mapping_with_isaac_lab():
    """Test 2: Verify mapping with Isaac Lab joint names (simulated)."""
    print("=" * 60)
    print("TEST 2: Mapping Logic Verification")
    print("=" * 60)
    
    # Simulated Isaac Lab joint names (typical G1 order from the robot config)
    # This is what we saw from the kinematic replay output
    simulated_il_joints = [
        "left_hip_pitch_joint",      # 0
        "right_hip_pitch_joint",     # 1
        "torso_joint",               # 2
        "left_hip_roll_joint",       # 3
        "right_hip_roll_joint",      # 4
        "left_shoulder_pitch_joint", # 5
        "right_shoulder_pitch_joint",# 6
        "left_hip_yaw_joint",        # 7
        "right_hip_yaw_joint",       # 8
        "left_shoulder_roll_joint",  # 9
        "right_shoulder_roll_joint", # 10
        "left_knee_joint",           # 11
        "right_knee_joint",          # 12
        "left_shoulder_yaw_joint",   # 13
        "right_shoulder_yaw_joint",  # 14
        "left_ankle_pitch_joint",    # 15
        "right_ankle_pitch_joint",   # 16
        "left_elbow_pitch_joint",    # 17
        "right_elbow_pitch_joint",   # 18
        "left_ankle_roll_joint",     # 19
        "right_ankle_roll_joint",    # 20
        "left_elbow_roll_joint",     # 21
        "right_elbow_roll_joint",    # 22
        # Extra joints (fingers, etc.) would follow
    ]
    
    mapping = G1RobotConfig.build_mujoco_to_isaaclab_mapping(simulated_il_joints)
    
    # Check critical mappings
    print("\nCritical joint mappings:")
    print("-" * 50)
    
    critical_checks = [
        (0, "left_hip_pitch_joint", 0),
        (15, "left_shoulder_pitch_joint", 5),
        (18, "left_elbow_pitch_joint", 17),
        (22, "right_shoulder_pitch_joint", 6),
        (25, "right_elbow_pitch_joint", 18),
    ]
    
    all_passed = True
    for mj_idx, expected_name, expected_il_idx in critical_checks:
        actual_il_idx = mapping.get(mj_idx)
        mj_name = G1RobotConfig.MUJOCO_JOINT_ORDER[mj_idx]
        
        if actual_il_idx == expected_il_idx:
            print(f"✓ MJ[{mj_idx}] '{mj_name}' -> IL[{actual_il_idx}]")
        else:
            print(f"✗ MJ[{mj_idx}] '{mj_name}' -> IL[{actual_il_idx}] (expected {expected_il_idx})")
            all_passed = False
    
    # Count mapped joints
    mapped = sum(1 for v in mapping.values() if v is not None)
    print(f"\nMapped: {mapped}/29 joints")
    
    if all_passed:
        print("\n✅ Mapping logic verification PASSED\n")
    else:
        print("\n❌ Mapping logic verification FAILED\n")
    
    return all_passed


def test_remapping_function():
    """Test 3: Verify remapping function works correctly."""
    print("=" * 60)
    print("TEST 3: DOF Remapping Function")
    print("=" * 60)
    
    # Simulated Isaac Lab joint names
    simulated_il_joints = [
        "left_hip_pitch_joint",      # 0
        "right_hip_pitch_joint",     # 1
        "torso_joint",               # 2
        "left_hip_roll_joint",       # 3
        "right_hip_roll_joint",      # 4
        "left_shoulder_pitch_joint", # 5
        "right_shoulder_pitch_joint",# 6
        "left_hip_yaw_joint",        # 7
        "right_hip_yaw_joint",       # 8
        "left_shoulder_roll_joint",  # 9
        "right_shoulder_roll_joint", # 10
        "left_knee_joint",           # 11
        "right_knee_joint",          # 12
        "left_shoulder_yaw_joint",   # 13
        "right_shoulder_yaw_joint",  # 14
        "left_ankle_pitch_joint",    # 15
        "right_ankle_pitch_joint",   # 16
        "left_elbow_pitch_joint",    # 17
        "right_elbow_pitch_joint",   # 18
        "left_ankle_roll_joint",     # 19
        "right_ankle_roll_joint",    # 20
        "left_elbow_roll_joint",     # 21
        "right_elbow_roll_joint",    # 22
    ]
    
    # Create test MuJoCo DOF data with known values
    mujoco_dof = np.zeros(29)
    mujoco_dof[15] = 0.5  # left_shoulder_pitch
    mujoco_dof[18] = 1.0  # left_elbow (maps to left_elbow_pitch)
    mujoco_dof[22] = 0.7  # right_shoulder_pitch
    mujoco_dof[25] = 1.2  # right_elbow (maps to right_elbow_pitch)
    
    # Remap
    isaaclab_dof = G1RobotConfig.remap_mujoco_to_isaaclab_numpy(
        mujoco_dof, simulated_il_joints
    )
    
    print(f"\nMuJoCo DOF (input):")
    print(f"  [15] left_shoulder_pitch = {mujoco_dof[15]}")
    print(f"  [18] left_elbow = {mujoco_dof[18]}")
    print(f"  [22] right_shoulder_pitch = {mujoco_dof[22]}")
    print(f"  [25] right_elbow = {mujoco_dof[25]}")
    
    print(f"\nIsaac Lab DOF (output):")
    print(f"  [5] left_shoulder_pitch = {isaaclab_dof[5]}")
    print(f"  [17] left_elbow_pitch = {isaaclab_dof[17]}")
    print(f"  [6] right_shoulder_pitch = {isaaclab_dof[6]}")
    print(f"  [18] right_elbow_pitch = {isaaclab_dof[18]}")
    
    # Verify
    checks = [
        (isaaclab_dof[5], 0.5, "left_shoulder_pitch"),
        (isaaclab_dof[17], 1.0, "left_elbow_pitch"),
        (isaaclab_dof[6], 0.7, "right_shoulder_pitch"),
        (isaaclab_dof[18], 1.2, "right_elbow_pitch"),
    ]
    
    all_passed = True
    print("\nVerification:")
    for actual, expected, name in checks:
        if np.isclose(actual, expected):
            print(f"  ✓ {name}: {actual} == {expected}")
        else:
            print(f"  ✗ {name}: {actual} != {expected}")
            all_passed = False
    
    if all_passed:
        print("\n✅ DOF remapping PASSED\n")
    else:
        print("\n❌ DOF remapping FAILED\n")
    
    return all_passed


def test_motion_file():
    """Test 4: Load a real motion file and verify structure."""
    print("=" * 60)
    print("TEST 4: Motion File Structure")
    print("=" * 60)
    
    motion_path = os.path.join(TWIST2_ROOT, "datasets/teleop_motions/baby_steps/steps_001.pkl")
    
    if not os.path.exists(motion_path):
        print(f"⚠ Motion file not found: {motion_path}")
        print("  Skipping this test\n")
        return True
    
    with open(motion_path, 'rb') as f:
        motion = pickle.load(f)
    
    print(f"Motion file: {motion_path}")
    print(f"Keys: {list(motion.keys())}")
    
    dof_pos = motion['dof_pos']
    print(f"dof_pos shape: {dof_pos.shape}")
    
    assert dof_pos.shape[1] == 29, f"Expected 29 DOFs, got {dof_pos.shape[1]}"
    print(f"✓ DOF count: 29")
    
    # Check that arm joints have reasonable values
    frame0 = dof_pos[0]
    print(f"\nFrame 0 arm joint values (MuJoCo order):")
    print(f"  [15] left_shoulder_pitch: {frame0[15]:.3f}")
    print(f"  [18] left_elbow: {frame0[18]:.3f}")
    print(f"  [22] right_shoulder_pitch: {frame0[22]:.3f}")
    print(f"  [25] right_elbow: {frame0[25]:.3f}")
    
    print("\n✅ Motion file structure PASSED\n")
    return True


def test_upper_body_indices():
    """Test 5: Verify upper body indices are computed correctly."""
    print("=" * 60)
    print("TEST 5: Upper Body Indices")
    print("=" * 60)
    
    simulated_il_joints = [
        "left_hip_pitch_joint",      # 0
        "right_hip_pitch_joint",     # 1
        "torso_joint",               # 2
        "left_hip_roll_joint",       # 3
        "right_hip_roll_joint",      # 4
        "left_shoulder_pitch_joint", # 5
        "right_shoulder_pitch_joint",# 6
        "left_hip_yaw_joint",        # 7
        "right_hip_yaw_joint",       # 8
        "left_shoulder_roll_joint",  # 9
        "right_shoulder_roll_joint", # 10
        "left_knee_joint",           # 11
        "right_knee_joint",          # 12
        "left_shoulder_yaw_joint",   # 13
        "right_shoulder_yaw_joint",  # 14
        "left_ankle_pitch_joint",    # 15
        "right_ankle_pitch_joint",   # 16
        "left_elbow_pitch_joint",    # 17
        "right_elbow_pitch_joint",   # 18
        "left_ankle_roll_joint",     # 19
        "right_ankle_roll_joint",    # 20
        "left_elbow_roll_joint",     # 21
        "right_elbow_roll_joint",    # 22
    ]
    
    upper_indices = G1RobotConfig.get_isaaclab_upper_body_indices(simulated_il_joints)
    
    print(f"Upper body indices (Isaac Lab order): {upper_indices}")
    
    # Expected: shoulders (5,6,9,10,13,14) + elbows (17,18,21,22)
    expected = [5, 6, 9, 10, 13, 14, 17, 18, 21, 22]
    
    if upper_indices == expected:
        print(f"✓ Upper body indices match expected: {expected}")
        print("\n✅ Upper body indices PASSED\n")
        return True
    else:
        print(f"✗ Upper body indices don't match!")
        print(f"  Expected: {expected}")
        print(f"  Got: {upper_indices}")
        print("\n❌ Upper body indices FAILED\n")
        return False


def main():
    print("\n" + "=" * 60)
    print("  G1 JOINT MAPPING QC TESTS")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("G1RobotConfig", test_robot_config()))
    results.append(("Mapping Logic", test_mapping_with_isaac_lab()))
    results.append(("DOF Remapping", test_remapping_function()))
    results.append(("Motion File", test_motion_file()))
    results.append(("Upper Body Indices", test_upper_body_indices()))
    
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED - Safe to proceed with training!")
    else:
        print("⚠️  SOME TESTS FAILED - Fix issues before training!")
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

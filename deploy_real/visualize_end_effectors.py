#!/usr/bin/env python3
"""
Visualize End-Effector Positions for Human and Robot

Compares pelvis-relative end-effector positions between:
- Human skeleton (from captured poses)
- Robot (from matched robot poses via forward kinematics)

This helps debug IK-based retargeting by showing where end-effectors should be.

Usage:
    python visualize_end_effectors.py
    python visualize_end_effectors.py --pose 1_20260123_162606
    
    # Test camera tilt correction (positive = rotate skeleton backward)
    python visualize_end_effectors.py --pose 1_20260123_162606 --pitch 15
"""

import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("Warning: MuJoCo not available. Robot visualization disabled.")

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
HUMAN_POSES_DIR = CALIBRATION_DIR / "captured_poses"
ROBOT_POSES_DIR = CALIBRATION_DIR / "robot_poses"
ASSETS_DIR = PROJECT_ROOT / "assets" / "g1"

# MediaPipe landmark indices
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_PINKY = 17
MP_RIGHT_PINKY = 18
MP_LEFT_INDEX = 19
MP_RIGHT_INDEX = 20
MP_LEFT_THUMB = 21
MP_RIGHT_THUMB = 22
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32

# End-effector colors
COLORS = {
    'left_foot': 'blue',
    'right_foot': 'cyan',
    'left_hand': 'red',
    'right_hand': 'orange',
    'neck': 'green',
    'pelvis': 'black',
}


def normalize(v):
    """Normalize a vector, return zero vector if magnitude is too small."""
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros(3)
    return v / norm


def compute_orientation_axes(skeleton: np.ndarray) -> dict:
    """
    Compute orientation axes for end-effectors from bone directions.
    
    Returns dict with 'forward', 'up', 'right' axes for each end-effector.
    These form a right-handed coordinate system at each end-effector.
    """
    orientations = {}
    
    # LEFT HAND orientation
    # Forward: wrist → finger midpoint (pointing direction)
    # Right: wrist → pinky direction  
    # Up: cross product (palm normal)
    l_wrist = skeleton[MP_LEFT_WRIST]
    l_index = skeleton[MP_LEFT_INDEX]
    l_pinky = skeleton[MP_LEFT_PINKY]
    l_elbow = skeleton[MP_LEFT_ELBOW]
    
    l_hand_forward = normalize(l_index - l_wrist)  # fingers pointing direction
    l_forearm = normalize(l_wrist - l_elbow)  # forearm direction
    l_hand_right = normalize(l_pinky - l_index)  # pinky side
    l_hand_up = normalize(np.cross(l_hand_forward, l_hand_right))  # palm normal
    # Recompute right to ensure orthogonal
    l_hand_right = normalize(np.cross(l_hand_up, l_hand_forward))
    
    orientations['left_hand'] = {
        'forward': l_hand_forward,
        'up': l_hand_up,
        'right': l_hand_right,
        'forearm': l_forearm,
    }
    
    # RIGHT HAND orientation
    r_wrist = skeleton[MP_RIGHT_WRIST]
    r_index = skeleton[MP_RIGHT_INDEX]
    r_pinky = skeleton[MP_RIGHT_PINKY]
    r_elbow = skeleton[MP_RIGHT_ELBOW]
    
    r_hand_forward = normalize(r_index - r_wrist)
    r_forearm = normalize(r_wrist - r_elbow)
    r_hand_right = normalize(r_pinky - r_index)
    r_hand_up = normalize(np.cross(r_hand_forward, r_hand_right))
    r_hand_right = normalize(np.cross(r_hand_up, r_hand_forward))
    
    orientations['right_hand'] = {
        'forward': r_hand_forward,
        'up': r_hand_up,
        'right': r_hand_right,
        'forearm': r_forearm,
    }
    
    # LEFT FOOT orientation
    # Forward: heel → toe
    # Up: perpendicular to foot plane (ankle-heel-toe plane)
    l_ankle = skeleton[MP_LEFT_ANKLE]
    l_heel = skeleton[MP_LEFT_HEEL]
    l_toe = skeleton[MP_LEFT_FOOT_INDEX]
    
    l_foot_forward = normalize(l_toe - l_heel)  # foot pointing direction
    l_foot_right = normalize(np.cross(l_foot_forward, np.array([0, 0, 1])))  # approximate
    # Use ankle position to refine up direction
    l_ankle_to_heel = normalize(l_heel - l_ankle)
    l_foot_up = normalize(np.cross(l_foot_right, l_foot_forward))
    
    orientations['left_foot'] = {
        'forward': l_foot_forward,
        'up': l_foot_up,
        'right': l_foot_right,
    }
    
    # RIGHT FOOT orientation
    r_ankle = skeleton[MP_RIGHT_ANKLE]
    r_heel = skeleton[MP_RIGHT_HEEL]
    r_toe = skeleton[MP_RIGHT_FOOT_INDEX]
    
    r_foot_forward = normalize(r_toe - r_heel)
    r_foot_right = normalize(np.cross(r_foot_forward, np.array([0, 0, 1])))
    r_foot_up = normalize(np.cross(r_foot_right, r_foot_forward))
    
    orientations['right_foot'] = {
        'forward': r_foot_forward,
        'up': r_foot_up,
        'right': r_foot_right,
    }
    
    # NECK orientation (torso facing direction)
    l_shoulder = skeleton[MP_LEFT_SHOULDER]
    r_shoulder = skeleton[MP_RIGHT_SHOULDER]
    l_hip = skeleton[MP_LEFT_HIP]
    r_hip = skeleton[MP_RIGHT_HIP]
    
    shoulder_mid = (l_shoulder + r_shoulder) / 2
    hip_mid = (l_hip + r_hip) / 2
    
    neck_up = normalize(shoulder_mid - hip_mid)  # spine direction
    neck_right = normalize(r_shoulder - l_shoulder)  # shoulder line
    neck_forward = normalize(np.cross(neck_up, neck_right))  # facing direction
    
    orientations['neck'] = {
        'forward': neck_forward,
        'up': neck_up,
        'right': neck_right,
    }
    
    return orientations


def load_human_pose(filepath: Path) -> dict:
    """Load a human pose JSON file."""
    with open(filepath) as f:
        return json.load(f)


def load_robot_pose(filepath: Path) -> dict:
    """Load a robot pose JSON file."""
    with open(filepath) as f:
        return json.load(f)


def extract_human_end_effectors(skeleton_3d: np.ndarray) -> dict:
    """
    Extract pelvis-relative end-effector positions from MediaPipe skeleton.
    
    Returns dict with positions and the pelvis position.
    """
    skeleton = np.array(skeleton_3d)
    
    # Pelvis = midpoint of hips
    pelvis = (skeleton[MP_LEFT_HIP] + skeleton[MP_RIGHT_HIP]) / 2
    
    # Neck = midpoint of shoulders
    neck = (skeleton[MP_LEFT_SHOULDER] + skeleton[MP_RIGHT_SHOULDER]) / 2
    
    # End-effector positions (absolute)
    end_effectors_abs = {
        'left_foot': skeleton[MP_LEFT_ANKLE],
        'right_foot': skeleton[MP_RIGHT_ANKLE],
        'left_hand': skeleton[MP_LEFT_WRIST],
        'right_hand': skeleton[MP_RIGHT_WRIST],
        'neck': neck,
        'pelvis': pelvis,
    }
    
    # Convert to pelvis-relative
    end_effectors_rel = {}
    for name, pos in end_effectors_abs.items():
        if name != 'pelvis':
            end_effectors_rel[name] = pos - pelvis
        else:
            end_effectors_rel[name] = np.zeros(3)  # Pelvis is origin
    
    # Compute height for scaling (ankle to neck/mid-shoulder)
    height = np.linalg.norm(neck - ((skeleton[MP_LEFT_ANKLE] + skeleton[MP_RIGHT_ANKLE]) / 2))
    
    return {
        'absolute': end_effectors_abs,
        'relative': end_effectors_rel,
        'pelvis': pelvis,
        'height': height,
        'skeleton': skeleton,
    }


def quat_to_rotation_matrix(quat):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def extract_robot_end_effectors(model, data, joint_angles_rad: dict) -> dict:
    """
    Extract pelvis-relative end-effector positions from robot via forward kinematics.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_angles_rad: Dict of joint_name -> angle_rad
    
    Returns dict with positions and orientations.
    """
    # Reset to default
    mujoco.mj_resetData(model, data)
    
    # Set base pose (standing)
    data.qpos[2] = 0.75  # Height
    data.qpos[3] = 1.0   # Quaternion w (upright)
    
    # Set joint angles
    joint_order = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    
    for i, joint_name in enumerate(joint_order):
        if joint_name in joint_angles_rad:
            data.qpos[7 + i] = joint_angles_rad[joint_name]
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Get body position and orientation
    def get_body_pos(name):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return data.xpos[body_id].copy()
        else:
            print(f"Warning: Body '{name}' not found")
            return np.zeros(3)
    
    def get_body_quat(name):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return data.xquat[body_id].copy()  # [w, x, y, z]
        else:
            return np.array([1, 0, 0, 0])  # Identity quaternion
    
    def get_body_orientation(name):
        """Get orientation axes (forward, up, right) from body quaternion."""
        quat = get_body_quat(name)
        R = quat_to_rotation_matrix(quat)
        # MuJoCo convention: X=forward, Y=left, Z=up for the robot
        return {
            'forward': R[:, 0],  # X axis
            'right': -R[:, 1],   # -Y axis (Y is left, so -Y is right)
            'up': R[:, 2],       # Z axis
        }
    
    # Try different body names (model-specific)
    pelvis = get_body_pos("pelvis")
    
    # Get shoulder positions to compute neck
    left_shoulder = get_body_pos("left_shoulder_pitch_link")
    right_shoulder = get_body_pos("right_shoulder_pitch_link")
    
    # Neck = midpoint between shoulders (like human mid-shoulder)
    neck_pos = (left_shoulder + right_shoulder) / 2
    
    end_effectors_abs = {
        'left_foot': get_body_pos("left_ankle_roll_link"),
        'right_foot': get_body_pos("right_ankle_roll_link"),
        'left_hand': get_body_pos("left_wrist_yaw_link"),
        'right_hand': get_body_pos("right_wrist_yaw_link"),
        'neck': neck_pos,
        'pelvis': pelvis,
    }
    
    # Get orientations for each end-effector
    orientations = {
        'left_foot': get_body_orientation("left_ankle_roll_link"),
        'right_foot': get_body_orientation("right_ankle_roll_link"),
        'left_hand': get_body_orientation("left_wrist_yaw_link"),
        'right_hand': get_body_orientation("right_wrist_yaw_link"),
        'neck': get_body_orientation("torso_link"),
    }
    
    # Convert to pelvis-relative
    end_effectors_rel = {}
    for name, pos in end_effectors_abs.items():
        if name != 'pelvis':
            end_effectors_rel[name] = pos - pelvis
        else:
            end_effectors_rel[name] = np.zeros(3)
    
    # Compute height (ankle to neck)
    foot_mid = (end_effectors_abs['left_foot'] + end_effectors_abs['right_foot']) / 2
    height = np.linalg.norm(end_effectors_abs['neck'] - foot_mid)
    
    return {
        'absolute': end_effectors_abs,
        'relative': end_effectors_rel,
        'orientations': orientations,
        'pelvis': pelvis,
        'height': height,
    }


def align_skeleton_upright(skeleton: np.ndarray) -> np.ndarray:
    """Rotate skeleton so torso is vertical."""
    skeleton = skeleton.copy()
    
    l_shoulder = skeleton[MP_LEFT_SHOULDER]
    r_shoulder = skeleton[MP_RIGHT_SHOULDER]
    l_hip = skeleton[MP_LEFT_HIP]
    r_hip = skeleton[MP_RIGHT_HIP]
    
    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_hip = (l_hip + r_hip) / 2
    
    spine = mid_shoulder - mid_hip
    spine_norm = np.linalg.norm(spine)
    if spine_norm < 0.01:
        return skeleton
    spine = spine / spine_norm
    
    up = np.array([0, 0, 1])
    axis = np.cross(spine, up)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:
        if np.dot(spine, up) < 0:
            skeleton[:, 2] = -skeleton[:, 2]
        return skeleton
    
    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(spine, up), -1, 1))
    
    def rotate_point(p, axis, angle):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return p * cos_a + np.cross(axis, p) * sin_a + axis * np.dot(axis, p) * (1 - cos_a)
    
    center = mid_hip.copy()
    for i in range(len(skeleton)):
        if not np.isnan(skeleton[i, 0]):
            skeleton[i] = rotate_point(skeleton[i] - center, axis, angle) + center
    
    skeleton = skeleton - center
    return skeleton


def draw_orientation_axes(ax, position, orientation, scale=0.1, alpha=0.8):
    """
    Draw orientation axes (forward=red, up=green, right=blue) at a position.
    """
    if orientation is None:
        return
    
    colors = {'forward': 'red', 'up': 'green', 'right': 'blue'}
    
    for axis_name, color in colors.items():
        if axis_name in orientation:
            direction = orientation[axis_name]
            if np.linalg.norm(direction) > 0.1:  # Only draw if valid
                end = position + direction * scale
                ax.plot([position[0], end[0]], 
                       [position[1], end[1]], 
                       [position[2], end[2]], 
                       color=color, linewidth=2, alpha=alpha)


def visualize_comparison(human_data: dict, robot_data: dict = None, title: str = ""):
    """
    Visualize human and robot end-effectors side by side.
    """
    fig = plt.figure(figsize=(16, 8))
    
    # === Human skeleton subplot ===
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"Human Skeleton + Orientations\n{title}")
    
    # Align and plot human skeleton
    skeleton = align_skeleton_upright(human_data['skeleton'])
    
    # Recompute end-effectors from aligned skeleton (pelvis-centered)
    pelvis = (skeleton[MP_LEFT_HIP] + skeleton[MP_RIGHT_HIP]) / 2
    neck = (skeleton[MP_LEFT_SHOULDER] + skeleton[MP_RIGHT_SHOULDER]) / 2
    
    # Center skeleton on pelvis
    skeleton_centered = skeleton - pelvis
    
    ee_human = {
        'left_foot': skeleton_centered[MP_LEFT_ANKLE],
        'right_foot': skeleton_centered[MP_RIGHT_ANKLE],
        'left_hand': skeleton_centered[MP_LEFT_WRIST],
        'right_hand': skeleton_centered[MP_RIGHT_WRIST],
        'neck': neck - pelvis,  # mid-shoulder
        'pelvis': np.zeros(3),
    }
    
    # Compute orientations from aligned skeleton
    # The skeleton is already aligned upright, use it directly
    orientations_centered = compute_orientation_axes(skeleton)
    
    # Debug: print end-effector positions and orientations
    print("\nHuman End-Effectors (pelvis-relative):")
    for name, pos in ee_human.items():
        print(f"  {name:12}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")
    
    print("\nEnd-Effector Orientations (forward direction):")
    for name in ['left_hand', 'right_hand', 'left_foot', 'right_foot', 'neck']:
        if name in orientations_centered:
            fwd = orientations_centered[name]['forward']
            print(f"  {name:12} forward: [{fwd[0]:+.3f}, {fwd[1]:+.3f}, {fwd[2]:+.3f}]")
    
    # Plot skeleton points (faded)
    valid = ~np.isnan(skeleton_centered[:, 0])
    ax1.scatter(skeleton_centered[valid, 0], skeleton_centered[valid, 1], skeleton_centered[valid, 2], 
               c='gray', s=20, alpha=0.3)
    
    # Plot end-effectors (highlighted) - with larger markers
    for name, pos in ee_human.items():
        if name == 'pelvis':
            # Pelvis at origin - use distinct diamond marker
            ax1.scatter(pos[0], pos[1], pos[2], c=COLORS[name], s=300, label=name, marker='D', edgecolors='white', linewidths=3)
        else:
            ax1.scatter(pos[0], pos[1], pos[2], c=COLORS[name], s=200, label=name, marker='o', edgecolors='black', linewidths=2)
            # Draw orientation axes for this end-effector
            if name in orientations_centered:
                draw_orientation_axes(ax1, pos, orientations_centered[name], scale=0.15)
        # Offset text slightly to avoid overlap
        ax1.text(pos[0] + 0.05, pos[1], pos[2] + 0.05, f"{name}", fontsize=9, fontweight='bold')
    
    # Plot connections - fuller skeleton
    connections = [
        # Spine
        (MP_LEFT_HIP, MP_LEFT_SHOULDER),
        (MP_RIGHT_HIP, MP_RIGHT_SHOULDER),
        (MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER),
        (MP_LEFT_HIP, MP_RIGHT_HIP),
        # Legs
        (MP_LEFT_HIP, 25),  # left knee
        (25, MP_LEFT_ANKLE),
        (MP_RIGHT_HIP, 26),  # right knee
        (26, MP_RIGHT_ANKLE),
        # Arms
        (MP_LEFT_SHOULDER, 13),  # left elbow
        (13, MP_LEFT_WRIST),
        (MP_RIGHT_SHOULDER, 14),  # right elbow
        (14, MP_RIGHT_WRIST),
        # Head
        (MP_LEFT_SHOULDER, MP_NOSE),
        (MP_RIGHT_SHOULDER, MP_NOSE),
    ]
    for i, j in connections:
        if i < len(skeleton_centered) and j < len(skeleton_centered):
            ax1.plot([skeleton_centered[i, 0], skeleton_centered[j, 0]],
                    [skeleton_centered[i, 1], skeleton_centered[j, 1]],
                    [skeleton_centered[i, 2], skeleton_centered[j, 2]], 'gray', alpha=0.5, linewidth=2)
    
    ax1.set_xlabel('X (right)')
    ax1.set_ylabel('Y (forward)')
    ax1.set_zlabel('Z (up)')
    ax1.legend(loc='upper left', fontsize=8)
    
    # Add orientation legend
    ax1.text2D(0.02, 0.02, "Orientation axes:\nRed=forward, Green=up, Blue=right", 
               transform=ax1.transAxes, fontsize=8, verticalalignment='bottom')
    
    # Set equal aspect - auto-scale based on data
    all_coords = skeleton_centered[valid]
    max_range = max(np.abs(all_coords).max() * 1.2, 0.8)
    ax1.set_xlim(-max_range, max_range)
    ax1.set_ylim(-max_range, max_range)
    ax1.set_zlim(-max_range, max_range)
    
    # === Robot or info subplot ===
    if robot_data:
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title("Robot End-Effectors + Orientations")
        
        print("\nRobot End-Effectors (pelvis-relative):")
        for name, pos in robot_data['relative'].items():
            print(f"  {name:12}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")
        
        # Print robot orientations
        if 'orientations' in robot_data:
            print("\nRobot End-Effector Orientations (forward direction):")
            for name in ['left_hand', 'right_hand', 'left_foot', 'right_foot', 'neck']:
                if name in robot_data['orientations']:
                    fwd = robot_data['orientations'][name]['forward']
                    print(f"  {name:12} forward: [{fwd[0]:+.3f}, {fwd[1]:+.3f}, {fwd[2]:+.3f}]")
        
        # Plot robot end-effectors
        for name, pos in robot_data['relative'].items():
            if name == 'pelvis':
                # Pelvis at origin - use distinct diamond marker
                ax2.scatter(pos[0], pos[1], pos[2], c=COLORS.get(name, 'purple'), s=300, 
                           label=name, marker='D', edgecolors='white', linewidths=3)
            else:
                ax2.scatter(pos[0], pos[1], pos[2], c=COLORS.get(name, 'purple'), s=200, 
                           label=name, marker='s', edgecolors='black', linewidths=2)
                # Draw orientation axes for robot end-effectors
                if 'orientations' in robot_data and name in robot_data['orientations']:
                    draw_orientation_axes(ax2, pos, robot_data['orientations'][name], scale=0.15)
            ax2.text(pos[0] + 0.05, pos[1], pos[2] + 0.05, f"{name}", fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('X (forward)')
        ax2.set_ylabel('Y (left)')
        ax2.set_zlabel('Z (up)')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.set_xlim(-max_range, max_range)
        ax2.set_ylim(-max_range, max_range)
        ax2.set_zlim(-max_range, max_range)
        
        # Add orientation legend
        ax2.text2D(0.02, 0.02, "Orientation axes:\nRed=forward, Green=up, Blue=right", 
                   transform=ax2.transAxes, fontsize=8, verticalalignment='bottom')
        
        # Print comparison
        print("\nEnd-Effector Position Errors:")
        for name in ['left_foot', 'right_foot', 'left_hand', 'right_hand', 'neck']:
            if name in ee_human and name in robot_data['relative']:
                h_pos = ee_human[name]
                r_pos = robot_data['relative'][name]
                error = np.linalg.norm(h_pos - r_pos)
                print(f"  {name:12}: {error:.3f}m")
    else:
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        
        # Show numeric values
        info = "Human End-Effector Positions (pelvis-relative):\n\n"
        for name, pos in ee_human.items():
            info += f"{name:12}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]\n"
        info += f"\nHeight: {human_data['height']:.3f}m"
        
        ax2.text(0.1, 0.9, info, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()


def compare_all_pairs():
    """Compare human and robot end-effectors for all matched pairs."""
    if not HAS_MUJOCO:
        print("MuJoCo required for robot comparison")
        return
    
    # Load robot model
    model_path = ASSETS_DIR / "g1_mocap_29dof.xml"
    if not model_path.exists():
        print(f"Robot model not found: {model_path}")
        return
    
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Find matched pairs
    robot_files = list(ROBOT_POSES_DIR.glob("*.json"))
    
    print("\n" + "="*70)
    print("END-EFFECTOR POSITION COMPARISON")
    print("="*70)
    print(f"\n{'Pose':<20} {'EE':<12} {'Human (rel)':<25} {'Robot (rel)':<25} {'Error':<8}")
    print("-"*90)
    
    for robot_file in sorted(robot_files):
        robot_pose = load_robot_pose(robot_file)
        human_ref = robot_pose.get("human_pose_ref", "")
        
        # Find human pose
        human_file = None
        for hf in HUMAN_POSES_DIR.glob("*.json"):
            if human_ref in hf.stem:
                human_file = hf
                break
        
        if not human_file:
            continue
        
        human_pose = load_human_pose(human_file)
        
        # Extract end-effectors
        human_ee = extract_human_end_effectors(human_pose["skeleton_3d"])
        robot_ee = extract_robot_end_effectors(model, data, robot_pose.get("joint_angles_rad", {}))
        
        # Scale human to robot height
        scale = robot_ee['height'] / human_ee['height'] if human_ee['height'] > 0.1 else 1.0
        
        # Compare each end-effector
        pose_name = human_file.stem[:15]
        for ee_name in ['left_hand', 'right_hand', 'left_foot', 'right_foot', 'neck']:
            h_pos = human_ee['relative'][ee_name] * scale
            r_pos = robot_ee['relative'][ee_name]
            error = np.linalg.norm(h_pos - r_pos)
            
            h_str = f"[{h_pos[0]:+.2f},{h_pos[1]:+.2f},{h_pos[2]:+.2f}]"
            r_str = f"[{r_pos[0]:+.2f},{r_pos[1]:+.2f},{r_pos[2]:+.2f}]"
            
            print(f"{pose_name:<20} {ee_name:<12} {h_str:<25} {r_str:<25} {error:.3f}m")
        print("-"*90)


def apply_pitch_correction(skeleton: np.ndarray, pitch_deg: float) -> np.ndarray:
    """
    Apply a pitch rotation to the skeleton to correct for camera tilt.
    
    Args:
        skeleton: (33, 3) array of MediaPipe landmarks
        pitch_deg: Rotation angle in degrees (positive = rotate backward)
    
    Returns:
        Rotated skeleton
    """
    if abs(pitch_deg) < 0.1:
        return skeleton
    
    pitch_rad = np.radians(pitch_deg)
    
    # Rotation around X-axis (pitch correction)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    # Apply rotation to each valid landmark
    rotated = skeleton.copy()
    for i in range(len(skeleton)):
        if not np.any(np.isnan(skeleton[i])):
            rotated[i] = Rx @ skeleton[i]
    
    return rotated


def apply_leg_pitch_correction(skeleton: np.ndarray, pitch_deg: float) -> np.ndarray:
    """
    Apply a pitch rotation ONLY to leg landmarks, relative to pelvis.
    
    This corrects for leg tilt without affecting the upper body.
    
    Args:
        skeleton: (33, 3) array of MediaPipe landmarks
        pitch_deg: Rotation angle in degrees (positive = rotate legs forward)
    
    Returns:
        Skeleton with rotated legs
    """
    if abs(pitch_deg) < 0.1:
        return skeleton
    
    # Leg landmark indices
    LEG_LANDMARKS = [
        MP_LEFT_HIP, MP_RIGHT_HIP,
        MP_LEFT_KNEE, MP_RIGHT_KNEE,
        MP_LEFT_ANKLE, MP_RIGHT_ANKLE,
        MP_LEFT_HEEL, MP_RIGHT_HEEL,
        MP_LEFT_FOOT_INDEX, MP_RIGHT_FOOT_INDEX,
    ]
    
    pitch_rad = np.radians(pitch_deg)
    
    # Rotation around X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    
    # Get pelvis as rotation center
    pelvis = (skeleton[MP_LEFT_HIP] + skeleton[MP_RIGHT_HIP]) / 2
    
    rotated = skeleton.copy()
    for i in LEG_LANDMARKS:
        if not np.any(np.isnan(skeleton[i])):
            # Rotate around pelvis
            rel_pos = skeleton[i] - pelvis
            rotated_rel = Rx @ rel_pos
            rotated[i] = pelvis + rotated_rel
    
    return rotated


def main():
    parser = argparse.ArgumentParser(description="Visualize end-effector positions")
    parser.add_argument("--pose", "-p", type=str, default=None,
                       help="Specific pose to visualize (e.g., '1_20260123_162606')")
    parser.add_argument("--compare", "-c", action="store_true",
                       help="Compare all matched human-robot pairs")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available poses")
    parser.add_argument("--pitch", type=float, default=0.0,
                       help="Pitch correction in degrees (positive = rotate backward, fixes forward tilt)")
    parser.add_argument("--leg-pitch", type=float, default=0.0,
                       help="Leg-only pitch correction (positive = rotate legs forward, fixes backward tilt)")
    args = parser.parse_args()
    
    if args.list:
        print("\nHuman poses:")
        for f in sorted(HUMAN_POSES_DIR.glob("*.json")):
            print(f"  {f.stem}")
        print("\nRobot poses:")
        for f in sorted(ROBOT_POSES_DIR.glob("*.json")):
            print(f"  {f.stem}")
        return
    
    if args.compare:
        compare_all_pairs()
        return
    
    # Visualize single pose
    if args.pose:
        # Find the pose file
        human_file = None
        for f in HUMAN_POSES_DIR.glob("*.json"):
            if args.pose in f.stem:
                human_file = f
                break
        
        if not human_file:
            print(f"Pose not found: {args.pose}")
            return
    else:
        # Use first available pose
        poses = sorted(HUMAN_POSES_DIR.glob("*.json"))
        if not poses:
            print("No poses found")
            return
        human_file = poses[0]
    
    print(f"\nLoading: {human_file.stem}")
    human_pose = load_human_pose(human_file)
    skeleton_3d = np.array(human_pose["skeleton_3d"])
    
    # Apply pitch correction if specified
    if abs(args.pitch) > 0.1:
        print(f"Applying pitch correction: {args.pitch:+.1f}°")
        skeleton_3d = apply_pitch_correction(skeleton_3d, args.pitch)
    
    # Apply leg-only pitch correction if specified
    if abs(args.leg_pitch) > 0.1:
        print(f"Applying leg pitch correction: {args.leg_pitch:+.1f}°")
        skeleton_3d = apply_leg_pitch_correction(skeleton_3d, args.leg_pitch)
    
    human_ee = extract_human_end_effectors(skeleton_3d)
    
    # Try to find matching robot pose
    robot_ee = None
    if HAS_MUJOCO:
        robot_file = ROBOT_POSES_DIR / f"{human_file.stem}.json"
        if robot_file.exists():
            print(f"Found robot pose: {robot_file.stem}")
            model = mujoco.MjModel.from_xml_path(str(ASSETS_DIR / "g1_mocap_29dof.xml"))
            data = mujoco.MjData(model)
            robot_pose = load_robot_pose(robot_file)
            robot_ee = extract_robot_end_effectors(model, data, robot_pose.get("joint_angles_rad", {}))
    
    visualize_comparison(human_ee, robot_ee, title=human_file.stem)


if __name__ == "__main__":
    main()

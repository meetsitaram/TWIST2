#!/usr/bin/env python3
"""
End-Effector IK Retargeting for Human to Robot Motion Transfer

Instead of matching all 14 body parts (like GMR), this simpler approach:
1. Extracts 5 end-effector positions + orientations from human pose
2. Scales to robot proportions
3. Uses mink IK solver to find robot joint angles

End-effectors tracked:
- Left/Right Hand: wrist position + palm orientation
- Left/Right Foot: ankle position + foot orientation (two-stage mode)
- Torso: mid-shoulder position + torso orientation

IK Modes:
- Single-stage (default): Upper body only, fixed base at Z=0.75m
- Two-stage (--two-stage): 
  1. Stage 1: Upper body with fixed base
  2. Stage 2: Lower body with arms frozen, base Z free

Usage:
    # Test with a captured pose (single-stage, upper body only)
    python end_effector_ik_retarget.py --pose 1_20260123_162606
    
    # Test with two-stage IK (whole body)
    python end_effector_ik_retarget.py --pose 1_20260123_162606 --two-stage
    
    # Test all captured poses with two-stage IK
    python end_effector_ik_retarget.py --all --two-stage
    
    # Visualize in MuJoCo
    python end_effector_ik_retarget.py --pose 1_20260123_162606 --viz --two-stage
    
    # Side-by-side comparison (human vs robot with end-effectors and orientations)
    python end_effector_ik_retarget.py --pose 1_20260123_162606 --compare
"""

import numpy as np
import json
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

try:
    import mink
    from mink.tasks import DofFreezingTask
    HAS_MINK = True
except ImportError:
    HAS_MINK = False
    DofFreezingTask = None
    print("Warning: mink not available. Install with: pip install mink")

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("Warning: mujoco not available.")

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
HUMAN_POSES_DIR = CALIBRATION_DIR / "captured_poses"
ROBOT_POSES_DIR = CALIBRATION_DIR / "robot_poses"
ASSETS_DIR = PROJECT_ROOT / "assets" / "g1"
ROBOT_MODEL_PATH = ASSETS_DIR / "g1_mocap_29dof.xml"

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

# Robot body names for each end-effector
# Upper body targets (used in Stage 1 of two-stage IK)
ROBOT_EE_BODIES_UPPER = {
    'left_hand': 'left_wrist_yaw_link',
    'right_hand': 'right_wrist_yaw_link',
    'left_elbow': 'left_elbow_link',
    'right_elbow': 'right_elbow_link',
}

# Lower body targets (used in Stage 2 of two-stage IK)
ROBOT_EE_BODIES_LOWER = {
    'left_foot': 'left_ankle_roll_link',
    'right_foot': 'right_ankle_roll_link',
    'left_knee': 'left_knee_link',      # Soft hint for leg pose
    'right_knee': 'right_knee_link',    # Soft hint for leg pose
}

# Combined for backward compatibility (upper body only by default)
ROBOT_EE_BODIES = ROBOT_EE_BODIES_UPPER.copy()

# Separate orientation-only constraint for torso (no position, just keep upright)
ROBOT_TORSO_BODY = 'waist_roll_link'  # Use waist for orientation constraint

# Joint order for the 29-DOF robot
JOINT_ORDER = [
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


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector, return zero vector if magnitude is too small."""
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros(3)
    return v / norm


def rotation_matrix_to_quat(R_mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    return R.from_matrix(R_mat).as_quat(scalar_first=True)


def axes_to_rotation_matrix(forward: np.ndarray, up: np.ndarray, right: np.ndarray = None) -> np.ndarray:
    """
    Create rotation matrix from forward and up axes.
    
    Uses Gram-Schmidt to ensure orthonormality and right-handed frame.
    MuJoCo convention: X=forward, Y=left, Z=up
    
    Returns identity matrix if vectors are degenerate.
    """
    # Start with forward as primary axis
    x_axis = normalize(forward)
    if np.linalg.norm(x_axis) < 0.5:
        # Degenerate forward, use default
        return np.eye(3)
    
    # Make up orthogonal to forward using Gram-Schmidt
    up_proj = up - np.dot(up, x_axis) * x_axis
    if np.linalg.norm(up_proj) < 1e-6:
        # up is parallel to forward, pick a different up
        if abs(x_axis[2]) < 0.9:
            up_proj = np.array([0, 0, 1]) - x_axis[2] * x_axis
        else:
            up_proj = np.array([1, 0, 0]) - x_axis[0] * x_axis
    
    z_axis = normalize(up_proj)
    if np.linalg.norm(z_axis) < 0.5:
        return np.eye(3)
    
    # Compute left axis (Y) using cross product for right-handed system
    y_axis = np.cross(z_axis, x_axis)
    y_axis = normalize(y_axis)
    if np.linalg.norm(y_axis) < 0.5:
        return np.eye(3)
    
    # Recompute z to ensure perfect orthonormality
    z_axis = np.cross(x_axis, y_axis)
    z_axis = normalize(z_axis)
    
    # Build rotation matrix with columns as axes
    R_mat = np.column_stack([x_axis, y_axis, z_axis])
    
    # Check for NaN or invalid values
    if np.any(np.isnan(R_mat)) or np.any(np.isinf(R_mat)):
        return np.eye(3)
    
    # Verify determinant is positive (right-handed) and close to 1
    det = np.linalg.det(R_mat)
    if det < 0:
        # Flip one axis to make it right-handed
        R_mat[:, 1] = -R_mat[:, 1]
    elif abs(det) < 0.5:
        # Degenerate, return identity
        return np.eye(3)
    
    return R_mat


def align_skeleton_upright(skeleton: np.ndarray) -> np.ndarray:
    """
    Rotate skeleton so torso is vertical (Z-up) and facing forward (X+).
    
    After alignment:
    - Skeleton is centered on pelvis (pelvis = origin)
    - Torso is vertical (Z-up)
    - Person faces X+ direction (MuJoCo convention: X=forward, Y=left, Z=up)
    """
    skeleton = skeleton.copy()
    
    l_shoulder = skeleton[MP_LEFT_SHOULDER]
    r_shoulder = skeleton[MP_RIGHT_SHOULDER]
    l_hip = skeleton[MP_LEFT_HIP]
    r_hip = skeleton[MP_RIGHT_HIP]
    
    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_hip = (l_hip + r_hip) / 2
    
    # Step 1: Rotate spine to be vertical (Z-up)
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
    else:
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
    
    # Center on pelvis first
    pelvis = (skeleton[MP_LEFT_HIP] + skeleton[MP_RIGHT_HIP]) / 2
    skeleton = skeleton - pelvis
    
    # Step 2: Rotate around Z so person faces X+ (forward)
    # In MuJoCo: X=forward, Y=left, Z=up
    # We want: left shoulder at positive Y, right shoulder at negative Y
    l_shoulder = skeleton[MP_LEFT_SHOULDER]
    r_shoulder = skeleton[MP_RIGHT_SHOULDER]
    
    # The direction from right shoulder to left shoulder should align with +Y
    shoulder_vec = l_shoulder - r_shoulder  # Left - Right, should point to +Y after rotation
    shoulder_vec[2] = 0  # Project to XY plane
    shoulder_norm = np.linalg.norm(shoulder_vec[:2])
    
    if shoulder_norm > 0.01:
        shoulder_vec = shoulder_vec / shoulder_norm
        # Target: shoulder vector should be (0, 1, 0) = +Y
        target = np.array([0, 1, 0])
        
        # Compute rotation angle around Z axis
        # angle from shoulder_vec to target
        cos_angle = shoulder_vec[0] * target[0] + shoulder_vec[1] * target[1]
        sin_angle = shoulder_vec[0] * target[1] - shoulder_vec[1] * target[0]
        angle_z = np.arctan2(sin_angle, cos_angle)
        
        # Rotate all points around Z axis
        cos_z = np.cos(angle_z)
        sin_z = np.sin(angle_z)
        rotation_z = np.array([
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1]
        ])
        
        for i in range(len(skeleton)):
            if not np.isnan(skeleton[i, 0]):
                skeleton[i] = rotation_z @ skeleton[i]
    
    return skeleton


def extract_human_end_effectors(skeleton_3d: np.ndarray) -> dict:
    """
    Extract 5 end-effector positions and orientations from MediaPipe skeleton.
    
    Returns:
        dict with 'left_hand', 'right_hand', 'left_foot', 'right_foot', 'torso'
        Each entry has 'position' (3D), 'orientation' (quaternion wxyz), 'R_mat' (3x3)
        
    Returns None if skeleton has invalid/NaN data.
    """
    skeleton_arr = np.array(skeleton_3d)
    
    # Check for NaN in critical landmarks
    critical_landmarks = [
        MP_LEFT_WRIST, MP_RIGHT_WRIST, MP_LEFT_ANKLE, MP_RIGHT_ANKLE,
        MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, MP_LEFT_HIP, MP_RIGHT_HIP,
        MP_LEFT_INDEX, MP_RIGHT_INDEX, MP_LEFT_HEEL, MP_RIGHT_HEEL,
        MP_LEFT_FOOT_INDEX, MP_RIGHT_FOOT_INDEX, MP_LEFT_PINKY, MP_RIGHT_PINKY,
        MP_LEFT_ELBOW, MP_RIGHT_ELBOW,  # Added for elbow tracking
    ]
    for idx in critical_landmarks:
        if idx < len(skeleton_arr) and np.any(np.isnan(skeleton_arr[idx])):
            return None
    
    # Align skeleton upright first
    skeleton = align_skeleton_upright(skeleton_arr)
    
    # Pelvis = midpoint of hips (already centered at origin after alignment)
    pelvis = (skeleton[MP_LEFT_HIP] + skeleton[MP_RIGHT_HIP]) / 2
    
    end_effectors = {}
    
    # === LEFT HAND ===
    l_wrist = skeleton[MP_LEFT_WRIST]
    l_index = skeleton[MP_LEFT_INDEX]
    l_pinky = skeleton[MP_LEFT_PINKY]
    l_elbow = skeleton[MP_LEFT_ELBOW]
    
    l_hand_forward = normalize(l_index - l_wrist)  # fingers pointing
    l_hand_right = normalize(l_pinky - l_index)    # across palm
    l_hand_up = normalize(np.cross(l_hand_forward, l_hand_right))  # palm normal
    l_hand_right = normalize(np.cross(l_hand_up, l_hand_forward))  # reorthogonalize
    
    l_hand_R = axes_to_rotation_matrix(l_hand_forward, l_hand_up, l_hand_right)
    end_effectors['left_hand'] = {
        'position': l_wrist - pelvis,
        'orientation': rotation_matrix_to_quat(l_hand_R),
        'R_mat': l_hand_R,
        'axes': {'forward': l_hand_forward, 'up': l_hand_up, 'right': l_hand_right}
    }
    
    # === RIGHT HAND ===
    r_wrist = skeleton[MP_RIGHT_WRIST]
    r_index = skeleton[MP_RIGHT_INDEX]
    r_pinky = skeleton[MP_RIGHT_PINKY]
    r_elbow = skeleton[MP_RIGHT_ELBOW]
    
    r_hand_forward = normalize(r_index - r_wrist)
    r_hand_right = normalize(r_pinky - r_index)
    r_hand_up = normalize(np.cross(r_hand_forward, r_hand_right))
    r_hand_right = normalize(np.cross(r_hand_up, r_hand_forward))
    
    r_hand_R = axes_to_rotation_matrix(r_hand_forward, r_hand_up, r_hand_right)
    end_effectors['right_hand'] = {
        'position': r_wrist - pelvis,
        'orientation': rotation_matrix_to_quat(r_hand_R),
        'R_mat': r_hand_R,
        'axes': {'forward': r_hand_forward, 'up': r_hand_up, 'right': r_hand_right}
    }
    
    # === LEFT ELBOW ===
    # Position only (no orientation needed for elbow)
    end_effectors['left_elbow'] = {
        'position': l_elbow - pelvis,
        'orientation': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity
        'R_mat': np.eye(3),
        'axes': None
    }
    
    # === RIGHT ELBOW ===
    end_effectors['right_elbow'] = {
        'position': r_elbow - pelvis,
        'orientation': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity
        'R_mat': np.eye(3),
        'axes': None
    }
    
    # === LEFT FOOT ===
    l_ankle = skeleton[MP_LEFT_ANKLE]
    l_heel = skeleton[MP_LEFT_HEEL]
    l_toe = skeleton[MP_LEFT_FOOT_INDEX]
    
    l_foot_forward = normalize(l_toe - l_heel)  # foot pointing direction
    # Use world up as reference, then orthogonalize
    l_foot_right = normalize(np.cross(l_foot_forward, np.array([0, 0, 1])))
    l_foot_up = normalize(np.cross(l_foot_right, l_foot_forward))
    
    l_foot_R = axes_to_rotation_matrix(l_foot_forward, l_foot_up, l_foot_right)
    end_effectors['left_foot'] = {
        'position': l_ankle - pelvis,
        'orientation': rotation_matrix_to_quat(l_foot_R),
        'R_mat': l_foot_R,
        'axes': {'forward': l_foot_forward, 'up': l_foot_up, 'right': l_foot_right}
    }
    
    # === RIGHT FOOT ===
    r_ankle = skeleton[MP_RIGHT_ANKLE]
    r_heel = skeleton[MP_RIGHT_HEEL]
    r_toe = skeleton[MP_RIGHT_FOOT_INDEX]
    
    r_foot_forward = normalize(r_toe - r_heel)
    r_foot_right = normalize(np.cross(r_foot_forward, np.array([0, 0, 1])))
    r_foot_up = normalize(np.cross(r_foot_right, r_foot_forward))
    
    r_foot_R = axes_to_rotation_matrix(r_foot_forward, r_foot_up, r_foot_right)
    end_effectors['right_foot'] = {
        'position': r_ankle - pelvis,
        'orientation': rotation_matrix_to_quat(r_foot_R),
        'R_mat': r_foot_R,
        'axes': {'forward': r_foot_forward, 'up': r_foot_up, 'right': r_foot_right}
    }
    
    # === LEFT KNEE (soft hint for leg pose) ===
    l_knee = skeleton[MP_LEFT_KNEE]
    end_effectors['left_knee'] = {
        'position': l_knee - pelvis,
        'orientation': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity
        'R_mat': np.eye(3),
        'axes': None
    }
    
    # === RIGHT KNEE (soft hint for leg pose) ===
    r_knee = skeleton[MP_RIGHT_KNEE]
    end_effectors['right_knee'] = {
        'position': r_knee - pelvis,
        'orientation': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity
        'R_mat': np.eye(3),
        'axes': None
    }
    
    # === TORSO (mid-shoulder with chest facing direction) ===
    l_shoulder = skeleton[MP_LEFT_SHOULDER]
    r_shoulder = skeleton[MP_RIGHT_SHOULDER]
    l_hip = skeleton[MP_LEFT_HIP]
    r_hip = skeleton[MP_RIGHT_HIP]
    
    neck = (l_shoulder + r_shoulder) / 2
    
    torso_up = normalize(neck - pelvis)  # spine direction
    torso_right = normalize(r_shoulder - l_shoulder)  # shoulder line
    torso_forward = normalize(np.cross(torso_up, torso_right))  # chest facing
    
    torso_R = axes_to_rotation_matrix(torso_forward, torso_up, torso_right)
    end_effectors['torso'] = {
        'position': neck - pelvis,
        'orientation': rotation_matrix_to_quat(torso_R),
        'R_mat': torso_R,
        'axes': {'forward': torso_forward, 'up': torso_up, 'right': torso_right}
    }
    
    # Compute height for scaling (ankle mid to neck)
    ankle_mid = (skeleton[MP_LEFT_ANKLE] + skeleton[MP_RIGHT_ANKLE]) / 2
    height = np.linalg.norm(neck - ankle_mid)
    
    # Compute limb lengths for per-limb scaling
    # Use segment lengths (upper + lower) rather than straight-line distance
    # This gives consistent measurements regardless of joint bend angles
    
    # Arm length: upper arm (shoulder to elbow) + forearm (elbow to wrist)
    l_upper_arm = np.linalg.norm(skeleton[MP_LEFT_ELBOW] - skeleton[MP_LEFT_SHOULDER])
    l_forearm = np.linalg.norm(skeleton[MP_LEFT_WRIST] - skeleton[MP_LEFT_ELBOW])
    r_upper_arm = np.linalg.norm(skeleton[MP_RIGHT_ELBOW] - skeleton[MP_RIGHT_SHOULDER])
    r_forearm = np.linalg.norm(skeleton[MP_RIGHT_WRIST] - skeleton[MP_RIGHT_ELBOW])
    arm_length = ((l_upper_arm + l_forearm) + (r_upper_arm + r_forearm)) / 2
    
    # Leg length: thigh (hip to knee) + shin (knee to ankle)
    l_thigh = np.linalg.norm(skeleton[MP_LEFT_KNEE] - skeleton[MP_LEFT_HIP])
    l_shin = np.linalg.norm(skeleton[MP_LEFT_ANKLE] - skeleton[MP_LEFT_KNEE])
    r_thigh = np.linalg.norm(skeleton[MP_RIGHT_KNEE] - skeleton[MP_RIGHT_HIP])
    r_shin = np.linalg.norm(skeleton[MP_RIGHT_ANKLE] - skeleton[MP_RIGHT_KNEE])
    leg_length = ((l_thigh + l_shin) + (r_thigh + r_shin)) / 2
    
    return {
        'end_effectors': end_effectors,
        'pelvis': pelvis,
        'height': height,
        'arm_length': arm_length,
        'leg_length': leg_length,
        'skeleton': skeleton,
    }


class EndEffectorIKRetargeter:
    """
    IK-based retargeting using only end-effector targets.
    
    Uses mink library for MuJoCo IK solving.
    """
    
    def __init__(
        self,
        model_path: str = None,
        solver: str = "daqp",
        damping: float = 0.5,
        max_iterations: int = 50,
        position_weight: float = 1.0,
        orientation_weight: float = 0.5,
        verbose: bool = True,
    ):
        if not HAS_MINK or not HAS_MUJOCO:
            raise RuntimeError("mink and mujoco are required. Install with: pip install mink mujoco")
        
        self.solver = solver
        self.damping = damping
        self.max_iterations = max_iterations
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.verbose = verbose
        
        # Load robot model
        if model_path is None:
            model_path = str(ROBOT_MODEL_PATH)
        
        if verbose:
            print(f"[IK] Loading robot model: {model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Get robot height for scaling
        self._compute_robot_height()
        
        # Setup mink configuration and tasks
        self._setup_ik()
        
        if verbose:
            print(f"[IK] Robot height: {self.robot_height:.3f}m")
            print(f"[IK] End-effector bodies: {list(ROBOT_EE_BODIES.keys())}")
    
    def _compute_robot_height(self):
        """Compute robot dimensions from default pose including limb lengths."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.75  # Base height
        self.data.qpos[3] = 1.0   # Quaternion w (upright)
        mujoco.mj_forward(self.model, self.data)
        
        def get_body_pos(name):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            return self.data.xpos[body_id].copy() if body_id >= 0 else np.zeros(3)
        
        # Get key positions in world frame
        self.robot_pelvis_world = get_body_pos("pelvis")
        left_ankle = get_body_pos("left_ankle_roll_link")
        right_ankle = get_body_pos("right_ankle_roll_link")
        left_shoulder = get_body_pos("left_shoulder_pitch_link")
        right_shoulder = get_body_pos("right_shoulder_pitch_link")
        left_wrist = get_body_pos("left_wrist_yaw_link")
        right_wrist = get_body_pos("right_wrist_yaw_link")
        left_hip = get_body_pos("left_hip_pitch_link")
        right_hip = get_body_pos("right_hip_pitch_link")
        left_elbow = get_body_pos("left_elbow_link")
        right_elbow = get_body_pos("right_elbow_link")
        left_knee = get_body_pos("left_knee_link")
        right_knee = get_body_pos("right_knee_link")
        
        ankle_mid = (left_ankle + right_ankle) / 2
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        
        # Robot height = ankle to shoulder mid (same as human measurement)
        self.robot_height = np.linalg.norm(shoulder_mid - ankle_mid)
        
        # Compute limb lengths as sum of segments (consistent with human measurement)
        # Arm length: upper arm (shoulder to elbow) + forearm (elbow to wrist)
        l_upper_arm = np.linalg.norm(left_elbow - left_shoulder)
        l_forearm = np.linalg.norm(left_wrist - left_elbow)
        r_upper_arm = np.linalg.norm(right_elbow - right_shoulder)
        r_forearm = np.linalg.norm(right_wrist - right_elbow)
        self.robot_arm_length = ((l_upper_arm + l_forearm) + (r_upper_arm + r_forearm)) / 2
        
        # Leg length: thigh (hip to knee) + shin (knee to ankle)
        l_thigh = np.linalg.norm(left_knee - left_hip)
        l_shin = np.linalg.norm(left_ankle - left_knee)
        r_thigh = np.linalg.norm(right_knee - right_hip)
        r_shin = np.linalg.norm(right_ankle - right_knee)
        self.robot_leg_length = ((l_thigh + l_shin) + (r_thigh + r_shin)) / 2
        
        # Store pelvis-relative positions for reference
        self.robot_ee_default = {
            'left_foot': left_ankle - self.robot_pelvis_world,
            'right_foot': right_ankle - self.robot_pelvis_world,
            'torso': shoulder_mid - self.robot_pelvis_world,
        }
        
        if self.verbose:
            print(f"[IK] Robot arm length: {self.robot_arm_length:.3f}m")
            print(f"[IK] Robot leg length: {self.robot_leg_length:.3f}m")
    
    def _setup_ik(self):
        """Setup mink IK configuration and all tasks."""
        self.configuration = mink.Configuration(self.model)
        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        
        # Setup individual components (each can be tuned independently)
        self._setup_end_effector_tasks()
        self._setup_posture_regularization()
        self._setup_waist_constraints()
        self._setup_base_freezing()
        
        # Combine all tasks for IK solver
        self.all_tasks = list(self.tasks.values())
        if self.posture_task is not None:
            self.all_tasks.append(self.posture_task)
    
    def _setup_end_effector_tasks(self):
        """
        Setup position tracking tasks for each end-effector.
        
        KEY INSIGHT: Position-only tracking for hands/elbows works much better
        than position+orientation. Orientation constraints fight against position
        goals, causing IK to fail to reach targets.
        
        Weights:
        - Hands: 100% weight (primary targets)
        - Elbows: 5% weight (soft hint to guide arm pose, prevents unnatural bends)
        - Feet: disabled (causes floating issues)
        """
        self.tasks = {}
        
        # Weight configuration (tune these to adjust behavior)
        HAND_POSITION_WEIGHT = 1.0       # Full weight for hand position
        HAND_ORIENTATION_WEIGHT = 0.0    # No orientation (was causing IK failures)
        ELBOW_POSITION_WEIGHT = 0.05     # Low weight - just a soft hint
        ELBOW_ORIENTATION_WEIGHT = 0.0   # No orientation
        
        for ee_name, body_name in ROBOT_EE_BODIES.items():
            if 'hand' in ee_name:
                position_cost = self.position_weight * HAND_POSITION_WEIGHT
                orientation_cost = HAND_ORIENTATION_WEIGHT
            elif 'elbow' in ee_name:
                position_cost = self.position_weight * ELBOW_POSITION_WEIGHT
                orientation_cost = ELBOW_ORIENTATION_WEIGHT
            else:
                # Default for other body parts (feet, etc.)
                position_cost = self.position_weight
                orientation_cost = self.orientation_weight
            
            task = mink.FrameTask(
                frame_name=body_name,
                frame_type="body",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=1.0,
            )
            self.tasks[ee_name] = task
    
    def _setup_posture_regularization(self):
        """
        Setup posture regularization to prevent joints from hitting limits.
        
        KEY INSIGHT: Without this, shoulder_roll joints were getting stuck at 
        ±129° (their limits) and couldn't recover. The posture task gently 
        pulls joints toward their default (neutral) positions, keeping them
        away from limits while still allowing tracking.
        
        Uses per-joint costs:
        - Shoulder/elbow: low cost (0.02) - allow tracking but prevent limits
        - Wrist joints: higher cost (0.1) - keep neutral (not tracked)
        - Leg joints: high cost (0.05) - keep stable (not tracked)
        - Waist/root: medium cost (0.01) - keep upright
        """
        # Per-DOF costs (length = nv = 35 for this robot)
        # nv uses 3 DOFs for orientation (not 4 like quaternion in qpos)
        # DOF layout: [x, y, z, rx, ry, rz, joint1, joint2, ...]
        # Leg DOFs breakdown:
        #   DOF 6 = left_hip_pitch, 7 = left_hip_roll, 8 = left_hip_yaw
        #   DOF 9 = left_knee
        #   DOF 10 = left_ankle_pitch, 11 = left_ankle_roll
        #   DOF 12 = right_hip_pitch, 13 = right_hip_roll, 14 = right_hip_yaw
        #   DOF 15 = right_knee
        #   DOF 16 = right_ankle_pitch, 17 = right_ankle_roll
        costs = np.ones(self.model.nv) * 0.01  # Default cost
        
        # Root position/orientation (DOFs 0-5): low cost (base is frozen anyway)
        costs[0:6] = 0.001
        
        # Leg joints - different costs for different functions:
        # Hip roll/yaw: higher cost (keep legs from splaying)
        costs[7] = 0.05   # left_hip_roll
        costs[8] = 0.05   # left_hip_yaw
        costs[13] = 0.05  # right_hip_roll
        costs[14] = 0.05  # right_hip_yaw
        
        # Hip pitch and knee: LOW cost (allow crouching/bending)
        costs[6] = 0.01   # left_hip_pitch - needed for crouching
        costs[9] = 0.01   # left_knee - needed for crouching
        costs[12] = 0.01  # right_hip_pitch - needed for crouching
        costs[15] = 0.01  # right_knee - needed for crouching
        
        # Ankles: medium cost (keep feet stable but allow some flex)
        costs[10] = 0.03  # left_ankle_pitch
        costs[11] = 0.05  # left_ankle_roll
        costs[16] = 0.03  # right_ankle_pitch
        costs[17] = 0.05  # right_ankle_roll
        
        # Waist joints (DOFs 18-20): higher cost to keep facing forward
        costs[18:21] = 0.05
        
        # Arm joints breakdown (each arm has 7 DOFs):
        # Left arm DOFs 21-27: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        # Right arm DOFs 28-34: same pattern
        
        # Shoulder and elbow (DOFs 21-24 left, 28-31 right): moderate cost
        # Provides resistance to prevent hitting joint limits
        costs[21:25] = 0.05  # Left shoulder/elbow
        costs[28:32] = 0.05  # Right shoulder/elbow
        
        # Wrist joints (DOFs 25-27 left, 32-34 right): higher cost
        # We're not tracking wrist orientation, so keep them near neutral
        costs[25:28] = 0.1   # Left wrist
        costs[32:35] = 0.1   # Right wrist
        
        self.posture_task = mink.PostureTask(model=self.model, cost=costs)
        
        # Default standing pose as target
        default_qpos = np.zeros(self.model.nq)
        default_qpos[2] = 0.75  # Standing height
        default_qpos[3] = 1.0   # Quaternion w (upright)
        self.posture_task.set_target(default_qpos)
    
    def _setup_waist_constraints(self):
        """
        Setup waist joint indices for clamping during IK.
        
        The waist roll/pitch are clamped to keep the torso upright:
        - waist_roll: ±5° (prevents leaning side to side)
        - waist_pitch: ±5° (prevents leaning forward/back)
        
        waist_yaw is NOT clamped - it's controlled by posture regularization
        so it can move freely but gently returns to center. This allows
        future use for whole-body teleop.
        """
        self.waist_joint_indices = {}
        # Only clamp roll and pitch - yaw is free to move
        for joint_name in ['waist_roll_joint', 'waist_pitch_joint']:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self.waist_joint_indices[joint_name] = self.model.jnt_qposadr[joint_id]
    
    def _setup_base_freezing(self):
        """
        Setup constraint to freeze the floating base DOFs during IK.
        
        KEY INSIGHT: Without this, the IK solver "cheats" by moving the pelvis
        (root body) to reach targets instead of using the arm joints properly.
        This causes:
        - Z position jumping around when moving arms
        - Cross-arm coupling (moving one arm affects the other via pelvis motion)
        - General instability
        
        The floating base has 6 DOFs in velocity space:
        - DOFs 0-2: Linear velocity (x, y, z)
        - DOFs 3-5: Angular velocity (roll, pitch, yaw)
        
        We freeze all 6 to force the IK to only use joint angles.
        """
        # Freeze all 6 floating base DOFs
        base_dof_indices = [0, 1, 2, 3, 4, 5]
        self.base_freezing_constraint = DofFreezingTask(
            model=self.model,
            dof_indices=base_dof_indices,
            gain=1.0,  # Strict enforcement
        )
        
        if self.verbose:
            print(f"[IK] Base freezing constraint: DOFs {base_dof_indices}")
    
    def reset_to_default(self):
        """Reset robot to default standing pose."""
        mujoco.mj_resetData(self.model, self.configuration.data)
        self.configuration.data.qpos[2] = 0.75  # Height
        self.configuration.data.qpos[3] = 1.0   # Quaternion w
        mujoco.mj_forward(self.model, self.configuration.data)
    
    def retarget(
        self,
        skeleton_3d: np.ndarray,
        human_height: float = None,
        reset_to_default: bool = True,
        fixed_base: bool = False,
    ) -> dict:
        """
        Retarget human pose to robot joint angles.
        
        Args:
            skeleton_3d: MediaPipe 33-landmark skeleton (33x3 array)
            human_height: Optional human height for scaling. If None, computed from skeleton.
            reset_to_default: Whether to reset robot to default pose before solving.
        
        Returns:
            dict with:
                - 'joint_angles_rad': dict of joint_name -> angle
                - 'qpos': full qpos array
                - 'error': final IK error
                - 'iterations': number of IK iterations
        """
        # Extract human end-effectors
        human_data = extract_human_end_effectors(skeleton_3d)
        
        if human_data is None:
            if self.verbose:
                print("[IK] Invalid skeleton data (contains NaN)")
            return {
                'joint_angles_rad': {},
                'qpos': None,
                'error': float('nan'),
                'iterations': 0,
                'scale': 1.0,
                'valid': False,
            }
        
        ee_data = human_data['end_effectors']
        
        if human_height is None:
            human_height = human_data['height']
        
        # Compute per-limb scale factors (more accurate than uniform height scaling)
        human_arm_length = human_data.get('arm_length', human_height * 0.4)
        human_leg_length = human_data.get('leg_length', human_height * 0.5)
        
        arm_scale = self.robot_arm_length / human_arm_length if human_arm_length > 0.1 else 1.0
        leg_scale = self.robot_leg_length / human_leg_length if human_leg_length > 0.1 else 1.0
        height_scale = self.robot_height / human_height if human_height > 0.1 else 1.0
        
        if self.verbose:
            print(f"[IK] Human: height={human_height:.3f}m, arm={human_arm_length:.3f}m, leg={human_leg_length:.3f}m")
            print(f"[IK] Scale: height={height_scale:.3f}, arm={arm_scale:.3f}, leg={leg_scale:.3f}")
        
        # Reset robot to default pose
        if reset_to_default:
            self.reset_to_default()
        
        # Get robot pelvis world position
        if fixed_base:
            # Keep pelvis at fixed height (for visualization/fixed-base teleop)
            robot_pelvis_z = 0.75
        else:
            # Compute robot pelvis height based on human pose
            # We want robot feet to be at ground level (Z ≈ 0)
            left_foot_rel_z = ee_data['left_foot']['position'][2] * leg_scale
            right_foot_rel_z = ee_data['right_foot']['position'][2] * leg_scale
            min_foot_rel_z = min(left_foot_rel_z, right_foot_rel_z)
            
            ground_clearance = 0.05  # Approximate ankle height above ground
            robot_pelvis_z = ground_clearance - min_foot_rel_z
            robot_pelvis_z = np.clip(robot_pelvis_z, 0.3, 0.9)
        
        # Update base height in qpos
        self.configuration.data.qpos[2] = robot_pelvis_z
        mujoco.mj_forward(self.model, self.configuration.data)
        
        pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        robot_pelvis_world = self.configuration.data.xpos[pelvis_body_id].copy()
        
        if self.verbose:
            mode = "fixed" if fixed_base else "adjusted"
            print(f"[IK] Robot pelvis at: [{robot_pelvis_world[0]:.3f}, {robot_pelvis_world[1]:.3f}, {robot_pelvis_world[2]:.3f}] ({mode})")
        
        # Set targets for each end-effector with per-limb scaling
        for ee_name, ee_info in ee_data.items():
            if ee_name not in self.tasks:
                continue
            
            # Human position is pelvis-relative
            human_pos_rel = ee_info['position'].copy()
            
            # Choose scale based on limb type
            if 'hand' in ee_name or 'elbow' in ee_name:
                scale = arm_scale
            elif 'foot' in ee_name:
                scale = leg_scale
            else:
                scale = height_scale
            
            # Scale to robot proportions (pelvis-relative)
            robot_pos_rel = human_pos_rel * scale
            
            # Convert to world coordinates by adding robot pelvis position
            target_pos = robot_pos_rel + robot_pelvis_world
            
            # For feet: clamp Z to ground level (ankle height)
            if 'foot' in ee_name:
                target_pos[2] = max(target_pos[2], 0.02)  # Ankle just above ground
                target_pos[2] = min(target_pos[2], 0.10)  # Max ankle height when grounded
            
            # Get orientation quaternion
            target_quat = ee_info['orientation']  # [w, x, y, z]
            
            # Set mink target
            rotation = mink.SO3(target_quat)
            pose = mink.SE3.from_rotation_and_translation(rotation, target_pos)
            self.tasks[ee_name].set_target(pose)
            
            if self.verbose:
                print(f"  {ee_name}: rel=[{robot_pos_rel[0]:+.3f}, {robot_pos_rel[1]:+.3f}, {robot_pos_rel[2]:+.3f}] (scale={scale:.3f})")
        
        # Solve IK iteratively
        dt = self.model.opt.timestep
        prev_error = self._compute_error()
        
        # Waist joint limits to keep torso upright (yaw is free, controlled by posture)
        max_waist_tilt = np.radians(5.0)  # ±5° for roll and pitch
        
        # Use base freezing constraint when fixed_base=True (arm-only teleop)
        # For whole-body teleop, set fixed_base=False to allow pelvis movement
        ik_constraints = [self.base_freezing_constraint] if fixed_base else None
        
        for i in range(self.max_iterations):
            vel = mink.solve_ik(
                self.configuration,
                self.all_tasks,
                dt,
                self.solver,
                damping=self.damping,
                limits=self.ik_limits,
                constraints=ik_constraints,
            )
            self.configuration.integrate_inplace(vel, dt)
            
            # Clamp waist roll and pitch to keep torso upright
            for joint_name, qpos_idx in self.waist_joint_indices.items():
                self.configuration.data.qpos[qpos_idx] = np.clip(
                    self.configuration.data.qpos[qpos_idx],
                    -max_waist_tilt,
                    max_waist_tilt
                )
            mujoco.mj_forward(self.model, self.configuration.data)
            
            curr_error = self._compute_error()
            
            # Check convergence
            if abs(prev_error - curr_error) < 0.001:
                break
            prev_error = curr_error
        
        iterations = i + 1
        
        if self.verbose:
            print(f"[IK] Converged in {iterations} iterations, error: {curr_error:.4f}")
        
        # Extract joint angles
        qpos = self.configuration.data.qpos.copy()
        
        # Fix root X/Y to 0 (we don't want base drift during standing teleop)
        qpos[0] = 0.0  # X position
        qpos[1] = 0.0  # Y position
        # qpos[2] is Z height - enforce fixed height if fixed_base was requested
        if fixed_base:
            qpos[2] = 0.75  # Fixed standing height
        # qpos[3:7] is quaternion - keep base orientation upright
        qpos[3] = 1.0  # w
        qpos[4] = 0.0  # x
        qpos[5] = 0.0  # y
        qpos[6] = 0.0  # z
        
        joint_angles_rad = {}
        for idx, joint_name in enumerate(JOINT_ORDER):
            joint_angles_rad[joint_name] = qpos[7 + idx]
        
        return {
            'joint_angles_rad': joint_angles_rad,
            'qpos': qpos,
            'error': curr_error,
            'iterations': iterations,
            'height_scale': height_scale,
            'arm_scale': arm_scale,
            'leg_scale': leg_scale,
            'valid': True,
        }
    
    def _compute_error(self) -> float:
        """Compute total IK error across all tasks."""
        errors = [task.compute_error(self.configuration) for task in self.all_tasks]
        return np.linalg.norm(np.concatenate(errors))
    
    def retarget_two_stage(
        self,
        skeleton_3d: np.ndarray,
        human_height: float = None,
        reset_to_default: bool = True,
    ) -> dict:
        """
        Two-stage IK retargeting for whole-body teleop.
        
        Stage 1: Upper body IK with fixed base
          - Solves for arm joints (shoulders, elbows, wrists)
          - Base (pelvis) frozen at fixed height
          - Waist roll/pitch constrained
        
        Stage 2: Lower body IK with base Z and yaw free
          - Arm joints frozen to Stage 1 values
          - Solves for leg joints (hips, knees, ankles)
          - Pelvis Z (height) derived from foot positions
          - Pelvis XY stays at origin
        
        This decoupling prevents lower body optimization from regressing upper body.
        
        Args:
            skeleton_3d: MediaPipe 33-landmark skeleton (33x3 array)
            human_height: Optional human height for scaling
            reset_to_default: Whether to reset robot to default pose before solving
        
        Returns:
            dict with joint angles, qpos, error, etc.
        """
        # Extract human end-effectors
        human_data = extract_human_end_effectors(skeleton_3d)
        
        if human_data is None:
            if self.verbose:
                print("[IK] Invalid skeleton data (contains NaN)")
            return {
                'joint_angles_rad': {},
                'qpos': None,
                'error': float('nan'),
                'iterations': 0,
                'valid': False,
            }
        
        ee_data = human_data['end_effectors']
        
        if human_height is None:
            human_height = human_data['height']
        
        # Compute per-limb scale factors
        human_arm_length = human_data.get('arm_length', human_height * 0.4)
        human_leg_length = human_data.get('leg_length', human_height * 0.5)
        
        arm_scale = self.robot_arm_length / human_arm_length if human_arm_length > 0.1 else 1.0
        leg_scale = self.robot_leg_length / human_leg_length if human_leg_length > 0.1 else 1.0
        height_scale = self.robot_height / human_height if human_height > 0.1 else 1.0
        
        if self.verbose:
            print(f"\n[Two-Stage IK] Starting...")
            print(f"  Human: height={human_height:.3f}m, arm={human_arm_length:.3f}m, leg={human_leg_length:.3f}m")
            print(f"  Scale: height={height_scale:.3f}, arm={arm_scale:.3f}, leg={leg_scale:.3f}")
        
        # Reset robot to default pose
        if reset_to_default:
            self.reset_to_default()
        
        # =====================================================================
        # STAGE 1: Upper Body IK (fixed base)
        # =====================================================================
        if self.verbose:
            print(f"\n[Stage 1] Upper Body IK (fixed base)")
        
        # Fixed pelvis height for Stage 1
        stage1_pelvis_z = 0.75
        self.configuration.data.qpos[2] = stage1_pelvis_z
        mujoco.mj_forward(self.model, self.configuration.data)
        
        pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        robot_pelvis_world = self.configuration.data.xpos[pelvis_body_id].copy()
        
        # Set upper body targets (hands and elbows)
        for ee_name in ROBOT_EE_BODIES_UPPER.keys():
            if ee_name not in ee_data or ee_name not in self.tasks:
                continue
            
            ee_info = ee_data[ee_name]
            human_pos_rel = ee_info['position'].copy()
            robot_pos_rel = human_pos_rel * arm_scale
            target_pos = robot_pos_rel + robot_pelvis_world
            target_quat = ee_info['orientation']
            
            rotation = mink.SO3(target_quat)
            pose = mink.SE3.from_rotation_and_translation(rotation, target_pos)
            self.tasks[ee_name].set_target(pose)
            
            if self.verbose:
                print(f"    {ee_name}: [{robot_pos_rel[0]:+.3f}, {robot_pos_rel[1]:+.3f}, {robot_pos_rel[2]:+.3f}]")
        
        # Solve Stage 1 (base frozen)
        dt = self.model.opt.timestep
        max_waist_tilt = np.radians(5.0)
        
        stage1_iterations = 0
        prev_error = self._compute_error()
        
        for i in range(self.max_iterations):
            vel = mink.solve_ik(
                self.configuration,
                self.all_tasks,
                dt,
                self.solver,
                damping=self.damping,
                limits=self.ik_limits,
                constraints=[self.base_freezing_constraint],  # Base frozen
            )
            self.configuration.integrate_inplace(vel, dt)
            
            # Clamp waist roll and pitch
            for joint_name, qpos_idx in self.waist_joint_indices.items():
                self.configuration.data.qpos[qpos_idx] = np.clip(
                    self.configuration.data.qpos[qpos_idx],
                    -max_waist_tilt, max_waist_tilt
                )
            mujoco.mj_forward(self.model, self.configuration.data)
            
            curr_error = self._compute_error()
            if abs(prev_error - curr_error) < 0.001:
                break
            prev_error = curr_error
            stage1_iterations = i + 1
        
        stage1_error = curr_error
        
        if self.verbose:
            print(f"  Stage 1 converged: {stage1_iterations} iters, error={stage1_error:.4f}")
        
        # Save Stage 1 arm joint values (DOFs 21-34 in velocity space = joints 15-28)
        # Joint indices in qpos: 7 (base) + 12 (legs) + 3 (waist) = 22 start for left arm
        # Left arm: qpos[22:29], Right arm: qpos[29:36]
        stage1_arm_qpos = self.configuration.data.qpos[22:36].copy()
        
        # =====================================================================
        # STAGE 2: Lower Body IK (base Z free, arms frozen)
        # =====================================================================
        if self.verbose:
            print(f"\n[Stage 2] Lower Body IK (base Z free, arms frozen)")
        
        # Setup lower body tracking tasks (create if not exist)
        # Feet: full weight (primary targets)
        # Knees: low weight (soft hints, like elbows for arms)
        if not hasattr(self, 'lower_body_tasks'):
            self.lower_body_tasks = {}
            
            # Weight configuration for lower body
            FOOT_POSITION_WEIGHT = 1.0    # Full weight for foot position
            KNEE_POSITION_WEIGHT = 0.3    # Moderate weight - helps guide knee bending for crouching
            
            for ee_name, body_name in ROBOT_EE_BODIES_LOWER.items():
                if 'foot' in ee_name:
                    position_cost = self.position_weight * FOOT_POSITION_WEIGHT
                elif 'knee' in ee_name:
                    position_cost = self.position_weight * KNEE_POSITION_WEIGHT
                else:
                    position_cost = self.position_weight
                
                task = mink.FrameTask(
                    frame_name=body_name,
                    frame_type="body",
                    position_cost=position_cost,
                    orientation_cost=0.0,  # Position only
                    lm_damping=1.0,
                )
                self.lower_body_tasks[ee_name] = task
            
            if self.verbose:
                print(f"    Lower body tasks: feet={FOOT_POSITION_WEIGHT}, knees={KNEE_POSITION_WEIGHT}")
        
        # Setup arm freezing constraint (freeze arm DOFs in velocity space)
        # DOFs 21-34 are left and right arm joints
        if not hasattr(self, 'arm_freezing_constraint'):
            arm_dof_indices = list(range(21, 35))  # DOFs 21-34 in velocity space
            self.arm_freezing_constraint = DofFreezingTask(
                model=self.model,
                dof_indices=arm_dof_indices,
                gain=1.0,
            )
            if self.verbose:
                print(f"    Arm freezing: DOFs {arm_dof_indices}")
        
        # Compute target pelvis height from human foot positions
        left_foot_rel_z = ee_data['left_foot']['position'][2] * leg_scale
        right_foot_rel_z = ee_data['right_foot']['position'][2] * leg_scale
        min_foot_rel_z = min(left_foot_rel_z, right_foot_rel_z)
        
        # Pelvis height = ground clearance - lowest foot relative Z
        ground_clearance = 0.05  # Ankle height above ground
        stage2_pelvis_z = ground_clearance - min_foot_rel_z
        stage2_pelvis_z = np.clip(stage2_pelvis_z, 0.4, 0.85)
        
        if self.verbose:
            print(f"    Pelvis Z: {stage1_pelvis_z:.3f} → {stage2_pelvis_z:.3f}")
        
        # Update pelvis height
        self.configuration.data.qpos[2] = stage2_pelvis_z
        mujoco.mj_forward(self.model, self.configuration.data)
        robot_pelvis_world = self.configuration.data.xpos[pelvis_body_id].copy()
        
        # Set lower body targets (feet and knees)
        for ee_name, task in self.lower_body_tasks.items():
            if ee_name not in ee_data:
                continue
            
            ee_info = ee_data[ee_name]
            human_pos_rel = ee_info['position'].copy()
            robot_pos_rel = human_pos_rel * leg_scale
            target_pos = robot_pos_rel + robot_pelvis_world
            
            # Clamp foot Z to ground level (but not knees)
            if 'foot' in ee_name:
                target_pos[2] = max(target_pos[2], 0.02)
                target_pos[2] = min(target_pos[2], 0.15)
            
            target_quat = ee_info['orientation']
            rotation = mink.SO3(target_quat)
            pose = mink.SE3.from_rotation_and_translation(rotation, target_pos)
            task.set_target(pose)
            
            if self.verbose:
                print(f"    {ee_name}: [{robot_pos_rel[0]:+.3f}, {robot_pos_rel[1]:+.3f}, {robot_pos_rel[2]:+.3f}]")
        
        # Combine tasks for Stage 2 (upper body tasks + lower body tasks + posture)
        stage2_tasks = list(self.tasks.values()) + list(self.lower_body_tasks.values())
        if self.posture_task is not None:
            stage2_tasks.append(self.posture_task)
        
        # Setup base constraint for Stage 2: freeze XY and orientation, allow Z
        # DOFs: 0=x, 1=y, 2=z, 3=rx, 4=ry, 5=rz
        # Freeze: XY (0,1) and full orientation (3,4,5)
        if not hasattr(self, 'base_xy_rot_freezing'):
            self.base_xy_rot_freezing = DofFreezingTask(
                model=self.model,
                dof_indices=[0, 1, 3, 4, 5],  # Keep Z (2) free
                gain=1.0,
            )
        
        # Solve Stage 2 (arms frozen, base XY frozen, base Z free)
        stage2_iterations = 0
        prev_error = sum(np.linalg.norm(t.compute_error(self.configuration)) for t in stage2_tasks)
        
        for i in range(self.max_iterations):
            vel = mink.solve_ik(
                self.configuration,
                stage2_tasks,
                dt,
                self.solver,
                damping=self.damping,
                limits=self.ik_limits,
                constraints=[self.arm_freezing_constraint, self.base_xy_rot_freezing],
            )
            self.configuration.integrate_inplace(vel, dt)
            
            # Re-enforce arm joint values from Stage 1 (belt and suspenders)
            self.configuration.data.qpos[22:36] = stage1_arm_qpos
            
            # Clamp waist roll and pitch
            for joint_name, qpos_idx in self.waist_joint_indices.items():
                self.configuration.data.qpos[qpos_idx] = np.clip(
                    self.configuration.data.qpos[qpos_idx],
                    -max_waist_tilt, max_waist_tilt
                )
            
            # Clamp pelvis Z to reasonable range
            self.configuration.data.qpos[2] = np.clip(
                self.configuration.data.qpos[2], 0.4, 0.85
            )
            
            mujoco.mj_forward(self.model, self.configuration.data)
            
            curr_error = sum(np.linalg.norm(t.compute_error(self.configuration)) for t in stage2_tasks)
            if abs(prev_error - curr_error) < 0.001:
                break
            prev_error = curr_error
            stage2_iterations = i + 1
        
        stage2_error = curr_error
        
        if self.verbose:
            print(f"  Stage 2 converged: {stage2_iterations} iters, error={stage2_error:.4f}")
        
        # =====================================================================
        # Extract final results
        # =====================================================================
        qpos = self.configuration.data.qpos.copy()
        
        # Fix base XY to 0 and ensure upright orientation
        qpos[0] = 0.0  # X
        qpos[1] = 0.0  # Y
        # qpos[2] is Z height from Stage 2
        qpos[3] = 1.0  # quat w
        qpos[4] = 0.0  # quat x
        qpos[5] = 0.0  # quat y
        qpos[6] = 0.0  # quat z
        
        joint_angles_rad = {}
        for idx, joint_name in enumerate(JOINT_ORDER):
            joint_angles_rad[joint_name] = qpos[7 + idx]
        
        total_iterations = stage1_iterations + stage2_iterations
        combined_error = stage1_error + stage2_error
        
        if self.verbose:
            print(f"\n[Two-Stage IK] Complete")
            print(f"  Final pelvis Z: {qpos[2]:.3f}m")
            print(f"  Total iterations: {total_iterations}")
            print(f"  Combined error: {combined_error:.4f}")
        
        return {
            'joint_angles_rad': joint_angles_rad,
            'qpos': qpos,
            'error': combined_error,
            'iterations': total_iterations,
            'stage1_error': stage1_error,
            'stage2_error': stage2_error,
            'stage1_iterations': stage1_iterations,
            'stage2_iterations': stage2_iterations,
            'height_scale': height_scale,
            'arm_scale': arm_scale,
            'leg_scale': leg_scale,
            'pelvis_z': qpos[2],
            'valid': True,
        }
    
    def get_ee_positions(self) -> dict:
        """Get current end-effector positions from robot state."""
        mujoco.mj_forward(self.model, self.configuration.data)
        
        def get_body_pos(name):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            return self.configuration.data.xpos[body_id].copy() if body_id >= 0 else np.zeros(3)
        
        pelvis = get_body_pos("pelvis")
        positions = {}
        for ee_name, body_name in ROBOT_EE_BODIES.items():
            positions[ee_name] = get_body_pos(body_name) - pelvis
        
        return positions


def load_human_pose(filepath: Path) -> dict:
    """Load a human pose JSON file."""
    with open(filepath) as f:
        return json.load(f)


def test_single_pose(pose_name: str, visualize: bool = False, two_stage: bool = False):
    """Test IK retargeting on a single captured pose."""
    # Find pose file
    pose_file = None
    for f in HUMAN_POSES_DIR.glob("*.json"):
        if pose_name in f.stem:
            pose_file = f
            break
    
    if not pose_file:
        print(f"Pose not found: {pose_name}")
        return
    
    mode_str = "Two-Stage" if two_stage else "Single-Stage (Upper Body)"
    print(f"\n{'='*60}")
    print(f"Testing pose: {pose_file.stem}")
    print(f"Mode: {mode_str}")
    print(f"{'='*60}")
    
    # Load human pose
    human_pose = load_human_pose(pose_file)
    skeleton_3d = np.array(human_pose["skeleton_3d"])
    
    # Create retargeter
    retargeter = EndEffectorIKRetargeter(verbose=True)
    
    # Retarget (choose method based on two_stage flag)
    if two_stage:
        result = retargeter.retarget_two_stage(skeleton_3d)
    else:
        result = retargeter.retarget(skeleton_3d, fixed_base=True)
    
    # Print results
    print(f"\n[Results]")
    print(f"  Height scale: {result['height_scale']:.3f}, Arm scale: {result['arm_scale']:.3f}, Leg scale: {result['leg_scale']:.3f}")
    print(f"  IK error: {result['error']:.4f}")
    print(f"  Iterations: {result['iterations']}")
    
    if two_stage:
        print(f"  Stage 1 (upper body): {result['stage1_iterations']} iters, error={result['stage1_error']:.4f}")
        print(f"  Stage 2 (lower body): {result['stage2_iterations']} iters, error={result['stage2_error']:.4f}")
        print(f"  Final pelvis Z: {result['pelvis_z']:.3f}m")
    
    print(f"\n[Joint Angles (degrees)]")
    for joint_name, angle_rad in result['joint_angles_rad'].items():
        angle_deg = np.degrees(angle_rad)
        print(f"  {joint_name}: {angle_deg:.1f}°")
    
    # Compare with human pose joint angles if available
    if "joint_angles" in human_pose:
        print(f"\n[Human pose joint angles (degrees)]")
        for name, angle in human_pose["joint_angles"].items():
            print(f"  {name}: {angle:.1f}°")
    
    # Visualize in MuJoCo if requested
    if visualize and HAS_MUJOCO:
        visualize_in_mujoco(retargeter.model, result['qpos'])
    
    return result


def test_all_poses(two_stage: bool = False):
    """Test IK retargeting on all captured poses."""
    pose_files = sorted(HUMAN_POSES_DIR.glob("*.json"))
    
    if not pose_files:
        print("No poses found")
        return
    
    mode_str = "Two-Stage" if two_stage else "Single-Stage (Upper Body)"
    print(f"\nTesting {len(pose_files)} poses with {mode_str} IK...")
    
    retargeter = EndEffectorIKRetargeter(verbose=False)
    
    results = []
    for pose_file in pose_files:
        human_pose = load_human_pose(pose_file)
        skeleton_3d = np.array(human_pose["skeleton_3d"])
        
        if two_stage:
            result = retargeter.retarget_two_stage(skeleton_3d, reset_to_default=True)
        else:
            result = retargeter.retarget(skeleton_3d, reset_to_default=True, fixed_base=True)
        result['pose_name'] = pose_file.stem
        results.append(result)
        
        if result.get('valid', False):
            if two_stage:
                print(f"  {pose_file.stem}: error={result['error']:.4f}, iters={result['iterations']}, pelvis_z={result['pelvis_z']:.3f}")
            else:
                print(f"  {pose_file.stem}: error={result['error']:.4f}, iters={result['iterations']}")
        else:
            print(f"  {pose_file.stem}: INVALID (NaN in skeleton)")
    
    # Summary (only valid results)
    valid_results = [r for r in results if r.get('valid', False)]
    if valid_results:
        avg_error = np.mean([r['error'] for r in valid_results])
        avg_iters = np.mean([r['iterations'] for r in valid_results])
        print(f"\nSummary ({len(valid_results)} valid / {len(results)} total):")
        print(f"  Average error: {avg_error:.4f}")
        print(f"  Average iterations: {avg_iters:.1f}")
        print(f"  Min error: {min(r['error'] for r in valid_results):.4f}")
        print(f"  Max error: {max(r['error'] for r in valid_results):.4f}")
        
        if two_stage:
            avg_pelvis_z = np.mean([r['pelvis_z'] for r in valid_results])
            avg_s1_error = np.mean([r['stage1_error'] for r in valid_results])
            avg_s2_error = np.mean([r['stage2_error'] for r in valid_results])
            print(f"  Average pelvis Z: {avg_pelvis_z:.3f}m")
            print(f"  Average Stage 1 error: {avg_s1_error:.4f}")
            print(f"  Average Stage 2 error: {avg_s2_error:.4f}")
    else:
        print("\nNo valid poses found.")
    
    return results


def visualize_in_mujoco(model, qpos):
    """Visualize robot pose in MuJoCo viewer (kinematics only, no physics)."""
    try:
        import mujoco.viewer
    except ImportError:
        print("MuJoCo viewer not available")
        return
    
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    
    print("\nLaunching MuJoCo viewer (kinematics only)... (close window to continue)")
    print("  Robot pose is static - no physics simulation")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Only update kinematics, no physics stepping
            mujoco.mj_forward(model, data)
            viewer.sync()
            # Small sleep to avoid busy loop
            import time
            time.sleep(0.016)  # ~60 FPS


def visualize_comparison(pose_name: str):
    """
    Visualize human skeleton and robot pose side-by-side with end-effectors and orientations.
    
    Shows:
    - Human skeleton with joints and end-effector orientation axes
    - Robot skeleton with joints and end-effector orientation axes
    - Position/orientation error comparison
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    # Find pose file
    pose_file = None
    for f in HUMAN_POSES_DIR.glob("*.json"):
        if pose_name in f.stem:
            pose_file = f
            break
    
    if not pose_file:
        print(f"Pose not found: {pose_name}")
        return
    
    print(f"\nVisualizing: {pose_file.stem}")
    
    # Load human pose
    human_pose = load_human_pose(pose_file)
    skeleton_3d = np.array(human_pose["skeleton_3d"])
    
    # Extract human end-effectors (this also aligns the skeleton)
    human_data = extract_human_end_effectors(skeleton_3d)
    if human_data is None:
        print("Invalid skeleton data")
        return
    
    human_skeleton = human_data['skeleton']
    human_ee = human_data['end_effectors']
    
    # Run IK retargeting
    retargeter = EndEffectorIKRetargeter(verbose=False)
    result = retargeter.retarget(skeleton_3d)
    
    if not result.get('valid', False):
        print("IK retargeting failed")
        return
    
    # Get per-limb scales
    height_scale = result['height_scale']
    arm_scale = result['arm_scale']
    leg_scale = result['leg_scale']
    
    # Get robot end-effector positions after IK
    mujoco.mj_forward(retargeter.model, retargeter.configuration.data)
    
    def get_robot_body_pos(name):
        body_id = mujoco.mj_name2id(retargeter.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return retargeter.configuration.data.xpos[body_id].copy() if body_id >= 0 else np.zeros(3)
    
    def get_robot_body_quat(name):
        body_id = mujoco.mj_name2id(retargeter.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return retargeter.configuration.data.xquat[body_id].copy() if body_id >= 0 else np.array([1,0,0,0])
    
    def quat_to_axes(quat):
        """Convert quaternion [w,x,y,z] to forward/up/right axes."""
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x,y,z,w]
        mat = rot.as_matrix()
        return {
            'forward': mat[:, 0],  # X axis
            'right': -mat[:, 1],   # -Y axis (Y is left)
            'up': mat[:, 2],       # Z axis
        }
    
    robot_pelvis = get_robot_body_pos("pelvis")
    
    # Get robot skeleton positions (key joints)
    left_shoulder_pos = get_robot_body_pos("left_shoulder_pitch_link") - robot_pelvis
    right_shoulder_pos = get_robot_body_pos("right_shoulder_pitch_link") - robot_pelvis
    shoulder_mid = (left_shoulder_pos + right_shoulder_pos) / 2  # Computed mid-shoulder
    
    robot_joints = {
        'pelvis': get_robot_body_pos("pelvis") - robot_pelvis,
        'torso': shoulder_mid,  # Use computed mid-shoulder, not torso_link
        'left_shoulder': left_shoulder_pos,
        'right_shoulder': right_shoulder_pos,
        'left_elbow': get_robot_body_pos("left_elbow_link") - robot_pelvis,
        'right_elbow': get_robot_body_pos("right_elbow_link") - robot_pelvis,
        'left_wrist': get_robot_body_pos("left_wrist_yaw_link") - robot_pelvis,
        'right_wrist': get_robot_body_pos("right_wrist_yaw_link") - robot_pelvis,
        'left_hip': get_robot_body_pos("left_hip_pitch_link") - robot_pelvis,
        'right_hip': get_robot_body_pos("right_hip_pitch_link") - robot_pelvis,
        'left_knee': get_robot_body_pos("left_knee_link") - robot_pelvis,
        'right_knee': get_robot_body_pos("right_knee_link") - robot_pelvis,
        'left_ankle': get_robot_body_pos("left_ankle_roll_link") - robot_pelvis,
        'right_ankle': get_robot_body_pos("right_ankle_roll_link") - robot_pelvis,
    }
    
    # Robot end-effector orientations
    # For torso, compute from shoulder positions
    torso_right = normalize(right_shoulder_pos - left_shoulder_pos)
    torso_up = np.array([0, 0, 1])  # Approximate
    torso_forward = normalize(np.cross(torso_up, torso_right))
    
    robot_ee_orientations = {
        'left_hand': quat_to_axes(get_robot_body_quat("left_wrist_yaw_link")),
        'right_hand': quat_to_axes(get_robot_body_quat("right_wrist_yaw_link")),
        'left_foot': quat_to_axes(get_robot_body_quat("left_ankle_roll_link")),
        'right_foot': quat_to_axes(get_robot_body_quat("right_ankle_roll_link")),
        'torso': {'forward': torso_forward, 'up': torso_up, 'right': torso_right},
    }
    
    # Create figure
    fig = plt.figure(figsize=(18, 8))
    
    # Colors for end-effectors
    ee_colors = {
        'left_hand': 'red',
        'right_hand': 'orange', 
        'left_foot': 'blue',
        'right_foot': 'cyan',
        'neck': 'green',  # Renamed from 'torso' for clarity
    }
    
    # Map internal name 'torso' to display name 'neck'
    def get_display_name(name):
        return 'neck' if name == 'torso' else name
    
    def draw_axes(ax, pos, axes, scale=0.1, alpha=0.8):
        """Draw orientation axes at a position."""
        colors = {'forward': 'red', 'up': 'green', 'right': 'blue'}
        for axis_name, color in colors.items():
            if axis_name in axes:
                direction = axes[axis_name]
                if np.linalg.norm(direction) > 0.1:
                    end = pos + direction * scale
                    ax.plot([pos[0], end[0]], [pos[1], end[1]], [pos[2], end[2]], 
                           color=color, linewidth=2, alpha=alpha)
    
    # === Human Skeleton (left subplot) ===
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title(f"Human Skeleton\n{pose_file.stem}", fontsize=12)
    
    # Scale human skeleton for comparison (use height_scale for skeleton display)
    human_skeleton_scaled = human_skeleton * height_scale
    
    # Plot human joints
    valid_mask = ~np.isnan(human_skeleton_scaled[:, 0])
    ax1.scatter(human_skeleton_scaled[valid_mask, 0], 
               human_skeleton_scaled[valid_mask, 1],
               human_skeleton_scaled[valid_mask, 2],
               c='gray', s=20, alpha=0.5)
    
    # Plot human end-effectors with per-limb scaling
    for ee_name, ee_info in human_ee.items():
        # Choose scale based on limb type
        if 'hand' in ee_name:
            ee_scale = arm_scale
        elif 'foot' in ee_name:
            ee_scale = leg_scale
        else:
            ee_scale = height_scale
        
        pos = ee_info['position'] * ee_scale
        display_name = get_display_name(ee_name)
        color = ee_colors.get(display_name, 'purple')
        ax1.scatter(pos[0], pos[1], pos[2], c=color, s=150, 
                   label=display_name, edgecolors='black', linewidths=1.5)
        draw_axes(ax1, pos, ee_info['axes'], scale=0.12)
    
    # Compute neck position for drawing spine connection
    neck_pos = human_ee['torso']['position'] * height_scale
    pelvis_pos = np.zeros(3)  # Pelvis is at origin after centering
    
    # Human skeleton connections (anatomically correct - no diagonal hip-to-shoulder)
    human_connections = [
        # Hips (pelvis width)
        (MP_LEFT_HIP, MP_RIGHT_HIP),
        # Legs
        (MP_LEFT_HIP, MP_LEFT_KNEE), (MP_LEFT_KNEE, MP_LEFT_ANKLE),
        (MP_RIGHT_HIP, MP_RIGHT_KNEE), (MP_RIGHT_KNEE, MP_RIGHT_ANKLE),
        # Arms (shoulders connected to neck separately below)
        (MP_LEFT_SHOULDER, MP_LEFT_ELBOW), (MP_LEFT_ELBOW, MP_LEFT_WRIST),
        (MP_RIGHT_SHOULDER, MP_RIGHT_ELBOW), (MP_RIGHT_ELBOW, MP_RIGHT_WRIST),
        # Note: spine and neck-to-shoulder connections drawn separately
    ]
    for i, j in human_connections:
        if i < len(human_skeleton_scaled) and j < len(human_skeleton_scaled):
            if not np.isnan(human_skeleton_scaled[i, 0]) and not np.isnan(human_skeleton_scaled[j, 0]):
                ax1.plot([human_skeleton_scaled[i, 0], human_skeleton_scaled[j, 0]],
                        [human_skeleton_scaled[i, 1], human_skeleton_scaled[j, 1]],
                        [human_skeleton_scaled[i, 2], human_skeleton_scaled[j, 2]], 
                        'gray', alpha=0.6, linewidth=2)
    
    # Draw spine (pelvis to neck) and neck to shoulders
    ax1.plot([pelvis_pos[0], neck_pos[0]], [pelvis_pos[1], neck_pos[1]], [pelvis_pos[2], neck_pos[2]],
            'gray', alpha=0.6, linewidth=2)
    
    # Connect neck to shoulders
    left_shoulder_pos = human_skeleton_scaled[MP_LEFT_SHOULDER]
    right_shoulder_pos = human_skeleton_scaled[MP_RIGHT_SHOULDER]
    if not np.isnan(left_shoulder_pos[0]):
        ax1.plot([neck_pos[0], left_shoulder_pos[0]], [neck_pos[1], left_shoulder_pos[1]], [neck_pos[2], left_shoulder_pos[2]],
                'gray', alpha=0.6, linewidth=2)
    if not np.isnan(right_shoulder_pos[0]):
        ax1.plot([neck_pos[0], right_shoulder_pos[0]], [neck_pos[1], right_shoulder_pos[1]], [neck_pos[2], right_shoulder_pos[2]],
                'gray', alpha=0.6, linewidth=2)
    
    ax1.set_xlabel('X (forward)')
    ax1.set_ylabel('Y (left)')
    ax1.set_zlabel('Z (up)')
    ax1.legend(loc='upper left', fontsize=8)
    
    # === Robot Skeleton (middle subplot) ===
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title(f"Robot Skeleton (IK Result)\nError: {result['error']:.3f}", fontsize=12)
    
    # Plot robot joints
    joint_positions = np.array(list(robot_joints.values()))
    ax2.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2],
               c='gray', s=30, alpha=0.7)
    
    # Plot robot end-effectors with orientation axes
    # Map internal 'torso' to 'neck' for display
    robot_ee_map = {
        'left_hand': robot_joints['left_wrist'],
        'right_hand': robot_joints['right_wrist'],
        'left_foot': robot_joints['left_ankle'],
        'right_foot': robot_joints['right_ankle'],
        'neck': robot_joints['torso'],  # Renamed for display
    }
    
    robot_ee_orient_display = {
        'left_hand': robot_ee_orientations['left_hand'],
        'right_hand': robot_ee_orientations['right_hand'],
        'left_foot': robot_ee_orientations['left_foot'],
        'right_foot': robot_ee_orientations['right_foot'],
        'neck': robot_ee_orientations['torso'],
    }
    
    for ee_name, pos in robot_ee_map.items():
        ax2.scatter(pos[0], pos[1], pos[2], c=ee_colors[ee_name], s=150,
                   marker='s', edgecolors='black', linewidths=1.5)
        draw_axes(ax2, pos, robot_ee_orient_display[ee_name], scale=0.12)
    
    # Also add 'neck' (torso) as a named joint for connections
    robot_joints['neck'] = robot_joints['torso']
    
    # Robot skeleton connections
    robot_connections = [
        ('pelvis', 'left_hip'), ('pelvis', 'right_hip'),
        ('pelvis', 'neck'),
        ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ]
    for j1, j2 in robot_connections:
        p1, p2 = robot_joints[j1], robot_joints[j2]
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                'darkblue', alpha=0.7, linewidth=2)
    
    ax2.set_xlabel('X (forward)')
    ax2.set_ylabel('Y (left)')
    ax2.set_zlabel('Z (up)')
    
    # === Error Comparison (right subplot) ===
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    
    # Compute errors
    error_text = "End-Effector Position Errors:\n" + "="*35 + "\n\n"
    total_pos_error = 0
    
    # Map display name to internal name for human_ee lookup
    ee_pairs = [
        ('left_hand', 'left_hand'),
        ('right_hand', 'right_hand'),
        ('left_foot', 'left_foot'),
        ('right_foot', 'right_foot'),
        ('neck', 'torso'),  # Display name -> internal name
    ]
    
    for display_name, internal_name in ee_pairs:
        # Choose scale based on limb type
        if 'hand' in internal_name:
            ee_scale = arm_scale
        elif 'foot' in internal_name:
            ee_scale = leg_scale
        else:
            ee_scale = height_scale
        
        human_pos = human_ee[internal_name]['position'] * ee_scale
        robot_pos = robot_ee_map[display_name]
        pos_error = np.linalg.norm(human_pos - robot_pos)
        total_pos_error += pos_error
        
        error_text += f"{display_name:12}: {pos_error:.3f}m\n"
        error_text += f"  Human: [{human_pos[0]:+.3f}, {human_pos[1]:+.3f}, {human_pos[2]:+.3f}]\n"
        error_text += f"  Robot: [{robot_pos[0]:+.3f}, {robot_pos[1]:+.3f}, {robot_pos[2]:+.3f}]\n\n"
    
    error_text += "="*35 + "\n"
    error_text += f"Total position error: {total_pos_error:.3f}m\n"
    error_text += f"IK solver error: {result['error']:.4f}\n"
    error_text += f"Iterations: {result['iterations']}\n"
    error_text += f"Scales: h={height_scale:.2f}, arm={arm_scale:.2f}, leg={leg_scale:.2f}\n\n"
    
    error_text += "Legend:\n"
    error_text += "  Red axis = Forward\n"
    error_text += "  Green axis = Up\n"
    error_text += "  Blue axis = Right\n"
    
    ax3.text(0.05, 0.95, error_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set equal aspect for 3D plots
    all_positions = np.vstack([human_skeleton_scaled[valid_mask], joint_positions])
    max_range = np.abs(all_positions).max() * 1.2
    max_range = max(max_range, 0.8)
    
    for ax in [ax1, ax2]:
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="End-Effector IK Retargeting")
    parser.add_argument("--pose", "-p", type=str, help="Pose name to test")
    parser.add_argument("--all", "-a", action="store_true", help="Test all poses")
    parser.add_argument("--viz", "-v", action="store_true", help="Visualize in MuJoCo")
    parser.add_argument("--compare", "-c", action="store_true", 
                       help="Visualize human vs robot side-by-side (matplotlib)")
    parser.add_argument("--list", "-l", action="store_true", help="List available poses")
    parser.add_argument("--two-stage", "-2", action="store_true",
                       help="Use two-stage IK (upper body first, then lower body)")
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable poses:")
        for f in sorted(HUMAN_POSES_DIR.glob("*.json")):
            print(f"  {f.stem}")
        return
    
    if not HAS_MINK:
        print("Error: mink is required. Install with: pip install mink")
        return
    
    if not HAS_MUJOCO:
        print("Error: mujoco is required. Install with: pip install mujoco")
        return
    
    if args.compare:
        # Side-by-side visualization
        if args.pose:
            visualize_comparison(args.pose)
        else:
            poses = sorted(HUMAN_POSES_DIR.glob("*.json"))
            if poses:
                visualize_comparison(poses[0].stem)
            else:
                print("No poses found.")
    elif args.all:
        test_all_poses(two_stage=args.two_stage)
    elif args.pose:
        test_single_pose(args.pose, visualize=args.viz, two_stage=args.two_stage)
    else:
        # Default: test first pose
        poses = sorted(HUMAN_POSES_DIR.glob("*.json"))
        if poses:
            test_single_pose(poses[0].stem, visualize=args.viz, two_stage=args.two_stage)
        else:
            print("No poses found. Capture some poses first with capture_pose_simple.py")


if __name__ == "__main__":
    main()

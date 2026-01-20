#!/usr/bin/env python3
"""
Direct MediaPipe 3D skeleton to G1 robot joint angles.

This bypasses SMPL-X conversion and computes G1 joint angles directly
from the triangulated 3D skeleton.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# MediaPipe landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors in radians."""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(cos_angle)


def compute_signed_angle(v1: np.ndarray, v2: np.ndarray, normal: np.ndarray) -> float:
    """Compute signed angle between two vectors given a normal for sign."""
    angle = compute_angle_between_vectors(v1, v2)
    cross = np.cross(v1, v2)
    if np.dot(cross, normal) < 0:
        angle = -angle
    return angle


import os
from datetime import datetime


def get_debug_logger():
    """Get or create a debug log file for joint tracking."""
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"joint_debug_{timestamp}.csv")
    
    # Create header if new file
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("timestamp,l_camera,l_send,l_actual,l_diff,r_camera,r_send,r_actual,r_diff\n")
    
    return log_file


class MediaPipeToG1Direct:
    """
    Convert MediaPipe 3D skeleton directly to G1 robot joint angles.
    
    mimic_obs format (35 dims):
    [root_vel_x, root_vel_y, root_z, roll, pitch, yaw_vel, dof_pos(29)]
    
    dof_pos order (29 dims):
    - Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (6)
    - Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (6)
    - Waist: yaw, pitch, roll (3)
    - Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw (7)
    - Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw (7)
    """
    
    # Robot's default standing pose (from DEFAULT_MIMIC_OBS)
    # These values keep the robot stable
    DEFAULT_DOF_POS = np.array([
        # Left leg (0-5)
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        # Right leg (6-11)
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        # Waist (12-14)
        0.0, 0.0, 0.0,
        # Left arm (15-21)
        0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
        # Right arm (22-28)
        0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
    ])
    
    def __init__(self, robot_height: float = 0.8, config_file: str = None):
        """
        Args:
            robot_height: Robot's pelvis height in default standing pose (meters)
            config_file: Path to joint_config.yaml (optional)
        """
        self.robot_height = robot_height
        self.last_pelvis_pos = None
        self.last_time = None
        self.last_yaw = None
        
        # Load config
        self.config = self._load_config(config_file)
        self.arms_only = self.config.get('arms_only', True)
        
        # Smoothing state for wrists (roll, pitch, yaw for each hand)
        self._smoothed_wrist_l = np.array([0.0, 0.0, 0.0])
        self._smoothed_wrist_r = np.array([0.0, 0.0, 0.0])
    
    def _load_config(self, config_file: str = None) -> dict:
        """Load joint configuration from YAML file."""
        import os
        
        # Default config
        default_config = {
            'arms_only': True,
            'elbow': {
                'human_max_angle': 150,
                'robot_max_straight': 2.0,
                'robot_min_bent': 0.0,
            },
            'shoulder_pitch': {'scale': 1.0, 'offset': 0.0},
            'shoulder_roll': {'scale': 1.0, 'offset': 0.0}
        }
        
        if config_file is None:
            # Try calibration directory first, then local
            calibration_dir = os.path.join(os.path.dirname(__file__), '..', 'calibration')
            config_file = os.path.join(calibration_dir, 'joint_config.yaml')
            if not os.path.exists(config_file):
                # Fallback to local directory
                config_file = os.path.join(os.path.dirname(__file__), 'joint_config.yaml')
        
        if os.path.exists(config_file):
            try:
                import yaml
                with open(config_file, 'r') as f:
                    loaded = yaml.safe_load(f)
                    if loaded:
                        # Merge with defaults
                        for key, value in loaded.items():
                            if isinstance(value, dict) and key in default_config:
                                default_config[key].update(value)
                            else:
                                default_config[key] = value
                        print(f"[INFO] Loaded joint config from {config_file}")
            except Exception as e:
                print(f"[WARN] Failed to load config: {e}, using defaults")
        
        return default_config
    
    def compute_wrist_angles(self, forearm_vec: np.ndarray, hand_landmarks: np.ndarray, 
                               body_up: np.ndarray = None, debug: bool = False) -> Tuple[float, float, float]:
        """
        Compute wrist roll, pitch, yaw from hand landmarks relative to forearm.
        
        Args:
            forearm_vec: (3,) forearm direction vector (elbow to wrist)
            hand_landmarks: (21, 3) MediaPipe hand landmarks
            body_up: (3,) body up direction for reference
            debug: print debug info
        
        Returns:
            (roll, pitch, yaw) in radians
        """
        if hand_landmarks is None or len(hand_landmarks) < 21:
            return 0.0, 0.0, 0.0
        
        # Hand landmark indices
        WRIST = 0
        INDEX_MCP = 5
        MIDDLE_MCP = 9
        PINKY_MCP = 17
        
        wrist = hand_landmarks[WRIST]
        index_mcp = hand_landmarks[INDEX_MCP]
        middle_mcp = hand_landmarks[MIDDLE_MCP]
        pinky_mcp = hand_landmarks[PINKY_MCP]
        
        # Check for valid landmarks
        if not (np.isfinite(wrist).all() and np.isfinite(middle_mcp).all() and 
                np.isfinite(index_mcp).all() and np.isfinite(pinky_mcp).all()):
            return 0.0, 0.0, 0.0
        
        # Check for degenerate hand (points too close)
        finger_len = np.linalg.norm(middle_mcp - wrist)
        hand_width = np.linalg.norm(index_mcp - pinky_mcp)
        if finger_len < 0.02 or hand_width < 0.01:  # Less than 2cm/1cm
            if debug:
                print(f"  [WRIST DEBUG] Degenerate hand: finger_len={finger_len:.3f}m, width={hand_width:.3f}m")
            return 0.0, 0.0, 0.0
        
        # Hand coordinate frame
        # Forward: fingers direction (wrist to middle finger)
        hand_forward = middle_mcp - wrist
        hand_forward = hand_forward / np.linalg.norm(hand_forward)
        
        # Right: pinky to index direction (palm width)
        hand_right = index_mcp - pinky_mcp
        hand_right = hand_right / np.linalg.norm(hand_right)
        
        # Up: palm normal (cross product)
        hand_up = np.cross(hand_forward, hand_right)
        hand_up = hand_up / np.linalg.norm(hand_up)
        
        # Recompute right for orthogonality
        hand_right = np.cross(hand_up, hand_forward)
        
        # Forearm direction
        forearm_len = np.linalg.norm(forearm_vec)
        if forearm_len < 0.01:
            return 0.0, 0.0, 0.0
        forearm_dir = forearm_vec / forearm_len
        
        # Use body up if provided, otherwise use world up
        if body_up is None:
            body_up = np.array([0, 0, 1])
        
        # Build forearm reference frame
        # Forward: along forearm
        forearm_forward = forearm_dir
        # Right: perpendicular to forearm and body up
        forearm_right = np.cross(forearm_forward, body_up)
        if np.linalg.norm(forearm_right) < 0.01:
            forearm_right = np.array([1, 0, 0])
        forearm_right = forearm_right / np.linalg.norm(forearm_right)
        # Up: complete the frame
        forearm_up = np.cross(forearm_right, forearm_forward)
        
        # Compute wrist angles as deviation of hand frame from forearm frame
        # Roll: rotation around forearm axis (palm facing up/down)
        # Project hand_up onto forearm's up-right plane
        roll = np.arctan2(np.dot(hand_up, forearm_right), np.dot(hand_up, forearm_up))
        
        # Pitch: hand tilted up/down (flexion/extension)
        # Angle between hand_forward and forearm_forward projected onto sagittal plane
        pitch = np.arcsin(np.clip(np.dot(hand_forward, forearm_up), -1, 1))
        
        # Yaw: hand rotated left/right (radial/ulnar deviation)
        yaw = np.arcsin(np.clip(np.dot(hand_forward, forearm_right), -1, 1))
        
        if debug:
            print(f"  [WRIST DEBUG] finger_len={finger_len:.3f}m, width={hand_width:.3f}m")
            print(f"  [WRIST DEBUG] roll={np.degrees(roll):.1f}° pitch={np.degrees(pitch):.1f}° yaw={np.degrees(yaw):.1f}°")
        
        return roll, pitch, yaw
    
    def skeleton_to_mimic_obs(self, skeleton: np.ndarray, 
                               left_hand: np.ndarray = None,
                               right_hand: np.ndarray = None) -> np.ndarray:
        """
        Convert 3D skeleton to mimic_obs format.
        
        Args:
            skeleton: (33, 3) MediaPipe 3D landmarks in meters
            left_hand: (21, 3) optional left hand landmarks
            right_hand: (21, 3) optional right hand landmarks
        
        Returns:
            mimic_obs: (35,) array for robot control
        """
        import time
        
        # Store hand data for wrist computation later
        self._left_hand = left_hand
        self._right_hand = right_hand
        
        # Validate skeleton
        if skeleton is None or not np.isfinite(skeleton[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]]).all():
            return None
        
        # Key landmarks
        left_shoulder = skeleton[LEFT_SHOULDER]
        right_shoulder = skeleton[RIGHT_SHOULDER]
        left_hip = skeleton[LEFT_HIP]
        right_hip = skeleton[RIGHT_HIP]
        
        # Pelvis center
        pelvis = (left_hip + right_hip) / 2.0
        
        current_time = time.time()
        
        # Root velocity (local frame)
        if self.last_pelvis_pos is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0.001:
                velocity = (pelvis - self.last_pelvis_pos) / dt
                root_vel_x = velocity[0]  # Forward
                root_vel_y = velocity[1]  # Left
            else:
                root_vel_x, root_vel_y = 0.0, 0.0
        else:
            root_vel_x, root_vel_y = 0.0, 0.0
        
        self.last_pelvis_pos = pelvis.copy()
        self.last_time = current_time
        
        # Root height - scale to robot's coordinate system
        # Human pelvis is typically at 0.9-1.1m, robot expects ~0.8m
        human_pelvis_height = pelvis[2]
        # Scale factor: assume human standing has pelvis at ~1.0m
        scale = self.robot_height / 1.0
        root_z = human_pelvis_height * scale
        # Clamp to reasonable range
        root_z = np.clip(root_z, 0.5, 1.0)
        
        # Roll and pitch from torso orientation
        roll, pitch, yaw = self._compute_torso_orientation(skeleton)
        
        # Yaw velocity
        if self.last_yaw is not None and self.last_time is not None:
            yaw_vel = (yaw - self.last_yaw) / dt if dt > 0.001 else 0.0
        else:
            yaw_vel = 0.0
        self.last_yaw = yaw
        
        # --- Joint angles (29 dims) ---
        dof_pos = self._compute_joint_angles(skeleton)
        
        # Build mimic_obs
        # Build mimic_obs
        if self.arms_only:
            # Use defaults for root and legs/torso, only keep computed arms
            mimic_obs = np.concatenate([
                [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],  # Fixed root state
                self.DEFAULT_DOF_POS.copy()
            ])
            # Override only arm joints (15-21 and 22-28) with computed values
            mimic_obs[6+15:6+22] = dof_pos[15:22]  # Left arm
            mimic_obs[6+22:6+29] = dof_pos[22:29]  # Right arm
        else:
            mimic_obs = np.concatenate([
                [root_vel_x, root_vel_y, root_z, roll, pitch, yaw_vel],
                dof_pos
            ])
        
        return mimic_obs
    
    def _compute_torso_orientation(self, skeleton: np.ndarray) -> Tuple[float, float, float]:
        """Compute torso roll, pitch, yaw from skeleton."""
        left_shoulder = skeleton[LEFT_SHOULDER]
        right_shoulder = skeleton[RIGHT_SHOULDER]
        left_hip = skeleton[LEFT_HIP]
        right_hip = skeleton[RIGHT_HIP]
        
        # Torso coordinate frame
        # X-axis: left to right (shoulder direction)
        x_axis = right_shoulder - left_shoulder
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
        
        # Y-axis: up (hip to shoulder)
        hip_center = (left_hip + right_hip) / 2.0
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        y_axis = shoulder_center - hip_center
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
        
        # Z-axis: forward
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
        
        # Recompute y for orthogonality
        y_axis = np.cross(z_axis, x_axis)
        
        # Build rotation matrix and extract Euler angles
        R_mat = np.column_stack([x_axis, y_axis, z_axis])
        
        try:
            r = R.from_matrix(R_mat)
            euler = r.as_euler('xyz', degrees=False)
            return euler[0], euler[1], euler[2]  # roll, pitch, yaw
        except:
            return 0.0, 0.0, 0.0
    
    def _compute_joint_angles(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Compute G1 joint angles from skeleton using body-relative coordinates.
        
        This computes joint angles relative to the body's own coordinate frame,
        making it robust to different world coordinate systems.
        """
        dof_pos = np.zeros(29)
        
        # Get key points
        left_shoulder = skeleton[LEFT_SHOULDER]
        right_shoulder = skeleton[RIGHT_SHOULDER]
        left_elbow = skeleton[LEFT_ELBOW]
        right_elbow = skeleton[RIGHT_ELBOW]
        left_wrist = skeleton[LEFT_WRIST]
        right_wrist = skeleton[RIGHT_WRIST]
        left_hip = skeleton[LEFT_HIP]
        right_hip = skeleton[RIGHT_HIP]
        left_knee = skeleton[LEFT_KNEE]
        right_knee = skeleton[RIGHT_KNEE]
        left_ankle = skeleton[LEFT_ANKLE]
        right_ankle = skeleton[RIGHT_ANKLE]
        
        # --- Build body-local coordinate frame ---
        # Use hip-to-shoulder as "up" direction (robust to world coords)
        hip_center = (left_hip + right_hip) / 2.0
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        
        # Up: from hips to shoulders
        body_up = shoulder_center - hip_center
        body_up = body_up / (np.linalg.norm(body_up) + 1e-6)
        
        # Right: from left hip to right hip
        body_right = right_hip - left_hip
        body_right = body_right / (np.linalg.norm(body_right) + 1e-6)
        
        # Forward: cross product (right x up = forward)
        body_forward = np.cross(body_right, body_up)
        body_forward = body_forward / (np.linalg.norm(body_forward) + 1e-6)
        
        # Recompute right for orthogonality
        body_right = np.cross(body_up, body_forward)
        
        # Body down (for leg computations)
        body_down = -body_up
        
        # --- Left Leg (indices 0-5) ---
        thigh_vec = left_knee - left_hip
        thigh_vec_norm = thigh_vec / (np.linalg.norm(thigh_vec) + 1e-6)
        
        # Hip pitch: angle between thigh and body_down, projected onto sagittal plane
        # Project thigh onto sagittal plane (forward-up plane)
        thigh_sagittal = thigh_vec - np.dot(thigh_vec, body_right) * body_right
        thigh_sagittal = thigh_sagittal / (np.linalg.norm(thigh_sagittal) + 1e-6)
        dof_pos[0] = compute_signed_angle(body_down, thigh_sagittal, body_right)
        
        # Hip roll: lateral splay
        thigh_frontal = thigh_vec - np.dot(thigh_vec, body_forward) * body_forward
        thigh_frontal = thigh_frontal / (np.linalg.norm(thigh_frontal) + 1e-6)
        dof_pos[1] = compute_signed_angle(body_down, thigh_frontal, body_forward)
        
        dof_pos[2] = 0.0  # hip yaw
        
        # Knee: angle between thigh and shin
        shin_vec = left_ankle - left_knee
        knee_angle = np.pi - compute_angle_between_vectors(thigh_vec, shin_vec)
        dof_pos[3] = knee_angle
        
        dof_pos[4] = 0.0  # ankle pitch
        dof_pos[5] = 0.0  # ankle roll
        
        # --- Right Leg (indices 6-11) ---
        thigh_vec = right_knee - right_hip
        
        thigh_sagittal = thigh_vec - np.dot(thigh_vec, body_right) * body_right
        thigh_sagittal = thigh_sagittal / (np.linalg.norm(thigh_sagittal) + 1e-6)
        dof_pos[6] = compute_signed_angle(body_down, thigh_sagittal, body_right)
        
        thigh_frontal = thigh_vec - np.dot(thigh_vec, body_forward) * body_forward
        thigh_frontal = thigh_frontal / (np.linalg.norm(thigh_frontal) + 1e-6)
        dof_pos[7] = -compute_signed_angle(body_down, thigh_frontal, body_forward)  # Flip sign for right
        
        dof_pos[8] = 0.0  # hip yaw
        
        shin_vec = right_ankle - right_knee
        knee_angle = np.pi - compute_angle_between_vectors(thigh_vec, shin_vec)
        dof_pos[9] = knee_angle
        
        dof_pos[10] = 0.0  # ankle pitch
        dof_pos[11] = 0.0  # ankle roll
        
        # --- Waist (indices 12-14) ---
        dof_pos[12] = 0.0  # waist yaw
        dof_pos[13] = 0.0  # waist pitch
        dof_pos[14] = 0.0  # waist roll
        
        # --- Left Arm (indices 15-21) ---
        upper_arm = left_elbow - left_shoulder
        forearm = left_wrist - left_elbow
        
        # Shoulder pitch: rotation forward/back
        # Positive = arm forward, Negative = arm backward
        upper_arm_sagittal = upper_arm - np.dot(upper_arm, body_right) * body_right
        if np.linalg.norm(upper_arm_sagittal) > 0.01:
            upper_arm_sagittal = upper_arm_sagittal / np.linalg.norm(upper_arm_sagittal)
            # Negate to match robot convention (forward = positive)
            dof_pos[15] = -compute_signed_angle(-body_up, upper_arm_sagittal, body_right)
        
        # Shoulder roll: how far arm is from body
        upper_arm_frontal = upper_arm - np.dot(upper_arm, body_forward) * body_forward
        if np.linalg.norm(upper_arm_frontal) > 0.01:
            upper_arm_frontal = upper_arm_frontal / np.linalg.norm(upper_arm_frontal)
            # Angle from straight down
            dof_pos[16] = compute_signed_angle(-body_up, upper_arm_frontal, -body_forward)
        
        dof_pos[17] = 0.0  # shoulder yaw
        
        # Elbow: flexion angle (uses config values - all in degrees, convert to radians)
        elbow_cfg = self.config.get('elbow', {})
        human_max = elbow_cfg.get('human_max_angle', 150) * np.pi / 180  # to radians
        robot_straight = elbow_cfg.get('robot_straight', 90) * np.pi / 180  # degrees to radians
        robot_bent = elbow_cfg.get('robot_bent', -60) * np.pi / 180  # degrees to radians
        invert_elbow = elbow_cfg.get('invert', False)
        
        forearm_valid = np.isfinite(forearm).all() and np.linalg.norm(forearm) > 0.01
        if forearm_valid:
            angle_between = compute_angle_between_vectors(upper_arm, forearm)
            # Interpolate: camera 0° → robot_straight, camera human_max → robot_bent
            t = angle_between / human_max  # 0 to 1 interpolation factor
            t = np.clip(t, 0, 1)
            
            if invert_elbow:
                # Inverted: camera 0° → robot_bent, camera max → robot_straight
                robot_elbow = robot_bent + t * (robot_straight - robot_bent)
            else:
                # Normal: camera 0° → robot_straight, camera max → robot_bent
                robot_elbow = robot_straight + t * (robot_bent - robot_straight)
            
            dof_pos[18] = robot_elbow
        else:
            dof_pos[18] = 1.2  # default slightly bent
        
        # Debug: log elbow tracking
        if hasattr(self, '_debug_frame_count'):
            self._debug_frame_count += 1
        else:
            self._debug_frame_count = 0
        
        if self._debug_frame_count % 30 == 0:  # Log every 30 frames (~1 sec)
            forearm_len = np.linalg.norm(forearm) if np.isfinite(forearm).all() else 0
            raw_angle = compute_angle_between_vectors(upper_arm, forearm) if forearm_valid else 0
            
            # Try to read robot's actual position from Redis
            robot_actual_l, robot_actual_r = None, None
            robot_actual_l_str, robot_actual_r_str = "?", "?"
            diff_warning_l, diff_warning_r = "", ""
            diff_l, diff_r = 0, 0
            try:
                import redis
                import json
                r = redis.Redis(host='localhost', port=6379, decode_responses=True)
                robot_dof = r.get("robot_current_dof_pos")
                if robot_dof:
                    robot_dof = json.loads(robot_dof)
                    robot_actual_l = robot_dof[18]
                    robot_actual_r = robot_dof[25]
                    robot_actual_l_str = f"{np.degrees(robot_actual_l):.0f}°"
                    robot_actual_r_str = f"{np.degrees(robot_actual_r):.0f}°"
                    
                    # Check for big differences (>20°)
                    diff_l = abs(np.degrees(dof_pos[18] - robot_actual_l))
                    diff_r = abs(np.degrees(dof_pos[25] - robot_actual_r))
                    if diff_l > 20:
                        diff_warning_l = f" ⚠ DIFF={diff_l:.0f}°"
                    if diff_r > 20:
                        diff_warning_r = f" ⚠ DIFF={diff_r:.0f}°"
            except:
                pass
            
            print(f"[ELBOW] L: camera={np.degrees(raw_angle):.0f}° → send={np.degrees(dof_pos[18]):.0f}° | actual={robot_actual_l_str}{diff_warning_l}")
            
            # Store for logging after right elbow is computed
            self._log_data_l = {
                'camera': np.degrees(raw_angle),
                'send': np.degrees(dof_pos[18]),
                'actual': np.degrees(robot_actual_l) if robot_actual_l else 0,
                'diff': diff_l
            }
        
        # Left wrist (uses hand landmarks if available)
        wrist_cfg = self.config.get('wrist', {})
        wrist_enabled = wrist_cfg.get('enabled', True)
        wrist_max = wrist_cfg.get('max_angle', 60) * np.pi / 180
        
        left_wrist_detected = False
        l_wrist_raw = (0.0, 0.0, 0.0)
        wrist_debug = (self._debug_frame_count % 30 == 0)
        
        # Debug: check if hand data exists
        if wrist_debug:
            l_hand = getattr(self, '_left_hand', None)
            r_hand = getattr(self, '_right_hand', None)
            l_info = f"shape={l_hand.shape}, valid={np.isfinite(l_hand).all()}" if l_hand is not None else "None"
            r_info = f"shape={r_hand.shape}, valid={np.isfinite(r_hand).all()}" if r_hand is not None else "None"
            print(f"  [HAND DATA] L: {l_info} | R: {r_info}")
        
        if wrist_enabled and hasattr(self, '_left_hand') and self._left_hand is not None:
            roll, pitch, yaw = self.compute_wrist_angles(forearm, self._left_hand, body_up, debug=wrist_debug)
            l_wrist_raw = (roll, pitch, yaw)
            left_wrist_detected = True
            
            # Apply per-DOF config (scale, offset, invert)
            roll_cfg = wrist_cfg.get('roll', {})
            pitch_cfg = wrist_cfg.get('pitch', {})
            yaw_cfg = wrist_cfg.get('yaw', {})
            
            # Roll
            r_val = roll * roll_cfg.get('scale', 1.0)
            if roll_cfg.get('invert', False):
                r_val = -r_val
            r_val += roll_cfg.get('offset', 0) * np.pi / 180
            
            # Pitch
            p_val = pitch * pitch_cfg.get('scale', 1.0)
            if pitch_cfg.get('invert', False):
                p_val = -p_val
            p_val += pitch_cfg.get('offset', 0) * np.pi / 180
            
            # Yaw
            y_val = yaw * yaw_cfg.get('scale', 1.0)
            if yaw_cfg.get('invert', False):
                y_val = -y_val
            y_val += yaw_cfg.get('offset', 0) * np.pi / 180
            
            # Apply smoothing (exponential moving average)
            smooth_alpha = wrist_cfg.get('smoothing', 0.3)
            new_wrist = np.array([r_val, p_val, y_val])
            self._smoothed_wrist_l = smooth_alpha * new_wrist + (1 - smooth_alpha) * self._smoothed_wrist_l
            
            dof_pos[19] = np.clip(self._smoothed_wrist_l[0], -wrist_max, wrist_max)
            dof_pos[20] = np.clip(self._smoothed_wrist_l[1], -wrist_max, wrist_max)
            dof_pos[21] = np.clip(self._smoothed_wrist_l[2], -wrist_max, wrist_max)
        else:
            # Smoothly return to neutral when hand not detected
            smooth_alpha = wrist_cfg.get('smoothing', 0.3)
            self._smoothed_wrist_l = (1 - smooth_alpha) * self._smoothed_wrist_l
            dof_pos[19] = self._smoothed_wrist_l[0]
            dof_pos[20] = self._smoothed_wrist_l[1]
            dof_pos[21] = self._smoothed_wrist_l[2]
        
        # --- Right Arm (indices 22-28) ---
        upper_arm = right_elbow - right_shoulder
        forearm = right_wrist - right_elbow
        
        # Shoulder pitch: same convention as left (forward = positive)
        upper_arm_sagittal = upper_arm - np.dot(upper_arm, body_right) * body_right
        if np.linalg.norm(upper_arm_sagittal) > 0.01:
            upper_arm_sagittal = upper_arm_sagittal / np.linalg.norm(upper_arm_sagittal)
            dof_pos[22] = -compute_signed_angle(-body_up, upper_arm_sagittal, body_right)
        
        # Right shoulder roll: arm naturally points opposite direction from left,
        # so the angle computation already gives opposite sign - no need to negate
        upper_arm_frontal = upper_arm - np.dot(upper_arm, body_forward) * body_forward
        if np.linalg.norm(upper_arm_frontal) > 0.01:
            upper_arm_frontal = upper_arm_frontal / np.linalg.norm(upper_arm_frontal)
            dof_pos[23] = compute_signed_angle(-body_up, upper_arm_frontal, -body_forward)
        
        dof_pos[24] = 0.0  # shoulder yaw
        
        # Right elbow (same config as left)
        forearm_valid_r = np.isfinite(forearm).all() and np.linalg.norm(forearm) > 0.01
        if forearm_valid_r:
            angle_between = compute_angle_between_vectors(upper_arm, forearm)
            t = angle_between / human_max
            t = np.clip(t, 0, 1)
            
            if invert_elbow:
                robot_elbow = robot_bent + t * (robot_straight - robot_bent)
            else:
                robot_elbow = robot_straight + t * (robot_bent - robot_straight)
            
            dof_pos[25] = robot_elbow
        else:
            dof_pos[25] = 1.2  # default slightly bent
        
        # Debug: log right elbow
        if self._debug_frame_count % 30 == 0:
            raw_angle_r = compute_angle_between_vectors(upper_arm, forearm) if forearm_valid_r else 0
            print(f"        R: camera={np.degrees(raw_angle_r):.0f}° → send={np.degrees(dof_pos[25]):.0f}° | actual={robot_actual_r_str}{diff_warning_r}")
            
            # Write to log file
            if not hasattr(self, '_log_file'):
                self._log_file = get_debug_logger()
                print(f"[INFO] Logging to: {self._log_file}")
            
            try:
                import time
                with open(self._log_file, 'a') as f:
                    l = self._log_data_l
                    r_actual = np.degrees(robot_actual_r) if robot_actual_r else 0
                    f.write(f"{time.time():.3f},{l['camera']:.1f},{l['send']:.1f},{l['actual']:.1f},{l['diff']:.1f},"
                            f"{np.degrees(raw_angle_r):.1f},{np.degrees(dof_pos[25]):.1f},{r_actual:.1f},{diff_r:.1f}\n")
            except:
                pass
        
        # Right wrist (uses hand landmarks if available)
        right_wrist_detected = False
        r_wrist_raw = (0.0, 0.0, 0.0)
        if wrist_enabled and hasattr(self, '_right_hand') and self._right_hand is not None:
            roll, pitch, yaw = self.compute_wrist_angles(forearm, self._right_hand, body_up, debug=wrist_debug)
            r_wrist_raw = (roll, pitch, yaw)
            right_wrist_detected = True
            
            # Apply per-DOF config (scale, offset, invert) - same as left
            roll_cfg = wrist_cfg.get('roll', {})
            pitch_cfg = wrist_cfg.get('pitch', {})
            yaw_cfg = wrist_cfg.get('yaw', {})
            
            # Roll
            r_val = roll * roll_cfg.get('scale', 1.0)
            if roll_cfg.get('invert', False):
                r_val = -r_val
            r_val += roll_cfg.get('offset', 0) * np.pi / 180
            
            # Pitch
            p_val = pitch * pitch_cfg.get('scale', 1.0)
            if pitch_cfg.get('invert', False):
                p_val = -p_val
            p_val += pitch_cfg.get('offset', 0) * np.pi / 180
            
            # Yaw
            y_val = yaw * yaw_cfg.get('scale', 1.0)
            if yaw_cfg.get('invert', False):
                y_val = -y_val
            y_val += yaw_cfg.get('offset', 0) * np.pi / 180
            
            # Apply smoothing (exponential moving average)
            smooth_alpha = wrist_cfg.get('smoothing', 0.3)
            new_wrist = np.array([r_val, p_val, y_val])
            self._smoothed_wrist_r = smooth_alpha * new_wrist + (1 - smooth_alpha) * self._smoothed_wrist_r
            
            dof_pos[26] = np.clip(self._smoothed_wrist_r[0], -wrist_max, wrist_max)
            dof_pos[27] = np.clip(self._smoothed_wrist_r[1], -wrist_max, wrist_max)
            dof_pos[28] = np.clip(self._smoothed_wrist_r[2], -wrist_max, wrist_max)
        else:
            # Smoothly return to neutral when hand not detected
            smooth_alpha = wrist_cfg.get('smoothing', 0.3)
            self._smoothed_wrist_r = (1 - smooth_alpha) * self._smoothed_wrist_r
            dof_pos[26] = self._smoothed_wrist_r[0]
            dof_pos[27] = self._smoothed_wrist_r[1]
            dof_pos[28] = self._smoothed_wrist_r[2]
        
        # Debug: log wrist values every 30 frames
        if self._debug_frame_count % 30 == 0:
            l_det = "✓" if left_wrist_detected else "✗"
            r_det = "✓" if right_wrist_detected else "✗"
            print(f"[WRIST] L{l_det}: roll={np.degrees(l_wrist_raw[0]):.0f}° pitch={np.degrees(l_wrist_raw[1]):.0f}° yaw={np.degrees(l_wrist_raw[2]):.0f}° → send: r={np.degrees(dof_pos[19]):.0f}° p={np.degrees(dof_pos[20]):.0f}° y={np.degrees(dof_pos[21]):.0f}°")
            print(f"        R{r_det}: roll={np.degrees(r_wrist_raw[0]):.0f}° pitch={np.degrees(r_wrist_raw[1]):.0f}° yaw={np.degrees(r_wrist_raw[2]):.0f}° → send: r={np.degrees(dof_pos[26]):.0f}° p={np.degrees(dof_pos[27]):.0f}° y={np.degrees(dof_pos[28]):.0f}°")
        
        # Clamp all angles to reasonable ranges
        dof_pos[0:6] = np.clip(dof_pos[0:6], -1.5, 1.5)
        dof_pos[6:12] = np.clip(dof_pos[6:12], -1.5, 1.5)
        dof_pos[12:15] = np.clip(dof_pos[12:15], -0.5, 0.5)
        dof_pos[15:22] = np.clip(dof_pos[15:22], -2.0, 2.0)
        dof_pos[22:29] = np.clip(dof_pos[22:29], -2.0, 2.0)
        
        return dof_pos
    
    def _compute_arm_angles(self, skeleton: np.ndarray, dof_pos: np.ndarray):
        """
        Compute only arm joint angles (indices 15-28), modifying dof_pos in place.
        
        Leaves legs and torso at their default values for stability.
        """
        # Get key points
        left_shoulder = skeleton[LEFT_SHOULDER]
        right_shoulder = skeleton[RIGHT_SHOULDER]
        left_elbow = skeleton[LEFT_ELBOW]
        right_elbow = skeleton[RIGHT_ELBOW]
        left_wrist = skeleton[LEFT_WRIST]
        right_wrist = skeleton[RIGHT_WRIST]
        left_hip = skeleton[LEFT_HIP]
        right_hip = skeleton[RIGHT_HIP]
        
        # Build body-local coordinate frame
        hip_center = (left_hip + right_hip) / 2.0
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        
        body_up = shoulder_center - hip_center
        body_up = body_up / (np.linalg.norm(body_up) + 1e-6)
        
        body_right = right_hip - left_hip
        body_right = body_right / (np.linalg.norm(body_right) + 1e-6)
        
        body_forward = np.cross(body_right, body_up)
        body_forward = body_forward / (np.linalg.norm(body_forward) + 1e-6)
        
        body_right = np.cross(body_up, body_forward)
        
        # --- Left Arm (indices 15-21) ---
        upper_arm = left_elbow - left_shoulder
        forearm = left_wrist - left_elbow
        
        # Shoulder pitch (forward = positive)
        upper_arm_sagittal = upper_arm - np.dot(upper_arm, body_right) * body_right
        if np.linalg.norm(upper_arm_sagittal) > 0.01:
            upper_arm_sagittal = upper_arm_sagittal / np.linalg.norm(upper_arm_sagittal)
            dof_pos[15] = -compute_signed_angle(-body_up, upper_arm_sagittal, body_right)
        
        # Shoulder roll
        upper_arm_frontal = upper_arm - np.dot(upper_arm, body_forward) * body_forward
        if np.linalg.norm(upper_arm_frontal) > 0.01:
            upper_arm_frontal = upper_arm_frontal / np.linalg.norm(upper_arm_frontal)
            dof_pos[16] = compute_signed_angle(-body_up, upper_arm_frontal, -body_forward)
        
        # Elbow
        forearm_valid = np.isfinite(forearm).all() and np.linalg.norm(forearm) > 0.01
        if forearm_valid:
            dof_pos[18] = compute_angle_between_vectors(upper_arm, forearm)
        
        # Debug
        if hasattr(self, '_debug_frame_count'):
            self._debug_frame_count += 1
        else:
            self._debug_frame_count = 0
        
        if self._debug_frame_count % 30 == 0:
            forearm_len = np.linalg.norm(forearm) if np.isfinite(forearm).all() else 0
            print(f"[ARMS ONLY] L: pitch={np.degrees(dof_pos[15]):.1f}°, roll={np.degrees(dof_pos[16]):.1f}°, elbow={np.degrees(dof_pos[18]):.1f}°")
        
        # --- Right Arm (indices 22-28) ---
        upper_arm = right_elbow - right_shoulder
        forearm = right_wrist - right_elbow
        
        # Shoulder pitch
        upper_arm_sagittal = upper_arm - np.dot(upper_arm, body_right) * body_right
        if np.linalg.norm(upper_arm_sagittal) > 0.01:
            upper_arm_sagittal = upper_arm_sagittal / np.linalg.norm(upper_arm_sagittal)
            dof_pos[22] = -compute_signed_angle(-body_up, upper_arm_sagittal, body_right)
        
        # Shoulder roll
        upper_arm_frontal = upper_arm - np.dot(upper_arm, body_forward) * body_forward
        if np.linalg.norm(upper_arm_frontal) > 0.01:
            upper_arm_frontal = upper_arm_frontal / np.linalg.norm(upper_arm_frontal)
            dof_pos[23] = compute_signed_angle(-body_up, upper_arm_frontal, -body_forward)
        
        # Elbow
        forearm_valid_r = np.isfinite(forearm).all() and np.linalg.norm(forearm) > 0.01
        if forearm_valid_r:
            dof_pos[25] = compute_angle_between_vectors(upper_arm, forearm)
        
        if self._debug_frame_count % 30 == 0:
            forearm_len_r = np.linalg.norm(forearm) if np.isfinite(forearm).all() else 0
            print(f"            R: pitch={np.degrees(dof_pos[22]):.1f}°, roll={np.degrees(dof_pos[23]):.1f}°, elbow={np.degrees(dof_pos[25]):.1f}°")
        
        # Clamp arm angles
        dof_pos[15:22] = np.clip(dof_pos[15:22], -2.0, 2.0)
        dof_pos[22:29] = np.clip(dof_pos[22:29], -2.0, 2.0)


def test_direct_mapping():
    """Test the direct mapping with a simple skeleton."""
    print("Testing direct MediaPipe to G1 mapping...")
    
    # Create a simple standing skeleton (arms at sides)
    # Coordinate system: X=right, Y=forward, Z=up
    skeleton = np.zeros((33, 3))
    
    # Standing person
    skeleton[LEFT_HIP] = [-0.1, 0, 0.9]
    skeleton[RIGHT_HIP] = [0.1, 0, 0.9]
    skeleton[LEFT_KNEE] = [-0.1, 0, 0.45]
    skeleton[RIGHT_KNEE] = [0.1, 0, 0.45]
    skeleton[LEFT_ANKLE] = [-0.1, 0, 0.05]
    skeleton[RIGHT_ANKLE] = [0.1, 0, 0.05]
    skeleton[LEFT_SHOULDER] = [-0.2, 0, 1.4]
    skeleton[RIGHT_SHOULDER] = [0.2, 0, 1.4]
    # Arms hanging down (not T-pose)
    skeleton[LEFT_ELBOW] = [-0.2, 0, 1.1]
    skeleton[RIGHT_ELBOW] = [0.2, 0, 1.1]
    skeleton[LEFT_WRIST] = [-0.2, 0, 0.8]
    skeleton[RIGHT_WRIST] = [0.2, 0, 0.8]
    skeleton[NOSE] = [0, 0, 1.6]
    
    converter = MediaPipeToG1Direct()
    mimic_obs = converter.skeleton_to_mimic_obs(skeleton)
    
    if mimic_obs is not None:
        print(f"  mimic_obs shape: {mimic_obs.shape}")
        print(f"  Root: vel_x={mimic_obs[0]:.3f}, vel_y={mimic_obs[1]:.3f}, z={mimic_obs[2]:.3f}")
        print(f"  Orient: roll={np.degrees(mimic_obs[3]):.1f}°, pitch={np.degrees(mimic_obs[4]):.1f}°")
        print("\n  Legs (should be ~0° for standing):")
        print(f"    L_hip_pitch={np.degrees(mimic_obs[6]):.1f}°, L_hip_roll={np.degrees(mimic_obs[7]):.1f}°")
        print(f"    R_hip_pitch={np.degrees(mimic_obs[12]):.1f}°, R_hip_roll={np.degrees(mimic_obs[13]):.1f}°")
        print(f"    L_knee={np.degrees(mimic_obs[9]):.1f}°, R_knee={np.degrees(mimic_obs[15]):.1f}°")
        print("\n  Arms (should be ~0° for arms down):")
        print(f"    L_shoulder_pitch={np.degrees(mimic_obs[21]):.1f}°, L_shoulder_roll={np.degrees(mimic_obs[22]):.1f}°")
        print(f"    R_shoulder_pitch={np.degrees(mimic_obs[28]):.1f}°, R_shoulder_roll={np.degrees(mimic_obs[29]):.1f}°")
        print(f"    L_elbow={np.degrees(mimic_obs[24]):.1f}°, R_elbow={np.degrees(mimic_obs[31]):.1f}°")
    else:
        print("  Failed to convert!")


if __name__ == "__main__":
    test_direct_mapping()

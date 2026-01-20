"""
MediaPipe to SMPL-X Converter

Convert MediaPipe Holistic skeleton to SMPL-X format for use with GMR.
This is a geometric approximation - for better accuracy, use optimization-based fitting.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Track validation state to prevent log spam
_last_validation_state = {"right_hand": True, "left_hand": True, "hands": True, "full_body": True}


def get_default_standing_pose() -> dict:
    """
    Get a safe default standing pose for the robot.
    Used as fallback when tracking quality is poor.
    
    Returns:
        smplx_data: SMPL-X parameters for neutral standing pose
    """
    # Standing pose at ground level
    pelvis_height = 0.9  # meters
    
    smplx_data = {
        'transl': np.array([0.0, 0.0, pelvis_height]),  # Center at ground level
        'global_orient': np.array([1.0, 0.0, 0.0, 0.0]),  # No rotation (w=1, x=y=z=0)
        'body_pose': np.zeros(127),  # All joints neutral
        'left_hand_pose': np.zeros(24),
        'right_hand_pose': np.zeros(24),
        'jaw_pose': np.zeros(3),
        'leye_pose': np.zeros(3),
        'reye_pose': np.zeros(3),
    }
    
    return smplx_data


def validate_lower_body(body_landmarks: np.ndarray, min_visibility: float = 0.5) -> Tuple[bool, str]:
    """
    Validate only lower body tracking quality.
    Allows upper body to continue if only legs are problematic.
    
    Args:
        body_landmarks: (33, 3) or (33, 4) MediaPipe landmarks
        min_visibility: Minimum visibility score (0-1)
    
    Returns:
        (is_valid, reason): Tuple of validation result and failure reason
    """
    # Check lower body critical joints: hips, knees, ankles
    lower_body_joints = [23, 24, 25, 26, 27, 28]  # Left/right hip, knee, ankle
    
    if body_landmarks.shape[1] >= 4:
        visibility = body_landmarks[:, 3]
        for joint_idx in lower_body_joints:
            if visibility[joint_idx] < min_visibility:
                return False, f"Lower body joint {joint_idx} not visible"
    
    # Check if legs are in reasonable positions
    landmarks_xyz = body_landmarks[:, :3]
    left_hip = landmarks_xyz[23]
    right_hip = landmarks_xyz[24]
    
    hip_width = np.linalg.norm(right_hip - left_hip)
    if hip_width < 0.03 or hip_width > 0.5:
        return False, f"Unrealistic hip width: {hip_width:.3f}"
    
    return True, "OK"


def validate_upper_body(body_landmarks: np.ndarray, min_visibility: float = 0.5) -> Tuple[bool, str]:
    """
    Validate only upper body tracking quality.
    
    Args:
        body_landmarks: (33, 3) or (33, 4) MediaPipe landmarks
        min_visibility: Minimum visibility score (0-1)
    
    Returns:
        (is_valid, reason): Tuple of validation result and failure reason
    """
    # Check upper body critical joints: shoulders, elbows, wrists
    upper_body_joints = [11, 12, 13, 14, 15, 16]  # Left/right shoulder, elbow, wrist
    
    if body_landmarks.shape[1] >= 4:
        visibility = body_landmarks[:, 3]
        for joint_idx in upper_body_joints:
            if visibility[joint_idx] < min_visibility:
                return False, f"Upper body joint {joint_idx} not visible"
    
    # Check if arms are in reasonable positions
    landmarks_xyz = body_landmarks[:, :3]
    left_shoulder = landmarks_xyz[11]
    right_shoulder = landmarks_xyz[12]
    
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    if shoulder_width < 0.05 or shoulder_width > 0.5:
        return False, f"Unrealistic shoulder width: {shoulder_width:.3f}"
    
    return True, "OK"


def validate_right_hand_only(
    body_landmarks: np.ndarray, 
    right_hand_landmarks: Optional[np.ndarray] = None,
    min_visibility: float = 0.5
) -> Tuple[bool, str]:
    """
    Validate only right hand tracking quality for hand-only mode.
    VERY LENIENT: Only fails if hand is completely lost.
    
    Args:
        body_landmarks: (33, 3) or (33, 4) MediaPipe landmarks
        right_hand_landmarks: (21, 3) or (21, 4) MediaPipe right hand landmarks
        min_visibility: Minimum visibility score (0-1)
    
    Returns:
        (is_valid, reason): Tuple of validation result and failure reason
    """
    # Only check if right hand is completely lost
    if right_hand_landmarks is None:
        return False, "Right hand completely lost"
    
    # If we have ANY hand landmarks, consider it valid
    # MediaPipe's visibility scores are often unreliable - we trust that if 
    # landmarks exist, they're useful enough for calibration
    return True, "OK"


def validate_left_hand_only(
    body_landmarks: np.ndarray, 
    left_hand_landmarks: Optional[np.ndarray] = None,
    min_visibility: float = 0.5
) -> Tuple[bool, str]:
    """
    Validate only left hand tracking quality for hand-only mode.
    VERY LENIENT: Only fails if hand is completely lost.
    
    Args:
        body_landmarks: (33, 3) or (33, 4) MediaPipe landmarks
        left_hand_landmarks: (21, 3) or (21, 4) MediaPipe left hand landmarks
        min_visibility: Minimum visibility score (0-1)
    
    Returns:
        (is_valid, reason): Tuple of validation result and failure reason
    """
    # Only check if left hand is completely lost
    if left_hand_landmarks is None:
        return False, "Left hand completely lost"
    
    # If we have ANY hand landmarks, consider it valid
    # MediaPipe's visibility scores are often unreliable - we trust that if 
    # landmarks exist, they're useful enough for calibration
    return True, "OK"


def validate_hands_only(
    body_landmarks: np.ndarray,
    left_hand_landmarks: Optional[np.ndarray] = None,
    right_hand_landmarks: Optional[np.ndarray] = None,
    min_visibility: float = 0.5
) -> Tuple[bool, str]:
    """
    Validate both hands tracking quality for hands-only mode.
    VERY LENIENT: Only fails if both hands are completely lost.
    
    Args:
        body_landmarks: (33, 3) or (33, 4) MediaPipe landmarks
        left_hand_landmarks: (21, 3) or (21, 4) MediaPipe left hand landmarks
        right_hand_landmarks: (21, 3) or (21, 4) MediaPipe right hand landmarks
        min_visibility: Minimum visibility score (0-1)
    
    Returns:
        (is_valid, reason): Tuple of validation result and failure reason
    """
    # Only check if BOTH hands are completely lost
    if left_hand_landmarks is None and right_hand_landmarks is None:
        return False, "Both hands completely lost"
    
    # If we have ANY hand landmarks, consider it valid
    # MediaPipe's visibility scores are often unreliable - we trust that if 
    # landmarks exist, they're useful enough for calibration
    return True, "OK"


def validate_upper_body_only(
    body_landmarks: np.ndarray,
    left_hand_landmarks: Optional[np.ndarray] = None,
    right_hand_landmarks: Optional[np.ndarray] = None,
    min_visibility: float = 0.5
) -> Tuple[bool, str]:
    """
    Validate upper body tracking quality for upper-body-only mode.
    This includes: both arms, both hands, and torso/waist.
    
    Args:
        body_landmarks: (33, 3) or (33, 4) MediaPipe landmarks
        left_hand_landmarks: (21, 3) or (21, 4) MediaPipe left hand landmarks
        right_hand_landmarks: (21, 3) or (21, 4) MediaPipe right hand landmarks
        min_visibility: Minimum visibility score (0-1)
    
    Returns:
        (is_valid, reason): Tuple of validation result and failure reason
    """
    # Check both arms: shoulders, elbows, wrists
    upper_body_joints = [11, 12, 13, 14, 15, 16]  # Left/right shoulder, elbow, wrist
    
    # Also check torso reference points for waist tracking
    torso_joints = [11, 12, 23, 24]  # Shoulders and hips for spine/torso orientation
    
    if body_landmarks.shape[1] >= 4:
        visibility = body_landmarks[:, 3]
        
        # Check arms
        for joint_idx in upper_body_joints:
            if visibility[joint_idx] < min_visibility:
                return False, f"Upper body joint {joint_idx} not visible"
        
        # Check torso reference points
        for joint_idx in torso_joints:
            if visibility[joint_idx] < min_visibility:
                return False, f"Torso joint {joint_idx} not visible"
    
    # Check hands if provided
    if left_hand_landmarks is not None and left_hand_landmarks.shape[1] >= 4:
        hand_visibility = left_hand_landmarks[:, 3]
        key_hand_joints = [0, 4, 8, 20]  # Wrist, thumb tip, index tip, pinky tip
        for joint_idx in key_hand_joints:
            if hand_visibility[joint_idx] < min_visibility:
                return False, f"Left hand joint {joint_idx} not visible"
    
    if right_hand_landmarks is not None and right_hand_landmarks.shape[1] >= 4:
        hand_visibility = right_hand_landmarks[:, 3]
        key_hand_joints = [0, 4, 8, 20]
        for joint_idx in key_hand_joints:
            if hand_visibility[joint_idx] < min_visibility:
                return False, f"Right hand joint {joint_idx} not visible"
    
    return True, "OK"


def validate_pose_quality(
    body_landmarks: np.ndarray,
    min_visibility: float = 0.5,
    scale_factor: float = 1.7
) -> Tuple[bool, str]:
    """
    Validate pose quality before processing.
    
    Args:
        body_landmarks: (33, 3) or (33, 4) MediaPipe landmarks
        min_visibility: Minimum visibility score (0-1)
        scale_factor: Scale factor for converting to metric coordinates
    
    Returns:
        (is_valid, reason): Tuple of validation result and failure reason
    """
    # Check if we have visibility scores (4th column)
    if body_landmarks.shape[1] >= 4:
        visibility = body_landmarks[:, 3]
        
        # Critical joints that must be visible
        critical_joints = [0, 11, 12, 23, 24, 25, 26]  # Nose, shoulders, hips, knees
        
        for joint_idx in critical_joints:
            if visibility[joint_idx] < min_visibility:
                return False, f"Critical joint {joint_idx} not visible"
    
    # Extract xyz coordinates
    landmarks_xyz = body_landmarks[:, :3]
    
    # Check for reasonable body proportions
    left_shoulder = landmarks_xyz[11]
    right_shoulder = landmarks_xyz[12]
    left_hip = landmarks_xyz[23]
    right_hip = landmarks_xyz[24]
    
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    hip_width = np.linalg.norm(right_hip - left_hip)
    
    # Shoulders should be wider than ~10cm and less than ~80cm (normalized coords)
    if shoulder_width < 0.05 or shoulder_width > 0.5:
        return False, f"Unrealistic shoulder width: {shoulder_width:.3f}"
    
    # Hip width should be similar to shoulder width
    if hip_width < 0.03 or hip_width > 0.5:
        return False, f"Unrealistic hip width: {hip_width:.3f}"
    
    # Torso should be vertical-ish (shoulders above hips in y-axis)
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_center_y = (left_hip[1] + right_hip[1]) / 2
    
    if shoulder_center_y >= hip_center_y:  # MediaPipe y increases downward
        return False, "Person appears upside down or sideways"
    
    # **NEW: Check pelvis height after scaling**
    pelvis_pos = (left_hip + right_hip) / 2.0
    pelvis_height_scaled = pelvis_pos[2] * scale_factor  # Convert to meters
    
    # Pelvis should be at reasonable height for standing person (0.6m - 1.2m)
    # This catches issues where scaling is wrong or person is on the ground
    if pelvis_height_scaled < 0.6:
        return False, f"Person too low/on ground: height={pelvis_height_scaled:.2f}m (expected 0.6-1.2m)"
    
    if pelvis_height_scaled > 1.2:
        return False, f"Person unrealistically tall: height={pelvis_height_scaled:.2f}m (expected 0.6-1.2m)"
    
    return True, "OK"


def mediapipe_to_smplx(
    body_landmarks: np.ndarray,
    left_hand_landmarks: Optional[np.ndarray] = None,
    right_hand_landmarks: Optional[np.ndarray] = None,
    scale_factor: float = 1.7,  # Average human height in meters
    validate: bool = True,
    tracking_mode: str = "full_body"  # "full_body", "right_hand_only", "left_hand_only"
) -> dict:
    """
    Convert MediaPipe skeleton to SMPL-X format.
    
    Args:
        body_landmarks: (33, 3) or (33, 4) MediaPipe body landmarks (normalized [0,1])
        left_hand_landmarks: (21, 3) MediaPipe left hand landmarks
        right_hand_landmarks: (21, 3) MediaPipe right hand landmarks
        scale_factor: Scale factor to convert normalized coords to meters
        validate: Whether to validate pose quality (returns default pose if invalid)
        tracking_mode: Tracking mode - "full_body", "right_hand_only", or "left_hand_only"
    
    Returns:
        smplx_data: Dictionary with SMPL-X parameters
    """
    
    # Validate pose quality with partial safety mode and hand-only modes
    use_safe_lower_body = False
    use_safe_upper_body = False
    use_safe_left_hand = False
    use_safe_right_hand = False
    
    # Hand-only modes
    if tracking_mode == "right_hand_only":
        if validate:
            is_valid, reason = validate_right_hand_only(body_landmarks, right_hand_landmarks)
            if not is_valid:
                # Only log when state changes (to prevent spam)
                if _last_validation_state["right_hand"]:
                    logger.warning(f"⚠️  Right hand tracking lost: {reason}. Keeping previous pose.")
                    _last_validation_state["right_hand"] = False
                return None  # Return None to keep previous pose
            else:
                # Log when tracking recovers
                if not _last_validation_state["right_hand"]:
                    logger.info("✅ Right hand tracking recovered")
                    _last_validation_state["right_hand"] = True
        # Safe everything except right hand
        use_safe_lower_body = True
        use_safe_left_hand = True
    
    elif tracking_mode == "left_hand_only":
        if validate:
            is_valid, reason = validate_left_hand_only(body_landmarks, left_hand_landmarks)
            if not is_valid:
                # Only log when state changes (to prevent spam)
                if _last_validation_state["left_hand"]:
                    logger.warning(f"⚠️  Left hand tracking lost: {reason}. Keeping previous pose.")
                    _last_validation_state["left_hand"] = False
                return None  # Return None to keep previous pose
            else:
                # Log when tracking recovers
                if not _last_validation_state["left_hand"]:
                    logger.info("✅ Left hand tracking recovered")
                    _last_validation_state["left_hand"] = True
        # Safe everything except left hand
        use_safe_lower_body = True
        use_safe_right_hand = True
    
    # Hands only mode (both hands, no torso)
    elif tracking_mode == "hands_only":
        if validate:
            is_valid, reason = validate_hands_only(body_landmarks, left_hand_landmarks, right_hand_landmarks)
            if not is_valid:
                # Only log when state changes (to prevent spam)
                if _last_validation_state["hands"]:
                    logger.warning(f"⚠️  Hands tracking lost: {reason}. Keeping previous pose.")
                    _last_validation_state["hands"] = False
                return None  # Return None to keep previous pose
            else:
                # Log when tracking recovers
                if not _last_validation_state["hands"]:
                    logger.info("✅ Hands tracking recovered")
                    _last_validation_state["hands"] = True
        # Lower body safe, no torso tracking, only arms
        use_safe_lower_body = True
    
    # Upper body only mode (both arms + waist/torso)
    elif tracking_mode == "upper_body_only":
        if validate:
            is_valid, reason = validate_upper_body_only(body_landmarks, left_hand_landmarks, right_hand_landmarks)
            if not is_valid:
                logger.warning(f"⚠️  Upper body tracking poor: {reason}. Using safe default pose.")
                return get_default_standing_pose()
        # Only lower body is safe, track upper body and torso
        use_safe_lower_body = True
        logger.info("💪 UPPER BODY ONLY MODE: Tracking arms, hands, and waist; legs in safe pose")
    
    # Full body mode (default)
    elif validate and tracking_mode == "full_body":
        # Check full pose first
        is_valid, reason = validate_pose_quality(body_landmarks, scale_factor=scale_factor)
        
        if not is_valid:
            # Full pose failed - check if we can use partial tracking
            lower_valid, lower_reason = validate_lower_body(body_landmarks)
            upper_valid, upper_reason = validate_upper_body(body_landmarks)
            
            if not lower_valid and not upper_valid:
                # Both failed - keep previous pose
                if _last_validation_state["full_body"]:
                    logger.warning(f"⚠️  Full body tracking lost: {reason}. Keeping previous pose.")
                    _last_validation_state["full_body"] = False
                return None
            elif not lower_valid:
                # Only lower body failed - use safe legs but track upper body
                if _last_validation_state["full_body"]:
                    logger.warning(f"⚠️  Lower body tracking poor: {lower_reason}. Using safe legs, tracking upper body.")
                use_safe_lower_body = True
            elif not upper_valid:
                # Only upper body failed - use safe arms but track lower body
                if _last_validation_state["full_body"]:
                    logger.warning(f"⚠️  Upper body tracking poor: {upper_reason}. Using safe arms, tracking lower body.")
                use_safe_upper_body = True
        else:
            # Log recovery
            if not _last_validation_state["full_body"]:
                logger.info("✅ Full body tracking recovered")
                _last_validation_state["full_body"] = True
    
    # Extract only xyz coordinates (ignore visibility if present)
    if body_landmarks.shape[1] > 3:
        body_landmarks = body_landmarks[:, :3]
    
    # Scale landmarks to metric coordinates
    mp = body_landmarks * scale_factor
    
    # 1. Global translation (pelvis position)
    if use_safe_lower_body:
        # Lower body unsafe - use safe standing position
        pelvis = np.array([0.0, 0.0, 0.9])
        logger.debug("Using safe pelvis position")
    else:
        left_hip = mp[23]
        right_hip = mp[24]
        pelvis = (left_hip + right_hip) / 2.0
    
    # 2. Global orientation (torso orientation)
    if use_safe_lower_body:
        # Lower body unsafe - use neutral orientation
        global_orient = np.array([1.0, 0.0, 0.0, 0.0])
        logger.debug("Using safe global orientation")
    else:
        global_orient = compute_torso_orientation(mp)
    
    # 3. Body pose (joint angles)
    # In hand-only mode, start with all zeros and only compute tracked arm
    # In hands-only mode, compute both arms but no torso
    # In upper-body-only mode, compute both arms and torso but keep legs safe
    if tracking_mode in ["right_hand_only", "left_hand_only", "hands_only", "upper_body_only"]:
        body_pose = np.zeros(127)  # Start with all joints at neutral
        logger.debug(f"{tracking_mode} mode: Starting with neutral body pose")
        
        # Hands only: compute both arms but no torso
        if tracking_mode == "hands_only":
            # Compute both arms
            # Left arm
            left_shoulder_angle = compute_joint_angle_from_points(
                mp[12],  # right shoulder (for reference)
                mp[11],  # left shoulder
                mp[13]   # left elbow
            )
            body_pose[48:51] = left_shoulder_angle
            
            left_elbow_angle = compute_joint_angle_from_points(
                mp[11],  # left shoulder
                mp[13],  # left elbow
                mp[15]   # left wrist
            )
            # Negate elbow angle to fix inward/outward direction
            body_pose[54:57] = -left_elbow_angle
            
            # Right arm
            right_shoulder_angle = compute_joint_angle_from_points(
                mp[11],  # left shoulder (for reference)
                mp[12],  # right shoulder
                mp[14]   # right elbow
            )
            body_pose[51:54] = right_shoulder_angle
            
            right_elbow_angle = compute_joint_angle_from_points(
                mp[12],  # right shoulder
                mp[14],  # right elbow
                mp[16]   # right wrist
            )
            # Negate elbow angle to fix inward/outward direction
            body_pose[57:60] = -right_elbow_angle
            
            logger.debug("Computed both arms (no torso)")
        
        # Upper body only: compute both arms and torso
        elif tracking_mode == "upper_body_only":
            # Compute spine/torso angles
            spine_angle = compute_joint_angle_from_points(
                (mp[23] + mp[24]) / 2,  # Pelvis center
                (mp[11] + mp[12]) / 2,  # Shoulder center
                mp[0]  # Nose (head reference)
            )
            body_pose[0:3] = spine_angle  # Spine1
            body_pose[3:6] = spine_angle * 0.5  # Spine2
            body_pose[6:9] = spine_angle * 0.3  # Spine3
            
            # Compute both arms
            # Left arm
            left_shoulder_angle = compute_joint_angle_from_points(
                mp[12],  # right shoulder (for reference)
                mp[11],  # left shoulder
                mp[13]   # left elbow
            )
            body_pose[48:51] = left_shoulder_angle
            
            left_elbow_angle = compute_joint_angle_from_points(
                mp[11],  # left shoulder
                mp[13],  # left elbow
                mp[15]   # left wrist
            )
            # Negate elbow angle to fix inward/outward direction
            body_pose[54:57] = -left_elbow_angle
            
            # Right arm
            right_shoulder_angle = compute_joint_angle_from_points(
                mp[11],  # left shoulder (for reference)
                mp[12],  # right shoulder
                mp[14]   # right elbow
            )
            body_pose[51:54] = right_shoulder_angle
            
            right_elbow_angle = compute_joint_angle_from_points(
                mp[12],  # right shoulder
                mp[14],  # right elbow
                mp[16]   # right wrist
            )
            # Negate elbow angle to fix inward/outward direction
            body_pose[57:60] = -right_elbow_angle
            
            logger.debug("Computed both arms and torso angles")
        
        # Only compute the tracked arm's angles for hand-only modes
        elif tracking_mode == "right_hand_only":
            # Compute only right arm angles
            right_shoulder_angle = compute_joint_angle_from_points(
                mp[11],  # left shoulder (for reference)
                mp[12],  # right shoulder
                mp[14]   # right elbow
            )
            # Negate shoulder angle to fix inward/outward direction
            body_pose[51:54] = -right_shoulder_angle
            
            right_elbow_angle = compute_joint_angle_from_points(
                mp[12],  # right shoulder
                mp[14],  # right elbow
                mp[16]   # right wrist
            )
            body_pose[57:60] = right_elbow_angle
            logger.debug("Computed right arm angles only")
            
        elif tracking_mode == "left_hand_only":
            # Compute only left arm angles
            left_shoulder_angle = compute_joint_angle_from_points(
                mp[12],  # right shoulder (for reference)
                mp[11],  # left shoulder
                mp[13]   # left elbow
            )
            body_pose[48:51] = left_shoulder_angle
            
            left_elbow_angle = compute_joint_angle_from_points(
                mp[11],  # left shoulder
                mp[13],  # left elbow
                mp[15]   # left wrist
            )
            # Negate elbow angle to fix inward/outward direction
            body_pose[54:57] = -left_elbow_angle
            logger.debug("Computed left arm angles only")
    else:
        # Full body or partial modes: compute all angles normally
        body_pose = compute_body_pose(mp)
        
        # Apply partial safety mode if needed
        if use_safe_lower_body:
            # Zero out lower body joints (legs): indices 0-11
            body_pose[0:12] = 0.0  # Left leg (0-5) + Right leg (6-11)
            logger.debug("Applied safe pose to lower body joints")
        
        if use_safe_upper_body:
            # Zero out upper body joints (arms): indices 48-66
            body_pose[48:66] = 0.0  # Left arm (48-59) + Right arm (51-65)
            logger.debug("Applied safe pose to upper body joints")
    
    # 4. Hand poses
    if use_safe_upper_body:
        # If upper body is unsafe, don't track hands either
        left_hand_pose = np.zeros(24)
        right_hand_pose = np.zeros(24)
    else:
        # Track hands based on mode
        if use_safe_left_hand:
            left_hand_pose = np.zeros(24)  # Don't track left hand
        else:
            left_hand_pose = compute_hand_pose(left_hand_landmarks) if left_hand_landmarks is not None else np.zeros(24)
        
        if use_safe_right_hand:
            right_hand_pose = np.zeros(24)  # Don't track right hand
        else:
            right_hand_pose = compute_hand_pose(right_hand_landmarks) if right_hand_landmarks is not None else np.zeros(24)
    
    smplx_data = {
        'transl': pelvis,  # (3,) in meters
        'global_orient': global_orient,  # (4,) quaternion [w,x,y,z]
        'body_pose': body_pose,  # (127,) joint angles in radians
        'left_hand_pose': left_hand_pose,  # (24,)
        'right_hand_pose': right_hand_pose,  # (24,)
        # Additional fields
        'jaw_pose': np.zeros(3),
        'leye_pose': np.zeros(3),
        'reye_pose': np.zeros(3),
    }
    
    return smplx_data


def compute_torso_orientation(mp: np.ndarray) -> np.ndarray:
    """
    Compute torso orientation from body landmarks.
    
    Args:
        mp: (33, 3) MediaPipe body landmarks
    
    Returns:
        Quaternion [w, x, y, z]
    """
    # Default identity quaternion (facing forward, upright)
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Get shoulder and hip positions
    left_shoulder = mp[11]
    right_shoulder = mp[12]
    left_hip = mp[23]
    right_hip = mp[24]
    
    # Check for NaN/inf values
    if not np.isfinite(np.array([left_shoulder, right_shoulder, left_hip, right_hip])).all():
        return identity_quat
    
    # Compute torso coordinate frame
    # X-axis: left to right shoulder
    x_axis = right_shoulder - left_shoulder
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 0.01:  # Shoulders too close together
        return identity_quat
    x_axis = x_axis / x_norm
    
    # Y-axis: hips to shoulders (up)
    hip_center = (left_hip + right_hip) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    y_axis = shoulder_center - hip_center
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 0.01:  # Torso too short
        return identity_quat
    y_axis = y_axis / y_norm
    
    # Z-axis: forward (perpendicular to X and Y)
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 0.01:  # X and Y nearly parallel (degenerate geometry)
        return identity_quat
    z_axis = z_axis / z_norm
    
    # Recompute Y to ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 0.01:
        return identity_quat
    y_axis = y_axis / y_norm
    
    # Build rotation matrix
    R_mat = np.column_stack([x_axis, y_axis, z_axis])
    
    # Verify it's a valid rotation matrix (determinant should be 1)
    det = np.linalg.det(R_mat)
    if not (0.9 < det < 1.1):  # Allow small numerical errors
        return identity_quat
    
    # Convert to quaternion
    try:
        r = R.from_matrix(R_mat)
        quat = r.as_quat()  # [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]
        return quat
    except Exception:
        return identity_quat


def compute_body_pose(mp: np.ndarray) -> np.ndarray:
    """
    Compute SMPL-X body pose (127 joint parameters) from MediaPipe landmarks.
    
    This is a simplified geometric approximation. For better results, use optimization.
    
    Args:
        mp: (33, 3) MediaPipe body landmarks
    
    Returns:
        body_pose: (127,) SMPL-X body pose parameters (axis-angle representation)
    """
    body_pose = np.zeros(127)
    
    # Lower body joints
    # Left leg (hip: 0-3, knee: 6-9)
    left_hip_angle = compute_joint_angle_from_points(
        mp[24],  # right hip (for reference)
        mp[23],  # left hip
        mp[25]   # left knee
    )
    body_pose[0:3] = left_hip_angle
    
    left_knee_angle = compute_joint_angle_from_points(
        mp[23],  # left hip
        mp[25],  # left knee
        mp[27]   # left ankle
    )
    body_pose[6:9] = left_knee_angle
    
    # Right leg (hip: 3-6, knee: 9-12)
    right_hip_angle = compute_joint_angle_from_points(
        mp[23],  # left hip (for reference)
        mp[24],  # right hip
        mp[26]   # right knee
    )
    body_pose[3:6] = right_hip_angle
    
    right_knee_angle = compute_joint_angle_from_points(
        mp[24],  # right hip
        mp[26],  # right knee
        mp[28]   # right ankle
    )
    body_pose[9:12] = right_knee_angle
    
    # Spine/torso (indices 12-21 for spine1, spine2, spine3)
    # Simplified: compute spine angle from hips to shoulders
    hip_center = (mp[23] + mp[24]) / 2.0
    shoulder_center = (mp[11] + mp[12]) / 2.0
    nose = mp[0]
    
    spine_angle = compute_joint_angle_from_points(
        hip_center,
        shoulder_center,
        nose
    )
    body_pose[12:15] = spine_angle * 0.3  # Distribute across spine1
    body_pose[15:18] = spine_angle * 0.3  # spine2
    body_pose[18:21] = spine_angle * 0.4  # spine3
    
    # Upper body joints
    # Left arm (shoulder: 48-51, elbow: 54-57, wrist: 60-63)
    left_shoulder_angle = compute_joint_angle_from_points(
        mp[12],  # right shoulder (for reference)
        mp[11],  # left shoulder
        mp[13]   # left elbow
    )
    body_pose[48:51] = left_shoulder_angle
    
    left_elbow_angle = compute_joint_angle_from_points(
        mp[11],  # left shoulder
        mp[13],  # left elbow
        mp[15]   # left wrist
    )
    body_pose[54:57] = left_elbow_angle
    
    # Right arm (shoulder: 51-54, elbow: 57-60, wrist: 63-66)
    right_shoulder_angle = compute_joint_angle_from_points(
        mp[11],  # left shoulder (for reference)
        mp[12],  # right shoulder
        mp[14]   # right elbow
    )
    body_pose[51:54] = right_shoulder_angle
    
    right_elbow_angle = compute_joint_angle_from_points(
        mp[12],  # right shoulder
        mp[14],  # right elbow
        mp[16]   # right wrist
    )
    body_pose[57:60] = right_elbow_angle
    
    # Other joints (ankles, feet, hands, neck, head) left at zero for now
    # TODO: Add more joints if needed for better accuracy
    
    return body_pose


def compute_joint_angle_from_points(
    parent: np.ndarray,
    joint: np.ndarray,
    child: np.ndarray
) -> np.ndarray:
    """
    Compute joint angle (axis-angle representation) from three points.
    
    Args:
        parent: (3,) or (4,) parent joint position (visibility optional)
        joint: (3,) or (4,) current joint position (visibility optional)
        child: (3,) or (4,) child joint position (visibility optional)
    
    Returns:
        angle: (3,) axis-angle representation [axis_x * angle, axis_y * angle, axis_z * angle]
    """
    # Ensure only 3D coordinates (strip visibility if present)
    parent = parent[:3] if len(parent) > 3 else parent
    joint = joint[:3] if len(joint) > 3 else joint
    child = child[:3] if len(child) > 3 else child
    
    vec1 = parent - joint
    vec2 = child - joint
    
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-6)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-6)
    
    # Rotation axis (cross product)
    axis = np.cross(vec1_norm, vec2_norm)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:
        # Vectors are parallel, no rotation
        return np.zeros(3)
    
    axis = axis / axis_norm
    
    # Rotation angle (dot product)
    cos_angle = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # Axis-angle representation
    axis_angle = axis * angle
    
    return axis_angle


def compute_hand_pose(hand_landmarks: Optional[np.ndarray]) -> np.ndarray:
    """
    Compute hand pose parameters from MediaPipe hand landmarks.
    
    Args:
        hand_landmarks: (21, 3) or (21, 4) hand joint positions (normalized, optional visibility)
    
    Returns:
        hand_pose: (24,) hand pose parameters (simplified)
    """
    if hand_landmarks is None:
        return np.zeros(24)
    
    # Strip visibility score if present (keep only x, y, z)
    if hand_landmarks.shape[1] == 4:
        hand_landmarks = hand_landmarks[:, :3]
    
    # Scale to metric coordinates (hand is ~0.2m)
    hand = hand_landmarks * 0.2
    
    # Simplified hand pose estimation
    # SMPL-X hand has 15 joints × 3 DoF = 45 params, but we use simplified 24
    hand_pose = np.zeros(24)
    
    # Thumb (joints 1-4)
    if len(hand) > 4:
        thumb_angles = compute_finger_angles(hand[1:5])
        hand_pose[0:9] = thumb_angles.flatten()[:9]
    
    # Index finger (joints 5-8)
    if len(hand) > 8:
        index_angles = compute_finger_angles(hand[5:9])
        hand_pose[9:12] = index_angles[0]  # First joint only
    
    # Middle, ring, pinky (simplified - using default poses)
    # TODO: Implement full hand pose if needed
    
    return hand_pose


def compute_finger_angles(finger_joints: np.ndarray) -> np.ndarray:
    """
    Compute angles for finger joints.
    
    Args:
        finger_joints: (4, 3) positions of finger joints (base to tip)
    
    Returns:
        angles: (3, 3) angles for 3 joints
    """
    angles = np.zeros((3, 3))
    
    for i in range(min(3, len(finger_joints) - 1)):
        if i == 0:
            parent = finger_joints[0]  # Base
        else:
            parent = finger_joints[i-1]
        joint = finger_joints[i]
        child = finger_joints[i+1]
        
        angles[i] = compute_joint_angle_from_points(parent, joint, child)
    
    return angles


class MediaPipeToTWIST2:
    """
    Convert MediaPipe landmarks directly to TWIST2 mimic_obs format (35 dimensions).
    
    mimic_obs format:
    [root_vel_x, root_vel_y, root_z, roll, pitch, yaw_vel, dof_pos(29)]
    """
    
    def __init__(self, human_height: float = 1.7, joint_config_file: str = None):
        """
        Args:
            human_height: Human height in meters for scaling
            joint_config_file: Path to joint_mapping_config.json (optional)
        """
        self.human_height = human_height
        self.last_pelvis_pos = None
        self.last_time = None
        
        # Load joint calibration config if provided
        self.joint_mappings = None
        if joint_config_file:
            try:
                import json
                with open(joint_config_file, 'r') as f:
                    config = json.load(f)
                self.joint_mappings = config.get('joint_mappings', {})
                logger.info(f"✅ Loaded {len(self.joint_mappings)} joint calibrations from {joint_config_file}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load joint config: {e}. Using default mappings.")
    
    def convert_to_mimic_obs(self, smplx_data: dict) -> np.ndarray:
        """
        Convert SMPL-X data to TWIST2 mimic_obs format (35 dims).
        
        Args:
            smplx_data: Dictionary with 'transl', 'global_orient', 'body_pose'
        
        Returns:
            mimic_obs: (35,) array [root_vel_x, root_vel_y, root_z, roll, pitch, yaw_vel, dof_pos(29)]
        """
        # Extract data
        pelvis_pos = smplx_data['transl']  # (3,) [x, y, z]
        global_orient_quat = smplx_data['global_orient']  # (4,) [w, x, y, z]
        body_pose = smplx_data['body_pose']  # (127,)
        
        # 1. Root velocity (xy) - compute from position delta
        import time
        current_time = time.time()
        
        if self.last_pelvis_pos is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                vel = (pelvis_pos[:2] - self.last_pelvis_pos[:2]) / dt
                root_vel_x, root_vel_y = vel
            else:
                root_vel_x, root_vel_y = 0.0, 0.0
        else:
            root_vel_x, root_vel_y = 0.0, 0.0
        
        self.last_pelvis_pos = pelvis_pos.copy()
        self.last_time = current_time
        
        # 2. Root height (z position)
        root_z = pelvis_pos[2]
        
        # 3. Roll, Pitch from quaternion
        r = R.from_quat([
            global_orient_quat[1],  # x
            global_orient_quat[2],  # y
            global_orient_quat[3],  # z
            global_orient_quat[0]   # w
        ])
        euler = r.as_euler('xyz', degrees=False)  # Roll, Pitch, Yaw
        roll, pitch, yaw = euler
        
        # 4. Yaw angular velocity (simplified - could compute from delta)
        yaw_vel = 0.0  # TODO: Compute from orientation delta
        
        # 5. Joint positions (29 DoF for G1)
        # Map from SMPL-X body_pose (127) to G1 joints (29)
        dof_pos = self._map_smplx_to_g1_joints(body_pose)
        
        # Construct mimic_obs
        mimic_obs = np.array([
            root_vel_x,
            root_vel_y,
            root_z,
            roll,
            pitch,
            yaw_vel,
            *dof_pos  # 29 joint angles
        ])
        
        assert mimic_obs.shape == (35,), f"Expected mimic_obs shape (35,), got {mimic_obs.shape}"
        
        return mimic_obs
    
    def _map_smplx_to_g1_joints(self, body_pose: np.ndarray) -> np.ndarray:
        """
        Map SMPL-X body pose (127 dims) to G1 joint angles (29 dims).
        
        G1 joint order (29 joints):
        - Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (6)
        - Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (6)
        - Waist: yaw, pitch, roll (3)
        - Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw (7)
        - Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw (7)
        
        Args:
            body_pose: (127,) SMPL-X body pose in axis-angle format
        
        Returns:
            dof_pos: (29,) G1 joint angles
        """
        dof_pos = np.zeros(29)
        
        # Convert axis-angle to Euler angles for easier mapping
        # SMPL-X body pose is axis-angle (3 values per joint)
        
        # Left leg (indices 0-5)
        left_hip_aa = body_pose[0:3]  # SMPL-X left hip
        left_knee_aa = body_pose[6:9]  # SMPL-X left knee
        
        dof_pos[0] = self._axis_angle_to_single_angle(left_hip_aa, axis=1)  # pitch
        dof_pos[1] = self._axis_angle_to_single_angle(left_hip_aa, axis=0)  # roll
        dof_pos[2] = self._axis_angle_to_single_angle(left_hip_aa, axis=2)  # yaw
        dof_pos[3] = self._axis_angle_to_single_angle(left_knee_aa, axis=1)  # knee pitch
        dof_pos[4] = 0.0  # ankle pitch (simplified)
        dof_pos[5] = 0.0  # ankle roll
        
        # Right leg (indices 6-11)
        right_hip_aa = body_pose[3:6]  # SMPL-X right hip
        right_knee_aa = body_pose[9:12]  # SMPL-X right knee
        
        dof_pos[6] = self._axis_angle_to_single_angle(right_hip_aa, axis=1)
        dof_pos[7] = self._axis_angle_to_single_angle(right_hip_aa, axis=0)
        dof_pos[8] = self._axis_angle_to_single_angle(right_hip_aa, axis=2)
        dof_pos[9] = self._axis_angle_to_single_angle(right_knee_aa, axis=1)
        dof_pos[10] = 0.0  # ankle pitch
        dof_pos[11] = 0.0  # ankle roll
        
        # Waist (indices 12-14)
        spine_aa = body_pose[12:15]  # SMPL-X spine1
        dof_pos[12] = self._axis_angle_to_single_angle(spine_aa, axis=2)  # yaw
        dof_pos[13] = self._axis_angle_to_single_angle(spine_aa, axis=1)  # pitch
        dof_pos[14] = self._axis_angle_to_single_angle(spine_aa, axis=0)  # roll
        
        # Left arm (indices 15-21)
        left_shoulder_aa = body_pose[48:51]  # SMPL-X left shoulder
        left_elbow_aa = body_pose[54:57]  # SMPL-X left elbow
        
        dof_pos[15] = self._axis_angle_to_single_angle(left_shoulder_aa, axis=1)  # pitch
        dof_pos[16] = self._axis_angle_to_single_angle(left_shoulder_aa, axis=0)  # roll
        dof_pos[17] = self._axis_angle_to_single_angle(left_shoulder_aa, axis=2)  # yaw
        dof_pos[18] = self._axis_angle_to_single_angle(left_elbow_aa, axis=1)  # elbow
        dof_pos[19] = 0.0  # wrist roll
        dof_pos[20] = 0.0  # wrist pitch
        dof_pos[21] = 0.0  # wrist yaw
        
        # Right arm (indices 22-28)
        right_shoulder_aa = body_pose[51:54]  # SMPL-X right shoulder
        right_elbow_aa = body_pose[57:60]  # SMPL-X right elbow
        
        # Extract raw angles
        right_shoulder_pitch_raw = self._axis_angle_to_single_angle(right_shoulder_aa, axis=1)
        right_shoulder_roll_raw = self._axis_angle_to_single_angle(right_shoulder_aa, axis=0)
        right_shoulder_yaw_raw = self._axis_angle_to_single_angle(right_shoulder_aa, axis=2)
        right_elbow_raw = self._axis_angle_to_single_angle(right_elbow_aa, axis=1)
        
        # Apply calibration if available
        dof_pos[22] = self._apply_joint_calibration('right_shoulder_pitch', right_shoulder_pitch_raw)
        dof_pos[23] = self._apply_joint_calibration('right_shoulder_roll', right_shoulder_roll_raw)
        dof_pos[24] = self._apply_joint_calibration('right_shoulder_yaw', right_shoulder_yaw_raw)
        dof_pos[25] = self._apply_joint_calibration('right_elbow', right_elbow_raw)
        dof_pos[26] = 0.0  # wrist roll
        dof_pos[27] = 0.0  # wrist pitch
        dof_pos[28] = 0.0  # wrist yaw
        
        return dof_pos
    
    def _apply_joint_calibration(self, joint_name: str, raw_angle: float) -> float:
        """
        Apply calibrated scale/offset/clamp to a joint angle
        
        Args:
            joint_name: Name of the joint (e.g., "right_shoulder_pitch")
            raw_angle: Raw angle from MediaPipe conversion
        
        Returns:
            Calibrated angle
        """
        # If no calibration loaded, return raw angle
        if not self.joint_mappings or joint_name not in self.joint_mappings:
            return raw_angle
        
        config = self.joint_mappings[joint_name]
        
        # Check if joint is enabled
        if not config.get('enabled', True):
            return raw_angle
        
        # Apply scale and offset
        scaled = raw_angle * config['scale'] + config['offset']
        
        # Clamp to robot limits
        clamped = np.clip(scaled, config['clamp_min'], config['clamp_max'])
        
        return float(clamped)
    
    def _axis_angle_to_single_angle(self, axis_angle: np.ndarray, axis: int) -> float:
        """
        Extract rotation around specific axis from axis-angle representation.
        
        Args:
            axis_angle: (3,) axis-angle representation
            axis: 0=x, 1=y, 2=z
        
        Returns:
            angle: Rotation angle around specified axis
        """
        if np.linalg.norm(axis_angle) < 1e-6:
            return 0.0
        
        # Convert to rotation matrix
        r = R.from_rotvec(axis_angle)
        euler = r.as_euler('xyz', degrees=False)
        
        return euler[axis]


if __name__ == '__main__':
    # Test the converter
    print("Testing MediaPipe to SMPL-X converter...")
    
    # Create dummy MediaPipe landmarks (T-pose)
    mp_landmarks = np.array([
        # Simple T-pose for testing
        [0.5, 0.2, 0.5],  # 0: nose
        *[[0, 0, 0]] * 10,  # 1-10: face landmarks (simplified)
        [0.3, 0.5, 0.5],  # 11: left shoulder
        [0.7, 0.5, 0.5],  # 12: right shoulder
        [0.2, 0.5, 0.5],  # 13: left elbow
        [0.8, 0.5, 0.5],  # 14: right elbow
        [0.1, 0.5, 0.5],  # 15: left wrist
        [0.9, 0.5, 0.5],  # 16: right wrist
        *[[0, 0, 0]] * 6,  # 17-22: hand landmarks (simplified)
        [0.4, 0.8, 0.5],  # 23: left hip
        [0.6, 0.8, 0.5],  # 24: right hip
        [0.4, 1.2, 0.5],  # 25: left knee
        [0.6, 1.2, 0.5],  # 26: right knee
        [0.4, 1.6, 0.5],  # 27: left ankle
        [0.6, 1.6, 0.5],  # 28: right ankle
        *[[0, 0, 0]] * 4,  # 29-32: foot landmarks
    ])
    
    smplx_data = mediapipe_to_smplx(mp_landmarks)
    
    print(f"✓ Conversion successful!")
    print(f"  - Global translation: {smplx_data['transl']}")
    print(f"  - Global orientation: {smplx_data['global_orient']}")
    print(f"  - Body pose shape: {smplx_data['body_pose'].shape}")
    print(f"  - Left hand pose shape: {smplx_data['left_hand_pose'].shape}")
    print(f"  - Right hand pose shape: {smplx_data['right_hand_pose'].shape}")



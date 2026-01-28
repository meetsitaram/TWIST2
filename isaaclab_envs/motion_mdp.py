# Copyright (c) 2025
# Motion Imitation MDP Functions for Isaac Lab
#
# Custom observation, reward, and termination functions for
# motion imitation training.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from .g1_motion_mimic_env import G1MotionMimicEnv


##############################################################################
# MOTION TARGET STATE ACCESS
##############################################################################

def get_target_state(env: ManagerBasedRLEnv, frame_offset: int = 0):
    """Get current target state from motion library.
    
    Uses the environment's get_target_state() method which handles
    motion timing and interpolation.
    
    Args:
        env: The environment instance.
        frame_offset: Optional frame offset for temporal windowing.
                      Positive = future frames, Negative = past frames.
                      At 30fps, ±2 frames = ±66ms.
    
    Returns dict with:
        - dof_pos: target joint positions
        - dof_vel: target joint velocities
        - root_pos: target root position
        - root_rot: target root rotation (quaternion)
        - keybody_pos: target key body positions
    """
    return env.get_target_state(frame_offset=frame_offset)


##############################################################################
# OBSERVATION FUNCTIONS
##############################################################################

def target_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get target joint positions from motion library.
    
    Returns:
        Tensor of shape (num_envs, num_joints)
    """
    target_state = get_target_state(env)
    return target_state["dof_pos"]


def target_keybody_pos_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get target key body positions in local (robot) frame.
    
    Returns:
        Tensor of shape (num_envs, num_key_bodies * 3)
    """
    target_state = get_target_state(env)
    keybody_pos = target_state["keybody_pos"]  # (num_envs, num_bodies, 3)
    return keybody_pos.reshape(keybody_pos.shape[0], -1)


def target_root_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get target root position.
    
    Returns:
        Tensor of shape (num_envs, 3)
    """
    target_state = get_target_state(env)
    return target_state["root_pos"]


def target_root_rot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get target root rotation (quaternion).
    
    Returns:
        Tensor of shape (num_envs, 4)
    """
    target_state = get_target_state(env)
    return target_state["root_rot"]


##############################################################################
# REWARD FUNCTIONS
##############################################################################

def _safe_reward(reward: torch.Tensor) -> torch.Tensor:
    """Clamp reward to valid range and replace NaN/Inf with 0.
    
    This prevents numerical instability from crashing training.
    """
    # Replace NaN/Inf with 0
    reward = torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))
    # Clamp to reasonable range
    return torch.clamp(reward, min=-100.0, max=100.0)


def tracking_joint_dof(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for tracking target joint positions.
    
    Uses exponential kernel: exp(-mean_squared_error / std^2)
    Note: Uses MEAN squared error (not sum) to normalize across different DOF counts.
    
    Args:
        env: The environment instance.
        std: Standard deviation for exponential kernel (per-joint scale).
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Skip if motion not initialized
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    target_state = get_target_state(env)
    target_dof = target_state["dof_pos"]
    
    # Get current joint positions from robot
    robot = env.scene["robot"]
    current_dof = robot.data.joint_pos
    
    # Compute MEAN squared error (normalized by number of joints)
    dof_error = torch.mean(torch.square(current_dof - target_dof), dim=1)
    
    reward = torch.exp(-dof_error / (std ** 2))
    return _safe_reward(reward)


def tracking_joint_vel(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
) -> torch.Tensor:
    """Reward for tracking target joint velocities.
    
    Uses MEAN squared error to normalize across different DOF counts.
    
    Args:
        env: The environment instance.
        std: Standard deviation for exponential kernel (per-joint scale).
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Skip if motion not initialized
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    target_state = get_target_state(env)
    target_vel = target_state["dof_vel"]
    
    robot = env.scene["robot"]
    current_vel = robot.data.joint_vel
    
    # Compute MEAN squared error (normalized by number of joints)
    vel_error = torch.mean(torch.square(current_vel - target_vel), dim=1)
    
    reward = torch.exp(-vel_error / (std ** 2))
    return _safe_reward(reward)


def tracking_keybody_pos(
    env: ManagerBasedRLEnv,
    key_bodies: list[str],
    std: float = 0.2,
) -> torch.Tensor:
    """Reward for tracking key body positions.
    
    Note: This reward compares the robot's current key body positions with
    the target positions from motion data.
    
    IMPORTANT: key_bodies should use Isaac Lab robot body names.
    The MotionLib uses motion data body names (stored in env.cfg.key_bodies).
    These must be semantically aligned (same order, same meaning).
    
    Args:
        env: The environment instance.
        key_bodies: List of Isaac Lab robot body names to track.
        std: Standard deviation for exponential kernel.
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Skip if motion not initialized
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    target_state = get_target_state(env)
    target_pos = target_state["keybody_pos"]  # (num_envs, num_key_bodies, 3)
    
    # Get current key body positions from robot using Isaac Lab body names
    robot = env.scene["robot"]
    
    # Use robot_key_bodies from config if available, otherwise use param
    robot_key_bodies = getattr(env.cfg, 'robot_key_bodies', key_bodies)
    
    # Get body indices for key bodies (only use bodies that are in the reward param list)
    body_ids = []
    for body_name in key_bodies:
        try:
            found_ids = robot.find_bodies(body_name)[0]
            if found_ids:
                body_ids.extend(found_ids)
        except ValueError:
            # Body not found - skip it
            pass
    
    if not body_ids:
        return torch.zeros(env.num_envs, device=env.device)
    
    current_pos = robot.data.body_pos_w[:, body_ids, :]  # (num_envs, num_bodies, 3)
    
    # Ensure shapes match - use the minimum of available bodies
    num_compare = min(current_pos.shape[1], target_pos.shape[1])
    current_pos = current_pos[:, :num_compare, :]
    target_pos = target_pos[:, :num_compare, :]
    
    # Compute MEAN position error (normalized by number of key bodies)
    pos_error = torch.mean(torch.norm(current_pos - target_pos, dim=-1), dim=1)
    
    reward = torch.exp(-pos_error / std)
    return _safe_reward(reward)


def tracking_root_pos_xy(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
) -> torch.Tensor:
    """Reward for tracking target root XY position (horizontal movement).
    
    This enables the robot to follow walking/locomotion targets.
    
    Args:
        env: The environment instance.
        std: Standard deviation for exponential kernel (meters).
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Skip if motion not initialized
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    target_state = get_target_state(env)
    target_xy = target_state["root_pos"][:, :2]  # XY components
    
    robot = env.scene["robot"]
    current_xy = robot.data.root_pos_w[:, :2]
    
    xy_error = torch.norm(current_xy - target_xy, dim=1)
    
    reward = torch.exp(-xy_error / std)
    return _safe_reward(reward)


def tracking_root_height(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
) -> torch.Tensor:
    """Reward for tracking target root height.
    
    Args:
        env: The environment instance.
        std: Standard deviation for exponential kernel.
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Skip if motion not initialized
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    target_state = get_target_state(env)
    target_height = target_state["root_pos"][:, 2]  # Z component
    
    robot = env.scene["robot"]
    current_height = robot.data.root_pos_w[:, 2]
    
    height_error = torch.square(current_height - target_height)
    
    reward = torch.exp(-height_error / (std ** 2))
    return _safe_reward(reward)


def tracking_root_orientation(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for tracking target root orientation.
    
    Uses quaternion distance metric.
    
    Args:
        env: The environment instance.
        std: Standard deviation for exponential kernel.
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Skip if motion not initialized
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    target_state = get_target_state(env)
    target_rot = target_state["root_rot"]  # (num_envs, 4) - quaternion
    
    robot = env.scene["robot"]
    current_rot = robot.data.root_quat_w  # (num_envs, 4)
    
    # Quaternion distance: 1 - |q1 · q2|
    dot_product = torch.sum(current_rot * target_rot, dim=1)
    rot_error = 1.0 - torch.abs(dot_product)
    
    reward = torch.exp(-rot_error / std)
    return _safe_reward(reward)


def feet_distance_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_dist: float = 0.1,
    max_dist: float = 0.5,
) -> torch.Tensor:
    """Penalty for feet being too far apart or too close together.
    
    Computes the lateral (Y-axis in robot frame) distance between feet
    and penalizes if outside the desired range.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        min_dist: Minimum acceptable distance between feet (meters).
        max_dist: Maximum acceptable distance between feet (meters).
    
    Returns:
        Penalty tensor of shape (num_envs,). Returns 0 if in range,
        positive value if out of range (to be used with negative weight).
    """
    robot = env.scene[asset_cfg.name]
    
    # Get foot body indices - assuming left and right ankle roll links
    left_foot_idx = robot.find_bodies("left_ankle_roll_link")[0][0]
    right_foot_idx = robot.find_bodies("right_ankle_roll_link")[0][0]
    
    # Get foot positions in world frame
    body_pos = robot.data.body_pos_w  # (num_envs, num_bodies, 3)
    left_foot_pos = body_pos[:, left_foot_idx, :]  # (num_envs, 3)
    right_foot_pos = body_pos[:, right_foot_idx, :]  # (num_envs, 3)
    
    # Compute lateral distance (in world Y, approximate for robot Y)
    # For more accuracy, could transform to robot frame
    feet_distance = torch.norm(left_foot_pos[:, :2] - right_foot_pos[:, :2], dim=1)
    
    # Penalty: 0 if in range, positive if out of range
    too_close = torch.clamp(min_dist - feet_distance, min=0.0)
    too_far = torch.clamp(feet_distance - max_dist, min=0.0)
    
    penalty = too_close + too_far
    
    return penalty


##############################################################################
# STAGE 3: ARM JOINT TRACKING REWARDS
##############################################################################

# Note: Motion data is now remapped to Isaac Lab order in get_target_state()
# so both target_dof and robot.data.joint_pos are in the same order.
# We use env._upper_body_indices which are computed at runtime for Isaac Lab.


def tracking_arm_joints(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
) -> torch.Tensor:
    """Reward for tracking arm joint angles specifically.
    
    Focuses on upper body arm joints (shoulders, elbows) for manipulation tasks.
    Uses tighter precision than full body tracking.
    
    Note: target_dof is now in Isaac Lab order (remapped in get_target_state),
    so we can directly compare with robot.data.joint_pos using Isaac Lab indices.
    
    Args:
        env: The environment instance.
        std: Standard deviation for exponential kernel (radians).
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Skip if motion not initialized
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    target_state = get_target_state(env)
    target_dof = target_state["dof_pos"]  # Now in Isaac Lab order
    
    robot = env.scene["robot"]
    current_dof = robot.data.joint_pos  # Isaac Lab order
    
    # Use Isaac Lab upper body indices from env (computed at init time)
    if hasattr(env, '_upper_body_indices') and env._upper_body_indices:
        arm_indices = env._upper_body_indices
    else:
        # Fallback: find arm joints by name pattern
        arm_indices = []
        for idx, name in enumerate(robot.joint_names):
            if any(p in name for p in ["shoulder", "elbow"]):
                arm_indices.append(idx)
    
    if not arm_indices:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Extract arm joints only
    current_arm = current_dof[:, arm_indices]
    target_arm = target_dof[:, arm_indices]
    
    # Compute MEAN squared error for arm joints
    arm_error = torch.mean(torch.square(current_arm - target_arm), dim=1)
    
    reward = torch.exp(-arm_error / (std ** 2))
    return _safe_reward(reward)


##############################################################################
# STAGE 3: END-EFFECTOR TRACKING REWARDS (kept for reference)
##############################################################################

def tracking_ee_pos_windowed(
    env: ManagerBasedRLEnv,
    ee_bodies: list[str],
    std: float = 0.03,
    window_frames: int = 2,
) -> torch.Tensor:
    """End-effector position tracking with temporal window tolerance.
    
    Instead of requiring exact frame-by-frame matching, allows the robot
    to match any target within a ±window_frames time window (~66ms at 30fps).
    
    This addresses the reality that:
    1. Physics simulation timing differs from recorded motion
    2. Robot dynamics may cause phase shifts  
    3. ~50ms tolerance is acceptable for manipulation tasks
    
    Args:
        env: The environment instance.
        ee_bodies: List of end-effector body names (e.g., wrist links).
        std: Standard deviation for exponential kernel (meters).
              3cm = precise manipulation, 5cm = general reaching.
        window_frames: Number of frames to search ±around current time.
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Skip if motion not initialized
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    robot = env.scene["robot"]
    
    # Get EE body indices
    ee_ids = []
    for body_name in ee_bodies:
        try:
            found = robot.find_bodies(body_name)[0]
            if found:
                ee_ids.extend(found)
        except ValueError:
            pass
    
    if not ee_ids:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Current end-effector positions (num_envs, num_ee, 3)
    current_ee_pos = robot.data.body_pos_w[:, ee_ids, :]
    
    # Get target positions - search within time window for best match
    # This allows temporal flexibility (~50ms window)
    best_error = None
    
    for offset in range(-window_frames, window_frames + 1):
        # Get target state at this time offset
        target_state = env.get_target_state(frame_offset=offset)
        
        # Extract EE positions from keybody_pos
        # We need the wrist positions which should be in local_body_pos
        # For now, use the target keybody positions
        target_keybody = target_state["keybody_pos"]  # (num_envs, num_bodies, 3)
        
        # Match EE positions - take the hand-related indices
        # Assuming hands are at specific indices in the keybody list
        num_ee = current_ee_pos.shape[1]
        num_target = target_keybody.shape[1]
        
        # Use first num_ee bodies from target (assumes order matches)
        # TODO: Better mapping between ee_bodies and keybody indices
        target_ee_pos = target_keybody[:, :min(num_ee, num_target), :]
        current_compare = current_ee_pos[:, :min(num_ee, num_target), :]
        
        # Compute position error
        error = torch.norm(current_compare - target_ee_pos, dim=-1).mean(dim=1)
        
        if best_error is None:
            best_error = error
        else:
            best_error = torch.minimum(best_error, error)
    
    reward = torch.exp(-best_error / std)
    return _safe_reward(reward)


def tracking_ee_pos_direct(
    env: ManagerBasedRLEnv,
    ee_bodies: list[str],
    std: float = 0.05,
) -> torch.Tensor:
    """Direct end-effector position tracking without time window.
    
    Simpler version that tracks current frame only.
    Use this for faster training if windowed version is too slow.
    
    Args:
        env: The environment instance.
        ee_bodies: List of end-effector body names.
        std: Standard deviation for exponential kernel (meters).
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Skip if motion not initialized
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    robot = env.scene["robot"]
    
    # Get EE body indices
    ee_ids = []
    for body_name in ee_bodies:
        try:
            found = robot.find_bodies(body_name)[0]
            if found:
                ee_ids.extend(found)
        except ValueError:
            pass
    
    if not ee_ids:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Current end-effector positions
    current_ee_pos = robot.data.body_pos_w[:, ee_ids, :]  # (num_envs, num_ee, 3)
    
    # Get target positions from keybody
    target_state = env.get_target_state()
    target_keybody = target_state["keybody_pos"]  # (num_envs, num_bodies, 3)
    
    # Match dimensions
    num_ee = current_ee_pos.shape[1]
    num_target = target_keybody.shape[1]
    num_compare = min(num_ee, num_target)
    
    target_ee_pos = target_keybody[:, :num_compare, :]
    current_compare = current_ee_pos[:, :num_compare, :]
    
    # Compute position error
    pos_error = torch.norm(current_compare - target_ee_pos, dim=-1).mean(dim=1)
    
    reward = torch.exp(-pos_error / std)
    return _safe_reward(reward)


def ee_velocity_direction(
    env: ManagerBasedRLEnv,
    ee_bodies: list[str],
) -> torch.Tensor:
    """Reward for matching end-effector velocity direction.
    
    Rather than exact position, reward moving in the right direction.
    This provides a smoother learning signal and is more forgiving
    of timing differences.
    
    Uses cosine similarity between current and target velocity vectors.
    
    Args:
        env: The environment instance.
        ee_bodies: List of end-effector body names.
    
    Returns:
        Reward tensor of shape (num_envs,) in range [0, 1].
    """
    # Skip if motion not initialized
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, device=env.device)
    
    robot = env.scene["robot"]
    
    # Get EE body indices
    ee_ids = []
    for body_name in ee_bodies:
        try:
            found = robot.find_bodies(body_name)[0]
            if found:
                ee_ids.extend(found)
        except ValueError:
            pass
    
    if not ee_ids:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Current end-effector velocities
    current_ee_vel = robot.data.body_lin_vel_w[:, ee_ids, :]  # (num_envs, num_ee, 3)
    
    # Get target velocities from motion
    target_state = env.get_target_state()
    
    # Target velocity from keybody motion (finite difference in MotionLib)
    # If not available, compute from position difference
    if "keybody_vel" in target_state:
        target_keybody_vel = target_state["keybody_vel"]
    else:
        # Approximate velocity from position difference between frames
        target_state_next = env.get_target_state(frame_offset=1)
        target_keybody = target_state["keybody_pos"]
        target_keybody_next = target_state_next["keybody_pos"]
        fps = getattr(env, '_motion_fps', 30.0)
        target_keybody_vel = (target_keybody_next - target_keybody) * fps
    
    # Match dimensions
    num_ee = current_ee_vel.shape[1]
    num_target = target_keybody_vel.shape[1]
    num_compare = min(num_ee, num_target)
    
    current_vel = current_ee_vel[:, :num_compare, :]
    target_vel = target_keybody_vel[:, :num_compare, :]
    
    # Compute cosine similarity
    current_norm = torch.norm(current_vel, dim=-1, keepdim=True) + 1e-6
    target_norm = torch.norm(target_vel, dim=-1, keepdim=True) + 1e-6
    
    cos_sim = torch.sum(
        (current_vel / current_norm) * (target_vel / target_norm),
        dim=-1
    )
    
    # Average across end-effectors, map from [-1,1] to [0,1]
    reward = (cos_sim.mean(dim=1) + 1.0) / 2.0
    return _safe_reward(reward)


def upper_body_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining stable upper body while arms move.
    
    Penalizes excessive torso angular velocity, encouraging the
    robot to keep its core stable while manipulating.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
    
    Returns:
        Reward tensor of shape (num_envs,) in range [0, 1].
    """
    robot = env.scene[asset_cfg.name]
    
    # Get torso angular velocity
    torso_idx = robot.find_bodies("torso_link")[0][0]
    torso_ang_vel = robot.data.body_ang_vel_w[:, torso_idx, :]  # (num_envs, 3)
    
    # Penalize angular velocity magnitude
    ang_vel_mag = torch.norm(torso_ang_vel, dim=1)
    
    # Exponential reward: 1 when stable, 0 when rotating fast
    reward = torch.exp(-ang_vel_mag / 0.5)
    return _safe_reward(reward)


##############################################################################
# TERMINATION FUNCTIONS
##############################################################################

def motion_tracking_failure(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate if robot deviates too far from target motion.
    
    Args:
        env: The environment instance.
        threshold: Maximum allowed deviation in meters.
    
    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    # Skip if motion not initialized
    if not hasattr(env, '_motion_initialized') or not env._motion_initialized:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    target_state = get_target_state(env)
    target_root_pos = target_state["root_pos"]
    
    robot = env.scene["robot"]
    current_root_pos = robot.data.root_pos_w
    
    # Compute root position error (only XY, ignore Z for now)
    root_error = torch.norm(current_root_pos[:, :2] - target_root_pos[:, :2], dim=1)
    
    return root_error > threshold

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

def get_target_state(env: ManagerBasedRLEnv):
    """Get current target state from motion library.
    
    Uses the environment's get_target_state() method which handles
    motion timing and interpolation.
    
    Returns dict with:
        - dof_pos: target joint positions
        - dof_vel: target joint velocities
        - root_pos: target root position
        - root_rot: target root rotation (quaternion)
        - keybody_pos: target key body positions
    """
    return env.get_target_state()


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
    
    return torch.exp(-dof_error / (std ** 2))


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
    
    return torch.exp(-vel_error / (std ** 2))


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
    
    return torch.exp(-pos_error / std)


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
    
    return torch.exp(-height_error / (std ** 2))


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
    
    return torch.exp(-rot_error / std)


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

# Copyright (c) 2025
# G1 Motion Imitation Environment for Isaac Lab
#
# Custom ManagerBasedRLEnv that integrates MotionLib for motion imitation.

from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv

from .motion_lib import MotionLib

# Import centralized robot configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_config import G1RobotConfig

if TYPE_CHECKING:
    from .g1_motion_mimic_env_cfg import G1MotionMimicEnvCfg


class G1MotionMimicEnv(ManagerBasedRLEnv):
    """G1 Motion Imitation Environment.
    
    Extends ManagerBasedRLEnv with motion library integration for
    imitation learning from recorded motion data.
    
    Key additions:
        - motion_lib: MotionLib instance for loading/sampling motions
        - motion_ids: Current motion index for each environment
        - motion_start_times: Start time offset for each environment
    """
    
    cfg: G1MotionMimicEnvCfg
    
    def __init__(self, cfg: G1MotionMimicEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the motion imitation environment.
        
        Args:
            cfg: Environment configuration.
            render_mode: Rendering mode (None, "human", "rgb_array").
        """
        # Pre-initialize motion attributes (needed before parent __init__ 
        # because load_managers() evaluates observation functions)
        self._motion_initialized = False
        self.motion_lib = None
        self.motion_ids = None
        self.motion_start_times = None
        
        # Teleop target override (for live streaming)
        self._teleop_targets = None  # Dict with dof_pos, etc. or None
        self._teleop_upper_body_only = True  # Only override upper body joints
        
        # Store config for later use
        self._motion_file = cfg.motion_file
        self._key_bodies = cfg.key_bodies
        
        # Initialize parent (sets up scene, managers, etc.)
        # Note: This calls load_managers() which evaluates observation functions
        super().__init__(cfg, render_mode, **kwargs)
        
        # Now initialize motion library (after parent sets up device, num_envs)
        self._init_motion_lib()
        
        # Compute upper body joint indices for teleop (Isaac Lab joint order)
        self._init_upper_body_indices()
        
        print(f"[G1MotionMimicEnv] Initialized with {self.motion_lib.num_motions} motions")
    
    def _init_motion_lib(self):
        """Initialize motion library and buffers."""
        # Resolve motion file path
        motion_file = self._motion_file
        if not os.path.isabs(motion_file):
            # Try relative to TWIST2 root
            twist2_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            motion_file = os.path.join(twist2_root, motion_file)
        
        # Initialize motion library
        print(f"[G1MotionMimicEnv] Loading motions from: {motion_file}")
        self.motion_lib = MotionLib(
            motion_file=motion_file,
            device=self.device,
            key_bodies=self._key_bodies,
        )
        
        # Motion tracking buffers
        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_start_times = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        
        # Sample initial motions
        self._reset_motion_ids(torch.arange(self.num_envs, device=self.device))
        
        self._motion_initialized = True
    
    def _init_upper_body_indices(self):
        """Initialize joint mapping using centralized G1RobotConfig.
        
        This is critical because MuJoCo motion data uses a different joint order
        than Isaac Lab's robot. We must remap motion data to Isaac Lab order.
        """
        robot = self.scene["robot"]
        self._isaaclab_joint_names = list(robot.joint_names)
        
        # Build mapping using centralized config
        self._mujoco_to_isaaclab_mapping = G1RobotConfig.build_mujoco_to_isaaclab_mapping(
            self._isaaclab_joint_names
        )
        
        # Get upper body indices (Isaac Lab order)
        self._upper_body_indices = G1RobotConfig.get_isaaclab_upper_body_indices(
            self._isaaclab_joint_names
        )
        
        mapped_count = sum(1 for v in self._mujoco_to_isaaclab_mapping.values() if v is not None)
        print(f"[G1MotionMimicEnv] Joint mapping: {mapped_count}/29 MuJoCo joints mapped to Isaac Lab")
        print(f"[G1MotionMimicEnv] Upper body indices (IL order): {self._upper_body_indices}")
    
    def _remap_mujoco_to_isaaclab(self, mujoco_dof: torch.Tensor) -> torch.Tensor:
        """Remap DOF positions from MuJoCo order to Isaac Lab order.
        
        Uses centralized G1RobotConfig for consistent joint mapping.
        
        Args:
            mujoco_dof: Joint positions in MuJoCo order, shape (num_envs, 29)
        
        Returns:
            Joint positions in Isaac Lab order, shape (num_envs, num_joints)
        """
        robot = self.scene["robot"]
        default_pos = robot.data.default_joint_pos[:1]
        
        return G1RobotConfig.remap_mujoco_to_isaaclab_torch(
            mujoco_dof,
            self._isaaclab_joint_names,
            default_pos,
        )
    
    def _reset_motion_ids(self, env_ids: torch.Tensor):
        """Reset motion assignments for specified environments.
        
        Args:
            env_ids: Environment indices to reset.
        """
        num_reset = len(env_ids)
        
        # Sample new motions
        self.motion_ids[env_ids] = self.motion_lib.sample_motions(num_reset)
        
        # Sample random start times within each motion
        self.motion_start_times[env_ids] = self.motion_lib.sample_start_times(
            self.motion_ids[env_ids]
        )
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments at given indices.
        
        Overrides parent to also reset motion assignments.
        
        Args:
            env_ids: Environment indices to reset.
        """
        # Reset motion tracking
        self._reset_motion_ids(env_ids)
        
        # Get target state to reset robot to motion pose
        motion_state = self.motion_lib.get_motion_state(
            self.motion_ids[env_ids],
            self.motion_start_times[env_ids],
        )
        
        # Call parent reset (resets physics, etc.)
        super()._reset_idx(env_ids)
        
        # Override robot state with motion reference (optional - for curriculum)
        # This helps early training by starting closer to target
        if hasattr(self.cfg, 'reset_to_motion') and self.cfg.reset_to_motion:
            robot = self.scene["robot"]
            
            # Set joint positions to motion target
            robot.write_joint_state_to_sim(
                position=motion_state["dof_pos"],
                velocity=torch.zeros_like(motion_state["dof_pos"]),
                env_ids=env_ids,
            )
    
    def get_motion_time(self) -> torch.Tensor:
        """Get current motion time for each environment.
        
        Returns:
            Tensor of shape (num_envs,) with current time in seconds.
        """
        if not self._motion_initialized:
            # Return zeros during initialization (for observation shape determination)
            return torch.zeros(self.num_envs, device=self.device)
        
        # Episode time plus start offset
        episode_time = self.episode_length_buf * self.step_dt
        return episode_time + self.motion_start_times
    
    def get_target_state(self, frame_offset: int = 0):
        """Get current target state from motion library.
        
        Args:
            frame_offset: Optional frame offset for temporal windowing.
                          Positive = future frames, Negative = past frames.
                          At 30fps, ±2 frames = ±66ms.
        
        Returns:
            Dict with target dof_pos, dof_vel, root_pos, root_rot, keybody_pos.
            DOF tensors are padded to match robot's joint count if motion has fewer DOFs.
        """
        robot = self.scene["robot"]
        num_joints = robot.num_joints
        
        if not self._motion_initialized:
            # Return zeros during initialization (for observation shape determination)
            return {
                "dof_pos": torch.zeros(self.num_envs, num_joints, device=self.device),
                "dof_vel": torch.zeros(self.num_envs, num_joints, device=self.device),
                "root_pos": torch.zeros(self.num_envs, 3, device=self.device),
                "root_rot": torch.zeros(self.num_envs, 4, device=self.device),
                "keybody_pos": torch.zeros(self.num_envs, len(self._key_bodies), 3, device=self.device),
            }
        
        motion_time = self.get_motion_time()
        
        # Apply frame offset for temporal windowing
        if frame_offset != 0:
            # Get FPS from motion lib or use default
            fps = getattr(self.motion_lib, 'fps', 30.0)
            time_offset = frame_offset / fps
            motion_time = motion_time + time_offset
            # Clamp to valid time range (handled by motion_lib internally)
        
        motion_state = self.motion_lib.get_motion_state(self.motion_ids, motion_time)
        
        # CRITICAL: Remap motion data from MuJoCo order to Isaac Lab order
        # Motion data uses MuJoCo joint order (29 DOFs), but robot uses Isaac Lab order
        # The joint indices are DIFFERENT between these two formats!
        motion_state["dof_pos"] = self._remap_mujoco_to_isaaclab(motion_state["dof_pos"])
        motion_state["dof_vel"] = self._remap_mujoco_to_isaaclab(motion_state["dof_vel"])
        
        # Apply teleop override if set (teleop data is also in MuJoCo order)
        if self._teleop_targets is not None:
            motion_state = self._apply_teleop_override(motion_state)
        
        return motion_state
    
    def set_teleop_targets(self, dof_pos: torch.Tensor, upper_body_only: bool = True):
        """Set external teleop targets to override motion library.
        
        Args:
            dof_pos: Target joint positions tensor of shape (num_envs, num_joints)
                     or (num_joints,) which will be broadcast to all envs.
            upper_body_only: If True, only override upper body joints (arms).
                            Lower body will still use motion library.
        """
        if dof_pos.dim() == 1:
            dof_pos = dof_pos.unsqueeze(0).expand(self.num_envs, -1)
        
        self._teleop_targets = {
            "dof_pos": dof_pos.to(self.device),
        }
        self._teleop_upper_body_only = upper_body_only
    
    def clear_teleop_targets(self):
        """Clear teleop targets, reverting to motion library."""
        self._teleop_targets = None
    
    def _apply_teleop_override(self, motion_state: dict) -> dict:
        """Apply teleop target override to motion state.
        
        For upper_body_only mode, blends teleop targets for arm joints
        with motion library targets for lower body.
        
        Note: teleop_dof comes in MuJoCo order (29 DOFs) and must be remapped
        to Isaac Lab order before merging with motion_state (already in IL order).
        """
        if self._teleop_targets is None:
            return motion_state
        
        # Teleop data is in MuJoCo order - remap to Isaac Lab order
        teleop_dof_mujoco = self._teleop_targets["dof_pos"]
        teleop_dof = self._remap_mujoco_to_isaaclab(teleop_dof_mujoco)
        
        motion_dof = motion_state["dof_pos"]  # Already in Isaac Lab order
        
        if self._teleop_upper_body_only:
            # Use Isaac Lab upper body indices (computed in _init_upper_body_indices)
            merged_dof = motion_dof.clone()
            
            # Override only upper body joints with teleop values
            for il_idx in self._upper_body_indices:
                if il_idx < teleop_dof.shape[1] and il_idx < merged_dof.shape[1]:
                    merged_dof[:, il_idx] = teleop_dof[:, il_idx]
            
            motion_state["dof_pos"] = merged_dof
        else:
            # Full override - use teleop for all joints
            motion_state["dof_pos"] = teleop_dof
        
        return motion_state

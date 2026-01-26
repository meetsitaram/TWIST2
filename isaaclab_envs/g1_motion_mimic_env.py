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
        
        # Store config for later use
        self._motion_file = cfg.motion_file
        self._key_bodies = cfg.key_bodies
        
        # Initialize parent (sets up scene, managers, etc.)
        # Note: This calls load_managers() which evaluates observation functions
        super().__init__(cfg, render_mode, **kwargs)
        
        # Now initialize motion library (after parent sets up device, num_envs)
        self._init_motion_lib()
        
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
    
    def get_target_state(self):
        """Get current target state from motion library.
        
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
        motion_state = self.motion_lib.get_motion_state(self.motion_ids, motion_time)
        
        # Handle DOF count mismatch (motion may have fewer DOFs than robot)
        motion_dof_count = motion_state["dof_pos"].shape[1]
        if motion_dof_count < num_joints:
            # Pad with robot's default joint positions for extra joints (fingers, etc.)
            default_pos = robot.data.default_joint_pos[0, motion_dof_count:]
            padded_pos = torch.zeros(self.num_envs, num_joints, device=self.device)
            padded_pos[:, :motion_dof_count] = motion_state["dof_pos"]
            padded_pos[:, motion_dof_count:] = default_pos
            motion_state["dof_pos"] = padded_pos
            
            padded_vel = torch.zeros(self.num_envs, num_joints, device=self.device)
            padded_vel[:, :motion_dof_count] = motion_state["dof_vel"]
            motion_state["dof_vel"] = padded_vel
        
        return motion_state

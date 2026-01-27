# Copyright (c) 2025
# Motion Library for Isaac Lab
#
# Loads and samples motion data from pkl files for motion imitation.
# Adapted from TWIST2's pose/utils/motion_lib_pkl.py for Isaac Lab.
# OPTIMIZED: Vectorized operations for GPU-efficient batch processing.

from __future__ import annotations

import os
import pickle
import sys
import torch
import yaml
import numpy as np
from types import ModuleType
from typing import Dict, List, Optional, Tuple

# Patch sys.modules to fake missing modules from numpy 2.x
# This is needed for loading pkl files saved with different numpy versions
class _FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)

# Apply patches before loading any pickles
if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = _FakeModule('numpy._core', np.core if hasattr(np, 'core') else np)
if 'numpy._core.multiarray' not in sys.modules:
    sys.modules['numpy._core.multiarray'] = _FakeModule('numpy._core.multiarray', getattr(np.core, 'multiarray', None))


class MotionLib:
    """Motion Library for loading and sampling motion data.
    
    Loads motion data from pkl files and provides methods to sample
    motion states at arbitrary times.
    
    OPTIMIZED: All motion data is stacked into padded tensors at load time
    for vectorized GPU operations. No Python for-loops during get_motion_state().
    
    Motion pkl format:
        - fps: frames per second
        - root_pos: (num_frames, 3) root position
        - root_rot: (num_frames, 4) root rotation quaternion [x, y, z, w]
        - dof_pos: (num_frames, num_joints) joint positions
        - local_body_pos: (num_frames, num_bodies, 3) body positions (optional)
    """
    
    # Default end-effector bodies for manipulation (wrists)
    DEFAULT_EE_BODIES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    
    def __init__(
        self,
        motion_file: str,
        device: str = "cuda:0",
        key_bodies: Optional[List[str]] = None,
        ee_bodies: Optional[List[str]] = None,
    ):
        """Initialize motion library.
        
        Args:
            motion_file: Path to motion YAML config or single pkl file.
            device: Torch device for tensors.
            key_bodies: List of key body names for tracking.
            ee_bodies: List of end-effector body names (default: wrists).
        """
        self.device = device
        self.key_bodies = key_bodies or []
        self.ee_bodies = ee_bodies or self.DEFAULT_EE_BODIES
        
        # Load motions into lists first
        self._motion_data_list = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_weights = []
        
        self._load_motions(motion_file)
        
        # Compute key body indices once (before stacking)
        self._key_body_indices = self.get_key_body_indices(self.key_bodies) if self.key_bodies else []
        
        # Compute EE body indices
        self._ee_body_indices = self.get_key_body_indices(self.ee_bodies) if self.ee_bodies else []
        
        # Stack all motion data into padded tensors for vectorized access
        self._build_stacked_tensors()
        
        # Compute sampling weights
        total_length = sum(self._motion_lengths)
        self._motion_probs = torch.tensor(
            [w * l / total_length for w, l in zip(self._motion_weights, self._motion_lengths)],
            device=device,
        )
        self._motion_probs /= self._motion_probs.sum()
        
        print(f"[MotionLib] Loaded {self.num_motions} motions, "
              f"total length: {total_length:.1f}s")
        if self._key_body_indices:
            print(f"[MotionLib] Key body indices: {self._key_body_indices}")
        if self._ee_body_indices:
            print(f"[MotionLib] EE body indices: {self._ee_body_indices}")
    
    def _build_stacked_tensors(self):
        """Stack all motion data into padded tensors for vectorized access."""
        num_motions = len(self._motion_data_list)
        
        # Find max frames across all motions
        max_frames = max(m["dof_pos"].shape[0] for m in self._motion_data_list)
        num_joints = self._motion_data_list[0]["dof_pos"].shape[1]
        
        # Check if we have key body data
        has_keybody = "local_body_pos" in self._motion_data_list[0] and self._key_body_indices
        num_key_bodies = len(self._key_body_indices) if has_keybody else len(self.key_bodies) if self.key_bodies else 1
        
        # Check if we have EE body data
        has_ee = "local_body_pos" in self._motion_data_list[0] and self._ee_body_indices
        num_ee_bodies = len(self._ee_body_indices) if has_ee else len(self.ee_bodies) if self.ee_bodies else 2
        
        # Pre-allocate stacked tensors (num_motions, max_frames, ...)
        self._stacked_dof_pos = torch.zeros(num_motions, max_frames, num_joints, device=self.device)
        self._stacked_dof_vel = torch.zeros(num_motions, max_frames, num_joints, device=self.device)
        self._stacked_root_pos = torch.zeros(num_motions, max_frames, 3, device=self.device)
        self._stacked_root_rot = torch.zeros(num_motions, max_frames, 4, device=self.device)
        self._stacked_keybody_pos = torch.zeros(num_motions, max_frames, num_key_bodies, 3, device=self.device)
        self._stacked_ee_pos = torch.zeros(num_motions, max_frames, num_ee_bodies, 3, device=self.device)
        
        # Store frame counts and fps as tensors for vectorized ops
        self._motion_num_frames = torch.zeros(num_motions, dtype=torch.long, device=self.device)
        self._motion_fps_tensor = torch.zeros(num_motions, device=self.device)
        self._motion_lengths_tensor = torch.zeros(num_motions, device=self.device)
        
        # Fill stacked tensors
        for i, motion in enumerate(self._motion_data_list):
            num_frames = motion["dof_pos"].shape[0]
            self._motion_num_frames[i] = num_frames
            self._motion_fps_tensor[i] = self._motion_fps[i]
            self._motion_lengths_tensor[i] = self._motion_lengths[i]
            
            # Copy data (padded with last frame for safety)
            self._stacked_dof_pos[i, :num_frames] = motion["dof_pos"]
            self._stacked_dof_vel[i, :num_frames] = motion["dof_vel"]
            self._stacked_root_pos[i, :num_frames] = motion["root_pos"]
            self._stacked_root_rot[i, :num_frames] = motion["root_rot"]
            
            # Pad with last frame to avoid index errors
            if num_frames < max_frames:
                self._stacked_dof_pos[i, num_frames:] = motion["dof_pos"][-1]
                self._stacked_dof_vel[i, num_frames:] = motion["dof_vel"][-1]
                self._stacked_root_pos[i, num_frames:] = motion["root_pos"][-1]
                self._stacked_root_rot[i, num_frames:] = motion["root_rot"][-1]
            
            # Key body positions
            if has_keybody:
                keybody_data = motion["local_body_pos"][:, self._key_body_indices, :]
                self._stacked_keybody_pos[i, :num_frames] = keybody_data
                if num_frames < max_frames:
                    self._stacked_keybody_pos[i, num_frames:] = keybody_data[-1]
        
        # Clear the list to free memory
        self._motion_data_list = None
        self._has_keybody = has_keybody
        self._num_joints = num_joints
        self._num_key_bodies = num_key_bodies
    
    def _load_motions(self, motion_file: str):
        """Load motions from YAML config, pkl file, or npz file."""
        if motion_file.endswith(".yaml"):
            self._load_from_yaml(motion_file)
        elif motion_file.endswith(".pkl"):
            self._load_single_motion(motion_file, weight=1.0)
        elif motion_file.endswith(".npz"):
            self._load_single_motion_npz(motion_file, weight=1.0)
        else:
            raise ValueError(f"Unsupported motion file format: {motion_file}")
    
    def _load_from_yaml(self, yaml_path: str):
        """Load motions from YAML config file."""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        
        root_path = config.get("root_path", "")
        
        # Handle relative paths
        if not os.path.isabs(root_path):
            yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
            root_path = os.path.join(yaml_dir, root_path)
        
        motions = config.get("motions", [])
        
        for motion_entry in motions:
            file_path = os.path.join(root_path, motion_entry["file"])
            weight = motion_entry.get("weight", 1.0)
            self._load_single_motion(file_path, weight)
    
    def _load_single_motion(self, pkl_path: str, weight: float):
        """Load a single motion from pkl file."""
        with open(pkl_path, "rb") as f:
            motion = pickle.load(f)
        
        fps = motion.get("fps", 50.0)
        num_frames = motion["dof_pos"].shape[0]
        duration = num_frames / fps
        
        # Convert to torch tensors
        motion_data = {
            "fps": fps,
            "root_pos": torch.tensor(motion["root_pos"], dtype=torch.float32, device=self.device),
            "root_rot": torch.tensor(motion["root_rot"], dtype=torch.float32, device=self.device),
            "dof_pos": torch.tensor(motion["dof_pos"], dtype=torch.float32, device=self.device),
        }
        
        # Load local body positions if available
        if "local_body_pos" in motion:
            motion_data["local_body_pos"] = torch.tensor(
                motion["local_body_pos"], dtype=torch.float32, device=self.device
            )
        
        # Store body link list for key body lookup
        if "link_body_list" in motion:
            motion_data["body_names"] = motion["link_body_list"]
        
        # Compute joint velocities via finite difference
        dt = 1.0 / fps
        dof_vel = torch.zeros_like(motion_data["dof_pos"])
        dof_vel[1:] = (motion_data["dof_pos"][1:] - motion_data["dof_pos"][:-1]) / dt
        dof_vel[0] = dof_vel[1]
        motion_data["dof_vel"] = dof_vel
        
        self._motion_data_list.append(motion_data)
        self._motion_lengths.append(duration)
        self._motion_fps.append(fps)
        self._motion_weights.append(weight)
    
    def _load_single_motion_npz(self, npz_path: str, weight: float):
        """Load a single motion from npz file (numpy 1.x compatible)."""
        import numpy as np
        
        npz = np.load(npz_path, allow_pickle=True)
        
        # Extract fps from metadata
        fps = float(npz.get("_meta_fps", [50.0])[0])
        num_frames = npz["dof_pos"].shape[0]
        duration = num_frames / fps
        
        # Convert to torch tensors
        motion_data = {
            "fps": fps,
            "root_pos": torch.tensor(npz["root_pos"], dtype=torch.float32, device=self.device),
            "root_rot": torch.tensor(npz["root_rot"], dtype=torch.float32, device=self.device),
            "dof_pos": torch.tensor(npz["dof_pos"], dtype=torch.float32, device=self.device),
        }
        
        # Load local body positions if available
        if "local_body_pos" in npz:
            motion_data["local_body_pos"] = torch.tensor(
                npz["local_body_pos"], dtype=torch.float32, device=self.device
            )
        
        # Store body link list for key body lookup
        if "link_body_list" in npz:
            # NPZ stores arrays, so convert back to list
            body_list = npz["link_body_list"]
            if hasattr(body_list, 'tolist'):
                body_list = body_list.tolist()
            motion_data["body_names"] = body_list
        
        # Compute joint velocities via finite difference
        dt = 1.0 / fps
        dof_vel = torch.zeros_like(motion_data["dof_pos"])
        dof_vel[1:] = (motion_data["dof_pos"][1:] - motion_data["dof_pos"][:-1]) / dt
        dof_vel[0] = dof_vel[1]
        motion_data["dof_vel"] = dof_vel
        
        self._motion_data_list.append(motion_data)
        self._motion_lengths.append(duration)
        self._motion_fps.append(fps)
        self._motion_weights.append(weight)
    
    def get_key_body_indices(self, key_body_names: List[str]) -> List[int]:
        """Get indices of key bodies in motion data.
        
        Args:
            key_body_names: List of body names to find.
            
        Returns:
            List of body indices in local_body_pos.
        """
        if not self._motion_data_list or "body_names" not in self._motion_data_list[0]:
            return list(range(len(key_body_names)))
        
        body_names = self._motion_data_list[0]["body_names"]
        indices = []
        for name in key_body_names:
            if name in body_names:
                indices.append(body_names.index(name))
            else:
                print(f"[MotionLib] Warning: key body '{name}' not found in motion data")
        return indices
    
    @property
    def num_motions(self) -> int:
        """Number of loaded motions."""
        return len(self._motion_lengths)
    
    def sample_motions(self, num_samples: int) -> torch.Tensor:
        """Sample motion indices based on weights.
        
        Args:
            num_samples: Number of motion indices to sample.
        
        Returns:
            Tensor of shape (num_samples,) with motion indices.
        """
        indices = torch.multinomial(
            self._motion_probs,
            num_samples,
            replacement=True,
        )
        return indices
    
    def sample_start_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Sample random start times for given motions (VECTORIZED).
        
        Args:
            motion_ids: Tensor of motion indices.
        
        Returns:
            Tensor of shape (num_samples,) with start times.
        """
        # Get durations for each motion_id (vectorized lookup)
        durations = self._motion_lengths_tensor[motion_ids]
        
        # Sample random times within duration
        start_times = torch.rand(motion_ids.shape[0], device=self.device) * durations
        
        return start_times
    
    def get_motion_state(
        self,
        motion_ids: torch.Tensor,
        times: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get motion state at specified times (FULLY VECTORIZED).
        
        Args:
            motion_ids: Tensor of motion indices, shape (num_envs,).
            times: Tensor of times within each motion, shape (num_envs,).
        
        Returns:
            Dict with:
                - dof_pos: (num_envs, num_joints) target joint positions
                - dof_vel: (num_envs, num_joints) target joint velocities
                - root_pos: (num_envs, 3) target root position
                - root_rot: (num_envs, 4) target root rotation
                - keybody_pos: (num_envs, num_key_bodies, 3) target key body positions
        """
        num_envs = motion_ids.shape[0]
        
        # Get fps and duration for each environment's motion (vectorized)
        fps = self._motion_fps_tensor[motion_ids]  # (num_envs,)
        durations = self._motion_lengths_tensor[motion_ids]  # (num_envs,)
        num_frames = self._motion_num_frames[motion_ids]  # (num_envs,)
        
        # Wrap time to motion duration
        wrapped_times = times % durations  # (num_envs,)
        
        # Compute frame indices with linear interpolation
        frame_f = wrapped_times * fps  # (num_envs,)
        frame_0 = frame_f.long()  # (num_envs,)
        
        # Clamp frame indices to valid range (use torch.minimum/maximum for tensor bounds)
        max_frame = num_frames - 1  # (num_envs,) tensor
        frame_0 = torch.clamp(frame_0, min=0)  # Lower bound with scalar
        frame_0 = torch.minimum(frame_0, max_frame)  # Upper bound with tensor
        frame_1 = torch.minimum(frame_0 + 1, max_frame)  # (num_envs,)
        
        blend = (frame_f - frame_0.float()).unsqueeze(-1)  # (num_envs, 1)
        
        # Advanced indexing to get all data at once
        # Index: [motion_id, frame] for each env
        dof_pos_0 = self._stacked_dof_pos[motion_ids, frame_0]  # (num_envs, num_joints)
        dof_pos_1 = self._stacked_dof_pos[motion_ids, frame_1]
        dof_pos = (1 - blend) * dof_pos_0 + blend * dof_pos_1
        
        dof_vel_0 = self._stacked_dof_vel[motion_ids, frame_0]
        dof_vel_1 = self._stacked_dof_vel[motion_ids, frame_1]
        dof_vel = (1 - blend) * dof_vel_0 + blend * dof_vel_1
        
        root_pos_0 = self._stacked_root_pos[motion_ids, frame_0]  # (num_envs, 3)
        root_pos_1 = self._stacked_root_pos[motion_ids, frame_1]
        root_pos = (1 - blend) * root_pos_0 + blend * root_pos_1
        
        # Quaternion interpolation (linear + normalize)
        rot_0 = self._stacked_root_rot[motion_ids, frame_0]  # (num_envs, 4)
        rot_1 = self._stacked_root_rot[motion_ids, frame_1]
        
        # Handle quaternion sign (dot product check, vectorized)
        dot = (rot_0 * rot_1).sum(dim=-1, keepdim=True)  # (num_envs, 1)
        rot_1 = torch.where(dot < 0, -rot_1, rot_1)
        
        root_rot = (1 - blend) * rot_0 + blend * rot_1
        root_rot = root_rot / root_rot.norm(dim=-1, keepdim=True)  # Normalize
        
        # Key body positions
        blend_3d = blend.unsqueeze(-1)  # (num_envs, 1, 1)
        keybody_pos_0 = self._stacked_keybody_pos[motion_ids, frame_0]  # (num_envs, num_key_bodies, 3)
        keybody_pos_1 = self._stacked_keybody_pos[motion_ids, frame_1]
        keybody_pos = (1 - blend_3d) * keybody_pos_0 + blend_3d * keybody_pos_1
        
        return {
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "keybody_pos": keybody_pos,
        }
    
    def get_motion_length(self, motion_id: int) -> float:
        """Get duration of a motion in seconds."""
        return self._motion_lengths[motion_id]

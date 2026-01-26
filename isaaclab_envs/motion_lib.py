# Copyright (c) 2025
# Motion Library for Isaac Lab
#
# Loads and samples motion data from pkl files for motion imitation.
# Adapted from TWIST2's pose/utils/motion_lib_pkl.py for Isaac Lab.

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
    
    Motion pkl format:
        - fps: frames per second
        - root_pos: (num_frames, 3) root position
        - root_rot: (num_frames, 4) root rotation quaternion [x, y, z, w]
        - dof_pos: (num_frames, num_joints) joint positions
        - local_body_pos: (num_frames, num_bodies, 3) body positions (optional)
    """
    
    def __init__(
        self,
        motion_file: str,
        device: str = "cuda:0",
        key_bodies: Optional[List[str]] = None,
    ):
        """Initialize motion library.
        
        Args:
            motion_file: Path to motion YAML config or single pkl file.
            device: Torch device for tensors.
            key_bodies: List of key body names for tracking.
        """
        self.device = device
        self.key_bodies = key_bodies or []
        
        # Load motions
        self._motion_data = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_weights = []
        
        self._load_motions(motion_file)
        
        # Compute sampling weights
        total_length = sum(self._motion_lengths)
        self._motion_probs = torch.tensor(
            [w * l / total_length for w, l in zip(self._motion_weights, self._motion_lengths)],
            device=device,
        )
        self._motion_probs /= self._motion_probs.sum()
        
        # Compute key body indices once
        self._key_body_indices = self.get_key_body_indices(self.key_bodies) if self.key_bodies else []
        
        print(f"[MotionLib] Loaded {len(self._motion_data)} motions, "
              f"total length: {total_length:.1f}s")
        if self._key_body_indices:
            print(f"[MotionLib] Key body indices: {self._key_body_indices}")
    
    def _load_motions(self, motion_file: str):
        """Load motions from YAML config or pkl file."""
        if motion_file.endswith(".yaml"):
            self._load_from_yaml(motion_file)
        elif motion_file.endswith(".pkl"):
            self._load_single_motion(motion_file, weight=1.0)
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
        
        self._motion_data.append(motion_data)
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
        if not self._motion_data or "body_names" not in self._motion_data[0]:
            return list(range(len(key_body_names)))
        
        body_names = self._motion_data[0]["body_names"]
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
        return len(self._motion_data)
    
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
        """Sample random start times for given motions.
        
        Args:
            motion_ids: Tensor of motion indices.
        
        Returns:
            Tensor of shape (num_samples,) with start times.
        """
        num_samples = motion_ids.shape[0]
        start_times = torch.zeros(num_samples, device=self.device)
        
        for i, motion_id in enumerate(motion_ids):
            duration = self._motion_lengths[motion_id.item()]
            start_times[i] = torch.rand(1, device=self.device).item() * duration
        
        return start_times
    
    def get_motion_state(
        self,
        motion_ids: torch.Tensor,
        times: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get motion state at specified times.
        
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
        
        # Get number of joints from first motion
        num_joints = self._motion_data[0]["dof_pos"].shape[1]
        
        # Initialize output tensors
        dof_pos = torch.zeros(num_envs, num_joints, device=self.device)
        dof_vel = torch.zeros(num_envs, num_joints, device=self.device)
        root_pos = torch.zeros(num_envs, 3, device=self.device)
        root_rot = torch.zeros(num_envs, 4, device=self.device)
        
        # Process each environment
        for i in range(num_envs):
            motion_id = motion_ids[i].item()
            time = times[i].item()
            
            motion = self._motion_data[motion_id]
            fps = self._motion_fps[motion_id]
            duration = self._motion_lengths[motion_id]
            
            # Wrap time to motion duration
            time = time % duration
            
            # Compute frame index with linear interpolation
            frame_f = time * fps
            frame_0 = int(frame_f)
            frame_1 = min(frame_0 + 1, motion["dof_pos"].shape[0] - 1)
            blend = frame_f - frame_0
            
            # Interpolate state
            dof_pos[i] = (1 - blend) * motion["dof_pos"][frame_0] + blend * motion["dof_pos"][frame_1]
            dof_vel[i] = (1 - blend) * motion["dof_vel"][frame_0] + blend * motion["dof_vel"][frame_1]
            root_pos[i] = (1 - blend) * motion["root_pos"][frame_0] + blend * motion["root_pos"][frame_1]
            
            # Quaternion SLERP (simplified - linear interpolation then normalize)
            rot_0 = motion["root_rot"][frame_0]
            rot_1 = motion["root_rot"][frame_1]
            # Handle quaternion sign
            if torch.dot(rot_0, rot_1) < 0:
                rot_1 = -rot_1
            root_rot[i] = (1 - blend) * rot_0 + blend * rot_1
            root_rot[i] = root_rot[i] / torch.norm(root_rot[i])
        
        # Key body positions (if available)
        if "local_body_pos" in self._motion_data[0] and self._key_body_indices:
            num_key_bodies = len(self._key_body_indices)
            keybody_pos = torch.zeros(num_envs, num_key_bodies, 3, device=self.device)
            
            for i in range(num_envs):
                motion_id = motion_ids[i].item()
                time = times[i].item() % self._motion_lengths[motion_id]
                
                motion = self._motion_data[motion_id]
                fps = self._motion_fps[motion_id]
                
                frame_f = time * fps
                frame_0 = int(frame_f)
                frame_1 = min(frame_0 + 1, motion["local_body_pos"].shape[0] - 1)
                blend = frame_f - frame_0
                
                # Get all body positions at this frame
                all_body_pos_0 = motion["local_body_pos"][frame_0]
                all_body_pos_1 = motion["local_body_pos"][frame_1]
                
                # Extract only key body positions
                for j, body_idx in enumerate(self._key_body_indices):
                    keybody_pos[i, j] = (1 - blend) * all_body_pos_0[body_idx] + \
                                        blend * all_body_pos_1[body_idx]
        else:
            # No key bodies or no body data - return zeros
            num_key_bodies = len(self.key_bodies) if self.key_bodies else 1
            keybody_pos = torch.zeros(num_envs, num_key_bodies, 3, device=self.device)
        
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

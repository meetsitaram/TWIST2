#!/usr/bin/env python3
"""
Episode → Motion Converter

Converts teleop episode NPZ files to TWIST2 motion pickle format for RL training.

Episode NPZ format (input):
    - robot_qpos: (N, 36) = [x, y, z, qw, qx, qy, qz, 29_joint_angles]
    - t_ms: timestamps in milliseconds
    - metadata: JSON with fps info

Motion pickle format (output):
    - fps: float (e.g., 30.0)
    - root_pos: (N, 3) - pelvis world position
    - root_rot: (N, 4) - pelvis quaternion [qx, qy, qz, qw] (scalar-last, TWIST2 convention)
    - dof_pos: (N, 29) - joint angles
    - local_body_pos: (N, 38, 3) - FK body positions relative to root
    - link_body_list: list of 38 body names

Note: Episode NPZ uses MuJoCo quaternion format [qw, qx, qy, qz] (scalar-first),
      but TWIST2 motion format uses [qx, qy, qz, qw] (scalar-last).
      The converter handles this transformation automatically.

Usage:
    # Convert single episode
    python convert_episodes_to_motion.py --episode elbow_track_007
    
    # Convert multiple episodes
    python convert_episodes_to_motion.py --episode elbow_track_007 elbow_track_008
    
    # Convert all episodes
    python convert_episodes_to_motion.py --all
    
    # Generate YAML config for training
    python convert_episodes_to_motion.py --all --yaml teleop_dataset
    
    # List available episodes
    python convert_episodes_to_motion.py --list
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
from tqdm import tqdm

# Paths
SCRIPT_DIR = Path(__file__).parent
TWIST2_ROOT = SCRIPT_DIR.parent
ASSETS_DIR = TWIST2_ROOT / "assets"
EPISODES_DIR = TWIST2_ROOT / "datasets" / "teleop_episodes"
MOTION_OUTPUT_DIR = TWIST2_ROOT / "datasets" / "teleop_motions"
G1_MODEL_PATH = ASSETS_DIR / "g1" / "g1_mocap_29dof.xml"


class EpisodeToMotionConverter:
    """Converts teleop episodes to TWIST2 motion format."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize with MuJoCo model for FK computation."""
        model_path = model_path or G1_MODEL_PATH
        
        if not model_path.exists():
            raise FileNotFoundError(f"G1 model not found: {model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        # Get body list (excluding 'world' at index 0)
        self.link_body_list = [
            self.model.body(i).name 
            for i in range(1, self.model.nbody)
        ]
        self.num_bodies = len(self.link_body_list)
        
        # Body ID mapping (for FK extraction)
        self.body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.link_body_list
        ]
        
        print(f"Loaded G1 model with {self.num_bodies} bodies")
    
    def load_episode(self, episode_path: Path) -> Dict:
        """Load episode NPZ file."""
        data = np.load(episode_path, allow_pickle=True)
        
        episode = {
            't_ms': data['t_ms'],
            'robot_qpos': data['robot_qpos'],
            'ik_error': data['ik_error'],
        }
        
        # Parse metadata
        try:
            metadata_raw = data['metadata']
            if isinstance(metadata_raw, np.ndarray):
                metadata_raw = metadata_raw.item()
            episode['metadata'] = json.loads(str(metadata_raw)) if isinstance(metadata_raw, str) else metadata_raw
        except:
            episode['metadata'] = {'fps': 30.0}
        
        return episode
    
    def compute_local_body_pos(self, qpos: np.ndarray) -> np.ndarray:
        """
        Compute body positions relative to pelvis using forward kinematics.
        
        Args:
            qpos: (N, 36) robot state [x, y, z, qw, qx, qy, qz, joints_29]
            
        Returns:
            local_body_pos: (N, 38, 3) body positions relative to pelvis
        """
        num_frames = qpos.shape[0]
        local_body_pos = np.zeros((num_frames, self.num_bodies, 3), dtype=np.float32)
        
        for i in range(num_frames):
            # Set robot state
            self.data.qpos[:] = qpos[i]
            
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # Get pelvis position (body index 1, since 0 is 'world')
            pelvis_pos = self.data.xpos[1].copy()
            
            # Get all body positions relative to pelvis
            for j, body_id in enumerate(self.body_ids):
                local_body_pos[i, j] = self.data.xpos[body_id] - pelvis_pos
        
        return local_body_pos
    
    def convert_episode(self, episode_name: str, output_dir: Optional[Path] = None) -> Path:
        """
        Convert a single episode to motion pickle format.
        
        Args:
            episode_name: Name of the episode (e.g., 'elbow_track_007')
            output_dir: Output directory (default: datasets/teleop_motions)
            
        Returns:
            Path to saved pickle file
        """
        output_dir = output_dir or MOTION_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find episode file
        episode_dir = EPISODES_DIR / episode_name
        episode_path = episode_dir / f"{episode_name}.npz"
        
        if not episode_path.exists():
            raise FileNotFoundError(f"Episode not found: {episode_path}")
        
        print(f"\nConverting: {episode_name}")
        
        # Load episode
        episode = self.load_episode(episode_path)
        robot_qpos = episode['robot_qpos']
        num_frames = robot_qpos.shape[0]
        
        # Extract root state and joint angles
        root_pos = robot_qpos[:, 0:3].astype(np.float32)  # (N, 3)
        
        # Episode qpos has quaternion in MuJoCo format [qw, qx, qy, qz] (scalar-first)
        # TWIST2 motion format expects [qx, qy, qz, qw] (scalar-last)
        root_rot_wxyz = robot_qpos[:, 3:7]
        root_rot = np.zeros_like(root_rot_wxyz, dtype=np.float32)
        root_rot[:, 0] = root_rot_wxyz[:, 1]  # qx
        root_rot[:, 1] = root_rot_wxyz[:, 2]  # qy
        root_rot[:, 2] = root_rot_wxyz[:, 3]  # qz
        root_rot[:, 3] = root_rot_wxyz[:, 0]  # qw
        
        dof_pos = robot_qpos[:, 7:36].astype(np.float32)  # (N, 29)
        
        # Get FPS from metadata
        fps = float(episode['metadata'].get('fps', 30.0))
        
        # Compute local body positions via FK
        print(f"  Computing FK for {num_frames} frames...")
        local_body_pos = self.compute_local_body_pos(robot_qpos)
        
        # Build motion data
        motion_data = {
            'fps': fps,
            'root_pos': root_pos,
            'root_rot': root_rot,
            'dof_pos': dof_pos,
            'local_body_pos': local_body_pos,
            'link_body_list': self.link_body_list,
        }
        
        # Save pickle
        output_path = output_dir / f"{episode_name}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(motion_data, f)
        
        # Print stats
        duration = num_frames / fps
        print(f"  Saved: {output_path}")
        print(f"  Frames: {num_frames}, Duration: {duration:.2f}s, FPS: {fps}")
        print(f"  root_pos: {root_pos.shape}, root_rot: {root_rot.shape}")
        print(f"  dof_pos: {dof_pos.shape}, local_body_pos: {local_body_pos.shape}")
        
        return output_path
    
    def convert_multiple(self, episode_names: List[str], output_dir: Optional[Path] = None) -> List[Path]:
        """Convert multiple episodes."""
        output_paths = []
        
        for name in tqdm(episode_names, desc="Converting episodes"):
            try:
                path = self.convert_episode(name, output_dir)
                output_paths.append(path)
            except Exception as e:
                print(f"  Error converting {name}: {e}")
        
        return output_paths
    
    @staticmethod
    def list_episodes() -> List[str]:
        """List all available episode names."""
        if not EPISODES_DIR.exists():
            return []
        
        episodes = []
        for d in sorted(EPISODES_DIR.iterdir()):
            if d.is_dir():
                npz_file = d / f"{d.name}.npz"
                if npz_file.exists():
                    episodes.append(d.name)
        
        return episodes
    
    @staticmethod
    def generate_yaml_config(motion_files: List[Path], config_name: str, 
                             output_dir: Optional[Path] = None) -> Path:
        """
        Generate YAML config file for TWIST2 training.
        
        Args:
            motion_files: List of motion pickle file paths
            config_name: Name for the config file
            output_dir: Output directory (default: motion_data_configs)
            
        Returns:
            Path to saved YAML file
        """
        output_dir = output_dir or (TWIST2_ROOT / "motion_data_configs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build YAML content
        yaml_content = f"""# TWIST2 Motion Dataset Config
# Generated from teleop episodes
# Config: {config_name}

root_path: {MOTION_OUTPUT_DIR}

motions:
"""
        for path in sorted(motion_files):
            yaml_content += f"  - file: {path.name}\n"
            yaml_content += f"    weight: 1.0\n"
        
        # Save YAML
        yaml_path = output_dir / f"{config_name}.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\nGenerated YAML config: {yaml_path}")
        print(f"  Motions: {len(motion_files)}")
        
        return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert teleop episodes to TWIST2 motion format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--episode', '-e', nargs='+',
        help='Episode name(s) to convert'
    )
    parser.add_argument(
        '--all', '-a', action='store_true',
        help='Convert all available episodes'
    )
    parser.add_argument(
        '--list', '-l', action='store_true',
        help='List available episodes'
    )
    parser.add_argument(
        '--yaml', '-y', type=str,
        help='Generate YAML config with given name'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output directory for motion files'
    )
    parser.add_argument(
        '--model', '-m', type=str,
        help='Path to G1 MuJoCo model XML'
    )
    parser.add_argument(
        '--subdir', '-s', type=str,
        help='Subdirectory within teleop_episodes (e.g., stage3_upper_body)'
    )
    
    args = parser.parse_args()
    
    # Update paths if subdir specified
    global EPISODES_DIR, MOTION_OUTPUT_DIR
    if args.subdir:
        EPISODES_DIR = TWIST2_ROOT / "datasets" / "teleop_episodes" / args.subdir
        MOTION_OUTPUT_DIR = TWIST2_ROOT / "datasets" / "teleop_motions" / args.subdir
        MOTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Using subdirectory: {args.subdir}")
        print(f"  Episodes: {EPISODES_DIR}")
        print(f"  Output:   {MOTION_OUTPUT_DIR}")
    
    # List episodes
    if args.list:
        episodes = EpisodeToMotionConverter.list_episodes()
        print(f"Available episodes ({len(episodes)}):")
        for ep in episodes:
            print(f"  - {ep}")
        return 0
    
    # Determine episodes to convert
    if args.all:
        episode_names = EpisodeToMotionConverter.list_episodes()
        if not episode_names:
            print("No episodes found!")
            return 1
        print(f"Converting all {len(episode_names)} episodes")
    elif args.episode:
        episode_names = args.episode
    else:
        parser.print_help()
        return 1
    
    # Setup converter
    model_path = Path(args.model) if args.model else None
    output_dir = Path(args.output) if args.output else None
    
    converter = EpisodeToMotionConverter(model_path)
    
    # Convert episodes
    if len(episode_names) == 1:
        motion_files = [converter.convert_episode(episode_names[0], output_dir)]
    else:
        motion_files = converter.convert_multiple(episode_names, output_dir)
    
    # Generate YAML config if requested
    if args.yaml and motion_files:
        converter.generate_yaml_config(motion_files, args.yaml)
    
    print(f"\nDone! Converted {len(motion_files)} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

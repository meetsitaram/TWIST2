#!/usr/bin/env python3
"""
Teleop Episode Recorder - Capture and store teleoperation episodes for analysis.

Records synchronized human skeleton and robot joint data during IK teleoperation.
Optionally records raw camera video for future reprocessing with improved pose estimation.
Designed for jitter analysis and smoothing evaluation.

Usage:
    from teleop_episode_recorder import TeleopEpisodeRecorder
    
    # Basic recording (skeleton + robot data only)
    recorder = TeleopEpisodeRecorder(name="baseline_001", fps=30)
    
    # With video recording (saves raw camera streams)
    recorder = TeleopEpisodeRecorder(
        name="baseline_001", 
        fps=30,
        record_video=True,
        camera_ids=[0, 2, 4],
    )
    
    # In streaming loop:
    recorder.add_frame(
        human_skeleton=skeleton_3d,
        robot_qpos=result['qpos'],
        ik_error=result['error'],
        camera_frames=latest_frames,  # Dict[int, np.ndarray] if recording video
    )
    
    # When done:
    recorder.save()
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
EPISODES_DIR = PROJECT_ROOT / "datasets" / "teleop_episodes"


@dataclass
class TeleopFrame:
    """Single frame of teleoperation data."""
    t_ms: int                          # Timestamp in milliseconds
    human_skeleton: np.ndarray         # (33, 3) MediaPipe landmarks
    robot_qpos: np.ndarray             # (36,) full configuration
    ik_error: float                    # IK solver error


@dataclass 
class TeleopEpisode:
    """Complete teleoperation episode with metadata."""
    name: str
    timestamp: str
    fps: int
    smoothing: str                     # "none", "ema", "one_euro", etc.
    smoothing_params: Dict             # Parameters used
    frames: List[TeleopFrame] = field(default_factory=list)
    
    @property
    def duration_sec(self) -> float:
        if not self.frames:
            return 0.0
        return (self.frames[-1].t_ms - self.frames[0].t_ms) / 1000.0
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)


class TeleopEpisodeRecorder:
    """
    Records teleoperation episodes for jitter analysis.
    
    Captures synchronized human skeleton and robot data at each frame.
    Optionally records raw camera video for future reprocessing.
    Saves skeleton/robot data to compressed NPZ format.
    Saves video as separate MP4 files per camera.
    """
    
    def __init__(
        self,
        name: str = None,
        fps: int = 30,
        smoothing: str = "none",
        smoothing_params: Dict = None,
        output_dir: Path = None,
        record_video: bool = False,
        camera_ids: List[int] = None,
        video_resolution: tuple = (1280, 720),
    ):
        """
        Initialize recorder.
        
        Args:
            name: Episode name (auto-generated if None)
            fps: Target frame rate
            smoothing: Smoothing method name for metadata
            smoothing_params: Smoothing parameters for metadata
            output_dir: Where to save episodes (default: datasets/teleop_episodes)
            record_video: Whether to record raw camera video
            camera_ids: List of camera IDs to record (required if record_video=True)
            video_resolution: Expected video resolution (width, height)
        """
        self.fps = fps
        self.smoothing = smoothing
        self.smoothing_params = smoothing_params or {}
        
        # Generate name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"episode_{timestamp}"
        self.name = name
        
        # Each episode gets its own subdirectory
        base_dir = Path(output_dir) if output_dir else EPISODES_DIR
        self.output_dir = base_dir / name
        
        # Video recording setup
        self.record_video = record_video
        self.camera_ids = camera_ids or []
        self.video_resolution = video_resolution
        self.video_writers: Dict[int, 'cv2.VideoWriter'] = {}
        
        # Frame storage
        self.frames: List[TeleopFrame] = []
        self.start_time_ms: Optional[int] = None
        self.recording = False
        self.video_frame_count = 0
        
    def start(self):
        """Start recording (resets any existing frames)."""
        self.frames = []
        self.start_time_ms = None
        self.video_frame_count = 0
        self.recording = True
        
        # Initialize video writers if recording video
        if self.record_video and self.camera_ids:
            self._init_video_writers()
        
    def _init_video_writers(self):
        """Initialize video writers for each camera."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use H.264 codec for good compression and compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        for cam_id in self.camera_ids:
            video_path = self.output_dir / f"{self.name}_cam{cam_id}.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.fps,
                self.video_resolution,
            )
            if writer.isOpened():
                self.video_writers[cam_id] = writer
                print(f"[Recorder] Video writer initialized: {video_path.name}")
            else:
                print(f"[Recorder] Warning: Failed to create video writer for camera {cam_id}")
        
    def _release_video_writers(self):
        """Release all video writers."""
        for cam_id, writer in self.video_writers.items():
            if writer is not None:
                writer.release()
        self.video_writers = {}
        
    def stop(self):
        """Stop recording."""
        self.recording = False
        self._release_video_writers()
        
    def add_frame(
        self,
        human_skeleton: np.ndarray,
        robot_qpos: np.ndarray,
        ik_error: float,
        timestamp_ms: int = None,
        camera_frames: Dict[int, np.ndarray] = None,
    ):
        """
        Add a frame to the recording.
        
        Args:
            human_skeleton: (33, 3) MediaPipe skeleton
            robot_qpos: (36,) robot configuration
            ik_error: IK solver error for this frame
            timestamp_ms: Optional explicit timestamp (auto-computed if None)
            camera_frames: Dict mapping camera_id -> frame image (for video recording)
        """
        if not self.recording:
            return
            
        # Auto-compute timestamp
        if timestamp_ms is None:
            import time
            timestamp_ms = int(time.time() * 1000)
        
        # Set start time on first frame
        if self.start_time_ms is None:
            self.start_time_ms = timestamp_ms
        
        # Store relative timestamp
        t_ms = timestamp_ms - self.start_time_ms
        
        frame = TeleopFrame(
            t_ms=t_ms,
            human_skeleton=np.array(human_skeleton, dtype=np.float32),
            robot_qpos=np.array(robot_qpos, dtype=np.float32),
            ik_error=float(ik_error),
        )
        self.frames.append(frame)
        
        # Write video frames if recording
        if self.record_video and camera_frames:
            for cam_id, writer in self.video_writers.items():
                if cam_id in camera_frames and camera_frames[cam_id] is not None:
                    frame_img = camera_frames[cam_id]
                    # Resize if needed
                    if frame_img.shape[:2][::-1] != self.video_resolution:
                        frame_img = cv2.resize(frame_img, self.video_resolution)
                    writer.write(frame_img)
            self.video_frame_count += 1
        
    def save(self, filepath: Path = None) -> Path:
        """
        Save episode to NPZ file (and finalize video files).
        
        Args:
            filepath: Optional explicit path (auto-generated if None)
            
        Returns:
            Path to saved NPZ file
        """
        # Release video writers first to ensure files are properly closed
        self._release_video_writers()
        
        if not self.frames:
            print("[Recorder] No frames to save")
            return None
            
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filepath
        if filepath is None:
            filepath = self.output_dir / f"{self.name}.npz"
        
        # Stack frame data into arrays
        t_ms = np.array([f.t_ms for f in self.frames], dtype=np.int32)
        human_skeleton = np.stack([f.human_skeleton for f in self.frames])
        robot_qpos = np.stack([f.robot_qpos for f in self.frames])
        ik_error = np.array([f.ik_error for f in self.frames], dtype=np.float32)
        
        # Build list of video files
        video_files = []
        if self.record_video and self.camera_ids:
            for cam_id in self.camera_ids:
                video_path = self.output_dir / f"{self.name}_cam{cam_id}.mp4"
                if video_path.exists():
                    video_files.append(str(video_path.name))
        
        # Metadata as JSON string
        metadata = {
            "name": self.name,
            "timestamp": datetime.now().isoformat(),
            "fps": self.fps,
            "num_frames": len(self.frames),
            "duration_sec": float(t_ms[-1] / 1000.0),
            "smoothing": self.smoothing,
            "smoothing_params": self.smoothing_params,
            "has_video": len(video_files) > 0,
            "video_files": video_files,
            "camera_ids": self.camera_ids,
            "video_resolution": list(self.video_resolution) if self.record_video else None,
        }
        metadata_json = json.dumps(metadata)
        
        # Save compressed
        np.savez_compressed(
            filepath,
            t_ms=t_ms,
            human_skeleton=human_skeleton,
            robot_qpos=robot_qpos,
            ik_error=ik_error,
            metadata=metadata_json,
        )
        
        duration = t_ms[-1] / 1000.0
        print(f"[Recorder] Saved {len(self.frames)} frames ({duration:.1f}s) to {filepath}")
        
        if video_files:
            print(f"[Recorder] Video files: {', '.join(video_files)}")
        
        return filepath
    
    @staticmethod
    def load(filepath: Path) -> 'TeleopEpisode':
        """
        Load episode from NPZ file.
        
        Args:
            filepath: Path to NPZ file
            
        Returns:
            TeleopEpisode object
        """
        data = np.load(filepath, allow_pickle=True)
        
        # Parse metadata
        metadata = json.loads(str(data['metadata']))
        
        # Reconstruct frames
        frames = []
        for i in range(len(data['t_ms'])):
            frame = TeleopFrame(
                t_ms=int(data['t_ms'][i]),
                human_skeleton=data['human_skeleton'][i],
                robot_qpos=data['robot_qpos'][i],
                ik_error=float(data['ik_error'][i]),
            )
            frames.append(frame)
        
        episode = TeleopEpisode(
            name=metadata['name'],
            timestamp=metadata['timestamp'],
            fps=metadata['fps'],
            smoothing=metadata['smoothing'],
            smoothing_params=metadata['smoothing_params'],
            frames=frames,
        )
        
        return episode
    
    @staticmethod
    def list_episodes(episodes_dir: Path = None) -> List[Path]:
        """List all episode files in directory (each episode is in its own subdir)."""
        if episodes_dir is None:
            episodes_dir = EPISODES_DIR
        
        if not episodes_dir.exists():
            return []
        
        # Each episode is in its own subdirectory: episodes_dir/episode_name/episode_name.npz
        episode_files = []
        for subdir in sorted(episodes_dir.iterdir()):
            if subdir.is_dir():
                npz_file = subdir / f"{subdir.name}.npz"
                if npz_file.exists():
                    episode_files.append(npz_file)
        
        return episode_files


def main():
    """Test the recorder with dummy data."""
    print("Testing TeleopEpisodeRecorder...")
    
    # Create recorder
    recorder = TeleopEpisodeRecorder(
        name="test_episode",
        fps=30,
        smoothing="none",
    )
    
    # Simulate recording
    recorder.start()
    
    import time
    for i in range(60):  # 2 seconds at 30 FPS
        # Dummy data
        skeleton = np.random.randn(33, 3).astype(np.float32) * 0.1
        qpos = np.random.randn(36).astype(np.float32) * 0.1
        
        recorder.add_frame(
            human_skeleton=skeleton,
            robot_qpos=qpos,
            ik_error=0.05 + np.random.rand() * 0.02,
        )
        time.sleep(1/30)
    
    recorder.stop()
    
    # Save
    filepath = recorder.save()
    
    # Load and verify
    if filepath:
        episode = TeleopEpisodeRecorder.load(filepath)
        print(f"\nLoaded episode: {episode.name}")
        print(f"  Frames: {episode.num_frames}")
        print(f"  Duration: {episode.duration_sec:.2f}s")
        print(f"  Smoothing: {episode.smoothing}")
        
        # Cleanup test file
        filepath.unlink()
        print(f"\nCleaned up test file")


if __name__ == "__main__":
    main()

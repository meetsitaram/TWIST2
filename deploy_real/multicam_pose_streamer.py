#!/usr/bin/env python3
"""
Multi-Camera Pose Streamer

Real-time 3D skeleton capture using multiple calibrated cameras.
Follows FreeMoCap's approach: Per-camera MediaPipe detection + DLT triangulation.

Architecture:
    Live Cameras → MediaPipe (per cam) → 2D landmarks → Triangulate → 3D skeleton

Usage:
    conda activate gmr
    python deploy_real/multicam_pose_streamer.py --display
"""

import cv2
import numpy as np
import toml
import yaml
import time
import logging
import argparse
import os
from threading import Thread, Lock
from typing import Dict, List, Optional, Tuple
from numba import jit

# MediaPipe import
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not found. Install with: pip install mediapipe")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import smoothing filter
try:
    from smoothing_filters import OneEuroFilter
    HAS_SMOOTHING = True
except ImportError:
    HAS_SMOOTHING = False
    logger.warning("smoothing_filters not available - skeleton smoothing disabled")


# =============================================================================
# Triangulation (from FreeMoCap/anipose)
# =============================================================================

@jit(nopython=True)
def triangulate_simple(points: np.ndarray, camera_mats: np.ndarray) -> np.ndarray:
    """
    Triangulate a 3D point from multiple 2D observations using DLT.
    
    This is the core triangulation algorithm from FreeMoCap/anipose.
    Uses SVD to solve the linear system.
    
    Args:
        points: (N, 2) array of 2D points from N cameras
        camera_mats: (N, 3, 4) array of projection matrices
    
    Returns:
        (3,) 3D point in world coordinates
    """
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[i * 2] = x * mat[2] - mat[0]
        A[i * 2 + 1] = y * mat[2] - mat[1]
    
    # SVD solution
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    
    return p3d


def undistort_points(points: np.ndarray, camera_matrix: np.ndarray, 
                     dist_coeffs: np.ndarray) -> np.ndarray:
    """Undistort 2D points using camera intrinsics."""
    if points is None or len(points) == 0:
        return points
    
    points_reshaped = points.reshape(-1, 1, 2).astype(np.float64)
    undistorted = cv2.undistortPoints(
        points_reshaped, 
        camera_matrix.astype(np.float64), 
        dist_coeffs.astype(np.float64)
    )
    return undistorted.reshape(-1, 2)


def project_point(point_3d: np.ndarray, camera_matrix: np.ndarray,
                  rotation: np.ndarray, translation: np.ndarray,
                  dist_coeffs: np.ndarray) -> np.ndarray:
    """Project a 3D point to 2D image coordinates."""
    point_3d = point_3d.reshape(1, 3)
    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(
        point_3d, rvec, translation.reshape(3, 1),
        camera_matrix, dist_coeffs
    )
    return projected.reshape(2)


# =============================================================================
# Camera and Calibration
# =============================================================================

class CameraCalibration:
    """Container for camera calibration data."""
    
    def __init__(self, cam_id: int, matrix: np.ndarray, distortions: np.ndarray,
                 rotation: np.ndarray, translation: np.ndarray, size: Tuple[int, int]):
        self.cam_id = cam_id
        self.matrix = np.array(matrix, dtype=np.float64)
        self.distortions = np.array(distortions, dtype=np.float64).ravel()
        self.rotation = np.array(rotation, dtype=np.float64)
        self.translation = np.array(translation, dtype=np.float64).ravel()
        self.size = size
        
        # Precompute extrinsics matrix [R | t] (3x4) - for triangulation with undistorted points
        self.extrinsics_matrix = np.hstack([self.rotation, self.translation.reshape(3, 1)])
        
        # Precompute full projection matrix P = K @ [R | t] - for reprojection to pixels
        self.projection_matrix = self.matrix @ self.extrinsics_matrix
    
    @classmethod
    def from_toml_dict(cls, cam_id: int, data: dict) -> 'CameraCalibration':
        """Create from TOML calibration data."""
        return cls(
            cam_id=cam_id,
            matrix=np.array(data['matrix']),
            distortions=np.array(data['distortions']),
            rotation=np.array(data.get('rotation', np.eye(3))),
            translation=np.array(data.get('translation', [0, 0, 0])),
            size=tuple(data.get('size', [1280, 720]))
        )


def load_calibration(calibration_path: str) -> Dict[int, CameraCalibration]:
    """Load camera calibration from TOML file."""
    with open(calibration_path, 'r') as f:
        data = toml.load(f)
    
    calibrations = {}
    for cam_key, cam_data in data.get('cameras', {}).items():
        cam_id = int(cam_key.replace('cam_', ''))
        calibrations[cam_id] = CameraCalibration.from_toml_dict(cam_id, cam_data)
    
    logger.info(f"Loaded calibration for cameras: {list(calibrations.keys())}")
    return calibrations


# =============================================================================
# Multi-Camera Capture
# =============================================================================

class MultiCameraCapture:
    """Threaded capture from multiple cameras."""
    
    def __init__(self, camera_ids: List[int], resolution: Tuple[int, int] = (1280, 720)):
        self.camera_ids = camera_ids
        self.resolution = resolution
        self.captures: Dict[int, cv2.VideoCapture] = {}
        self.frames: Dict[int, np.ndarray] = {}
        self.timestamps: Dict[int, float] = {}
        self.frame_lock = Lock()
        self.running = False
        self.threads: List[Thread] = []
    
    def start(self):
        """Start capture threads for all cameras."""
        self.running = True
        
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            
            # Set MJPEG mode for better FPS
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                logger.warning(f"Could not open camera {cam_id}")
                continue
            
            # Warm up camera - discard first frames to ensure stable capture
            for _ in range(10):
                cap.read()
            
            # Verify camera is returning frames
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(f"Camera {cam_id} opened but not returning frames")
                cap.release()
                continue
            
            self.captures[cam_id] = cap
            
            # Start capture thread
            t = Thread(target=self._capture_loop, args=(cam_id, cap), daemon=True)
            t.start()
            self.threads.append(t)
        
        logger.info(f"Started {len(self.captures)} cameras: {list(self.captures.keys())}")
    
    def _capture_loop(self, cam_id: int, cap: cv2.VideoCapture):
        """Capture loop for a single camera."""
        while self.running:
            ret, frame = cap.read()
            if ret:
                timestamp = time.time()
                with self.frame_lock:
                    self.frames[cam_id] = frame.copy()
                    self.timestamps[cam_id] = timestamp
            time.sleep(0.001)
    
    def get_synchronized_frames(self, max_time_diff: float = 0.05) -> Tuple[Dict[int, np.ndarray], float, float]:
        """
        Get frames from all cameras with approximate synchronization.
        
        Args:
            max_time_diff: Maximum allowed time difference between frames (seconds)
                           Default 50ms - tighter than calibration for real-time accuracy
        
        Returns:
            (frames_dict, timestamp, time_spread): 
                - Dictionary of camera_id -> frame
                - Average timestamp
                - Time spread between oldest and newest frame (for quality check)
        """
        with self.frame_lock:
            if not self.frames:
                return {}, 0.0, 0.0
            
            # Check time spread between cameras
            timestamps = list(self.timestamps.values())
            cam_ids = list(self.timestamps.keys())
            
            if len(timestamps) < 2:
                frames = {k: v.copy() for k, v in self.frames.items()}
                return frames, timestamps[0] if timestamps else 0.0, 0.0
            
            time_spread = max(timestamps) - min(timestamps)
            avg_timestamp = np.mean(timestamps)
            
            # If frames are too out of sync, only return cameras within tolerance
            if time_spread > max_time_diff:
                # Find the newest timestamp and include only cameras within max_time_diff
                newest_ts = max(timestamps)
                synced_frames = {}
                for cam_id, ts in self.timestamps.items():
                    if newest_ts - ts <= max_time_diff:
                        synced_frames[cam_id] = self.frames[cam_id].copy()
                
                if len(synced_frames) >= 2:
                    logger.debug(f"Frame sync: {len(synced_frames)}/{len(self.frames)} cameras within {max_time_diff*1000:.0f}ms")
                    return synced_frames, avg_timestamp, time_spread
                else:
                    # Not enough synced cameras, return all but flag it
                    logger.debug(f"Poor sync: {time_spread*1000:.1f}ms spread across {len(self.frames)} cameras")
            
            frames = {k: v.copy() for k, v in self.frames.items()}
            return frames, avg_timestamp, time_spread
    
    def get_all_frames(self) -> Dict[int, np.ndarray]:
        """
        Get latest frames from all cameras without sync filtering.
        Use this for display purposes where sync timing doesn't matter.
        
        Returns:
            Dictionary of camera_id -> frame
        """
        with self.frame_lock:
            if not self.frames:
                return {}
            return {k: v.copy() for k, v in self.frames.items()}
    
    def stop(self):
        """Stop all capture threads."""
        self.running = False
        time.sleep(0.1)
        for cap in self.captures.values():
            cap.release()


# =============================================================================
# Per-Camera MediaPipe Detection
# =============================================================================

class MediaPipeDetector:
    """MediaPipe Holistic detector for a single camera."""
    
    # MediaPipe pose landmark indices
    NUM_POSE_LANDMARKS = 33
    NUM_HAND_LANDMARKS = 21
    
    def __init__(self, static_mode: bool = False, model_complexity: int = 1):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required")
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect(self, frame: np.ndarray) -> dict:
        """
        Run MediaPipe detection on a frame.
        
        Args:
            frame: BGR image
        
        Returns:
            dict with 'pose', 'left_hand', 'right_hand' as (N, 3) arrays or None
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.holistic.process(frame_rgb)
        
        output = {
            'pose': None,
            'left_hand': None,
            'right_hand': None,
            'pose_visibility': None,
            'raw_results': results  # Keep for visualization
        }
        
        # Extract pose landmarks
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            visibility = np.array([lm.visibility for lm in results.pose_landmarks.landmark])
            output['pose'] = pose  # (33, 3) normalized [0,1]
            output['pose_visibility'] = visibility  # (33,)
        
        # Extract hand landmarks
        if results.left_hand_landmarks:
            left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
            output['left_hand'] = left_hand  # (21, 3)
        
        if results.right_hand_landmarks:
            right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            output['right_hand'] = right_hand  # (21, 3)
        
        return output
    
    def close(self):
        """Release resources."""
        self.holistic.close()


# =============================================================================
# Multi-Camera Pose Triangulator
# =============================================================================

class MultiCameraPoseTriangulator:
    """
    Triangulate 3D pose from multiple camera 2D detections.
    
    Follows FreeMoCap's approach:
    1. Collect 2D landmarks from each camera
    2. Undistort points using calibration
    3. Triangulate using DLT
    4. Compute reprojection error for quality assessment
    """
    
    def __init__(self, calibrations: Dict[int, CameraCalibration]):
        self.calibrations = calibrations
        self.camera_ids = sorted(calibrations.keys())
        
        # Precompute extrinsics matrices [R | t] for triangulation (after undistortion)
        self.extrinsics_matrices = np.array([
            calibrations[cam_id].extrinsics_matrix 
            for cam_id in self.camera_ids
        ])
    
    def triangulate_landmarks(
        self, 
        landmarks_2d: Dict[int, np.ndarray],
        visibility: Optional[Dict[int, np.ndarray]] = None,
        min_cameras: int = 2,
        visibility_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangulate 2D landmarks from multiple cameras to 3D.
        
        Args:
            landmarks_2d: Dict of camera_id -> (N, 2) 2D landmarks in pixel coordinates
            visibility: Optional dict of camera_id -> (N,) visibility scores
            min_cameras: Minimum cameras that must see a point for triangulation
            visibility_threshold: Minimum visibility score to use a detection
        
        Returns:
            landmarks_3d: (N, 3) 3D landmarks in world coordinates (meters)
            reprojection_errors: (N,) average reprojection error per landmark
        """
        if not landmarks_2d:
            return np.array([]), np.array([])
        
        # Get number of landmarks from first detection
        n_landmarks = next(iter(landmarks_2d.values())).shape[0]
        
        # Prepare output
        landmarks_3d = np.full((n_landmarks, 3), np.nan)
        reprojection_errors = np.full(n_landmarks, np.nan)
        
        # Triangulate each landmark
        for i in range(n_landmarks):
            # Collect 2D observations from all cameras
            points_2d = []
            cam_indices = []
            
            for cam_idx, cam_id in enumerate(self.camera_ids):
                if cam_id not in landmarks_2d:
                    continue
                
                lm = landmarks_2d[cam_id]
                if i >= len(lm):
                    continue
                
                # Check visibility
                if visibility and cam_id in visibility:
                    if visibility[cam_id][i] < visibility_threshold:
                        continue
                
                # Check for NaN
                point = lm[i]
                if np.any(np.isnan(point)):
                    continue
                
                points_2d.append(point)
                cam_indices.append(cam_idx)
            
            # Need at least min_cameras to triangulate
            if len(points_2d) < min_cameras:
                continue
            
            # Undistort points
            undistorted_points = []
            for point, cam_idx in zip(points_2d, cam_indices):
                cam_id = self.camera_ids[cam_idx]
                cal = self.calibrations[cam_id]
                undist = undistort_points(point.reshape(1, 2), cal.matrix, cal.distortions)
                undistorted_points.append(undist[0])
            
            undistorted_points = np.array(undistorted_points)
            
            # Get extrinsics matrices for visible cameras (use [R|t] after undistortion)
            ext_mats = self.extrinsics_matrices[cam_indices]
            
            # Triangulate
            try:
                point_3d = triangulate_simple(undistorted_points, ext_mats)
                landmarks_3d[i] = point_3d
                
                # Compute reprojection error
                errors = []
                for point_orig, cam_idx in zip(points_2d, cam_indices):
                    cam_id = self.camera_ids[cam_idx]
                    cal = self.calibrations[cam_id]
                    projected = project_point(
                        point_3d, cal.matrix, cal.rotation, 
                        cal.translation, cal.distortions
                    )
                    error = np.linalg.norm(point_orig - projected)
                    errors.append(error)
                
                reprojection_errors[i] = np.mean(errors)
                
            except Exception as e:
                logger.debug(f"Triangulation failed for landmark {i}: {e}")
                continue
        
        return landmarks_3d, reprojection_errors
    
    def normalized_to_pixel(
        self, 
        landmarks_normalized: np.ndarray, 
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Convert normalized [0,1] landmarks to pixel coordinates.
        
        Args:
            landmarks_normalized: (N, 2 or 3) normalized landmarks
            image_size: (width, height) of image
        
        Returns:
            (N, 2) pixel coordinates
        """
        w, h = image_size
        landmarks_pixel = landmarks_normalized[:, :2].copy()
        landmarks_pixel[:, 0] *= w
        landmarks_pixel[:, 1] *= h
        return landmarks_pixel


# =============================================================================
# Main Multi-Camera Pose Streamer
# =============================================================================

class MultiCamPoseStreamer:
    """
    Real-time 3D skeleton streaming from multiple calibrated cameras.
    
    Drop-in replacement for single-camera FreeMoCapStreamer.
    """
    
    def __init__(
        self,
        camera_ids: List[int],
        calibration_file: str,
        resolution: Tuple[int, int] = (1280, 720),
        enable_hands: bool = True,
        enable_display: bool = False,
        target_fps: int = 30,
        use_gmr: bool = False,
        skeleton_smoothing: str = "none",
        smoothing_min_cutoff: float = 1.0,
        smoothing_beta: float = 0.007,
        # World frame correction for camera tilt
        # Rotates triangulated skeleton to correct for cameras pointing downward
        # Positive pitch_correction rotates skeleton backward (fixes forward tilt)
        world_pitch_correction_deg: float = 0.0,
        world_roll_correction_deg: float = 0.0,
        # Leg-only pitch correction (rotates legs around pelvis)
        # Positive = rotate legs forward, Negative = rotate legs backward
        leg_pitch_correction_deg: float = 0.0,
    ):
        """
        Initialize multi-camera pose streamer.
        
        Args:
            camera_ids: List of camera device IDs
            calibration_file: Path to calibration.toml
            resolution: Capture resolution (width, height)
            enable_hands: Whether to track hands
            enable_display: Show visualization window
            target_fps: Target frame rate
            use_gmr: Use GMR IK-based retargeting instead of direct mapping
            skeleton_smoothing: Smoothing method - "none", "one_euro"
            smoothing_min_cutoff: One Euro min_cutoff parameter (lower = smoother)
            smoothing_beta: One Euro beta parameter (higher = more responsive to speed)
            world_pitch_correction_deg: Pitch correction in degrees (positive = rotate backward)
            world_roll_correction_deg: Roll correction in degrees (positive = roll right)
            leg_pitch_correction_deg: Leg-only pitch correction (positive = forward, negative = backward)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe required. Install with: pip install mediapipe")
        
        self.camera_ids = camera_ids
        self.resolution = resolution
        self.enable_hands = enable_hands
        self.enable_display = enable_display
        self.target_fps = target_fps
        self.use_gmr = use_gmr
        
        # Load calibration
        self.calibrations = load_calibration(calibration_file)
        
        # Verify all cameras have calibration
        for cam_id in camera_ids:
            if cam_id not in self.calibrations:
                raise ValueError(f"Camera {cam_id} not found in calibration file")
        
        # Initialize components
        self.capture = MultiCameraCapture(camera_ids, resolution)
        self.detectors = {cam_id: MediaPipeDetector() for cam_id in camera_ids}
        self.triangulator = MultiCameraPoseTriangulator(self.calibrations)
        
        # State
        self.latest_3d_skeleton: Optional[np.ndarray] = None
        self._prev_skeleton: Optional[np.ndarray] = None  # For outlier rejection
        self.latest_hands_3d: Dict[str, Optional[np.ndarray]] = {
            'left': None, 'right': None
        }
        self.latest_reprojection_error: float = 0.0
        self.latest_frames: Dict[int, np.ndarray] = {}
        self.latest_detections: Dict[int, dict] = {}
        self._latest_display_frame: Optional[np.ndarray] = None  # Pre-rendered display frame
        self.data_lock = Lock()
        
        # Skeleton smoothing
        self.skeleton_smoothing = skeleton_smoothing
        self.skeleton_filter: Optional['OneEuroFilter'] = None
        if skeleton_smoothing == "one_euro" and HAS_SMOOTHING:
            # 33 landmarks * 3 coordinates = 99 dimensions
            self.skeleton_filter = OneEuroFilter(
                min_cutoff=smoothing_min_cutoff,
                beta=smoothing_beta,
                num_dims=99,
            )
            logger.info(f"Skeleton smoothing enabled: {skeleton_smoothing} "
                       f"(min_cutoff={smoothing_min_cutoff}, beta={smoothing_beta})")
        elif skeleton_smoothing != "none":
            logger.warning(f"Unknown smoothing method '{skeleton_smoothing}' or smoothing not available")
        
        # World frame rotation correction (for camera tilt)
        # This rotates the triangulated skeleton to align with world Z-up
        self.world_rotation_matrix = None
        if abs(world_pitch_correction_deg) > 0.1 or abs(world_roll_correction_deg) > 0.1:
            pitch_rad = np.radians(world_pitch_correction_deg)
            roll_rad = np.radians(world_roll_correction_deg)
            
            # Rotation around X-axis (pitch correction - forward/backward tilt)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                [0, np.sin(pitch_rad), np.cos(pitch_rad)]
            ])
            
            # Rotation around Y-axis (roll correction - side tilt)
            Ry = np.array([
                [np.cos(roll_rad), 0, np.sin(roll_rad)],
                [0, 1, 0],
                [-np.sin(roll_rad), 0, np.cos(roll_rad)]
            ])
            
            # Combined rotation: first roll, then pitch
            self.world_rotation_matrix = Rx @ Ry
            logger.info(f"World frame correction enabled: pitch={world_pitch_correction_deg}°, roll={world_roll_correction_deg}°")
        
        # Leg-only pitch correction
        self.leg_pitch_correction_deg = leg_pitch_correction_deg
        self.leg_rotation_matrix = None
        if abs(leg_pitch_correction_deg) > 0.1:
            leg_pitch_rad = np.radians(leg_pitch_correction_deg)
            self.leg_rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(leg_pitch_rad), -np.sin(leg_pitch_rad)],
                [0, np.sin(leg_pitch_rad), np.cos(leg_pitch_rad)]
            ])
            logger.info(f"Leg pitch correction enabled: {leg_pitch_correction_deg}°")
        
        # Processing thread
        self.is_running = False
        self.process_thread: Optional[Thread] = None
        
        # FPS tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Display window
        self.window_name = "Multi-Camera Pose - Press 'q' to quit"
        self._window_created = False
        self._last_time_spread = 0.0  # Track frame sync quality
    
    def start(self):
        """Start streaming."""
        self.capture.start()
        
        self.is_running = True
        self.process_thread = Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        logger.info("Multi-camera pose streamer started")
    
    def _process_loop(self):
        """Main processing loop."""
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create resizable window in processing thread (OpenCV GUI must be in same thread)
        if self.enable_display and not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 960)  # Default size, user can resize/maximize
            self._window_created = True
        
        while self.is_running:
            start_time = time.time()
            
            # Get ALL frames for display (no sync filtering - prevents flashing)
            all_frames = self.capture.get_all_frames()
            
            if not all_frames:
                time.sleep(0.01)
                continue
            
            # Get synchronized frames for triangulation (with sync filtering for accuracy)
            synced_frames, timestamp, time_spread = self.capture.get_synchronized_frames()
            
            # Store sync quality for display
            self._last_time_spread = time_spread
            
            # Run MediaPipe detection on ALL frames (for display)
            all_detections = {}
            for cam_id, frame in all_frames.items():
                if cam_id in self.detectors:
                    all_detections[cam_id] = self.detectors[cam_id].detect(frame)
            
            # Always pre-render display frame using ALL cameras
            display_frame = self._render_display_frame(all_frames, all_detections, mp_drawing, mp_holistic, mp_drawing_styles)
            if display_frame is not None:
                with self.data_lock:
                    self._latest_display_frame = display_frame
                    # Store raw frames for external access (e.g., video recording)
                    self.latest_frames = {cam_id: frame.copy() for cam_id, frame in all_frames.items()}
                    self.latest_detections = dict(all_detections)
            
            # Need at least 2 synced cameras for triangulation
            if len(synced_frames) < 2:
                time.sleep(0.01)
                continue
            
            # Use detections from synced frames for triangulation (filter from all_detections)
            synced_detections = {cam_id: all_detections[cam_id] 
                                 for cam_id in synced_frames.keys() 
                                 if cam_id in all_detections}
            
            # Collect 2D pose landmarks from synced cameras only
            landmarks_2d = {}
            visibility = {}
            
            for cam_id, det in synced_detections.items():
                if det['pose'] is not None:
                    # Convert normalized to pixel coordinates
                    pose_normalized = det['pose']
                    pose_pixel = self.triangulator.normalized_to_pixel(
                        pose_normalized, self.resolution
                    )
                    landmarks_2d[cam_id] = pose_pixel
                    
                    if det['pose_visibility'] is not None:
                        visibility[cam_id] = det['pose_visibility']
            
            # Triangulate to 3D
            if len(landmarks_2d) >= 2:
                skeleton_3d, reproj_errors = self.triangulator.triangulate_landmarks(
                    landmarks_2d, visibility, min_cameras=2
                )
                
                avg_error = np.nanmean(reproj_errors) if len(reproj_errors) > 0 else 0.0
                
                # Apply skeleton smoothing if enabled
                if self.skeleton_filter is not None and skeleton_3d is not None:
                    # Only smooth valid (non-NaN) values
                    valid_mask = ~np.isnan(skeleton_3d).any(axis=1)
                    if valid_mask.sum() > 0:
                        # Flatten for filtering, then reshape
                        skeleton_flat = skeleton_3d.flatten()
                        # Replace NaN with 0 for filtering (will be restored after)
                        nan_mask = np.isnan(skeleton_flat)
                        skeleton_flat[nan_mask] = 0.0
                        
                        smoothed_flat = self.skeleton_filter.filter(skeleton_flat, timestamp)
                        skeleton_3d = smoothed_flat.reshape(33, 3)
                        
                        # Restore NaN values for invalid landmarks
                        for i in range(33):
                            if not valid_mask[i]:
                                skeleton_3d[i] = np.nan
                
                # Apply world frame rotation correction if configured
                if self.world_rotation_matrix is not None:
                    # Rotate each valid landmark
                    for i in range(len(skeleton_3d)):
                        if not np.isnan(skeleton_3d[i, 0]):
                            skeleton_3d[i] = self.world_rotation_matrix @ skeleton_3d[i]
                
                # Apply leg-only pitch correction if configured
                if self.leg_rotation_matrix is not None:
                    # MediaPipe leg landmark indices
                    LEG_LANDMARKS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]  # hips, knees, ankles, heels, feet
                    # Get pelvis as rotation center
                    left_hip = skeleton_3d[23]
                    right_hip = skeleton_3d[24]
                    if not np.isnan(left_hip[0]) and not np.isnan(right_hip[0]):
                        pelvis = (left_hip + right_hip) / 2
                        for i in LEG_LANDMARKS:
                            if not np.isnan(skeleton_3d[i, 0]):
                                rel_pos = skeleton_3d[i] - pelvis
                                rotated_rel = self.leg_rotation_matrix @ rel_pos
                                skeleton_3d[i] = pelvis + rotated_rel
                
                with self.data_lock:
                    self.latest_3d_skeleton = skeleton_3d
                    self.latest_reprojection_error = avg_error
                
                # Triangulate hands if enabled
                if self.enable_hands:
                    self._triangulate_hands(synced_detections)
            
            # Display if enabled (use all frames, not just synced)
            if self.enable_display:
                self._display_results(all_frames, all_detections, mp_drawing, mp_holistic, mp_drawing_styles)
            
            # Update FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.fps_start_time
                self.current_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_start_time = time.time()
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / self.target_fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _triangulate_hands(self, detections: Dict[int, dict]):
        """Triangulate hand landmarks."""
        for hand_name in ['left_hand', 'right_hand']:
            landmarks_2d = {}
            
            for cam_id, det in detections.items():
                if det[hand_name] is not None:
                    hand_normalized = det[hand_name]
                    hand_pixel = self.triangulator.normalized_to_pixel(
                        hand_normalized, self.resolution
                    )
                    landmarks_2d[cam_id] = hand_pixel
            
            if len(landmarks_2d) >= 2:
                hand_3d, _ = self.triangulator.triangulate_landmarks(
                    landmarks_2d, min_cameras=2
                )
                
                key = 'left' if hand_name == 'left_hand' else 'right'
                with self.data_lock:
                    self.latest_hands_3d[key] = hand_3d
    
    def _display_results(self, frames, detections, mp_drawing, mp_holistic, mp_drawing_styles):
        """Display visualization."""
        vis_frames = []
        
        for cam_id in sorted(frames.keys()):
            frame = frames[cam_id].copy()
            
            if cam_id in detections:
                det = detections[cam_id]
                results = det.get('raw_results')
                
                if results and results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                if self.enable_hands and results:
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS
                        )
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS
                        )
            
            # Add camera label and FPS
            cv2.putText(frame, f"Cam {cam_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show reprojection error
            with self.data_lock:
                error = self.latest_reprojection_error
            error_color = (0, 255, 0) if error < 10 else (0, 255, 255) if error < 20 else (0, 0, 255)
            cv2.putText(frame, f"Reproj: {error:.1f}px", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, error_color, 1)
            
            # Show sync status (time spread between cameras)
            sync_ms = self._last_time_spread * 1000
            sync_color = (0, 255, 0) if sync_ms < 30 else (0, 255, 255) if sync_ms < 50 else (0, 0, 255)
            cv2.putText(frame, f"Sync: {sync_ms:.0f}ms", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, sync_color, 1)
            
            # Resize for display
            scale = 640 / frame.shape[1]
            resized = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
            vis_frames.append(resized)
        
        # Arrange frames
        if len(vis_frames) == 3:
            top_row = np.hstack(vis_frames[:2])
            pad_width = (top_row.shape[1] - vis_frames[2].shape[1]) // 2
            bottom = np.pad(vis_frames[2], 
                           ((0, 0), (pad_width, top_row.shape[1] - vis_frames[2].shape[1] - pad_width), (0, 0)),
                           mode='constant')
            combined = np.vstack([top_row, bottom])
        elif len(vis_frames) == 2:
            combined = np.hstack(vis_frames)
        else:
            combined = vis_frames[0] if vis_frames else np.zeros((480, 640, 3), dtype=np.uint8)
        
        cv2.imshow(self.window_name, combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.is_running = False
    
    def _render_display_frame(self, frames, detections, mp_drawing, mp_holistic, mp_drawing_styles) -> Optional[np.ndarray]:
        """Render display frame for external use (called from processing thread)."""
        if not frames:
            return None
        
        # Fixed output size to prevent window resizing/flashing
        # Layout: 2 columns, up to 2 rows (for up to 4 cameras)
        CELL_W, CELL_H = 640, 360
        OUTPUT_W, OUTPUT_H = CELL_W * 2, CELL_H * 2  # 1280 x 720
        
        # Create fixed-size black canvas
        combined = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)
        
        # Process each camera and place in grid
        sorted_cam_ids = sorted(frames.keys())
        
        for idx, cam_id in enumerate(sorted_cam_ids):
            if idx >= 4:  # Max 4 cameras in 2x2 grid
                break
                
            frame = frames[cam_id].copy()
            
            if cam_id in detections:
                det = detections[cam_id]
                results = det.get('raw_results')
                
                if results and results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                if self.enable_hands and results:
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS
                        )
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, results.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS
                        )
            
            # Add camera label and FPS
            cv2.putText(frame, f"Cam {cam_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show reprojection error (don't lock here, we're in processing thread)
            error = self.latest_reprojection_error
            error_color = (0, 255, 0) if error < 10 else (0, 255, 255) if error < 20 else (0, 0, 255)
            cv2.putText(frame, f"Reproj: {error:.1f}px", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, error_color, 1)
            
            # Show sync status
            sync_ms = self._last_time_spread * 1000
            sync_color = (0, 255, 0) if sync_ms < 30 else (0, 255, 255) if sync_ms < 50 else (0, 0, 255)
            cv2.putText(frame, f"Sync: {sync_ms:.0f}ms", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, sync_color, 1)
            
            # Resize to fit cell
            resized = cv2.resize(frame, (CELL_W, CELL_H))
            
            # Place in grid (row 0: cameras 0,1; row 1: cameras 2,3)
            row = idx // 2
            col = idx % 2
            y_start = row * CELL_H
            x_start = col * CELL_W
            combined[y_start:y_start+CELL_H, x_start:x_start+CELL_W] = resized
        
        return combined
    
    def get_3d_skeleton(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the current 3D skeleton.
        
        Returns:
            (skeleton_3d, reprojection_error): (33, 3) skeleton in meters, avg error in pixels
        """
        with self.data_lock:
            skeleton = self.latest_3d_skeleton.copy() if self.latest_3d_skeleton is not None else None
            error = self.latest_reprojection_error
        return skeleton, error
    
    def get_3d_hands(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the current 3D hand landmarks.
        
        Returns:
            (left_hand, right_hand): Each (21, 3) in meters or None
        """
        with self.data_lock:
            left = self.latest_hands_3d['left'].copy() if self.latest_hands_3d['left'] is not None else None
            right = self.latest_hands_3d['right'].copy() if self.latest_hands_3d['right'] is not None else None
        return left, right
    
    def get_latest_frames(self) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
        """
        Get the latest camera frames and detections for external display.
        
        Returns:
            (frames, detections): Dictionaries keyed by camera_id
        """
        with self.data_lock:
            # Data is already copied in the processing loop, just copy the references
            frames = dict(self.latest_frames) if self.latest_frames else {}
            detections = dict(self.latest_detections) if self.latest_detections else {}
        return frames, detections
    
    def get_display_frame(self) -> Optional[np.ndarray]:
        """
        Get a combined visualization frame for external display.
        
        The frame is pre-rendered by the processing thread to avoid threading
        issues with MediaPipe objects.
        
        Returns:
            Combined BGR image with all camera views and pose overlays, or None
        """
        with self.data_lock:
            if self._latest_display_frame is not None:
                return self._latest_display_frame.copy()
            return None
    
    def _fill_missing_landmarks(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Fill missing (NaN) landmarks with interpolated values.
        
        Uses parent/child joint relationships to estimate missing positions.
        """
        skeleton = skeleton.copy()
        
        # MediaPipe landmark relationships (parent -> child for interpolation)
        # If a landmark is NaN, estimate from nearby landmarks
        interpolation_rules = {
            # Face landmarks - use nose or nearby
            0: [11, 12],  # nose from shoulders
            1: [0],       # left eye from nose
            2: [0],       # right eye from nose
            3: [1],       # left ear from left eye
            4: [2],       # right ear from right eye
            5: [0],       # left mouth from nose
            6: [0],       # right mouth from nose
            7: [3],       # left ear from ear
            8: [4],       # right ear
            9: [5],       # mouth
            10: [6],      # mouth
            
            # Body landmarks - interpolate from parent joints
            13: [11],     # left elbow from shoulder
            14: [12],     # right elbow from shoulder
            # Wrists handled specially below (need extrapolation, not interpolation)
            # 15: left wrist - extrapolate from shoulder->elbow
            # 16: right wrist - extrapolate from shoulder->elbow
            17: [15],     # left pinky from wrist
            18: [16],     # right pinky from wrist
            19: [15],     # left index from wrist
            20: [16],     # right index from wrist
            21: [15],     # left thumb from wrist
            22: [16],     # right thumb from wrist
            25: [23],     # left knee from hip
            26: [24],     # right knee from hip
            27: [25, 23], # left ankle from knee, hip
            28: [26, 24], # right ankle from knee, hip
            29: [27],     # left heel from ankle
            30: [28],     # right heel
            31: [27],     # left foot index
            32: [28],     # right foot index
        }
        
        for landmark_idx, parent_indices in interpolation_rules.items():
            if landmark_idx >= len(skeleton):
                continue
            if np.isnan(skeleton[landmark_idx]).any():
                # Try to interpolate from parent landmarks
                valid_parents = []
                for parent_idx in parent_indices:
                    if parent_idx < len(skeleton) and np.isfinite(skeleton[parent_idx]).all():
                        valid_parents.append(skeleton[parent_idx])
                
                if valid_parents:
                    # Average of valid parents
                    skeleton[landmark_idx] = np.mean(valid_parents, axis=0)
        
        # Special handling for wrists: extrapolate from shoulder->elbow direction
        # Left wrist (15) from left shoulder (11) and left elbow (13)
        if np.isnan(skeleton[15]).any():
            if np.isfinite(skeleton[11]).all() and np.isfinite(skeleton[13]).all():
                # Extrapolate: wrist = elbow + (elbow - shoulder) * 0.8
                arm_direction = skeleton[13] - skeleton[11]
                skeleton[15] = skeleton[13] + arm_direction * 0.8
        
        # Right wrist (16) from right shoulder (12) and right elbow (14)
        if np.isnan(skeleton[16]).any():
            if np.isfinite(skeleton[12]).all() and np.isfinite(skeleton[14]).all():
                arm_direction = skeleton[14] - skeleton[12]
                skeleton[16] = skeleton[14] + arm_direction * 0.8
        
        return skeleton
    
    def get_current_frame(self) -> Tuple:
        """
        Get current data in FreeMoCapStreamer-compatible format.
        
        Returns:
            (smplx_data, left_hand_data, right_hand_data, None, None)
        """
        skeleton_3d, reproj_error = self.get_3d_skeleton()
        left_hand, right_hand = self.get_3d_hands()
        
        if skeleton_3d is None:
            return (None, None, None, None, None)
        
        # CORE landmarks - absolutely required (shoulders + hips define the torso)
        core_landmarks = [11, 12, 23, 24]  # left_shoulder, right_shoulder, left_hip, right_hip
        core_points = skeleton_3d[core_landmarks]
        
        # Only require core landmarks to be valid
        if not np.isfinite(core_points).all():
            valid_count = np.isfinite(core_points).all(axis=1).sum()
            logger.debug(f"Core landmarks missing: {4 - valid_count}/4 invalid")
            return (None, None, None, None, None)
        
        # Fill NaN values with reasonable defaults (interpolate from nearby joints)
        skeleton_3d = self._fill_missing_landmarks(skeleton_3d)
        
        # Check for degenerate geometry (points too close together)
        left_shoulder = skeleton_3d[11]
        right_shoulder = skeleton_3d[12]
        left_hip = skeleton_3d[23]
        right_hip = skeleton_3d[24]
        
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        hip_width = np.linalg.norm(right_hip - left_hip)
        torso_height = np.linalg.norm((left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2)
        
        # Sanity checks (in meters) - use wider ranges
        if shoulder_width < 0.05 or shoulder_width > 1.0:
            logger.debug(f"Bad shoulder width: {shoulder_width:.3f}m")
            return (None, None, None, None, None)
        if hip_width < 0.05 or hip_width > 0.8:
            logger.debug(f"Bad hip width: {hip_width:.3f}m")
            return (None, None, None, None, None)
        if torso_height < 0.1 or torso_height > 1.2:
            logger.debug(f"Bad torso height: {torso_height:.3f}m")
            return (None, None, None, None, None)
        
        # Skip frames with very high reprojection error
        if reproj_error > 100:
            logger.debug(f"High reprojection error: {reproj_error:.1f}px")
            return (None, None, None, None, None)
        
        # Return skeleton and hands for direct G1 conversion
        # The caller (multicam_to_twist2.py) will handle the conversion
        return (skeleton_3d, left_hand, right_hand, None, None)
    
    def get_mimic_obs(self, use_gmr: bool = None) -> Optional[np.ndarray]:
        """
        Get current skeleton as mimic_obs directly (bypassing SMPL-X).
        
        Args:
            use_gmr: If True, use GMR IK-based retargeting. If False, use direct mapping.
                     If None, uses the instance default (self.use_gmr).
        
        Returns:
            mimic_obs: (35,) array for robot control, or None if invalid
        """
        # Use instance default if not specified
        if use_gmr is None:
            use_gmr = self.use_gmr
        
        skeleton_3d, reproj_error = self.get_3d_skeleton()
        
        if skeleton_3d is None:
            return None
        
        # CORE landmarks required
        core_landmarks = [11, 12, 23, 24]
        core_points = skeleton_3d[core_landmarks]
        if not np.isfinite(core_points).all():
            return None
        
        # Fill missing landmarks
        skeleton_3d = self._fill_missing_landmarks(skeleton_3d)
        
        # Get hand landmarks for wrist control
        left_hand, right_hand = self.get_3d_hands()
        
        # Choose converter based on mode
        if use_gmr:
            # GMR IK-based retargeting
            if not hasattr(self, '_gmr_converter'):
                from mediapipe_to_g1_gmr import MediaPipeToG1GMR
                self._gmr_converter = MediaPipeToG1GMR(
                    tpose_calibration_path='../calibration/tpose_calibration.json',
                    human_height=1.7,
                    robot_height=0.8,
                    arms_only=True,
                    use_gmr_ik=True
                )
            converter = self._gmr_converter
        else:
            # Direct mapping (original method)
            if not hasattr(self, '_direct_converter'):
                from mediapipe_to_g1_direct import MediaPipeToG1Direct
                self._direct_converter = MediaPipeToG1Direct(robot_height=0.8)
            converter = self._direct_converter
        
        # Convert to mimic_obs
        try:
            mimic_obs = converter.skeleton_to_mimic_obs(
                skeleton_3d, 
                left_hand=left_hand, 
                right_hand=right_hand
            )
            return mimic_obs
        except Exception as e:
            logger.debug(f"G1 conversion failed: {e}")
            return None
    
    def stop(self):
        """Stop streaming."""
        logger.info("Stopping multi-camera pose streamer...")
        self.is_running = False
        
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        self.capture.stop()
        
        for detector in self.detectors.values():
            detector.close()
        
        if self.enable_display:
            cv2.destroyAllWindows()
        
        logger.info("Multi-camera pose streamer stopped")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-camera 3D pose streaming")
    parser.add_argument("--cameras", type=int, nargs='+', default=None,
                       help="Camera IDs to use (default: from camera_config.yaml)")
    parser.add_argument("--calibration", type=str, default=None,
                       help="Path to calibration.toml")
    parser.add_argument("--display", action="store_true",
                       help="Show visualization window")
    parser.add_argument("--no-hands", action="store_true",
                       help="Disable hand tracking")
    args = parser.parse_args()
    
    # Find paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    calibration_dir = os.path.join(project_dir, "calibration")
    
    # Load camera config if cameras not specified
    if args.cameras is None:
        config_path = os.path.join(calibration_dir, "camera_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        camera_ids = config['camera_ids']
    else:
        camera_ids = args.cameras
    
    # Calibration path
    if args.calibration:
        calibration_path = args.calibration
    else:
        calibration_path = os.path.join(calibration_dir, "calibration.toml")
    
    print("=" * 60)
    print("  Multi-Camera 3D Pose Streaming")
    print("=" * 60)
    print(f"\nCameras: {camera_ids}")
    print(f"Calibration: {calibration_path}")
    print(f"Display: {args.display}")
    print(f"Hands: {not args.no_hands}")
    print("\nPress 'q' to quit")
    print("-" * 60)
    
    # Create streamer
    streamer = MultiCamPoseStreamer(
        camera_ids=camera_ids,
        calibration_file=calibration_path,
        enable_hands=not args.no_hands,
        enable_display=args.display
    )
    
    try:
        streamer.start()
        
        # Main loop - print skeleton info
        while streamer.is_running:
            skeleton, error = streamer.get_3d_skeleton()
            
            if skeleton is not None:
                # Count valid (non-NaN) landmarks
                valid_count = np.sum(~np.isnan(skeleton[:, 0]))
                
                # Get pelvis position (landmark 23+24 / 2)
                if not np.isnan(skeleton[23, 0]) and not np.isnan(skeleton[24, 0]):
                    pelvis = (skeleton[23] + skeleton[24]) / 2
                    print(f"\rPelvis: [{pelvis[0]:.2f}, {pelvis[1]:.2f}, {pelvis[2]:.2f}]m | "
                          f"Valid: {valid_count}/33 | Reproj: {error:.1f}px | "
                          f"FPS: {streamer.current_fps:.1f}", end="")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        streamer.stop()
    
    print("\nDone!")


if __name__ == "__main__":
    main()

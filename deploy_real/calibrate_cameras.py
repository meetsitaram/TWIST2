#!/usr/bin/env python3
"""
Multi-Camera Calibration Tool

This script:
1. Records synchronized video from all cameras while you wave the Charuco board
2. Detects Charuco corners in each frame
3. Saves detected corners to disk (so you don't have to recapture)
4. Calibrates intrinsic parameters for each camera
5. Calibrates extrinsic parameters (relative positions between cameras)
6. Saves calibration to a TOML file for use in triangulation

Usage:
    conda activate gmr
    python calibrate_cameras.py [--recapture] [--target-frames 100]

Options:
    --recapture      Force new capture even if saved data exists
    --target-frames  Auto-stop after this many DIVERSE frames per camera (default: 60)

Prerequisites:
    - Print and mount the Charuco board (generate_charuco_board.py)
    - Measure the actual printed square size and update camera_config.yaml
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os
import sys
import signal
import yaml
import time
import toml
import pickle
import argparse
from threading import Thread, Lock
from datetime import datetime
from collections import defaultdict


def signal_handler(sig, frame):
    """Handle Ctrl+C by forcing immediate exit."""
    print("\n\nInterrupted by user. Exiting...")
    sys.exit(1)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)


def print_progress(current, total, prefix="", suffix="", bar_length=30):
    """Print a progress bar that updates in place."""
    percent = current / total if total > 0 else 0
    filled = int(bar_length * percent)
    bar = "█" * filled + "░" * (bar_length - filled)
    sys.stdout.write(f"\r    {prefix} [{bar}] {current}/{total} {suffix}")
    sys.stdout.flush()
    if current >= total:
        print()  # New line when complete


class MultiCameraRecorder:
    """Record synchronized video from multiple cameras."""
    
    def __init__(self, camera_ids, resolution=(1280, 720), fps=30):
        self.camera_ids = camera_ids
        self.resolution = resolution
        self.fps = fps
        self.captures = []
        self.frames = {}
        self.frame_lock = Lock()
        self.running = False
        self.threads = []
        
    def start(self):
        """Start all camera capture threads."""
        self.running = True
        
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            
            # Set MJPEG mode for better FPS
            # MJPEG uses compression, reducing USB bandwidth
            # Without this: ~10 FPS (raw YUYV), With this: ~26 FPS
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not cap.isOpened():
                print(f"Warning: Could not open camera {cam_id}")
                continue
                
            self.captures.append((cam_id, cap))
            
            # Start capture thread
            t = Thread(target=self._capture_loop, args=(cam_id, cap))
            t.daemon = True
            t.start()
            self.threads.append(t)
        
        print(f"Started {len(self.captures)} cameras")
        
    def _capture_loop(self, cam_id, cap):
        """Capture loop for a single camera."""
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.frame_lock:
                    self.frames[cam_id] = frame.copy()
            time.sleep(0.001)
    
    def get_frames(self):
        """Get current frames from all cameras."""
        with self.frame_lock:
            return {k: v.copy() for k, v in self.frames.items()}
    
    def stop(self):
        """Stop all captures."""
        self.running = False
        for t in self.threads:
            t.join(timeout=1)
        for cam_id, cap in self.captures:
            cap.release()


def is_pose_diverse(new_corners, existing_corners_list, min_distance=30):
    """Check if new corners represent a diverse pose from existing ones.
    
    Compares the centroid and spread of corners to existing frames.
    Returns True if the new frame is sufficiently different.
    
    Args:
        new_corners: New detected corners (Nx1x2 array)
        existing_corners_list: List of previously accepted corners
        min_distance: Minimum pixel distance to consider diverse
    """
    if len(existing_corners_list) == 0:
        return True
    
    # Calculate centroid and spread of new detection
    new_pts = new_corners.reshape(-1, 2)
    new_centroid = np.mean(new_pts, axis=0)
    new_spread = np.std(new_pts)  # Indicates board size/distance
    
    # Compare against recent frames (check last 50 to save time)
    for existing_corners in existing_corners_list[-50:]:
        existing_pts = existing_corners.reshape(-1, 2)
        existing_centroid = np.mean(existing_pts, axis=0)
        existing_spread = np.std(existing_pts)
        
        # Check centroid distance (board position in frame)
        centroid_dist = np.linalg.norm(new_centroid - existing_centroid)
        
        # Check spread difference (board size/tilt)
        spread_diff = abs(new_spread - existing_spread)
        
        # If both are too similar, this is not diverse
        if centroid_dist < min_distance and spread_diff < min_distance / 2:
            return False
    
    return True


class CharucoDetector:
    """Detect Charuco board in images."""
    
    def __init__(self, squares_x, squares_y, square_size, marker_size, dictionary_name):
        # Get ArUco dictionary
        dict_map = {
            "DICT_4X4_50": aruco.DICT_4X4_50,
            "DICT_4X4_100": aruco.DICT_4X4_100,
            "DICT_5X5_50": aruco.DICT_5X5_50,
            "DICT_5X5_100": aruco.DICT_5X5_100,
        }
        
        dictionary = aruco.getPredefinedDictionary(dict_map.get(dictionary_name, aruco.DICT_4X4_50))
        
        self.board = aruco.CharucoBoard(
            (squares_x, squares_y),
            square_size / 1000.0,  # Convert mm to meters
            marker_size / 1000.0,
            dictionary
        )
        
        self.detector_params = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(dictionary, self.detector_params)
        
    def detect(self, frame):
        """Detect Charuco corners in frame.
        
        Returns:
            corners: Nx1x2 array of corner positions, or None
            ids: Nx1 array of corner IDs, or None
            frame_with_markers: Frame with detected markers drawn
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        frame_vis = frame.copy()
        
        if ids is not None and len(ids) > 0:
            # Draw detected markers
            aruco.drawDetectedMarkers(frame_vis, corners, ids)
            
            # Interpolate Charuco corners
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )
            
            if ret > 4:  # Need at least 4 corners for calibration
                # Draw Charuco corners
                aruco.drawDetectedCornersCharuco(frame_vis, charuco_corners, charuco_ids)
                return charuco_corners, charuco_ids, frame_vis
        
        return None, None, frame_vis


def save_captured_data(data, output_path):
    """Save captured corners and IDs to disk."""
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Captured data saved to: {output_path}")


def load_captured_data(input_path):
    """Load captured corners and IDs from disk."""
    if os.path.exists(input_path):
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded captured data from: {input_path}")
        return data
    return None


def analyze_pose_diversity(corners_list):
    """
    Analyze the diversity of board poses in captured frames.
    Returns diversity metrics and a filtered list of diverse frame indices.
    """
    if len(corners_list) < 5:
        return None
    
    # Compute centroid and spread for each frame
    centroids = []
    spreads = []
    for corners in corners_list:
        if corners is not None and len(corners) >= 4:
            pts = corners.reshape(-1, 2)
            centroid = np.mean(pts, axis=0)
            spread = np.std(pts, axis=0).mean()  # Average spread in x and y
            centroids.append(centroid)
            spreads.append(spread)
        else:
            centroids.append(None)
            spreads.append(None)
    
    # Filter valid entries
    valid_indices = [i for i, c in enumerate(centroids) if c is not None]
    if len(valid_indices) < 5:
        return None
    
    valid_centroids = np.array([centroids[i] for i in valid_indices])
    valid_spreads = np.array([spreads[i] for i in valid_indices])
    
    # Compute diversity metrics
    centroid_std = np.std(valid_centroids, axis=0)  # [std_x, std_y]
    spread_std = np.std(valid_spreads)
    spread_range = valid_spreads.max() - valid_spreads.min()
    
    return {
        'centroid_std_x': centroid_std[0],
        'centroid_std_y': centroid_std[1],
        'spread_std': spread_std,
        'spread_range': spread_range,
        'spread_min': valid_spreads.min(),
        'spread_max': valid_spreads.max(),
        'n_frames': len(valid_indices),
        'valid_indices': valid_indices,
        'centroids': valid_centroids,
        'spreads': valid_spreads
    }


def select_diverse_frames(corners_list, ids_list, max_frames=50):
    """
    Select a diverse subset of frames using greedy farthest-point sampling.
    This helps ensure we have frames from different positions and distances.
    """
    diversity = analyze_pose_diversity(corners_list)
    if diversity is None or diversity['n_frames'] < 10:
        return corners_list, ids_list
    
    valid_indices = diversity['valid_indices']
    centroids = diversity['centroids']
    spreads = diversity['spreads']
    
    # Combine centroid and spread into feature vector
    features = np.column_stack([
        centroids / 100,  # Normalize position
        spreads.reshape(-1, 1) / 50  # Normalize spread
    ])
    
    # Greedy farthest-point sampling
    n_select = min(max_frames, len(valid_indices))
    selected = [0]  # Start with first frame
    
    for _ in range(n_select - 1):
        # Find point farthest from all selected points
        max_min_dist = -1
        best_idx = -1
        
        for i, feat in enumerate(features):
            if i in selected:
                continue
            # Min distance to any selected point
            min_dist = min(np.linalg.norm(feat - features[j]) for j in selected)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i
        
        if best_idx >= 0:
            selected.append(best_idx)
    
    # Map back to original indices
    selected_original = [valid_indices[i] for i in selected]
    
    return (
        [corners_list[i] for i in selected_original],
        [ids_list[i] for i in selected_original]
    )


def calibrate_camera_intrinsics(all_corners, all_ids, image_size, board, cam_id=None):
    """Calibrate intrinsic parameters for a single camera with robust fallbacks."""
    
    total_frames = len(all_corners)
    if total_frames < 10:
        return None, None, None
    
    # Analyze pose diversity first
    diversity = analyze_pose_diversity(all_corners)
    if diversity:
        print(f"    Pose diversity analysis:")
        print(f"      Position variation: X={diversity['centroid_std_x']:.1f}px, Y={diversity['centroid_std_y']:.1f}px")
        print(f"      Distance variation: {diversity['spread_min']:.1f} - {diversity['spread_max']:.1f}px (range={diversity['spread_range']:.1f})")
        
        # Check if diversity is sufficient
        needs_more_position = diversity['centroid_std_x'] < 100 or diversity['centroid_std_y'] < 100
        needs_more_distance = diversity['spread_range'] < 50
        
        if needs_more_position:
            print(f"      ⚠ Low position variation - move board to different parts of frame")
        if needs_more_distance:
            print(f"      ⚠ Low distance variation - move board closer and farther from camera")
    
    # Subsample if too many frames (speeds up calibration)
    max_frames = 200
    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        all_corners = [all_corners[i] for i in indices]
        all_ids = [all_ids[i] for i in indices]
        print(f"    Subsampled {total_frames} -> {max_frames} frames")
    
    # Show progress for validation phase
    print(f"    Validating {len(all_corners)} frames...")
    valid_corners = []
    valid_ids = []
    for i, (corners, ids) in enumerate(zip(all_corners, all_ids)):
        if i % 20 == 0 or i == len(all_corners) - 1:
            print_progress(i + 1, len(all_corners), "Validating")
        if corners is not None and ids is not None and len(corners) >= 4:
            valid_corners.append(corners)
            valid_ids.append(ids)
    
    if len(valid_corners) < 10:
        print(f"    Only {len(valid_corners)} valid frames (need 10+)")
        return None, None, None
    
    print(f"    Running OpenCV calibration on {len(valid_corners)} frames...")
    start_time = time.time()
    
    try:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            valid_corners, valid_ids, board,
            image_size, None, None
        )
        
        elapsed = time.time() - start_time
        print(f"    Calibration took {elapsed:.1f}s")
        
        if ret:
            return camera_matrix, dist_coeffs, ret
        return None, None, None
        
    except cv2.error as e:
        print(f"    OpenCV error: {e}")
        print(f"    Trying with diverse frame subset...")
        
        # Try with a carefully selected diverse subset
        diverse_corners, diverse_ids = select_diverse_frames(valid_corners, valid_ids, max_frames=40)
        print(f"    Selected {len(diverse_corners)} diverse frames")
        
        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                diverse_corners, diverse_ids, board,
                image_size, None, None
            )
            
            if ret:
                print(f"    ✓ Succeeded with diverse subset! Error = {ret:.3f}px")
                return camera_matrix, dist_coeffs, ret
        except cv2.error:
            pass
        
        # Last resort: try with initial camera matrix guess
        print(f"    Trying with initial camera matrix guess...")
        try:
            # Create reasonable initial guess
            fx = fy = max(image_size) * 1.2  # Approximate focal length
            cx, cy = image_size[0] / 2, image_size[1] / 2
            camera_matrix_init = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs_init = np.zeros(5, dtype=np.float64)
            
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                diverse_corners, diverse_ids, board,
                image_size, camera_matrix_init, dist_coeffs_init,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS
            )
            
            if ret:
                print(f"    ✓ Succeeded with initial guess! Error = {ret:.3f}px")
                return camera_matrix, dist_coeffs, ret
        except cv2.error as e2:
            print(f"    Still failed: {e2}")
        
        print(f"    ✗ All attempts failed. Recommendations:")
        print(f"      1. Tilt the board at 30-45° angles (not just flat)")
        print(f"      2. Move the board to all corners of the frame")
        print(f"      3. Vary distance: some poses close, some far")
        return None, None, None


def match_frames_by_timestamp(timestamps1, timestamps2, max_diff_ms=100):
    """
    Match frames from two cameras by timestamp.
    
    Returns list of (idx1, idx2) pairs where timestamps are within max_diff_ms.
    """
    matched_pairs = []
    
    for i, t1 in enumerate(timestamps1):
        best_j = None
        best_diff = float('inf')
        
        for j, t2 in enumerate(timestamps2):
            diff = abs(t1 - t2) * 1000  # Convert to ms
            if diff < best_diff:
                best_diff = diff
                best_j = j
        
        if best_j is not None and best_diff <= max_diff_ms:
            matched_pairs.append((i, best_j, best_diff))
    
    return matched_pairs


def calibrate_stereo_pnp(corners1, corners2, ids1, ids2,
                         camera_matrix1, dist_coeffs1,
                         camera_matrix2, dist_coeffs2,
                         image_size, board,
                         timestamps1=None, timestamps2=None):
    """
    Alternative stereo calibration using PnP.
    
    Instead of OpenCV's stereoCalibrate, we:
    1. For each frame, estimate board pose relative to each camera using solvePnP
    2. Compute the relative transform between cameras
    3. Average/median filter the results for robustness
    
    If timestamps are provided, uses timestamp-based frame matching.
    """
    board_corners = board.getChessboardCorners()
    
    relative_Rs = []
    relative_Ts = []
    
    # For diagnostics
    all_img_pts1 = []
    all_img_pts2 = []
    total_correspondences = 0
    
    # Match frames by timestamp if available, otherwise by index
    if timestamps1 is not None and timestamps2 is not None and len(timestamps1) > 0 and len(timestamps2) > 0:
        matched_pairs = match_frames_by_timestamp(timestamps1, timestamps2, max_diff_ms=100)
        print(f"    Timestamp matching: {len(matched_pairs)} pairs within 100ms")
        frame_pairs = [(i, j) for i, j, _ in matched_pairs]
    else:
        min_len = min(len(corners1), len(corners2))
        frame_pairs = [(i, i) for i in range(min_len)]
    
    for idx1, idx2 in frame_pairs:
        c1, id1 = corners1[idx1], ids1[idx1]
        c2, id2 = corners2[idx2], ids2[idx2]
        
        if c1 is None or c2 is None:
            continue
        
        # Find common IDs
        common_ids = np.intersect1d(id1.flatten(), id2.flatten())
        if len(common_ids) < 6:
            continue
        
        # Get corresponding points
        obj_pts = []
        img_pts1 = []
        img_pts2 = []
        
        for cid in common_ids:
            idx1 = np.where(id1.flatten() == cid)[0]
            idx2 = np.where(id2.flatten() == cid)[0]
            if len(idx1) > 0 and len(idx2) > 0:
                obj_pts.append(board_corners[cid].flatten())
                img_pts1.append(c1[idx1[0]].flatten())
                img_pts2.append(c2[idx2[0]].flatten())
        
        if len(obj_pts) < 6:
            continue
        
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts1 = np.array(img_pts1, dtype=np.float32)
        img_pts2 = np.array(img_pts2, dtype=np.float32)
        
        # Collect for diagnostics
        all_img_pts1.extend(img_pts1.tolist())
        all_img_pts2.extend(img_pts2.tolist())
        total_correspondences += len(img_pts1)
        
        # Solve PnP for each camera
        success1, rvec1, tvec1 = cv2.solvePnP(obj_pts, img_pts1, camera_matrix1, dist_coeffs1)
        success2, rvec2, tvec2 = cv2.solvePnP(obj_pts, img_pts2, camera_matrix2, dist_coeffs2)
        
        if not success1 or not success2:
            continue
        
        # Convert to rotation matrices
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        
        # Relative transform: T_cam2_from_cam1 = T_cam2_from_board @ T_board_from_cam1
        # T_cam_from_board = [R|t], T_board_from_cam = [R^T | -R^T @ t]
        R_board_from_cam1 = R1.T
        t_board_from_cam1 = -R1.T @ tvec1
        
        # T_cam2_from_cam1 = T_cam2_from_board @ T_board_from_cam1
        R_rel = R2 @ R_board_from_cam1
        t_rel = R2 @ t_board_from_cam1 + tvec2
        
        relative_Rs.append(R_rel)
        relative_Ts.append(t_rel)
    
    if len(relative_Rs) < 5:
        return None, None, float('inf'), 0, 0
    
    # Print diagnostics
    if len(all_img_pts1) > 0:
        pts1 = np.array(all_img_pts1)
        pts2 = np.array(all_img_pts2)
        print(f"\n    === DIAGNOSTICS ===")
        print(f"    Total point correspondences: {total_correspondences}")
        print(f"    Cam1 points range: X[{pts1[:,0].min():.0f}-{pts1[:,0].max():.0f}], Y[{pts1[:,1].min():.0f}-{pts1[:,1].max():.0f}]")
        print(f"    Cam2 points range: X[{pts2[:,0].min():.0f}-{pts2[:,0].max():.0f}], Y[{pts2[:,1].min():.0f}-{pts2[:,1].max():.0f}]")
        x_diff = np.abs(pts1[:,0] - pts2[:,0])
        y_diff = np.abs(pts1[:,1] - pts2[:,1])
        print(f"    Point disparity (same ID, different cams):")
        print(f"      Mean X diff: {x_diff.mean():.1f}px, Y diff: {y_diff.mean():.1f}px")
        print(f"      Max X diff: {x_diff.max():.1f}px, Y diff: {y_diff.max():.1f}px")
        print(f"    === END DIAGNOSTICS ===")
    
    # Use median for robustness
    R_avg = np.median(np.array(relative_Rs), axis=0)
    T_avg = np.median(np.array(relative_Ts), axis=0)
    
    # Re-orthogonalize R using SVD
    U, _, Vt = np.linalg.svd(R_avg)
    R_final = U @ Vt
    
    # Compute reprojection error to validate
    errors = []
    for i, (R_rel, t_rel) in enumerate(zip(relative_Rs, relative_Ts)):
        # Compare to average
        err_R = np.linalg.norm(R_rel - R_final, 'fro')
        err_t = np.linalg.norm(t_rel - T_avg)
        errors.append(err_t)
    
    median_error = np.median(errors)
    baseline = np.linalg.norm(T_avg)
    
    return R_final, T_avg, median_error, baseline, len(relative_Rs)


def calibrate_stereo(corners1, corners2, ids1, ids2, 
                     camera_matrix1, dist_coeffs1,
                     camera_matrix2, dist_coeffs2,
                     image_size, board,
                     timestamps1=None, timestamps2=None):
    """Calibrate extrinsic parameters between two cameras with timestamp matching."""
    
    # First try PnP-based approach (more robust)
    result = calibrate_stereo_pnp(corners1, corners2, ids1, ids2,
                                   camera_matrix1, dist_coeffs1,
                                   camera_matrix2, dist_coeffs2,
                                   image_size, board,
                                   timestamps1, timestamps2)
    
    if result[0] is not None:
        R_pnp, T_pnp, error_pnp, baseline_pnp, n_frames = result
        print(f"\n    === PnP-based calibration ===")
        print(f"    Used {n_frames} frame pairs")
        print(f"    Baseline: {baseline_pnp*1000:.1f}mm")
        print(f"    Translation: [{T_pnp[0,0]*1000:.1f}, {T_pnp[1,0]*1000:.1f}, {T_pnp[2,0]*1000:.1f}]mm")
        print(f"    Consistency error: {error_pnp*1000:.1f}mm")
        
        # If baseline seems reasonable (30cm to 3m for ~1m apart cameras), accept it
        # Be more lenient with error since cameras aren't hardware-synced
        if 0.3 < baseline_pnp < 3.0 and error_pnp < 1.0:
            print(f"    ✓ PnP calibration accepted (baseline={baseline_pnp:.2f}m)")
            if error_pnp > 0.3:
                print(f"    Note: High consistency error ({error_pnp*1000:.0f}mm) - cameras may not be perfectly synced")
            return R_pnp, T_pnp
        else:
            print(f"    PnP result seems off (baseline={baseline_pnp:.2f}m, error={error_pnp:.3f}m)")
            print(f"    Trying OpenCV stereoCalibrate...")
    
    # Fall back to OpenCV stereoCalibrate
    # Find common frames where both cameras detected the board
    common_frames = []
    
    # Use timestamp matching if available
    if timestamps1 is not None and timestamps2 is not None and len(timestamps1) > 0 and len(timestamps2) > 0:
        matched_pairs = match_frames_by_timestamp(timestamps1, timestamps2, max_diff_ms=100)
        print(f"    Using timestamp matching: {len(matched_pairs)} pairs within 100ms")
        
        for idx1, idx2, _ in matched_pairs:
            c1, id1 = corners1[idx1], ids1[idx1]
            c2, id2 = corners2[idx2], ids2[idx2]
            
            if c1 is not None and c2 is not None:
                common_ids = np.intersect1d(id1.flatten(), id2.flatten())
                if len(common_ids) >= 6:
                    common_frames.append((c1, c2, id1, id2, common_ids))
    else:
        # Fallback to index-based matching
        min_len = min(len(corners1), len(corners2))
        print(f"    Searching {min_len} frames for common detections...")
        
        for i in range(min_len):
            if i % 50 == 0 or i == min_len - 1:
                print_progress(i + 1, min_len, "Scanning")
            
            c1, id1 = corners1[i], ids1[i]
            c2, id2 = corners2[i], ids2[i]
            
            if c1 is not None and c2 is not None:
                common_ids = np.intersect1d(id1.flatten(), id2.flatten())
                if len(common_ids) >= 6:
                    common_frames.append((c1, c2, id1, id2, common_ids))
    
    print(f"    Found {len(common_frames)} frames with common detections")
    
    if len(common_frames) < 10:
        return None, None
    
    # Subsample if too many
    max_frames = 300
    if len(common_frames) > max_frames:
        indices = np.linspace(0, len(common_frames) - 1, max_frames, dtype=int)
        common_frames = [common_frames[i] for i in indices]
        print(f"    Subsampled to {max_frames} frame pairs")
        print(f"    Subsampled to {max_frames} frame pairs")
    
    # Prepare object points and image points
    obj_points = []
    img_points1 = []
    img_points2 = []
    
    # Get all charuco corner positions
    board_corners = board.getChessboardCorners()
    
    print(f"    Extracting point correspondences...")
    for i, (c1, c2, id1, id2, common_ids) in enumerate(common_frames):
        if i % 20 == 0 or i == len(common_frames) - 1:
            print_progress(i + 1, len(common_frames), "Processing")
        
        pts1 = []
        pts2 = []
        obj_pts = []
        
        for cid in common_ids:
            idx1 = np.where(id1.flatten() == cid)[0]
            idx2 = np.where(id2.flatten() == cid)[0]
            if len(idx1) > 0 and len(idx2) > 0:
                pts1.append(c1[idx1[0]].flatten())
                pts2.append(c2[idx2[0]].flatten())
                obj_pts.append(board_corners[cid].flatten())
        
        if len(pts1) >= 6:  # OpenCV stereoCalibrate requires at least 6 points
            obj_points.append(np.array(obj_pts, dtype=np.float32))
            img_points1.append(np.array(pts1, dtype=np.float32))
            img_points2.append(np.array(pts2, dtype=np.float32))
    
    if len(obj_points) < 10:
        return None, None
    
    print(f"    Running stereo calibration on {len(obj_points)} frame pairs...")
    start_time = time.time()
    
    # Analyze point distribution first
    all_pts1 = np.vstack(img_points1)
    all_pts2 = np.vstack(img_points2)
    print(f"\n    === DIAGNOSTICS ===")
    print(f"    Total point correspondences: {len(all_pts1)}")
    print(f"    Cam1 points range: X[{all_pts1[:,0].min():.0f}-{all_pts1[:,0].max():.0f}], Y[{all_pts1[:,1].min():.0f}-{all_pts1[:,1].max():.0f}]")
    print(f"    Cam2 points range: X[{all_pts2[:,0].min():.0f}-{all_pts2[:,0].max():.0f}], Y[{all_pts2[:,1].min():.0f}-{all_pts2[:,1].max():.0f}]")
    
    # Check disparity between corresponding points
    disparities = np.abs(all_pts1 - all_pts2)
    print(f"    Point disparity (same ID, different cams):")
    print(f"      Mean X diff: {disparities[:,0].mean():.1f}px, Y diff: {disparities[:,1].mean():.1f}px")
    print(f"      Max X diff: {disparities[:,0].max():.1f}px, Y diff: {disparities[:,1].max():.1f}px")
    
    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points1, img_points2,
        camera_matrix1, dist_coeffs1,
        camera_matrix2, dist_coeffs2,
        image_size, flags=flags
    )
    
    elapsed = time.time() - start_time
    print(f"    === END DIAGNOSTICS ===\n")
    print(f"    Stereo calibration took {elapsed:.1f}s")
    print(f"    Stereo reprojection error: {ret:.3f} pixels")
    
    # Show translation (baseline) for sanity check
    baseline = np.linalg.norm(T)
    print(f"    Estimated baseline: {baseline*1000:.1f}mm")
    print(f"    Translation vector: [{T[0,0]*1000:.1f}, {T[1,0]*1000:.1f}, {T[2,0]*1000:.1f}]mm")
    
    # Accept if reprojection error is under 50 pixels
    # Without hardware-synced cameras, we may not get perfect calibration
    if ret < 50:
        if ret > 15:
            print(f"    Note: Higher than ideal error - triangulation may have some noise")
        return R, T
    
    # If failed, try with more relaxed flags
    print(f"\n    Trying with CALIB_USE_INTRINSIC_GUESS (re-optimize intrinsics)...")
    flags2 = cv2.CALIB_USE_INTRINSIC_GUESS
    
    ret2, mtx1_new, dist1_new, mtx2_new, dist2_new, R2, T2, E2, F2 = cv2.stereoCalibrate(
        obj_points, img_points1, img_points2,
        camera_matrix1.copy(), dist_coeffs1.copy(),
        camera_matrix2.copy(), dist_coeffs2.copy(),
        image_size, flags=flags2
    )
    
    print(f"    New reprojection error: {ret2:.3f} pixels")
    
    if ret2 < ret and ret2 < 20:
        print(f"    Using re-optimized calibration")
        return R2, T2
    
    return None, None


def save_calibration(calibration_data, output_path):
    """Save calibration to TOML file (anipose format)."""
    
    # Convert numpy arrays to lists for TOML
    def numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [numpy_to_list(i) for i in obj]
        return obj
    
    calibration_data = numpy_to_list(calibration_data)
    
    with open(output_path, 'w') as f:
        toml.dump(calibration_data, f)
    
    print(f"Calibration saved to: {output_path}")


def capture_calibration_data(camera_ids, resolution, detector, target_frames, calibration_dir, slow_mode=False, sync_mode=False):
    """Capture calibration data from cameras.
    
    Args:
        slow_mode: If True, wait 0.5s between saves to allow better coverage
        sync_mode: If True, only capture when board is stable and visible to multiple cameras
    """
    
    # Start cameras
    recorder = MultiCameraRecorder(camera_ids, resolution)
    recorder.start()
    
    # Timing for slow mode
    min_interval = 0.5 if slow_mode else 0.0
    last_save_time = {cam_id: 0 for cam_id in camera_ids}
    
    # Sync mode tracking - detect when board is stationary
    prev_detections = {}  # Previous frame detections for motion detection
    stable_count = 0  # How many frames the board has been stable
    STABLE_THRESHOLD = 3  # Frames board must be stable before capture
    MOTION_THRESHOLD = 25.0  # Pixel movement threshold (generous for camera noise)
    last_sync_capture = 0  # Time of last sync capture
    SYNC_COOLDOWN = 0.3  # Minimum seconds between sync captures
    
    # Collect calibration data with timestamps
    all_corners = {cam_id: [] for cam_id in camera_ids}
    all_ids = {cam_id: [] for cam_id in camera_ids}
    all_timestamps = {cam_id: [] for cam_id in camera_ids}  # Track capture times
    
    if sync_mode:
        print(f"\n*** SYNC MODE - For better stereo calibration ***")
        print(f"Hold the board STILL briefly - captures when stable")
        print(f"Move to new position, pause, repeat.")
        print(f"Target: {target_frames} synchronized captures (quality > quantity)")
    else:
        print(f"\nWave the Charuco board so all cameras can see it.")
        print(f"Auto-stop after {target_frames} DIVERSE frames per camera.")
        print(f"\nDiversity filter is ON - frames too similar to existing ones are rejected.")
        print(f"If you see RED + 'MOVE!', change the board's position/angle/distance!")
        if slow_mode:
            print(f"SLOW MODE: Saving 1 frame every {min_interval}s - take your time to cover the whole area!")
    print("\nPress 'q' to stop early, 'c' to continue past target.")
    print("-" * 60)
    
    # Create resizable window
    window_name = "Calibration - Press 'q' to finish"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 960)  # Default size, but user can resize/maximize
    
    auto_stop = True
    
    try:
        while True:
            frames = recorder.get_frames()
            
            if not frames:
                time.sleep(0.01)
                continue
            
            # Detect in each frame
            vis_frames = []
            current_detections = {}
            current_time = time.time()
            
            for cam_id in camera_ids:
                if cam_id not in frames:
                    continue
                    
                frame = frames[cam_id]
                corners, ids, vis_frame = detector.detect(frame)
                
                if corners is not None:
                    current_detections[cam_id] = {'corners': corners, 'ids': ids}
                
                vis_frames.append((cam_id, vis_frame))
            
            # Sync mode: check if board is stable and visible to multiple cameras
            sync_captured = False
            if sync_mode:
                cams_seeing_board = list(current_detections.keys())
                
                if len(cams_seeing_board) >= 2:
                    # Check if board is stable (not moving much)
                    is_stable = True
                    for cam_id in cams_seeing_board:
                        if cam_id in prev_detections:
                            # Compare current and previous centroid
                            curr_pts = current_detections[cam_id]['corners'].reshape(-1, 2)
                            prev_pts = prev_detections[cam_id]['corners'].reshape(-1, 2)
                            curr_centroid = np.mean(curr_pts, axis=0)
                            prev_centroid = np.mean(prev_pts, axis=0)
                            motion = np.linalg.norm(curr_centroid - prev_centroid)
                            if motion > MOTION_THRESHOLD:
                                is_stable = False
                                break
                    
                    if is_stable:
                        stable_count += 1
                    else:
                        stable_count = 0
                    
                    # Capture when stable for enough frames
                    if stable_count >= STABLE_THRESHOLD:
                        # Check cooldown to avoid rapid captures
                        time_since_last = current_time - last_sync_capture
                        if time_since_last >= SYNC_COOLDOWN:
                            # Use relaxed diversity check - just need some movement
                            any_new = False
                            for cam_id in cams_seeing_board:
                                corners = current_detections[cam_id]['corners']
                                if is_pose_diverse(corners, all_corners[cam_id], min_distance=20):
                                    any_new = True
                                    break
                            
                            # If no cameras have frames yet, always accept
                            if all(len(all_corners[cam_id]) == 0 for cam_id in cams_seeing_board):
                                any_new = True
                            
                            if any_new:
                                # Store with shared timestamp for sync mode
                                capture_timestamp = current_time
                                for cam_id in cams_seeing_board:
                                    corners = current_detections[cam_id]['corners']
                                    ids = current_detections[cam_id]['ids']
                                    all_corners[cam_id].append(corners)
                                    all_ids[cam_id].append(ids)
                                    all_timestamps[cam_id].append(capture_timestamp)
                                sync_captured = True
                                last_sync_capture = current_time
                                stable_count = 0  # Reset to require new stable position
                else:
                    stable_count = 0
                
                # Store for next frame comparison
                prev_detections = {k: v.copy() for k, v in current_detections.items()}
            
            # Non-sync mode: original per-camera capture
            frame_status = {}
            if not sync_mode:
                for cam_id in current_detections:
                    corners = current_detections[cam_id]['corners']
                    ids = current_detections[cam_id]['ids']
                    
                    time_since_last = current_time - last_save_time[cam_id]
                    if time_since_last >= min_interval:
                        if is_pose_diverse(corners, all_corners[cam_id]):
                            all_corners[cam_id].append(corners)
                            all_ids[cam_id].append(ids)
                            all_timestamps[cam_id].append(current_time)
                            last_save_time[cam_id] = current_time
                            frame_status[cam_id] = 'accepted'
                        else:
                            frame_status[cam_id] = 'similar'
            
            # Add status text to vis_frames
            updated_vis_frames = []
            for cam_id, vis_frame in vis_frames:
                count = len(all_corners[cam_id])
                
                if sync_mode:
                    status = f"Cam {cam_id}: {count}"
                    if sync_captured:
                        color = (0, 255, 0)  # Green - just captured
                        cv2.putText(vis_frame, "CAPTURED!", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    elif cam_id in current_detections and len(current_detections) >= 2:
                        if stable_count > 0:
                            color = (0, 255, 255)  # Yellow - stabilizing
                            status += f" (HOLD {stable_count}/{STABLE_THRESHOLD})"
                        else:
                            color = (0, 165, 255)  # Orange - detected, moving
                            status += " (HOLD STILL)"
                    else:
                        color = (128, 128, 128)  # Gray - not enough cameras
                        status += " (need 2+ cams)"
                else:
                    status = f"Cam {cam_id}: {count}/{target_frames}"
                    if count >= target_frames:
                        color = (0, 255, 0)
                    elif frame_status.get(cam_id) == 'accepted':
                        color = (0, 255, 0)
                    elif frame_status.get(cam_id) == 'similar':
                        color = (0, 0, 255)
                        status += " (MOVE!)"
                    else:
                        color = (0, 165, 255)
                    
                cv2.putText(vis_frame, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                updated_vis_frames.append(vis_frame)
            
            vis_frames = updated_vis_frames
            
            # Check if all cameras have enough frames
            if sync_mode:
                # In sync mode, check min frames across cameras with detections
                min_frames = min(len(all_corners[cam_id]) for cam_id in camera_ids if len(all_corners[cam_id]) > 0) if any(len(all_corners[cam_id]) > 0 for cam_id in camera_ids) else 0
                all_ready = min_frames >= target_frames
            else:
                all_ready = all(len(all_corners[cam_id]) >= target_frames for cam_id in camera_ids)
            
            # Show combined view
            if vis_frames:
                # Resize frames for display - larger for better visibility
                display_width = 640  # Width per camera view
                display_frames = []
                for f in vis_frames:
                    h, w = f.shape[:2]
                    scale = display_width / w
                    resized = cv2.resize(f, (display_width, int(h * scale)))
                    display_frames.append(resized)
                
                # Layout: 2x2 grid for 3-4 cameras, horizontal for 2
                if len(display_frames) == 3:
                    # 2 on top, 1 centered on bottom
                    top_row = np.hstack(display_frames[:2])
                    # Center the bottom frame by adding padding
                    bottom_frame = display_frames[2]
                    pad_width = (top_row.shape[1] - bottom_frame.shape[1]) // 2
                    bottom_row = np.pad(bottom_frame, 
                                       ((0, 0), (pad_width, top_row.shape[1] - bottom_frame.shape[1] - pad_width), (0, 0)),
                                       mode='constant', constant_values=0)
                    combined = np.vstack([top_row, bottom_row])
                elif len(display_frames) == 4:
                    # 2x2 grid
                    top_row = np.hstack(display_frames[:2])
                    bottom_row = np.hstack(display_frames[2:4])
                    combined = np.vstack([top_row, bottom_row])
                else:
                    combined = np.hstack(display_frames)
                
                # Add status bar at top
                if all_ready:
                    cv2.putText(combined, "TARGET REACHED - Press 'q' to finish or 'c' to continue", 
                               (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(window_name, combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                auto_stop = False
                print("Continuing past target... Press 'q' to stop.")
            
            # Auto-stop when target reached
            if all_ready and auto_stop:
                print("\nTarget frames reached for all cameras!")
                time.sleep(0.5)  # Brief pause to show final state
                break
            
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    
    finally:
        recorder.stop()
        cv2.destroyAllWindows()
    
    # Save captured data with per-frame timestamps
    captured_data = {
        'corners': all_corners,
        'ids': all_ids,
        'timestamps': all_timestamps,  # Per-frame timestamps for sync matching
        'camera_ids': camera_ids,
        'resolution': resolution,
        'timestamp': datetime.now().isoformat()
    }
    
    data_path = os.path.join(calibration_dir, "captured_corners.pkl")
    save_captured_data(captured_data, data_path)
    
    return all_corners, all_ids, all_timestamps


def run_calibration(all_corners, all_ids, all_timestamps, camera_ids, resolution, detector, calibration_dir, board_config):
    """Run calibration on captured data with timestamp-based frame matching."""
    
    calibration_start = time.time()
    
    # Check if we have enough data
    print("\n" + "=" * 60)
    print("  CALIBRATION PHASE")
    print("=" * 60)
    print("\nCaptured data summary:")
    total_frames = 0
    for cam_id in camera_ids:
        count = len(all_corners[cam_id])
        total_frames += count
        status = "✓" if count >= 10 else "✗"
        print(f"  Camera {cam_id}: {count} frames [{status}]")
    print(f"  Total: {total_frames} frames across {len(camera_ids)} cameras")
    
    min_collected = min(len(all_corners[cam_id]) for cam_id in camera_ids)
    if min_collected < 10:
        print("\nError: Not enough calibration data. Need at least 10 frames per camera.")
        return False
    
    # Calibrate each camera
    print("\n" + "-" * 60)
    print("STEP 1/2: Intrinsic calibration (per-camera)")
    print("-" * 60)
    intrinsics = {}
    
    for i, cam_id in enumerate(camera_ids):
        print(f"\n[{i+1}/{len(camera_ids)}] Camera {cam_id} ({len(all_corners[cam_id])} frames)")
        camera_matrix, dist_coeffs, error = calibrate_camera_intrinsics(
            all_corners[cam_id], all_ids[cam_id],
            tuple(resolution), detector.board, cam_id
        )
        
        if camera_matrix is not None:
            intrinsics[cam_id] = {
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'error': error
            }
            print(f"    ✓ Success! Reprojection error = {error:.3f} pixels")
        else:
            print(f"    ✗ FAILED")
    
    if len(intrinsics) < 2:
        print("\nError: Need at least 2 calibrated cameras for stereo calibration.")
        return False
    
    # Calibrate stereo pairs
    print("\n" + "-" * 60)
    print("STEP 2/2: Extrinsic calibration (stereo pairs)")
    print("-" * 60)
    extrinsics = {}
    
    # Use CENTER camera as reference (better overlap with all cameras)
    # For 3 cameras [left, center, right], use the middle one
    if len(camera_ids) == 3:
        ref_cam = camera_ids[1]  # Center camera
    else:
        ref_cam = camera_ids[0]  # Fallback to first
    
    print(f"\nUsing Camera {ref_cam} (center) as reference")
    
    # Only calibrate pairs where both cameras have intrinsics
    other_cams = [cam_id for cam_id in camera_ids if cam_id != ref_cam]
    valid_pairs = [cam_id for cam_id in other_cams if cam_id in intrinsics]
    num_pairs = len(valid_pairs)
    
    if ref_cam not in intrinsics:
        print(f"\nError: Reference camera {ref_cam} failed intrinsic calibration.")
        return False
    
    if num_pairs == 0:
        print(f"\nError: No other cameras have valid intrinsics for stereo calibration.")
        return False
    
    for i, cam_id in enumerate(valid_pairs):
        print(f"\n[{i+1}/{num_pairs}] Pair: Camera {ref_cam} -> Camera {cam_id}")
        
        # Get timestamps for each camera if available
        ts_ref = all_timestamps.get(ref_cam, []) if all_timestamps else []
        ts_cam = all_timestamps.get(cam_id, []) if all_timestamps else []
        
        R, T = calibrate_stereo(
            all_corners[ref_cam], all_corners[cam_id],
            all_ids[ref_cam], all_ids[cam_id],
            intrinsics[ref_cam]['camera_matrix'], intrinsics[ref_cam]['dist_coeffs'],
            intrinsics[cam_id]['camera_matrix'], intrinsics[cam_id]['dist_coeffs'],
            tuple(resolution), detector.board,
            ts_ref, ts_cam
        )
        
        if R is not None:
            extrinsics[(ref_cam, cam_id)] = {'R': R, 'T': T}
            print(f"    ✓ Success!")
        else:
            print(f"    ✗ FAILED (need more common views)")
    
    # Build calibration data structure
    calibration_data = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'cameras': camera_ids,
            'resolution': list(resolution),
            'board': board_config
        },
        'cameras': {}
    }
    
    for cam_id in camera_ids:
        if cam_id in intrinsics:
            cam_data = {
                'matrix': intrinsics[cam_id]['camera_matrix'].tolist(),
                'distortions': intrinsics[cam_id]['dist_coeffs'].flatten().tolist(),
                'size': list(resolution),
                'error': float(intrinsics[cam_id]['error'])
            }
            
            # Add extrinsics (rotation and translation relative to reference)
            if cam_id == ref_cam:
                cam_data['rotation'] = np.eye(3).tolist()
                cam_data['translation'] = [0.0, 0.0, 0.0]
            elif (ref_cam, cam_id) in extrinsics:
                cam_data['rotation'] = extrinsics[(ref_cam, cam_id)]['R'].tolist()
                cam_data['translation'] = extrinsics[(ref_cam, cam_id)]['T'].flatten().tolist()
            
            calibration_data['cameras'][f'cam_{cam_id}'] = cam_data
    
    # Save calibration
    output_path = os.path.join(calibration_dir, "calibration.toml")
    save_calibration(calibration_data, output_path)
    
    # Report total time
    total_time = time.time() - calibration_start
    print(f"\nTotal calibration time: {total_time:.1f}s")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Multi-camera calibration")
    parser.add_argument("--recapture", action="store_true",
                       help="Force new capture even if saved data exists")
    parser.add_argument("--target-frames", type=int, default=60,
                       help="Auto-stop after this many DIVERSE frames per camera (default: 60)")
    parser.add_argument("--slow", action="store_true",
                       help="Slow capture mode - waits 0.5s between saves (better coverage)")
    parser.add_argument("--sync", action="store_true",
                       help="Sync mode - only captures when board is still AND visible to 2+ cameras (RECOMMENDED)")
    args = parser.parse_args()
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    calibration_dir = os.path.join(project_dir, "calibration")
    config_path = os.path.join(calibration_dir, "camera_config.yaml")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    camera_ids = config['camera_ids']
    resolution = tuple(config['capture']['resolution'])
    board_config = config['calibration']['board']
    
    print("=" * 60)
    print("  Multi-Camera Calibration")
    print("=" * 60)
    print(f"\nCameras: {camera_ids}")
    print(f"Resolution: {resolution}")
    print(f"Board: {board_config['squares_x']}x{board_config['squares_y']}, "
          f"{board_config['square_size_mm']}mm squares")
    print(f"Target frames: {args.target_frames}")
    
    # Create detector
    detector = CharucoDetector(
        squares_x=board_config['squares_x'],
        squares_y=board_config['squares_y'],
        square_size=board_config['square_size_mm'],
        marker_size=board_config['marker_size_mm'],
        dictionary_name=board_config['dictionary']
    )
    
    # Check for saved data
    data_path = os.path.join(calibration_dir, "captured_corners.pkl")
    
    if os.path.exists(data_path) and not args.recapture:
        print(f"\nFound saved capture data: {data_path}")
        response = input("Use saved data? [Y/n]: ").strip().lower()
        
        if response != 'n':
            captured_data = load_captured_data(data_path)
            if captured_data:
                all_corners = captured_data['corners']
                all_ids = captured_data['ids']
                all_timestamps = captured_data.get('timestamps', {cam_id: [] for cam_id in camera_ids})
            else:
                all_corners, all_ids, all_timestamps = capture_calibration_data(
                    camera_ids, resolution, detector, args.target_frames, calibration_dir, args.slow, args.sync
                )
        else:
            all_corners, all_ids, all_timestamps = capture_calibration_data(
                camera_ids, resolution, detector, args.target_frames, calibration_dir, args.slow, args.sync
            )
    else:
        all_corners, all_ids, all_timestamps = capture_calibration_data(
            camera_ids, resolution, detector, args.target_frames, calibration_dir, args.slow, args.sync
        )
    
    # Run calibration with timestamp-based frame matching
    success = run_calibration(
        all_corners, all_ids, all_timestamps, camera_ids, resolution, 
        detector, calibration_dir, board_config
    )
    
    if success:
        print("\n" + "=" * 60)
        print("  ✓ CALIBRATION COMPLETE!")
        print("=" * 60)
        print(f"\nOutput files:")
        print(f"  Calibration: {os.path.join(calibration_dir, 'calibration.toml')}")
        print(f"  Captured data: {data_path}")
        print(f"\nTo recapture (discard saved data):")
        print(f"  python deploy_real/calibrate_cameras.py --recapture")
        print(f"\nNext step: Test triangulation")
        print(f"  python deploy_real/test_triangulation.py")
    else:
        print("\n" + "=" * 60)
        print("  ✗ CALIBRATION FAILED")
        print("=" * 60)
        print("\nTry recapturing with:")
        print("  - More overlap between camera views")
        print("  - Slower board movement")
        print("  - Better lighting")


if __name__ == "__main__":
    main()

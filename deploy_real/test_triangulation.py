#!/usr/bin/env python3
"""
Test triangulation with calibrated cameras.

This script:
1. Loads calibration data
2. Captures frames from cameras
3. Detects the Charuco board in each view
4. Triangulates corner positions to 3D
5. Displays the reprojection error as a quality metric

Usage:
    conda activate gmr
    python deploy_real/test_triangulation.py

Good triangulation should show:
- Low reprojection errors (< 5 pixels)
- Consistent 3D positions as you move the board
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os
import yaml
import toml
import time
from threading import Thread, Lock


class CameraCapture:
    """Capture from multiple cameras."""
    
    def __init__(self, camera_ids, resolution=(1280, 720)):
        self.camera_ids = camera_ids
        self.resolution = resolution
        self.captures = {}
        self.frames = {}
        self.frame_lock = Lock()
        self.running = False
        
    def start(self):
        self.running = True
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            if cap.isOpened():
                self.captures[cam_id] = cap
                t = Thread(target=self._capture_loop, args=(cam_id, cap))
                t.daemon = True
                t.start()
        print(f"Started {len(self.captures)} cameras")
        
    def _capture_loop(self, cam_id, cap):
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.frame_lock:
                    self.frames[cam_id] = frame.copy()
            time.sleep(0.001)
    
    def get_frames(self):
        with self.frame_lock:
            return {k: v.copy() for k, v in self.frames.items()}
    
    def stop(self):
        self.running = False
        time.sleep(0.1)
        for cap in self.captures.values():
            cap.release()


def load_calibration(calibration_path):
    """Load camera calibration from TOML file."""
    with open(calibration_path, 'r') as f:
        data = toml.load(f)
    
    calibration = {}
    for cam_key, cam_data in data.get('cameras', {}).items():
        cam_id = int(cam_key.replace('cam_', ''))
        
        calibration[cam_id] = {
            'matrix': np.array(cam_data['matrix']),
            'distortions': np.array(cam_data['distortions']),
            'rotation': np.array(cam_data.get('rotation', np.eye(3))),
            'translation': np.array(cam_data.get('translation', [0, 0, 0])),
        }
    
    return calibration, data.get('metadata', {})


def triangulate_point(pt1, pt2, P1, P2):
    """Triangulate a 3D point from two 2D observations."""
    pts1 = np.array([[pt1[0], pt1[1]]], dtype=np.float64)
    pts2 = np.array([[pt2[0], pt2[1]]], dtype=np.float64)
    
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = pts4d[:3] / pts4d[3]
    
    return pts3d.flatten()


def project_point(pt3d, K, R, t):
    """Project a 3D point to 2D."""
    pt3d = pt3d.reshape(3, 1)
    pt_cam = R @ pt3d + t.reshape(3, 1)
    pt_proj = K @ pt_cam
    pt2d = pt_proj[:2] / pt_proj[2]
    return pt2d.flatten()


def compute_reprojection_error(pt3d, pt2d_observed, K, R, t):
    """Compute reprojection error for a triangulated point."""
    pt2d_projected = project_point(pt3d, K, R, t)
    error = np.linalg.norm(pt2d_observed - pt2d_projected)
    return error


def main():
    # Load paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    calibration_dir = os.path.join(project_dir, "calibration")
    config_path = os.path.join(calibration_dir, "camera_config.yaml")
    calibration_path = os.path.join(calibration_dir, "calibration.toml")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    camera_ids = config['camera_ids']
    resolution = tuple(config['capture']['resolution'])
    board_config = config['calibration']['board']
    
    # Load calibration
    print("=" * 60)
    print("  Triangulation Test")
    print("=" * 60)
    
    if not os.path.exists(calibration_path):
        print(f"\nError: Calibration file not found: {calibration_path}")
        print("Run calibration first: python deploy_real/calibrate_cameras.py")
        return
    
    calibration, metadata = load_calibration(calibration_path)
    print(f"\nLoaded calibration for cameras: {list(calibration.keys())}")
    
    # Check which camera pairs have stereo calibration
    # Use center camera as reference (same as calibration script)
    if len(camera_ids) == 3:
        ref_cam = camera_ids[1]  # Center camera
    else:
        ref_cam = camera_ids[0]
    
    other_cams = [c for c in camera_ids if c != ref_cam]
    
    valid_pairs = []
    for cam_id in other_cams:
        if cam_id in calibration:
            R = calibration[cam_id].get('rotation')
            t = calibration[cam_id].get('translation')
            if R is not None and t is not None and not np.allclose(R, np.eye(3)):
                valid_pairs.append((ref_cam, cam_id))
                print(f"  Stereo pair: {ref_cam} -> {cam_id} ✓")
            else:
                print(f"  Stereo pair: {ref_cam} -> {cam_id} ✗ (no extrinsics)")
    
    if not valid_pairs:
        print("\nError: No valid stereo pairs found. Need at least 2 calibrated cameras.")
        return
    
    # Create Charuco detector
    dict_map = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
    }
    dictionary = aruco.getPredefinedDictionary(
        dict_map.get(board_config['dictionary'], aruco.DICT_4X4_50)
    )
    
    board = aruco.CharucoBoard(
        (board_config['squares_x'], board_config['squares_y']),
        board_config['square_size_mm'] / 1000.0,
        board_config['marker_size_mm'] / 1000.0,
        dictionary
    )
    detector_params = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(dictionary, detector_params)
    
    # Build projection matrices
    projection_matrices = {}
    for cam_id in calibration:
        K = calibration[cam_id]['matrix']
        R = calibration[cam_id]['rotation']
        t = calibration[cam_id]['translation']
        
        # Projection matrix P = K * [R | t]
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt
        projection_matrices[cam_id] = P
    
    # Start cameras
    capture = CameraCapture(camera_ids, resolution)
    capture.start()
    time.sleep(1)  # Warm up
    
    print(f"\nHold the Charuco board so both cameras can see it.")
    print("The display shows triangulation quality.")
    print("Press 'q' to quit.")
    print("-" * 60)
    
    # Use first valid pair for triangulation
    cam1, cam2 = valid_pairs[0]
    
    # Create resizable window
    window_name = "Triangulation Test - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 960)
    
    try:
        while True:
            frames = capture.get_frames()
            
            if not frames:
                time.sleep(0.01)
                continue
            
            # Detect Charuco in all frames
            detections = {}
            vis_frames = {}
            
            for cam_id in camera_ids:
                if cam_id not in frames:
                    continue
                frame = frames[cam_id]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                corners, ids, rejected = aruco_detector.detectMarkers(gray)
                
                vis = frame.copy()
                
                if ids is not None and len(ids) > 0:
                    aruco.drawDetectedMarkers(vis, corners, ids)
                    
                    ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                        corners, ids, gray, board
                    )
                    
                    if ret > 4:
                        aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
                        detections[cam_id] = {
                            'corners': charuco_corners,
                            'ids': charuco_ids.flatten()
                        }
                
                vis_frames[cam_id] = vis
            
            # Triangulate if both cameras detected the board
            if cam1 in detections and cam2 in detections:
                # Find common corner IDs
                ids1 = detections[cam1]['ids']
                ids2 = detections[cam2]['ids']
                common_ids = np.intersect1d(ids1, ids2)
                
                if len(common_ids) >= 4:
                    errors = []
                    
                    for cid in common_ids[:10]:  # Limit to 10 corners for speed
                        idx1 = np.where(ids1 == cid)[0][0]
                        idx2 = np.where(ids2 == cid)[0][0]
                        
                        pt1 = detections[cam1]['corners'][idx1].flatten()
                        pt2 = detections[cam2]['corners'][idx2].flatten()
                        
                        # Triangulate
                        pt3d = triangulate_point(
                            pt1, pt2,
                            projection_matrices[cam1],
                            projection_matrices[cam2]
                        )
                        
                        # Compute reprojection errors
                        err1 = compute_reprojection_error(
                            pt3d, pt1,
                            calibration[cam1]['matrix'],
                            calibration[cam1]['rotation'],
                            calibration[cam1]['translation']
                        )
                        err2 = compute_reprojection_error(
                            pt3d, pt2,
                            calibration[cam2]['matrix'],
                            calibration[cam2]['rotation'],
                            calibration[cam2]['translation']
                        )
                        
                        errors.append((err1 + err2) / 2)
                    
                    avg_error = np.mean(errors)
                    max_error = np.max(errors)
                    
                    # Display quality (relaxed thresholds for non-hardware-synced cameras)
                    # <10px = GOOD (~2cm at 1m), <20px = FAIR (~4cm at 1m), >=20px = POOR
                    quality_color = (0, 255, 0) if avg_error < 10 else (0, 255, 255) if avg_error < 20 else (0, 0, 255)
                    quality_text = "GOOD" if avg_error < 10 else "FAIR" if avg_error < 20 else "POOR"
                    
                    for cam_id in [cam1, cam2]:
                        cv2.putText(vis_frames[cam_id], 
                                   f"Triangulation: {quality_text}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, quality_color, 2)
                        cv2.putText(vis_frames[cam_id],
                                   f"Reproj Error: {avg_error:.2f}px (max: {max_error:.2f}px)", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
                        cv2.putText(vis_frames[cam_id],
                                   f"Common points: {len(common_ids)}", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    for cam_id in [cam1, cam2]:
                        cv2.putText(vis_frames[cam_id],
                                   "Need more common points", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                for cam_id in [cam1, cam2]:
                    if cam_id in vis_frames:
                        status = "Board detected" if cam_id in detections else "No board detected"
                        cv2.putText(vis_frames[cam_id], status, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # Stack and display all cameras
            display_width = 640
            display_frames = []
            
            for cam_id in camera_ids:
                if cam_id in vis_frames:
                    vis = vis_frames[cam_id].copy()
                else:
                    # Create placeholder for missing camera
                    vis = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                    cv2.putText(vis, f"Camera {cam_id} - No frame", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                
                # Add camera label
                cv2.putText(vis, f"Camera {cam_id}", (10, vis.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Resize
                scale = display_width / resolution[0]
                resized = cv2.resize(vis, (display_width, int(resolution[1] * scale)))
                display_frames.append(resized)
            
            # Layout: 2x2 grid for 3 cameras (2 on top, 1 centered bottom)
            if len(display_frames) == 3:
                top_row = np.hstack(display_frames[:2])
                bottom_frame = display_frames[2]
                pad_width = (top_row.shape[1] - bottom_frame.shape[1]) // 2
                bottom_row = np.pad(bottom_frame,
                                   ((0, 0), (pad_width, top_row.shape[1] - bottom_frame.shape[1] - pad_width), (0, 0)),
                                   mode='constant', constant_values=0)
                combined = np.vstack([top_row, bottom_row])
            elif len(display_frames) == 2:
                combined = np.hstack(display_frames)
            elif len(display_frames) == 4:
                top_row = np.hstack(display_frames[:2])
                bottom_row = np.hstack(display_frames[2:4])
                combined = np.vstack([top_row, bottom_row])
            else:
                combined = display_frames[0] if display_frames else np.zeros((480, 640, 3), dtype=np.uint8)
            
            cv2.imshow(window_name, combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        capture.stop()
        cv2.destroyAllWindows()
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()

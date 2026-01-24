#!/usr/bin/env python3
"""
Skeleton Visualizer for Captured Poses

View captured pose JSON files to identify what pose each represents.

Usage:
    python view_captured_poses.py                    # View all poses in sequence
    python view_captured_poses.py --file pose.json  # View specific file
"""

import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
POSES_DIR = PROJECT_ROOT / "calibration" / "captured_poses"


# MediaPipe skeleton connections for visualization
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye to ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye to ear
    (9, 10),  # Mouth
    
    # Torso
    (11, 12),  # Shoulders
    (11, 23), (12, 24),  # Shoulders to hips
    (23, 24),  # Hips
    
    # Left arm
    (11, 13), (13, 15),  # Shoulder -> elbow -> wrist
    (15, 17), (15, 19), (15, 21),  # Wrist to fingers
    
    # Right arm
    (12, 14), (14, 16),  # Shoulder -> elbow -> wrist
    (16, 18), (16, 20), (16, 22),  # Wrist to fingers
    
    # Left leg
    (23, 25), (25, 27),  # Hip -> knee -> ankle
    (27, 29), (27, 31),  # Ankle to foot
    
    # Right leg
    (24, 26), (26, 28),  # Hip -> knee -> ankle
    (28, 30), (28, 32),  # Ankle to foot
]

# Key landmarks for labeling
LANDMARK_NAMES = {
    0: "nose",
    11: "L_shoulder",
    12: "R_shoulder",
    13: "L_elbow",
    14: "R_elbow",
    15: "L_wrist",
    16: "R_wrist",
    23: "L_hip",
    24: "R_hip",
    25: "L_knee",
    26: "R_knee",
    27: "L_ankle",
    28: "R_ankle",
}


def load_pose(filepath: Path) -> dict:
    """Load a captured pose JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def align_skeleton_upright(skeleton: np.ndarray) -> np.ndarray:
    """
    Rotate skeleton so the torso is vertical (standing upright).
    Uses the spine direction (mid-hip to mid-shoulder) to determine rotation.
    """
    skeleton = skeleton.copy()
    
    # Get torso landmarks
    l_shoulder = skeleton[11]
    r_shoulder = skeleton[12]
    l_hip = skeleton[23]
    r_hip = skeleton[24]
    
    # Check if we have valid landmarks
    if any(np.isnan(l_shoulder)) or any(np.isnan(r_shoulder)) or \
       any(np.isnan(l_hip)) or any(np.isnan(r_hip)):
        return skeleton
    
    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_hip = (l_hip + r_hip) / 2
    
    # Spine direction (should point UP when standing)
    spine = mid_shoulder - mid_hip
    spine_norm = np.linalg.norm(spine)
    if spine_norm < 0.01:
        return skeleton
    spine = spine / spine_norm
    
    # Target up direction (positive Z after our transform)
    up = np.array([0, 0, 1])
    
    # Compute rotation axis and angle
    axis = np.cross(spine, up)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:
        # Already aligned or opposite
        if np.dot(spine, up) < 0:
            # Flip 180 degrees around X axis
            skeleton[:, 2] = -skeleton[:, 2]
        return skeleton
    
    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(spine, up), -1, 1))
    
    # Rodrigues' rotation formula
    def rotate_point(p, axis, angle):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return p * cos_a + np.cross(axis, p) * sin_a + axis * np.dot(axis, p) * (1 - cos_a)
    
    # Center at mid-hip, rotate, then translate back
    center = mid_hip.copy()
    
    for i in range(len(skeleton)):
        if not np.isnan(skeleton[i, 0]):
            skeleton[i] = rotate_point(skeleton[i] - center, axis, angle) + center
    
    return skeleton


def visualize_skeleton(ax, skeleton: np.ndarray, title: str = ""):
    """Draw skeleton on 3D axes."""
    ax.clear()
    
    # Align skeleton to stand upright
    skeleton = align_skeleton_upright(skeleton)
    
    # Get valid points (non-NaN)
    valid_mask = ~np.isnan(skeleton[:, 0])
    
    # After alignment:
    # X = left/right
    # Y = forward/back  
    # Z = up/down
    
    # Plot points
    ax.scatter(
        skeleton[valid_mask, 0],   # X
        skeleton[valid_mask, 1],   # Y (forward)
        skeleton[valid_mask, 2],   # Z (up)
        c='blue', s=50, alpha=0.8
    )
    
    # Plot connections
    for i, j in POSE_CONNECTIONS:
        if i < len(skeleton) and j < len(skeleton):
            if valid_mask[i] and valid_mask[j]:
                ax.plot(
                    [skeleton[i, 0], skeleton[j, 0]],
                    [skeleton[i, 1], skeleton[j, 1]],
                    [skeleton[i, 2], skeleton[j, 2]],
                    'b-', linewidth=2, alpha=0.7
                )
    
    # Label key landmarks
    for idx, name in LANDMARK_NAMES.items():
        if idx < len(skeleton) and valid_mask[idx]:
            ax.text(
                skeleton[idx, 0],
                skeleton[idx, 1],
                skeleton[idx, 2],
                name, fontsize=8, alpha=0.7
            )
    
    # Set labels and title
    ax.set_xlabel('X (Left/Right)')
    ax.set_ylabel('Y (Forward)')
    ax.set_zlabel('Z (Up)')
    ax.set_title(title)
    
    # Set equal aspect ratio
    skeleton_valid = skeleton[valid_mask]
    if len(skeleton_valid) > 0:
        max_range = np.max([
            skeleton_valid[:, 0].max() - skeleton_valid[:, 0].min(),
            skeleton_valid[:, 1].max() - skeleton_valid[:, 1].min(),
            skeleton_valid[:, 2].max() - skeleton_valid[:, 2].min()
        ]) / 2.0
        
        mid_x = (skeleton_valid[:, 0].max() + skeleton_valid[:, 0].min()) / 2
        mid_y = (skeleton_valid[:, 1].max() + skeleton_valid[:, 1].min()) / 2
        mid_z = (skeleton_valid[:, 2].max() + skeleton_valid[:, 2].min()) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


def format_angles(angles: dict) -> str:
    """Format joint angles for display."""
    lines = []
    
    # Group by body part
    groups = {
        "Shoulders": ["left_shoulder_pitch", "left_shoulder_roll", "right_shoulder_pitch", "right_shoulder_roll"],
        "Elbows": ["left_elbow", "right_elbow"],
        "Hips": ["left_hip_pitch", "left_hip_roll", "right_hip_pitch", "right_hip_roll"],
        "Knees": ["left_knee", "right_knee"],
    }
    
    for group_name, keys in groups.items():
        lines.append(f"\n{group_name}:")
        for key in keys:
            if key in angles:
                short_name = key.replace("left_", "L_").replace("right_", "R_")
                lines.append(f"  {short_name}: {angles[key]:+.1f}°")
    
    return "\n".join(lines)


def view_single_pose(filepath: Path):
    """View a single pose file."""
    pose_data = load_pose(filepath)
    skeleton = np.array(pose_data['skeleton_3d'])
    
    fig = plt.figure(figsize=(14, 8))
    
    # 3D skeleton view
    ax = fig.add_subplot(121, projection='3d')
    
    title = f"{pose_data['name']}\n{filepath.name}"
    visualize_skeleton(ax, skeleton, title)
    
    # Joint angles text
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    info_text = f"File: {filepath.name}\n"
    info_text += f"Name: {pose_data['name']}\n"
    info_text += f"Timestamp: {pose_data['timestamp']}\n"
    info_text += f"Frames: {pose_data['num_frames']}\n"
    info_text += f"Std Dev: {pose_data['avg_std_cm']:.2f} cm\n"
    info_text += "\n" + "="*30
    info_text += format_angles(pose_data['joint_angles'])
    
    ax2.text(0.1, 0.95, info_text, transform=ax2.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()


def view_all_poses(poses_dir: Path):
    """View all poses in the directory with navigation."""
    pose_files = sorted(poses_dir.glob("*.json"))
    
    if not pose_files:
        print(f"No pose files found in {poses_dir}")
        return
    
    print(f"\nFound {len(pose_files)} pose files")
    print("\nControls:")
    print("  LEFT/RIGHT arrows - Previous/Next pose")
    print("  R - Rename current pose")
    print("  Q or ESC - Quit")
    print("  Close window to quit")
    print("="*50)
    
    current_idx = [0]  # Use list to allow modification in nested function
    
    fig = plt.figure(figsize=(14, 8))
    
    def update_display():
        filepath = pose_files[current_idx[0]]
        pose_data = load_pose(filepath)
        skeleton = np.array(pose_data['skeleton_3d'])
        
        fig.clear()
        
        # 3D skeleton view
        ax = fig.add_subplot(121, projection='3d')
        title = f"[{current_idx[0]+1}/{len(pose_files)}] {pose_data['name']}\n{filepath.name}"
        visualize_skeleton(ax, skeleton, title)
        
        # Set a good viewing angle
        ax.view_init(elev=20, azim=-60)
        
        # Joint angles text
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        
        info_text = f"File: {filepath.name}\n"
        info_text += f"Name: {pose_data['name']}\n"
        info_text += f"Timestamp: {pose_data['timestamp']}\n"
        info_text += f"Frames: {pose_data['num_frames']}\n"
        info_text += f"Std Dev: {pose_data['avg_std_cm']:.2f} cm\n"
        info_text += "\n" + "="*30
        info_text += format_angles(pose_data['joint_angles'])
        info_text += "\n\n" + "="*30
        info_text += "\n\nControls: ←/→ Navigate, R Rename, Q Quit"
        
        ax2.text(0.1, 0.95, info_text, transform=ax2.transAxes, 
                 fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        fig.canvas.draw()
    
    def on_key(event):
        if event.key in ['right', 'n']:
            current_idx[0] = (current_idx[0] + 1) % len(pose_files)
            update_display()
        elif event.key in ['left', 'p']:
            current_idx[0] = (current_idx[0] - 1) % len(pose_files)
            update_display()
        elif event.key == 'r':
            # Rename pose
            filepath = pose_files[current_idx[0]]
            pose_data = load_pose(filepath)
            
            print(f"\nCurrent name: {pose_data['name']}")
            new_name = input("Enter new name (or press Enter to cancel): ").strip()
            
            if new_name:
                # Update the JSON file
                pose_data['name'] = new_name
                with open(filepath, 'w') as f:
                    json.dump(pose_data, f, indent=2)
                print(f"Renamed to: {new_name}")
                update_display()
            else:
                print("Cancelled")
        elif event.key in ['q', 'escape']:
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    update_display()
    plt.show()
    
    print("\nDone viewing poses.")


def main():
    parser = argparse.ArgumentParser(description="View captured pose skeletons")
    parser.add_argument("--file", "-f", type=str, default=None,
                       help="Specific pose JSON file to view")
    parser.add_argument("--dir", "-d", type=str, default=None,
                       help="Directory containing pose files (default: calibration/captured_poses)")
    args = parser.parse_args()
    
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            # Try relative to poses dir
            filepath = POSES_DIR / args.file
        
        if not filepath.exists():
            print(f"Error: File not found: {args.file}")
            return
        
        view_single_pose(filepath)
    else:
        poses_dir = Path(args.dir) if args.dir else POSES_DIR
        
        if not poses_dir.exists():
            print(f"Error: Directory not found: {poses_dir}")
            return
        
        view_all_poses(poses_dir)


if __name__ == "__main__":
    main()

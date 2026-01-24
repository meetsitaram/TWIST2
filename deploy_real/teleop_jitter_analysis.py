#!/usr/bin/env python3
"""
Teleop Jitter Analysis - Analyze and compare teleoperation episode quality.

Computes jitter metrics for recorded episodes:
- Velocity variance (per joint)
- Acceleration RMS
- High-frequency power (FFT)
- Smoothness index (spectral arc length)
- Frame-to-frame fluctuation analysis

Usage:
    # Analyze single episode with time-series plots
    python teleop_jitter_analysis.py --episode baseline_001 --plot
    
    # Detailed time-series analysis (frame-by-frame fluctuations)
    python teleop_jitter_analysis.py --episode baseline_001 --timeseries
    
    # Compare multiple episodes
    python teleop_jitter_analysis.py --compare baseline_001 ema_001 oneeuro_001
    
    # List all episodes
    python teleop_jitter_analysis.py --list
    
    # Generate summary report
    python teleop_jitter_analysis.py --report
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from teleop_episode_recorder import TeleopEpisodeRecorder, TeleopEpisode, EPISODES_DIR

# MediaPipe landmark indices for key body points
SKELETON_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_index": 19,
    "right_index": 20,
}

# Key landmarks for end-effector tracking
EE_LANDMARKS = {
    "left_wrist": 15,
    "right_wrist": 16,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Joint names for the 29-DOF robot (indices 7-35 in qpos)
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# Group joints for summary
JOINT_GROUPS = {
    "left_leg": [0, 1, 2, 3, 4, 5],
    "right_leg": [6, 7, 8, 9, 10, 11],
    "waist": [12, 13, 14],
    "left_arm": [15, 16, 17, 18, 19, 20, 21],
    "right_arm": [22, 23, 24, 25, 26, 27, 28],
}


def compute_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocity from position time series using central differences."""
    if len(positions) < 3:
        return np.zeros_like(positions)
    
    velocity = np.zeros_like(positions)
    velocity[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    velocity[0] = (positions[1] - positions[0]) / dt
    velocity[-1] = (positions[-1] - positions[-2]) / dt
    
    return velocity


def compute_acceleration(velocity: np.ndarray, dt: float) -> np.ndarray:
    """Compute acceleration from velocity time series."""
    return compute_velocity(velocity, dt)


def compute_spectral_arc_length(velocity: np.ndarray, dt: float) -> float:
    """
    Compute spectral arc length (smoothness measure from motor control).
    
    Lower values indicate smoother motion.
    Reference: Balasubramanian et al. (2012) "On the analysis of movement smoothness"
    """
    if len(velocity) < 10:
        return 0.0
    
    # FFT of velocity
    n = len(velocity)
    freq = np.fft.rfftfreq(n, dt)
    fft_vel = np.fft.rfft(velocity)
    magnitude = np.abs(fft_vel)
    
    # Normalize magnitude
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
    
    # Compute arc length in frequency domain (up to 20 Hz)
    freq_cutoff = 20.0  # Hz
    valid = freq <= freq_cutoff
    
    if valid.sum() < 2:
        return 0.0
    
    freq_valid = freq[valid]
    mag_valid = magnitude[valid]
    
    # Arc length = integral of sqrt(1 + (dM/df)^2)
    df = freq_valid[1] - freq_valid[0] if len(freq_valid) > 1 else 1.0
    dM = np.diff(mag_valid) / df
    arc_length = -np.sum(np.sqrt(1 + dM**2) * df)
    
    return arc_length


def compute_high_freq_power(signal: np.ndarray, dt: float, cutoff_hz: float = 5.0) -> float:
    """Compute fraction of power above cutoff frequency (noise/jitter indicator)."""
    if len(signal) < 10:
        return 0.0
    
    n = len(signal)
    freq = np.fft.rfftfreq(n, dt)
    fft_signal = np.fft.rfft(signal)
    power = np.abs(fft_signal) ** 2
    
    total_power = power.sum()
    if total_power < 1e-10:
        return 0.0
    
    high_freq_mask = freq > cutoff_hz
    high_freq_power = power[high_freq_mask].sum()
    
    return high_freq_power / total_power


class JitterMetrics:
    """Container for jitter analysis metrics."""
    
    def __init__(self):
        self.velocity_variance: Dict[str, float] = {}
        self.acceleration_rms: Dict[str, float] = {}
        self.high_freq_power: Dict[str, float] = {}
        self.smoothness_index: Dict[str, float] = {}
        self.max_velocity: Dict[str, float] = {}
        
        # Aggregate metrics
        self.mean_velocity_variance: float = 0.0
        self.mean_acceleration_rms: float = 0.0
        self.mean_high_freq_power: float = 0.0
        self.mean_smoothness_index: float = 0.0
        self.max_joint_velocity: float = 0.0


def analyze_episode(episode: TeleopEpisode) -> JitterMetrics:
    """
    Compute jitter metrics for an episode.
    
    Returns:
        JitterMetrics object with per-joint and aggregate metrics
    """
    metrics = JitterMetrics()
    
    if episode.num_frames < 10:
        print(f"[Warning] Episode too short ({episode.num_frames} frames)")
        return metrics
    
    # Extract joint angles (indices 7-35 in qpos)
    qpos_all = np.array([f.robot_qpos for f in episode.frames])
    joint_angles = qpos_all[:, 7:36]  # 29 joints
    
    # Compute dt from timestamps
    t_ms = np.array([f.t_ms for f in episode.frames])
    dt = np.median(np.diff(t_ms)) / 1000.0  # Convert to seconds
    
    if dt <= 0:
        dt = 1.0 / episode.fps
    
    # Analyze each joint
    for i, joint_name in enumerate(JOINT_NAMES):
        positions = joint_angles[:, i]
        
        # Compute velocity and acceleration
        velocity = compute_velocity(positions, dt)
        acceleration = compute_acceleration(velocity, dt)
        
        # Metrics
        metrics.velocity_variance[joint_name] = float(np.var(velocity))
        metrics.acceleration_rms[joint_name] = float(np.sqrt(np.mean(acceleration**2)))
        metrics.high_freq_power[joint_name] = compute_high_freq_power(velocity, dt)
        metrics.smoothness_index[joint_name] = compute_spectral_arc_length(velocity, dt)
        metrics.max_velocity[joint_name] = float(np.max(np.abs(velocity)))
    
    # Aggregate metrics
    metrics.mean_velocity_variance = np.mean(list(metrics.velocity_variance.values()))
    metrics.mean_acceleration_rms = np.mean(list(metrics.acceleration_rms.values()))
    metrics.mean_high_freq_power = np.mean(list(metrics.high_freq_power.values()))
    metrics.mean_smoothness_index = np.mean(list(metrics.smoothness_index.values()))
    metrics.max_joint_velocity = max(metrics.max_velocity.values())
    
    return metrics


def print_metrics(metrics: JitterMetrics, name: str = "Episode"):
    """Print metrics summary."""
    print(f"\n{'='*60}")
    print(f"  Jitter Analysis: {name}")
    print(f"{'='*60}")
    
    print(f"\n  Aggregate Metrics:")
    print(f"  {'─'*40}")
    print(f"  Mean Velocity Variance:  {metrics.mean_velocity_variance:.6f}")
    print(f"  Mean Acceleration RMS:   {metrics.mean_acceleration_rms:.4f}")
    print(f"  Mean High-Freq Power:    {metrics.mean_high_freq_power:.4f}")
    print(f"  Mean Smoothness Index:   {metrics.mean_smoothness_index:.2f}")
    print(f"  Max Joint Velocity:      {metrics.max_joint_velocity:.2f} rad/s")
    
    # Per-group summary
    print(f"\n  Per-Group Velocity Variance:")
    print(f"  {'─'*40}")
    for group_name, joint_indices in JOINT_GROUPS.items():
        group_vars = [metrics.velocity_variance[JOINT_NAMES[i]] for i in joint_indices]
        print(f"  {group_name:15}: {np.mean(group_vars):.6f}")


def plot_episode(episode: TeleopEpisode, metrics: JitterMetrics = None):
    """Plot episode data and metrics."""
    if not HAS_MATPLOTLIB:
        print("[Warning] matplotlib not available for plotting")
        return
    
    # Extract data
    qpos_all = np.array([f.robot_qpos for f in episode.frames])
    joint_angles = qpos_all[:, 7:36]
    t_sec = np.array([f.t_ms for f in episode.frames]) / 1000.0
    
    dt = np.median(np.diff(t_sec)) if len(t_sec) > 1 else 1/30
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Episode: {episode.name} | Smoothing: {episode.smoothing}", fontsize=14)
    
    # Plot 1: Joint angles over time (arms only)
    ax = axes[0, 0]
    arm_joints = JOINT_GROUPS["left_arm"] + JOINT_GROUPS["right_arm"]
    for i in arm_joints[:6]:  # First 6 arm joints
        ax.plot(t_sec, np.degrees(joint_angles[:, i]), label=JOINT_NAMES[i], alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Arm Joint Angles")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Joint velocities
    ax = axes[0, 1]
    for i in arm_joints[:6]:
        velocity = compute_velocity(joint_angles[:, i], dt)
        ax.plot(t_sec, velocity, label=JOINT_NAMES[i], alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Arm Joint Velocities")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Velocity variance by joint group
    ax = axes[1, 0]
    if metrics:
        group_names = list(JOINT_GROUPS.keys())
        group_vars = []
        for group_name in group_names:
            joint_indices = JOINT_GROUPS[group_name]
            vars_ = [metrics.velocity_variance[JOINT_NAMES[i]] for i in joint_indices]
            group_vars.append(np.mean(vars_))
        ax.bar(group_names, group_vars)
        ax.set_ylabel("Mean Velocity Variance")
        ax.set_title("Velocity Variance by Group")
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 4: FFT of a sample joint
    ax = axes[1, 1]
    sample_joint = JOINT_GROUPS["left_arm"][0]  # left_shoulder_pitch
    velocity = compute_velocity(joint_angles[:, sample_joint], dt)
    
    n = len(velocity)
    freq = np.fft.rfftfreq(n, dt)
    fft_vel = np.fft.rfft(velocity)
    power = np.abs(fft_vel) ** 2
    
    # Normalize
    if power.max() > 0:
        power = power / power.max()
    
    ax.semilogy(freq, power)
    ax.axvline(x=5.0, color='r', linestyle='--', label='5 Hz cutoff')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Power")
    ax.set_title(f"FFT: {JOINT_NAMES[sample_joint]}")
    ax.set_xlim(0, 30)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: IK error over time
    ax = axes[2, 0]
    ik_errors = [f.ik_error for f in episode.frames]
    ax.plot(t_sec, ik_errors)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("IK Error")
    ax.set_title("IK Solver Error")
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Smoothness index by group
    ax = axes[2, 1]
    if metrics:
        group_names = list(JOINT_GROUPS.keys())
        group_smooth = []
        for group_name in group_names:
            joint_indices = JOINT_GROUPS[group_name]
            smooth = [metrics.smoothness_index[JOINT_NAMES[i]] for i in joint_indices]
            group_smooth.append(np.mean(smooth))
        ax.bar(group_names, np.abs(group_smooth))
        ax.set_ylabel("|Smoothness Index|")
        ax.set_title("Smoothness by Group (lower = smoother)")
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def compare_episodes(episode_names: List[str]):
    """Compare multiple episodes side by side."""
    episodes = []
    metrics_list = []
    
    for name in episode_names:
        filepath = EPISODES_DIR / f"{name}.npz"
        if not filepath.exists():
            print(f"[Warning] Episode not found: {name}")
            continue
        
        episode = TeleopEpisodeRecorder.load(filepath)
        metrics = analyze_episode(episode)
        
        episodes.append(episode)
        metrics_list.append(metrics)
    
    if not episodes:
        print("No episodes to compare")
        return
    
    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  Episode Comparison")
    print(f"{'='*80}")
    
    header = f"{'Metric':<25}"
    for e in episodes:
        header += f" | {e.name:>12}"
    print(header)
    print("-" * 80)
    
    # Metrics to compare
    metric_rows = [
        ("Velocity Variance", [m.mean_velocity_variance for m in metrics_list]),
        ("Acceleration RMS", [m.mean_acceleration_rms for m in metrics_list]),
        ("High-Freq Power", [m.mean_high_freq_power for m in metrics_list]),
        ("Smoothness Index", [m.mean_smoothness_index for m in metrics_list]),
        ("Max Velocity (rad/s)", [m.max_joint_velocity for m in metrics_list]),
        ("Duration (s)", [e.duration_sec for e in episodes]),
        ("Frames", [e.num_frames for e in episodes]),
    ]
    
    for label, values in metric_rows:
        row = f"{label:<25}"
        for v in values:
            row += f" | {v:>12.4f}"
        print(row)
    
    # Plot comparison if matplotlib available
    if HAS_MATPLOTLIB:
        plot_comparison(episodes, metrics_list)


def plot_comparison(episodes: List[TeleopEpisode], metrics_list: List[JitterMetrics]):
    """Plot comparison of multiple episodes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Episode Comparison", fontsize=14)
    
    names = [e.name for e in episodes]
    
    # Bar plots for each metric
    metrics_to_plot = [
        ("Velocity Variance", [m.mean_velocity_variance for m in metrics_list]),
        ("Acceleration RMS", [m.mean_acceleration_rms for m in metrics_list]),
        ("High-Freq Power", [m.mean_high_freq_power for m in metrics_list]),
        ("|Smoothness Index|", [abs(m.mean_smoothness_index) for m in metrics_list]),
    ]
    
    for ax, (label, values) in zip(axes.flat, metrics_to_plot):
        bars = ax.bar(names, values)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.tick_params(axis='x', rotation=45)
        
        # Highlight best (lowest) value
        min_idx = np.argmin(values)
        bars[min_idx].set_color('green')
    
    plt.tight_layout()
    plt.show()


def plot_timeseries_detailed(episode: TeleopEpisode, metrics: JitterMetrics = None):
    """
    Generate comprehensive time-series plots showing frame-by-frame fluctuations.
    
    Creates multiple figure windows:
    1. Human skeleton landmark positions (X, Y, Z)
    2. Human skeleton frame-to-frame deltas
    3. Robot joint angles by group
    4. Robot joint velocities and accelerations
    5. End-effector positions (computed from skeleton)
    6. Summary statistics
    """
    if not HAS_MATPLOTLIB:
        print("[Warning] matplotlib not available for plotting")
        return
    
    print(f"\n[Analysis] Generating detailed time-series plots for: {episode.name}")
    print(f"           Frames: {episode.num_frames}, Duration: {episode.duration_sec:.1f}s")
    
    # Extract all data
    n_frames = episode.num_frames
    t_ms = np.array([f.t_ms for f in episode.frames])
    t_sec = t_ms / 1000.0
    
    # Human skeleton data: (n_frames, 33, 3)
    skeleton_all = np.stack([f.human_skeleton for f in episode.frames])
    
    # Robot qpos: (n_frames, 36)
    qpos_all = np.stack([f.robot_qpos for f in episode.frames])
    joint_angles = qpos_all[:, 7:36]  # 29 joints
    root_pos = qpos_all[:, :3]  # x, y, z
    root_quat = qpos_all[:, 3:7]  # quaternion
    
    # IK errors
    ik_errors = np.array([f.ik_error for f in episode.frames])
    
    # Compute dt
    dt = np.median(np.diff(t_sec)) if len(t_sec) > 1 else 1/30
    actual_fps = 1.0 / dt if dt > 0 else 30
    
    print(f"           Actual FPS: {actual_fps:.1f}, dt: {dt*1000:.1f}ms")
    
    # =========================================================================
    # FIGURE 1: Human Skeleton Landmark Positions
    # =========================================================================
    fig1, axes1 = plt.subplots(3, 2, figsize=(16, 10))
    fig1.suptitle(f"Human Skeleton Positions - {episode.name}", fontsize=14, fontweight='bold')
    
    # Left side: Wrist positions (X, Y, Z)
    ax = axes1[0, 0]
    for name, idx in [("left_wrist", 15), ("right_wrist", 16)]:
        ax.plot(t_sec, skeleton_all[:, idx, 0], label=f"{name} X", alpha=0.8)
    ax.set_ylabel("X Position (m)")
    ax.set_title("Wrist X Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes1[1, 0]
    for name, idx in [("left_wrist", 15), ("right_wrist", 16)]:
        ax.plot(t_sec, skeleton_all[:, idx, 1], label=f"{name} Y", alpha=0.8)
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Wrist Y Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes1[2, 0]
    for name, idx in [("left_wrist", 15), ("right_wrist", 16)]:
        ax.plot(t_sec, skeleton_all[:, idx, 2], label=f"{name} Z", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z Position (m)")
    ax.set_title("Wrist Z Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Right side: Shoulder/Elbow positions
    ax = axes1[0, 1]
    for name, idx in [("left_shoulder", 11), ("right_shoulder", 12)]:
        ax.plot(t_sec, skeleton_all[:, idx, 0], label=f"{name} X", alpha=0.8)
    ax.set_ylabel("X Position (m)")
    ax.set_title("Shoulder X Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes1[1, 1]
    for name, idx in [("left_elbow", 13), ("right_elbow", 14)]:
        ax.plot(t_sec, skeleton_all[:, idx, 0], label=f"{name} X", alpha=0.8)
    ax.set_ylabel("X Position (m)")
    ax.set_title("Elbow X Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes1[2, 1]
    ax.plot(t_sec, ik_errors, 'r-', alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("IK Error")
    ax.set_title("IK Solver Error Over Time")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # =========================================================================
    # FIGURE 2: Frame-to-Frame Deltas (Skeleton)
    # =========================================================================
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 10))
    fig2.suptitle(f"Human Skeleton Frame-to-Frame Changes - {episode.name}", fontsize=14, fontweight='bold')
    
    # Compute deltas
    skeleton_delta = np.diff(skeleton_all, axis=0)  # (n-1, 33, 3)
    skeleton_delta_mag = np.linalg.norm(skeleton_delta, axis=2)  # (n-1, 33)
    t_delta = t_sec[1:]
    
    # Wrist deltas
    ax = axes2[0, 0]
    for name, idx in [("left_wrist", 15), ("right_wrist", 16)]:
        ax.plot(t_delta, skeleton_delta_mag[:, idx] * 1000, label=name, alpha=0.8)
    ax.set_ylabel("Delta (mm)")
    ax.set_title("Wrist Position Change per Frame")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5mm threshold')
    
    # Elbow deltas
    ax = axes2[1, 0]
    for name, idx in [("left_elbow", 13), ("right_elbow", 14)]:
        ax.plot(t_delta, skeleton_delta_mag[:, idx] * 1000, label=name, alpha=0.8)
    ax.set_ylabel("Delta (mm)")
    ax.set_title("Elbow Position Change per Frame")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Shoulder deltas
    ax = axes2[2, 0]
    for name, idx in [("left_shoulder", 11), ("right_shoulder", 12)]:
        ax.plot(t_delta, skeleton_delta_mag[:, idx] * 1000, label=name, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Delta (mm)")
    ax.set_title("Shoulder Position Change per Frame")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Histograms of deltas
    ax = axes2[0, 1]
    wrist_deltas = np.concatenate([skeleton_delta_mag[:, 15], skeleton_delta_mag[:, 16]]) * 1000
    ax.hist(wrist_deltas, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=np.median(wrist_deltas), color='r', linestyle='--', label=f'Median: {np.median(wrist_deltas):.1f}mm')
    ax.axvline(x=np.percentile(wrist_deltas, 95), color='orange', linestyle='--', label=f'95th: {np.percentile(wrist_deltas, 95):.1f}mm')
    ax.set_xlabel("Delta (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Wrist Delta Distribution")
    ax.legend(fontsize=8)
    
    # Statistics text
    ax = axes2[1, 1]
    ax.axis('off')
    stats_text = "Frame-to-Frame Delta Statistics (mm)\n"
    stats_text += "=" * 40 + "\n\n"
    for name, idx in EE_LANDMARKS.items():
        deltas_mm = skeleton_delta_mag[:, idx] * 1000
        stats_text += f"{name}:\n"
        stats_text += f"  Mean: {np.mean(deltas_mm):.2f}  Std: {np.std(deltas_mm):.2f}\n"
        stats_text += f"  Max:  {np.max(deltas_mm):.2f}  95th: {np.percentile(deltas_mm, 95):.2f}\n\n"
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Velocity from deltas
    ax = axes2[2, 1]
    wrist_velocity = skeleton_delta_mag[:, 15] / dt  # m/s
    ax.plot(t_delta, wrist_velocity, 'b-', alpha=0.8, label='Left wrist velocity')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Left Wrist Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # =========================================================================
    # FIGURE 3: Robot Joint Angles by Group
    # =========================================================================
    fig3, axes3 = plt.subplots(3, 2, figsize=(16, 10))
    fig3.suptitle(f"Robot Joint Angles - {episode.name}", fontsize=14, fontweight='bold')
    
    # Left arm
    ax = axes3[0, 0]
    for i in JOINT_GROUPS["left_arm"]:
        ax.plot(t_sec, np.degrees(joint_angles[:, i]), label=JOINT_NAMES[i].replace("left_", ""), alpha=0.7)
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Left Arm Joints")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Right arm
    ax = axes3[0, 1]
    for i in JOINT_GROUPS["right_arm"]:
        ax.plot(t_sec, np.degrees(joint_angles[:, i]), label=JOINT_NAMES[i].replace("right_", ""), alpha=0.7)
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Right Arm Joints")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Waist
    ax = axes3[1, 0]
    for i in JOINT_GROUPS["waist"]:
        ax.plot(t_sec, np.degrees(joint_angles[:, i]), label=JOINT_NAMES[i].replace("waist_", ""), alpha=0.8, linewidth=2)
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Waist Joints")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Root position
    ax = axes3[1, 1]
    ax.plot(t_sec, root_pos[:, 0], label='X', alpha=0.8)
    ax.plot(t_sec, root_pos[:, 1], label='Y', alpha=0.8)
    ax.plot(t_sec, root_pos[:, 2], label='Z (height)', alpha=0.8)
    ax.set_ylabel("Position (m)")
    ax.set_title("Robot Root Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Left leg
    ax = axes3[2, 0]
    for i in JOINT_GROUPS["left_leg"][:4]:  # First 4 leg joints
        ax.plot(t_sec, np.degrees(joint_angles[:, i]), label=JOINT_NAMES[i].replace("left_", ""), alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Left Leg Joints")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Right leg
    ax = axes3[2, 1]
    for i in JOINT_GROUPS["right_leg"][:4]:
        ax.plot(t_sec, np.degrees(joint_angles[:, i]), label=JOINT_NAMES[i].replace("right_", ""), alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Right Leg Joints")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # =========================================================================
    # FIGURE 4: Robot Joint Velocities and Accelerations
    # =========================================================================
    fig4, axes4 = plt.subplots(3, 2, figsize=(16, 10))
    fig4.suptitle(f"Robot Joint Velocities & Accelerations - {episode.name}", fontsize=14, fontweight='bold')
    
    # Compute velocities and accelerations for arm joints
    arm_joints_idx = JOINT_GROUPS["left_arm"] + JOINT_GROUPS["right_arm"]
    
    # Left arm velocities
    ax = axes4[0, 0]
    for i in JOINT_GROUPS["left_arm"][:4]:
        vel = compute_velocity(joint_angles[:, i], dt)
        ax.plot(t_sec, vel, label=JOINT_NAMES[i].replace("left_", ""), alpha=0.7)
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Left Arm Joint Velocities")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Right arm velocities
    ax = axes4[0, 1]
    for i in JOINT_GROUPS["right_arm"][:4]:
        vel = compute_velocity(joint_angles[:, i], dt)
        ax.plot(t_sec, vel, label=JOINT_NAMES[i].replace("right_", ""), alpha=0.7)
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Right Arm Joint Velocities")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Left arm accelerations
    ax = axes4[1, 0]
    for i in JOINT_GROUPS["left_arm"][:4]:
        vel = compute_velocity(joint_angles[:, i], dt)
        acc = compute_acceleration(vel, dt)
        ax.plot(t_sec, acc, label=JOINT_NAMES[i].replace("left_", ""), alpha=0.7)
    ax.set_ylabel("Acceleration (rad/s²)")
    ax.set_title("Left Arm Joint Accelerations")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Right arm accelerations
    ax = axes4[1, 1]
    for i in JOINT_GROUPS["right_arm"][:4]:
        vel = compute_velocity(joint_angles[:, i], dt)
        acc = compute_acceleration(vel, dt)
        ax.plot(t_sec, acc, label=JOINT_NAMES[i].replace("right_", ""), alpha=0.7)
    ax.set_ylabel("Acceleration (rad/s²)")
    ax.set_title("Right Arm Joint Accelerations")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Joint angle deltas (frame-to-frame change in degrees)
    joint_delta = np.abs(np.diff(joint_angles, axis=0))  # (n-1, 29)
    
    ax = axes4[2, 0]
    # Show max delta per frame across arm joints
    arm_max_delta = np.max(joint_delta[:, arm_joints_idx], axis=1)
    ax.plot(t_delta, np.degrees(arm_max_delta), 'r-', alpha=0.8, label='Max arm joint')
    ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='2° threshold')
    ax.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, label='5° threshold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Delta (deg)")
    ax.set_title("Max Joint Angle Change per Frame (Arms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Histogram of joint deltas
    ax = axes4[2, 1]
    all_arm_deltas = np.degrees(joint_delta[:, arm_joints_idx].flatten())
    ax.hist(all_arm_deltas, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(x=np.median(all_arm_deltas), color='r', linestyle='--', 
               label=f'Median: {np.median(all_arm_deltas):.2f}°')
    ax.axvline(x=np.percentile(all_arm_deltas, 95), color='orange', linestyle='--',
               label=f'95th: {np.percentile(all_arm_deltas, 95):.2f}°')
    ax.set_xlabel("Delta (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Arm Joint Delta Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, np.percentile(all_arm_deltas, 99))
    
    plt.tight_layout()
    
    # =========================================================================
    # FIGURE 5: End Effector Analysis
    # =========================================================================
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 8))
    fig5.suptitle(f"End Effector Analysis (from Human Skeleton) - {episode.name}", fontsize=14, fontweight='bold')
    
    # Left hand trajectory (XY plane)
    ax = axes5[0, 0]
    left_wrist = skeleton_all[:, 15, :]
    scatter = ax.scatter(left_wrist[:, 0], left_wrist[:, 1], c=t_sec, cmap='viridis', s=2, alpha=0.5)
    ax.plot(left_wrist[0, 0], left_wrist[0, 1], 'go', markersize=10, label='Start')
    ax.plot(left_wrist[-1, 0], left_wrist[-1, 1], 'ro', markersize=10, label='End')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Left Wrist Trajectory (XY)")
    ax.legend(fontsize=8)
    ax.axis('equal')
    plt.colorbar(scatter, ax=ax, label='Time (s)')
    
    # Right hand trajectory (XY plane)
    ax = axes5[0, 1]
    right_wrist = skeleton_all[:, 16, :]
    scatter = ax.scatter(right_wrist[:, 0], right_wrist[:, 1], c=t_sec, cmap='viridis', s=2, alpha=0.5)
    ax.plot(right_wrist[0, 0], right_wrist[0, 1], 'go', markersize=10, label='Start')
    ax.plot(right_wrist[-1, 0], right_wrist[-1, 1], 'ro', markersize=10, label='End')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Right Wrist Trajectory (XY)")
    ax.legend(fontsize=8)
    ax.axis('equal')
    plt.colorbar(scatter, ax=ax, label='Time (s)')
    
    # Wrist height (Z) over time
    ax = axes5[1, 0]
    ax.plot(t_sec, left_wrist[:, 2], label='Left wrist Z', alpha=0.8)
    ax.plot(t_sec, right_wrist[:, 2], label='Right wrist Z', alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Wrist Heights Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Distance between hands
    ax = axes5[1, 1]
    hand_dist = np.linalg.norm(left_wrist - right_wrist, axis=1)
    ax.plot(t_sec, hand_dist, 'b-', alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance Between Hands")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # =========================================================================
    # FIGURE 6: Summary Metrics
    # =========================================================================
    fig6, axes6 = plt.subplots(2, 3, figsize=(16, 8))
    fig6.suptitle(f"Summary Metrics - {episode.name}", fontsize=14, fontweight='bold')
    
    # Velocity variance by joint group
    if metrics:
        ax = axes6[0, 0]
        group_names = list(JOINT_GROUPS.keys())
        group_vars = []
        for group_name in group_names:
            joint_indices = JOINT_GROUPS[group_name]
            vars_ = [metrics.velocity_variance[JOINT_NAMES[i]] for i in joint_indices]
            group_vars.append(np.mean(vars_))
        bars = ax.bar(group_names, group_vars, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_ylabel("Velocity Variance")
        ax.set_title("Velocity Variance by Joint Group")
        ax.tick_params(axis='x', rotation=45)
        
        # Acceleration RMS by group
        ax = axes6[0, 1]
        group_acc = []
        for group_name in group_names:
            joint_indices = JOINT_GROUPS[group_name]
            accs = [metrics.acceleration_rms[JOINT_NAMES[i]] for i in joint_indices]
            group_acc.append(np.mean(accs))
        bars = ax.bar(group_names, group_acc, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_ylabel("Acceleration RMS (rad/s²)")
        ax.set_title("Acceleration RMS by Joint Group")
        ax.tick_params(axis='x', rotation=45)
        
        # High frequency power
        ax = axes6[0, 2]
        group_hf = []
        for group_name in group_names:
            joint_indices = JOINT_GROUPS[group_name]
            hfs = [metrics.high_freq_power[JOINT_NAMES[i]] for i in joint_indices]
            group_hf.append(np.mean(hfs))
        bars = ax.bar(group_names, group_hf, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_ylabel("High-Freq Power Ratio")
        ax.set_title("High-Frequency Power (>5Hz) by Group")
        ax.tick_params(axis='x', rotation=45)
    
    # FFT of left shoulder pitch
    ax = axes6[1, 0]
    sample_joint = JOINT_GROUPS["left_arm"][0]  # left_shoulder_pitch
    vel = compute_velocity(joint_angles[:, sample_joint], dt)
    n = len(vel)
    freq = np.fft.rfftfreq(n, dt)
    fft_vel = np.fft.rfft(vel)
    power = np.abs(fft_vel) ** 2
    if power.max() > 0:
        power = power / power.max()
    ax.semilogy(freq, power, 'b-', alpha=0.8)
    ax.axvline(x=5.0, color='r', linestyle='--', alpha=0.5, label='5 Hz')
    ax.axvline(x=10.0, color='orange', linestyle='--', alpha=0.5, label='10 Hz')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Power")
    ax.set_title(f"FFT: {JOINT_NAMES[sample_joint]}")
    ax.set_xlim(0, 30)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # IK error distribution
    ax = axes6[1, 1]
    ax.hist(ik_errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(ik_errors), color='r', linestyle='--', label=f'Mean: {np.mean(ik_errors):.4f}')
    ax.set_xlabel("IK Error")
    ax.set_ylabel("Count")
    ax.set_title("IK Error Distribution")
    ax.legend(fontsize=8)
    
    # Text summary
    ax = axes6[1, 2]
    ax.axis('off')
    
    summary_text = f"Episode Summary\n"
    summary_text += "=" * 35 + "\n\n"
    summary_text += f"Name:     {episode.name}\n"
    summary_text += f"Frames:   {episode.num_frames}\n"
    summary_text += f"Duration: {episode.duration_sec:.1f}s\n"
    summary_text += f"FPS:      {actual_fps:.1f}\n"
    summary_text += f"Smoothing: {episode.smoothing}\n\n"
    
    if metrics:
        summary_text += "Aggregate Metrics\n"
        summary_text += "-" * 35 + "\n"
        summary_text += f"Vel Variance:  {metrics.mean_velocity_variance:.6f}\n"
        summary_text += f"Accel RMS:     {metrics.mean_acceleration_rms:.4f}\n"
        summary_text += f"High-Freq Pwr: {metrics.mean_high_freq_power:.4f}\n"
        summary_text += f"Smoothness:    {metrics.mean_smoothness_index:.2f}\n"
        summary_text += f"Max Velocity:  {metrics.max_joint_velocity:.2f} rad/s\n"
    
    summary_text += f"\nIK Error Stats\n"
    summary_text += "-" * 35 + "\n"
    summary_text += f"Mean:   {np.mean(ik_errors):.4f}\n"
    summary_text += f"Std:    {np.std(ik_errors):.4f}\n"
    summary_text += f"Max:    {np.max(ik_errors):.4f}\n"
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    print(f"\n[Analysis] Generated 6 figure windows. Close all to continue.")
    plt.show()


def list_episodes():
    
    if not episodes:
        print("No episodes found")
        return
    
    print(f"\n{'='*60}")
    print(f"  Available Episodes ({len(episodes)})")
    print(f"{'='*60}")
    print(f"{'Name':<30} {'Frames':>8} {'Duration':>10} {'Smoothing':<15}")
    print("-" * 60)
    
    for filepath in episodes:
        try:
            episode = TeleopEpisodeRecorder.load(filepath)
            print(f"{episode.name:<30} {episode.num_frames:>8} {episode.duration_sec:>9.1f}s {episode.smoothing:<15}")
        except Exception as e:
            print(f"{filepath.stem:<30} [Error loading: {e}]")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze teleop episode jitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all episodes
    python teleop_jitter_analysis.py --list
    
    # Analyze single episode with basic plots
    python teleop_jitter_analysis.py --episode baseline_001 --plot
    
    # Detailed time-series analysis (shows fluctuations)
    python teleop_jitter_analysis.py --episode baseline_001 --timeseries
    
    # Compare multiple episodes
    python teleop_jitter_analysis.py --compare baseline_001 baseline_002
    
    # Generate report for all episodes
    python teleop_jitter_analysis.py --report
"""
    )
    parser.add_argument("--episode", "-e", type=str, help="Episode name to analyze")
    parser.add_argument("--compare", "-c", nargs="+", help="Episode names to compare")
    parser.add_argument("--list", "-l", action="store_true", help="List all episodes")
    parser.add_argument("--report", "-r", action="store_true", help="Generate report for all episodes")
    parser.add_argument("--plot", "-p", action="store_true", help="Show basic plots")
    parser.add_argument("--timeseries", "-t", action="store_true", 
                       help="Show detailed time-series plots (skeleton, joints, velocities)")
    args = parser.parse_args()
    
    if args.list:
        list_episodes()
        return
    
    if args.compare:
        compare_episodes(args.compare)
        return
    
    if args.report:
        episodes = TeleopEpisodeRecorder.list_episodes()
        if not episodes:
            print("No episodes found")
            return
        compare_episodes([e.stem for e in episodes])
        return
    
    if args.episode:
        filepath = EPISODES_DIR / f"{args.episode}.npz"
        if not filepath.exists():
            print(f"Episode not found: {args.episode}")
            print(f"Looking in: {EPISODES_DIR}")
            return
        
        episode = TeleopEpisodeRecorder.load(filepath)
        metrics = analyze_episode(episode)
        print_metrics(metrics, episode.name)
        
        if args.timeseries and HAS_MATPLOTLIB:
            plot_timeseries_detailed(episode, metrics)
        elif args.plot and HAS_MATPLOTLIB:
            plot_episode(episode, metrics)
        return
    
    # Default: list episodes
    list_episodes()


if __name__ == "__main__":
    main()

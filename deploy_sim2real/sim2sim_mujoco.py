#!/usr/bin/env python3
"""
Sim2Sim: Test policy in MuJoCo simulation.

This script loads the trained ONNX policy and runs it in MuJoCo
to verify behavior before deploying to real robot.

Observation structure (178 dims) - matches Isaac Lab G1MotionMimicEnv:
  - base_lin_vel (3)
  - base_ang_vel (3)
  - projected_gravity (3)
  - joint_pos (37) - relative to default, Isaac Lab order
  - joint_vel (37) - Isaac Lab order
  - last_action (37)
  - target_joint_pos (37) - motion targets
  - target_keybody_pos (21) - 7 key bodies × 3D

Usage:
    python sim2sim_mujoco.py --model policy.onnx
    python sim2sim_mujoco.py --model policy.onnx --mujoco_model path/to/g1.xml
"""

import argparse
import os
import sys
import time
import numpy as np

try:
    import mujoco
    import mujoco.viewer
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("Warning: mujoco not installed. Install with: pip install mujoco")

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime")


# ============================================================================
# JOINT MAPPING: MuJoCo (29 DOF) <-> Isaac Lab (37 DOF)
# ============================================================================

# MuJoCo joint order (29 DOF)
MUJOCO_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Isaac Lab joint order (37 DOF, alphabetical from URDF)
ISAACLAB_JOINT_ORDER = [
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint",
    "left_five_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_one_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_three_joint", "left_zero_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint",
    "right_five_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_one_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_three_joint", "right_zero_joint",
    "torso_joint",
    # Additional head/other joints if present (31-36)
    "head_joint_0", "head_joint_1", "head_joint_2", "head_joint_3", "head_joint_4", "head_joint_5",
]

# MuJoCo index -> Isaac Lab index mapping
# From docs/ISAACLAB_OBSERVATION_SPEC.md
MUJOCO_TO_ISAACLAB = {
    0: 5,   # left_hip_pitch
    1: 6,   # left_hip_roll
    2: 7,   # left_hip_yaw
    3: 8,   # left_knee
    4: 0,   # left_ankle_pitch
    5: 1,   # left_ankle_roll
    6: 20,  # right_hip_pitch
    7: 21,  # right_hip_roll
    8: 22,  # right_hip_yaw
    9: 23,  # right_knee
    10: 15, # right_ankle_pitch
    11: 16, # right_ankle_roll
    12: 30, # torso_joint (waist_yaw)
    # 13, 14: waist_roll/pitch - no Isaac Lab equivalent
    15: 10, # left_shoulder_pitch
    16: 11, # left_shoulder_roll
    17: 12, # left_shoulder_yaw
    18: 2,  # left_elbow_pitch
    19: 3,  # left_elbow_roll (wrist_roll -> elbow_roll in Isaac Lab)
    # 20, 21: left_wrist_pitch/yaw - no equivalent
    22: 25, # right_shoulder_pitch
    23: 26, # right_shoulder_roll
    24: 27, # right_shoulder_yaw
    25: 17, # right_elbow_pitch
    26: 18, # right_elbow_roll
    # 27, 28: right_wrist_pitch/yaw - no equivalent
}

# Isaac Lab index -> MuJoCo index (reverse mapping)
ISAACLAB_TO_MUJOCO = {v: k for k, v in MUJOCO_TO_ISAACLAB.items()}


# ============================================================================
# DEFAULT JOINT POSITIONS
# ============================================================================

# Isaac Lab default joint positions (37 DOF)
# Using ISAACLAB_JOINT_ORDER indices (alphabetical from URDF):
#   0: left_ankle_pitch, 1: left_ankle_roll, 2: left_elbow_pitch, 3: left_elbow_roll,
#   4: left_five (hand), 5: left_hip_pitch, 6: left_hip_roll, 7: left_hip_yaw,
#   8: left_knee, 9: left_one (hand), 10: left_shoulder_pitch, etc.
ISAACLAB_DEFAULT_POS = np.zeros(37, dtype=np.float32)
# Left leg
ISAACLAB_DEFAULT_POS[0] = -0.2   # left_ankle_pitch
ISAACLAB_DEFAULT_POS[1] = 0.0    # left_ankle_roll
ISAACLAB_DEFAULT_POS[5] = 0.0    # left_hip_pitch (neutral)
ISAACLAB_DEFAULT_POS[6] = 0.0    # left_hip_roll
ISAACLAB_DEFAULT_POS[7] = 0.0    # left_hip_yaw
ISAACLAB_DEFAULT_POS[8] = 0.4    # left_knee
# Right leg
ISAACLAB_DEFAULT_POS[15] = -0.2  # right_ankle_pitch
ISAACLAB_DEFAULT_POS[16] = 0.0   # right_ankle_roll
ISAACLAB_DEFAULT_POS[20] = 0.0   # right_hip_pitch (neutral)
ISAACLAB_DEFAULT_POS[21] = 0.0   # right_hip_roll
ISAACLAB_DEFAULT_POS[22] = 0.0   # right_hip_yaw
ISAACLAB_DEFAULT_POS[23] = 0.4   # right_knee
# Torso
ISAACLAB_DEFAULT_POS[30] = 0.0   # torso_joint
# Left arm
ISAACLAB_DEFAULT_POS[10] = 0.35  # left_shoulder_pitch
ISAACLAB_DEFAULT_POS[11] = 0.16  # left_shoulder_roll
ISAACLAB_DEFAULT_POS[12] = 0.0   # left_shoulder_yaw
ISAACLAB_DEFAULT_POS[2] = 0.52   # left_elbow_pitch
ISAACLAB_DEFAULT_POS[3] = 0.0    # left_elbow_roll
# Right arm
ISAACLAB_DEFAULT_POS[25] = 0.35  # right_shoulder_pitch
ISAACLAB_DEFAULT_POS[26] = 0.16  # right_shoulder_roll
ISAACLAB_DEFAULT_POS[27] = 0.0   # right_shoulder_yaw
ISAACLAB_DEFAULT_POS[17] = 0.52  # right_elbow_pitch
ISAACLAB_DEFAULT_POS[18] = 0.0   # right_elbow_roll

# MuJoCo default joint positions (29 DOF)
# Using MUJOCO_JOINT_ORDER indices:
#   0: left_hip_pitch, 1: left_hip_roll, 2: left_hip_yaw, 3: left_knee,
#   4: left_ankle_pitch, 5: left_ankle_roll, 6-11: right leg, 12-14: waist,
#   15-21: left arm, 22-28: right arm
MUJOCO_DEFAULT_POS = np.zeros(29, dtype=np.float32)
# Left leg
MUJOCO_DEFAULT_POS[0] = 0.0    # left_hip_pitch
MUJOCO_DEFAULT_POS[1] = 0.0    # left_hip_roll
MUJOCO_DEFAULT_POS[2] = 0.0    # left_hip_yaw
MUJOCO_DEFAULT_POS[3] = 0.4    # left_knee
MUJOCO_DEFAULT_POS[4] = -0.2   # left_ankle_pitch
MUJOCO_DEFAULT_POS[5] = 0.0    # left_ankle_roll
# Right leg
MUJOCO_DEFAULT_POS[6] = 0.0    # right_hip_pitch
MUJOCO_DEFAULT_POS[7] = 0.0    # right_hip_roll
MUJOCO_DEFAULT_POS[8] = 0.0    # right_hip_yaw
MUJOCO_DEFAULT_POS[9] = 0.4    # right_knee
MUJOCO_DEFAULT_POS[10] = -0.2  # right_ankle_pitch
MUJOCO_DEFAULT_POS[11] = 0.0   # right_ankle_roll
# Waist
MUJOCO_DEFAULT_POS[12] = 0.0   # waist_yaw
MUJOCO_DEFAULT_POS[13] = 0.0   # waist_roll
MUJOCO_DEFAULT_POS[14] = 0.0   # waist_pitch
# Left arm
MUJOCO_DEFAULT_POS[15] = 0.35  # left_shoulder_pitch
MUJOCO_DEFAULT_POS[16] = 0.16  # left_shoulder_roll
MUJOCO_DEFAULT_POS[17] = 0.0   # left_shoulder_yaw
MUJOCO_DEFAULT_POS[18] = 0.52  # left_elbow
MUJOCO_DEFAULT_POS[19] = 0.0   # left_wrist_roll
MUJOCO_DEFAULT_POS[20] = 0.0   # left_wrist_pitch
MUJOCO_DEFAULT_POS[21] = 0.0   # left_wrist_yaw
# Right arm
MUJOCO_DEFAULT_POS[22] = 0.35  # right_shoulder_pitch
MUJOCO_DEFAULT_POS[23] = 0.16  # right_shoulder_roll
MUJOCO_DEFAULT_POS[24] = 0.0   # right_shoulder_yaw
MUJOCO_DEFAULT_POS[25] = 0.52  # right_elbow
MUJOCO_DEFAULT_POS[26] = 0.0   # right_wrist_roll
MUJOCO_DEFAULT_POS[27] = 0.0   # right_wrist_pitch
MUJOCO_DEFAULT_POS[28] = 0.0   # right_wrist_yaw


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quat_to_euler(quat):
    """Convert quaternion (w,x,y,z) to euler angles (roll, pitch, yaw)."""
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q (w,x,y,z format)."""
    w, x, y, z = q
    
    # Quaternion conjugate for inverse rotation
    q_vec = np.array([x, y, z])
    
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * (2.0 * w)
    c = q_vec * (2.0 * np.dot(q_vec, v))
    
    return a - b + c


def compute_projected_gravity(quat):
    """Compute gravity vector in body frame from quaternion (w,x,y,z)."""
    # World gravity direction (normalized)
    gravity_world = np.array([0.0, 0.0, -1.0])
    
    # Rotate gravity to body frame
    proj_gravity = quat_rotate_inverse(quat, gravity_world)
    
    return proj_gravity.astype(np.float32)


def mujoco_to_isaaclab_joints(mujoco_joints):
    """Convert 29-DOF MuJoCo joints to 37-DOF Isaac Lab order."""
    isaaclab_joints = np.zeros(37, dtype=np.float32)
    
    for mj_idx, il_idx in MUJOCO_TO_ISAACLAB.items():
        if mj_idx < len(mujoco_joints):
            isaaclab_joints[il_idx] = mujoco_joints[mj_idx]
    
    return isaaclab_joints


def isaaclab_to_mujoco_joints(isaaclab_joints):
    """Convert 37-DOF Isaac Lab joints to 29-DOF MuJoCo order."""
    mujoco_joints = np.zeros(29, dtype=np.float32)
    
    for il_idx, mj_idx in ISAACLAB_TO_MUJOCO.items():
        if il_idx < len(isaaclab_joints):
            mujoco_joints[mj_idx] = isaaclab_joints[il_idx]
    
    return mujoco_joints


# ============================================================================
# SIMULATION CLASS
# ============================================================================

class G1Sim2Sim:
    """Run trained Isaac Lab policy in MuJoCo simulation."""
    
    def __init__(self, policy_path: str, mujoco_model_path: str):
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime is required")
        if not HAS_MUJOCO:
            raise RuntimeError("mujoco is required")
        
        # Load policy
        self.session = ort.InferenceSession(policy_path)
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        self.obs_dim = input_info.shape[1]
        self.act_dim = output_info.shape[1]
        print(f"Policy loaded: obs_dim={self.obs_dim}, act_dim={self.act_dim}")
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        
        # Set simulation parameters
        self.model.opt.timestep = 0.001  # 1kHz physics
        self.control_dt = 0.02  # 50Hz control (matches Isaac Lab default)
        self.steps_per_control = int(self.control_dt / self.model.opt.timestep)
        
        print(f"MuJoCo model loaded:")
        print(f"  DOFs: {self.model.nq}")
        print(f"  Actuators: {self.model.nu}")
        print(f"  Physics dt: {self.model.opt.timestep}")
        print(f"  Control dt: {self.control_dt}")
        
        # PD control gains (from server_low_level_g1_sim.py)
        self.kp = np.array([
            100, 100, 100, 150, 40, 40,   # left leg
            100, 100, 100, 150, 40, 40,   # right leg
            150, 150, 150,                 # waist
            40, 40, 40, 40, 4.0, 4.0, 4.0, # left arm
            40, 40, 40, 40, 4.0, 4.0, 4.0, # right arm
        ], dtype=np.float32)
        
        self.kd = np.array([
            2, 2, 2, 4, 2, 2,
            2, 2, 2, 4, 2, 2,
            4, 4, 4,
            5, 5, 5, 5, 0.2, 0.2, 0.2,
            5, 5, 5, 5, 0.2, 0.2, 0.2,
        ], dtype=np.float32)
        
        self.torque_limits = np.array([
            100, 100, 100, 150, 40, 40,
            100, 100, 100, 150, 40, 40,
            150, 150, 150,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
        ], dtype=np.float32)
        
        # Action scale
        self.action_scale = 0.5
        
        # State
        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        
        # Target pose - use default standing pose as target
        # The policy was trained to track motion targets, so we give it the default pose
        self.target_joint_pos_il = ISAACLAB_DEFAULT_POS.copy()
        
        # Target key body positions (7 bodies × 3D = 21 dims)
        # These are relative positions in local frame
        # For standing, approximate positions based on robot kinematics
        self.target_keybody_pos = np.zeros(21, dtype=np.float32)
        # Feet (roughly at ankle height below pelvis)
        self.target_keybody_pos[0:3] = [0.0, 0.1, -0.75]   # left foot
        self.target_keybody_pos[3:6] = [0.0, -0.1, -0.75]  # right foot
        # Elbows (roughly at arm positions)
        self.target_keybody_pos[6:9] = [0.15, 0.3, 0.1]    # left elbow
        self.target_keybody_pos[9:12] = [0.15, -0.3, 0.1]  # right elbow
        # Shoulders
        self.target_keybody_pos[12:15] = [0.0, 0.15, 0.25]  # left shoulder
        self.target_keybody_pos[15:18] = [0.0, -0.15, 0.25] # right shoulder
        # Torso
        self.target_keybody_pos[18:21] = [0.0, 0.0, 0.15]   # torso
        
        self.debug = True
    
    def reset(self):
        """Reset simulation to standing pose."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial pose
        # qpos: [x, y, z, qw, qx, qy, qz, joint1, joint2, ...]
        self.data.qpos[0:3] = [0, 0, 1.0]  # Position (INIT_HEIGHT=1.0m matches training)
        self.data.qpos[3:7] = [1, 0, 0, 0]   # Quaternion (w, x, y, z)
        self.data.qpos[7:7+29] = MUJOCO_DEFAULT_POS
        
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        
        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
    
    def get_state(self):
        """Extract robot state from MuJoCo."""
        quat = self.data.qpos[3:7]  # w, x, y, z
        joint_pos = self.data.qpos[7:7+29]
        joint_vel = self.data.qvel[6:6+29]
        
        # Base velocities in world frame
        lin_vel_world = self.data.qvel[0:3]
        ang_vel_world = self.data.qvel[3:6]
        
        # Rotate to body frame
        lin_vel_body = quat_rotate_inverse(quat, lin_vel_world)
        ang_vel_body = quat_rotate_inverse(quat, ang_vel_world)
        
        return {
            'quat': quat,
            'joint_pos': joint_pos,
            'joint_vel': joint_vel,
            'lin_vel_body': lin_vel_body,
            'ang_vel_body': ang_vel_body,
            'base_height': self.data.qpos[2],
        }
    
    def build_observation(self, state):
        """
        Build 178-dim observation vector matching Isaac Lab G1MotionMimicEnv.
        
        Structure:
          [0-2]    base_lin_vel (3)
          [3-5]    base_ang_vel (3)
          [6-8]    projected_gravity (3)
          [9-45]   joint_pos relative to default (37)
          [46-82]  joint_vel (37)
          [83-119] last_action (37)
          [120-156] target_joint_pos (37)
          [157-177] target_keybody_pos (21)
        """
        # Convert MuJoCo joints to Isaac Lab order
        joint_pos_il = mujoco_to_isaaclab_joints(state['joint_pos'])
        joint_vel_il = mujoco_to_isaaclab_joints(state['joint_vel'])
        
        # Relative joint positions
        joint_pos_rel = joint_pos_il - ISAACLAB_DEFAULT_POS
        
        # Projected gravity
        proj_gravity = compute_projected_gravity(state['quat'])
        
        # Build observation
        obs = np.concatenate([
            state['lin_vel_body'].astype(np.float32),   # 0-2
            state['ang_vel_body'].astype(np.float32),   # 3-5
            proj_gravity,                                # 6-8
            joint_pos_rel,                               # 9-45
            joint_vel_il,                                # 46-82
            self.last_action,                            # 83-119
            self.target_joint_pos_il,                    # 120-156
            self.target_keybody_pos,                     # 157-177
        ])
        
        # Verify dimension
        assert obs.shape[0] == 178, f"Expected 178 obs dims, got {obs.shape[0]}"
        
        return obs
    
    def forward(self, observation):
        """Run policy inference."""
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        outputs = self.session.run(
            None,
            {'observation': observation.astype(np.float32)}
        )
        
        return outputs[0].squeeze()
    
    def apply_action(self, action, state):
        """Apply action using PD control."""
        # Clip action to [-1, 1] range (policy can output larger values)
        action = np.clip(action, -1.0, 1.0)
        
        # Update last_action with clipped value (this is what next obs should use)
        self.last_action = action.astype(np.float32)
        
        # Action is position delta in Isaac Lab order (37 DOF)
        # Convert to absolute position target
        target_pos_il = ISAACLAB_DEFAULT_POS + action * self.action_scale
        
        # Convert to MuJoCo order (29 DOF)
        target_pos_mj = isaaclab_to_mujoco_joints(target_pos_il)
        
        # Fill in default positions for unmapped joints
        # This ensures joints without Isaac Lab mapping use defaults
        for i in range(29):
            if target_pos_mj[i] == 0 and MUJOCO_DEFAULT_POS[i] != 0:
                target_pos_mj[i] = MUJOCO_DEFAULT_POS[i]
        
        # PD control
        pos_error = target_pos_mj - state['joint_pos']
        vel = state['joint_vel']
        
        torque = pos_error * self.kp - vel * self.kd
        torque = np.clip(torque, -self.torque_limits, self.torque_limits)
        
        self.data.ctrl[:29] = torque
    
    def check_fallen(self, state):
        """Check if robot has fallen."""
        height = state['base_height']
        rpy = quat_to_euler(state['quat'])
        roll, pitch = rpy[0], rpy[1]
        
        if height < 0.4:
            return True, f"height={height:.2f}m"
        if abs(roll) > 0.8:
            return True, f"roll={np.degrees(roll):.1f}°"
        if abs(pitch) > 0.8:
            return True, f"pitch={np.degrees(pitch):.1f}°"
        
        return False, None
    
    def run(self, duration: float = 30.0):
        """Run simulation with viewer."""
        print(f"\nStarting sim2sim simulation for {duration}s...")
        print("Press Ctrl+C or close viewer to stop\n")
        
        self.reset()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            step_count = 0
            policy_count = 0
            
            while viewer.is_running() and (time.time() - start_time) < duration:
                t_start = time.time()
                
                state = self.get_state()
                
                # Check for falls
                fallen, reason = self.check_fallen(state)
                if fallen:
                    print(f"Robot fell! ({reason}) - Resetting...")
                    self.reset()
                    time.sleep(0.1)
                    continue
                
                # Run policy at control frequency
                if step_count % self.steps_per_control == 0:
                    obs = self.build_observation(state)
                    action = self.forward(obs)
                    self.last_action = action.copy()
                    policy_count += 1
                    
                    # Print status periodically
                    if policy_count % 50 == 0:
                        print(f"Step {policy_count}: height={state['base_height']:.3f}m, "
                              f"roll={np.degrees(quat_to_euler(state['quat'])[0]):.1f}°")
                
                # Apply action via PD control
                self.apply_action(self.last_action, state)
                
                # Step physics
                mujoco.mj_step(self.model, self.data)
                step_count += 1
                
                # Sync viewer
                viewer.sync()
                
                # Real-time pacing
                elapsed = time.time() - t_start
                if elapsed < self.model.opt.timestep:
                    time.sleep(self.model.opt.timestep - elapsed)
        
        print(f"\nSimulation completed: {policy_count} policy steps")


def find_g1_model():
    """Find G1 MuJoCo model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_paths = [
        os.path.join(script_dir, "assets/g1/g1_sim2sim_29dof.xml"),
        os.path.join(script_dir, "../assets/g1/g1_sim2sim_29dof.xml"),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim: Test policy in MuJoCo")
    parser.add_argument("--model", type=str, default="policy_stage4_82000.onnx",
                        help="Path to ONNX policy model")
    parser.add_argument("--mujoco_model", type=str, default=None,
                        help="Path to MuJoCo XML model")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Simulation duration in seconds")
    
    args = parser.parse_args()
    
    if not HAS_MUJOCO:
        print("Error: mujoco is required. Install with: pip install mujoco")
        sys.exit(1)
    
    if not HAS_ONNX:
        print("Error: onnxruntime is required. Install with: pip install onnxruntime")
        sys.exit(1)
    
    if args.mujoco_model is None:
        args.mujoco_model = find_g1_model()
        if args.mujoco_model is None:
            print("Error: Could not find G1 MuJoCo model.")
            sys.exit(1)
    
    if not os.path.exists(args.mujoco_model):
        print(f"Error: MuJoCo model not found: {args.mujoco_model}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Policy model not found: {args.model}")
        sys.exit(1)
    
    sim = G1Sim2Sim(args.model, args.mujoco_model)
    sim.run(args.duration)


if __name__ == "__main__":
    main()

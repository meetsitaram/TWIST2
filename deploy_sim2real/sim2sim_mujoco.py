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
    python sim2sim_mujoco.py --model policy.onnx --motion sample_motions/open-doors.pkl
"""

import argparse
import os
import pickle
import sys
import time
from types import ModuleType
import numpy as np

# Add TWIST2 root for importing robot_config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWIST2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TWIST2_ROOT)

# Patch sys.modules to handle numpy version differences when loading pickles
class _FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)

if 'numpy._core' not in sys.modules:
    sys.modules['numpy._core'] = _FakeModule('numpy._core', np.core if hasattr(np, 'core') else np)
if 'numpy._core.multiarray' not in sys.modules:
    sys.modules['numpy._core.multiarray'] = _FakeModule('numpy._core.multiarray', getattr(np.core, 'multiarray', None))

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

# Import centralized robot config for consistent joint mapping
try:
    from robot_config import G1RobotConfig
    HAS_ROBOT_CONFIG = True
except ImportError:
    HAS_ROBOT_CONFIG = False
    print("Warning: robot_config.py not found, using fallback mappings")


# ============================================================================
# JOINT MAPPING: MuJoCo (29 DOF) <-> Isaac Lab (37 DOF)
# ============================================================================

# Use centralized config if available
if HAS_ROBOT_CONFIG:
    MUJOCO_JOINT_ORDER = G1RobotConfig.MUJOCO_JOINT_ORDER
else:
    # Fallback MuJoCo joint order (29 DOF)
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
# This is the actual order from the Isaac Lab G1 robot model
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

# Build mapping using centralized config
if HAS_ROBOT_CONFIG:
    MUJOCO_TO_ISAACLAB = G1RobotConfig.build_mujoco_to_isaaclab_mapping(ISAACLAB_JOINT_ORDER)
    # Filter out None values for the index mapping
    MUJOCO_TO_ISAACLAB = {k: v for k, v in MUJOCO_TO_ISAACLAB.items() if v is not None}
else:
    # Fallback mapping
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
        15: 10, # left_shoulder_pitch
        16: 11, # left_shoulder_roll
        17: 12, # left_shoulder_yaw
        18: 2,  # left_elbow_pitch
        19: 3,  # left_elbow_roll
        22: 25, # right_shoulder_pitch
        23: 26, # right_shoulder_roll
        24: 27, # right_shoulder_yaw
        25: 17, # right_elbow_pitch
        26: 18, # right_elbow_roll
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
# MOTION LOADER
# ============================================================================

class MotionLoader:
    """Simple motion loader for teleop pkl files."""
    
    def __init__(self, motion_file: str):
        """
        Load motion from a pkl file.
        
        Motion pkl format:
            - fps: frames per second
            - root_pos: (num_frames, 3) root position
            - root_rot: (num_frames, 4) root rotation quaternion [x, y, z, w]
            - dof_pos: (num_frames, num_joints) joint positions (29 DOF MuJoCo order)
            - local_body_pos: (num_frames, num_bodies, 3) body positions (optional)
            - link_body_list: list of body names
        """
        print(f"Loading motion from {motion_file}...")
        with open(motion_file, 'rb') as f:
            motion = pickle.load(f)
        
        self.fps = motion.get("fps", 50.0)
        self.dt = 1.0 / self.fps
        
        # Core motion data
        self.root_pos = motion["root_pos"].astype(np.float32)
        self.root_rot = motion["root_rot"].astype(np.float32)  # [x, y, z, w] format
        self.dof_pos = motion["dof_pos"].astype(np.float32)    # 29 DOF MuJoCo order
        
        # Optional key body positions
        self.local_body_pos = None
        self.body_names = None
        if "local_body_pos" in motion:
            self.local_body_pos = motion["local_body_pos"].astype(np.float32)
        if "link_body_list" in motion:
            self.body_names = motion["link_body_list"]
        
        self.num_frames = self.dof_pos.shape[0]
        self.duration = self.num_frames / self.fps
        self.num_joints = self.dof_pos.shape[1]
        
        # Compute joint velocities via finite difference
        self.dof_vel = np.zeros_like(self.dof_pos)
        self.dof_vel[1:] = (self.dof_pos[1:] - self.dof_pos[:-1]) / self.dt
        self.dof_vel[0] = self.dof_vel[1]
        
        print(f"  Motion loaded: {self.num_frames} frames, {self.duration:.2f}s, {self.num_joints} DOF")
        print(f"  FPS: {self.fps}")
        if self.body_names:
            print(f"  Bodies: {len(self.body_names)}")
    
    def get_state_at_time(self, t: float, loop: bool = True) -> dict:
        """
        Get motion state at a given time.
        
        Args:
            t: Time in seconds
            loop: If True, loop the motion; otherwise clamp to last frame
            
        Returns:
            Dict with dof_pos, dof_vel, root_pos, root_rot, keybody_pos (if available)
        """
        if loop:
            t = t % self.duration
        else:
            t = min(t, self.duration - self.dt)
        
        # Compute frame with linear interpolation
        frame_f = t * self.fps
        frame_0 = int(frame_f)
        frame_1 = min(frame_0 + 1, self.num_frames - 1)
        blend = frame_f - frame_0
        
        frame_0 = max(0, min(frame_0, self.num_frames - 1))
        
        # Interpolate
        dof_pos = (1 - blend) * self.dof_pos[frame_0] + blend * self.dof_pos[frame_1]
        dof_vel = (1 - blend) * self.dof_vel[frame_0] + blend * self.dof_vel[frame_1]
        root_pos = (1 - blend) * self.root_pos[frame_0] + blend * self.root_pos[frame_1]
        
        # Quaternion interpolation (linear + normalize)
        rot_0 = self.root_rot[frame_0]
        rot_1 = self.root_rot[frame_1]
        # Handle quaternion sign
        if np.dot(rot_0, rot_1) < 0:
            rot_1 = -rot_1
        root_rot = (1 - blend) * rot_0 + blend * rot_1
        root_rot = root_rot / np.linalg.norm(root_rot)
        
        result = {
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "root_pos": root_pos,
            "root_rot": root_rot,  # [x, y, z, w] format from pkl
        }
        
        if self.local_body_pos is not None:
            keybody_0 = self.local_body_pos[frame_0]
            keybody_1 = self.local_body_pos[frame_1]
            result["keybody_pos"] = (1 - blend) * keybody_0 + blend * keybody_1
        
        return result
    
    def get_keybody_pos_for_obs(self, keybody_pos: np.ndarray, key_body_indices: list) -> np.ndarray:
        """
        Extract key body positions for observation.
        
        Args:
            keybody_pos: (num_bodies, 3) all body positions
            key_body_indices: List of body indices to extract
            
        Returns:
            (num_key_bodies * 3,) flattened key body positions
        """
        if keybody_pos is None or len(key_body_indices) == 0:
            return np.zeros(21, dtype=np.float32)
        
        selected = keybody_pos[key_body_indices]  # (num_key_bodies, 3)
        return selected.flatten().astype(np.float32)


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


def mujoco_to_isaaclab_joints(mujoco_joints, default_pos=None):
    """Convert 29-DOF MuJoCo joints to 37-DOF Isaac Lab order.
    
    Args:
        mujoco_joints: 29 DOF MuJoCo joint positions
        default_pos: Optional default positions for unmapped joints
        
    Returns:
        37 DOF Isaac Lab joint positions
    """
    if HAS_ROBOT_CONFIG:
        if default_pos is None:
            default_pos = ISAACLAB_DEFAULT_POS
        return G1RobotConfig.remap_mujoco_to_isaaclab_numpy(
            mujoco_joints,
            ISAACLAB_JOINT_ORDER,
            default_pos
        )
    else:
        # Fallback implementation
        if default_pos is None:
            isaaclab_joints = np.zeros(37, dtype=np.float32)
        else:
            isaaclab_joints = default_pos.copy()
        
        for mj_idx, il_idx in MUJOCO_TO_ISAACLAB.items():
            if mj_idx < len(mujoco_joints):
                isaaclab_joints[il_idx] = mujoco_joints[mj_idx]
        
        return isaaclab_joints


def isaaclab_to_mujoco_joints(isaaclab_joints, default_pos=None):
    """Convert 37-DOF Isaac Lab joints to 29-DOF MuJoCo order.
    
    Args:
        isaaclab_joints: 37 DOF Isaac Lab joint positions
        default_pos: Optional default positions for unmapped joints
        
    Returns:
        29 DOF MuJoCo joint positions
    """
    if HAS_ROBOT_CONFIG:
        if default_pos is None:
            default_pos = MUJOCO_DEFAULT_POS
        return G1RobotConfig.remap_isaaclab_to_mujoco_numpy(
            isaaclab_joints,
            ISAACLAB_JOINT_ORDER,
            default_pos
        )
    else:
        # Fallback implementation
        if default_pos is None:
            mujoco_joints = np.zeros(29, dtype=np.float32)
        else:
            mujoco_joints = default_pos.copy()
        
        for il_idx, mj_idx in ISAACLAB_TO_MUJOCO.items():
            if il_idx < len(isaaclab_joints):
                mujoco_joints[mj_idx] = isaaclab_joints[il_idx]
        
        return mujoco_joints


# ============================================================================
# SIMULATION CLASS
# ============================================================================

class G1Sim2Sim:
    """Run trained Isaac Lab policy in MuJoCo simulation."""
    
    # Key body names used in training (from Isaac Lab config)
    KEY_BODY_NAMES = [
        "left_ankle_roll_link",   # left foot
        "right_ankle_roll_link",  # right foot
        "left_elbow_link",        # left elbow
        "right_elbow_link",       # right elbow
        "left_shoulder_yaw_link", # left shoulder
        "right_shoulder_yaw_link",# right shoulder
        "torso_link",             # torso
    ]
    
    def __init__(self, policy_path: str, mujoco_model_path: str, motion_file: str = None):
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
        
        # Load motion if provided
        self.motion = None
        self.motion_time = 0.0
        self.key_body_indices = []
        if motion_file:
            self.motion = MotionLoader(motion_file)
            # Find key body indices in motion data
            if self.motion.body_names:
                for name in self.KEY_BODY_NAMES:
                    if name in self.motion.body_names:
                        self.key_body_indices.append(self.motion.body_names.index(name))
                    else:
                        # Try without _link suffix
                        alt_name = name.replace("_link", "")
                        if alt_name in self.motion.body_names:
                            self.key_body_indices.append(self.motion.body_names.index(alt_name))
                        else:
                            print(f"  Warning: key body '{name}' not found in motion")
                print(f"  Key body indices: {self.key_body_indices}")
        
        # PD control gains (from server_low_level_g1_sim.py)
        # PD gains matched to Isaac Lab G1_MINIMAL_CFG (from isaaclab_assets/robots/unitree.py)
        # Legs: hip_yaw/roll=150, hip_pitch/knee=200, ankle=20
        # Arms: all=40
        # Damping: legs=5, feet=2, arms=10
        self.kp = np.array([
            200, 150, 150, 200, 20, 20,   # left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
            200, 150, 150, 200, 20, 20,   # right leg
            200, 200, 200,                 # waist (torso_joint uses 200)
            40, 40, 40, 40, 40, 40, 40,   # left arm: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
            40, 40, 40, 40, 40, 40, 40,   # right arm
        ], dtype=np.float32)
        
        self.kd = np.array([
            5, 5, 5, 5, 2, 2,             # left leg
            5, 5, 5, 5, 2, 2,             # right leg
            5, 5, 5,                       # waist
            10, 10, 10, 10, 10, 10, 10,   # left arm (Isaac Lab uses 10 for arms)
            10, 10, 10, 10, 10, 10, 10,   # right arm
        ], dtype=np.float32)
        
        # Torque limits from Isaac Lab (effort_limit_sim): legs=300, feet=20, arms=300
        self.torque_limits = np.array([
            300, 300, 300, 300, 20, 20,   # left leg
            300, 300, 300, 300, 20, 20,   # right leg
            300, 300, 300,                 # waist (same as legs in Isaac Lab)
            300, 300, 300, 300, 300, 300, 300,  # left arm (effort_limit_sim=300)
            300, 300, 300, 300, 300, 300, 300,  # right arm
        ], dtype=np.float32)
        
        # Action scale
        self.action_scale = 0.5
        
        # State
        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        
        # Target pose - use default standing pose as target (updated per-frame if motion is loaded)
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
    
    def reset(self, start_from_motion: bool = True):
        """Reset simulation to standing pose or first frame of motion."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset motion time
        self.motion_time = 0.0
        
        # Set initial pose from motion or default
        if self.motion and start_from_motion:
            motion_state = self.motion.get_state_at_time(0.0)
            # Motion root_rot is [x, y, z, w], MuJoCo wants [w, x, y, z]
            root_rot_xyzw = motion_state["root_rot"]
            root_rot_wxyz = np.array([root_rot_xyzw[3], root_rot_xyzw[0], root_rot_xyzw[1], root_rot_xyzw[2]])
            
            # Start with default pose
            init_dof_pos = MUJOCO_DEFAULT_POS.copy()
            
            # Overlay motion data, but keep default for zero joints (upper-body only motions)
            motion_dof = motion_state["dof_pos"]
            leg_joints = slice(0, 12)  # First 12 joints are legs
            
            # For upper body only motions, legs are zeros - use defaults
            if np.allclose(motion_dof[leg_joints], 0, atol=0.01):
                # Keep default legs, use motion for upper body
                init_dof_pos[12:] = motion_dof[12:]
                print(f"  Upper-body only motion detected - using default leg pose")
            else:
                # Full body motion
                init_dof_pos = motion_dof
            
            # Use motion XY but ensure proper standing height (training uses ~1.0m)
            init_pos = motion_state["root_pos"].copy()
            if init_pos[2] < 0.9:
                init_pos[2] = 1.0  # Override low height with standing height
                print(f"  Overriding motion height {motion_state['root_pos'][2]:.2f}m -> 1.0m")
            
            self.data.qpos[0:3] = init_pos
            self.data.qpos[3:7] = root_rot_wxyz
            self.data.qpos[7:7+29] = init_dof_pos
            print(f"  Initialized from motion frame 0: pos={init_pos}")
        else:
            # Set initial pose
            # qpos: [x, y, z, qw, qx, qy, qz, joint1, joint2, ...]
            self.data.qpos[0:3] = [0, 0, 1.0]  # Position (INIT_HEIGHT=1.0m matches training)
            self.data.qpos[3:7] = [1, 0, 0, 0]   # Quaternion (w, x, y, z)
            self.data.qpos[7:7+29] = MUJOCO_DEFAULT_POS
        
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        
        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        
        # Update targets from first frame if using motion
        if self.motion:
            self.update_targets_from_motion()
    
    def update_targets_from_motion(self):
        """Update target joint positions and key body positions from motion.
        
        Only shoulder and elbow joints are taken from motion data.
        All other joints use default positions.
        """
        if not self.motion:
            return
        
        motion_state = self.motion.get_state_at_time(self.motion_time, loop=True)
        
        # Motion dof_pos is in MuJoCo order (29 DOF)
        motion_mujoco = motion_state["dof_pos"]
        
        # Start with default positions
        target_mujoco = MUJOCO_DEFAULT_POS.copy()
        
        # Only copy shoulder and elbow joints from motion (MuJoCo indices):
        # Left arm:  15=shoulder_pitch, 16=shoulder_roll, 17=shoulder_yaw, 18=elbow
        # Right arm: 22=shoulder_pitch, 23=shoulder_roll, 24=shoulder_yaw, 25=elbow
        arm_indices_mj = [15, 16, 17, 18, 22, 23, 24, 25]
        for idx in arm_indices_mj:
            target_mujoco[idx] = motion_mujoco[idx]
        
        # Convert to Isaac Lab order (37 DOF)
        self.target_joint_pos_il = mujoco_to_isaaclab_joints(target_mujoco)
        
        # Update key body positions if available
        if "keybody_pos" in motion_state and len(self.key_body_indices) > 0:
            self.target_keybody_pos = self.motion.get_keybody_pos_for_obs(
                motion_state["keybody_pos"], 
                self.key_body_indices
            )
    
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
    
    def print_observation(self, obs):
        """Print detailed breakdown of observation vector."""
        print("\n" + "="*70)
        print("OBSERVATION BREAKDOWN (178 dims)")
        print("="*70)
        
        # Base velocities
        print(f"\n[0-2] Base Linear Velocity (body frame):")
        print(f"  x={obs[0]:.4f}, y={obs[1]:.4f}, z={obs[2]:.4f}")
        
        print(f"\n[3-5] Base Angular Velocity (body frame):")
        print(f"  x={obs[3]:.4f}, y={obs[4]:.4f}, z={obs[5]:.4f}")
        
        print(f"\n[6-8] Projected Gravity:")
        print(f"  x={obs[6]:.4f}, y={obs[7]:.4f}, z={obs[8]:.4f}")
        
        # Joint positions relative to default
        print(f"\n[9-45] Joint Positions (relative to default) - 37 DOF Isaac Lab order:")
        joint_pos_rel = obs[9:46]
        for i, name in enumerate(ISAACLAB_JOINT_ORDER):
            if abs(joint_pos_rel[i]) > 0.01:  # Only print non-zero
                print(f"  [{i:2d}] {name}: {joint_pos_rel[i]:.4f}")
        
        # Joint velocities
        print(f"\n[46-82] Joint Velocities - 37 DOF Isaac Lab order:")
        joint_vel = obs[46:83]
        non_zero_vels = [(i, name, joint_vel[i]) for i, name in enumerate(ISAACLAB_JOINT_ORDER) if abs(joint_vel[i]) > 0.1]
        if non_zero_vels:
            for i, name, val in non_zero_vels:
                print(f"  [{i:2d}] {name}: {val:.4f}")
        else:
            print("  (all near zero)")
        
        # Last action
        print(f"\n[83-119] Last Action - 37 DOF:")
        last_action = obs[83:120]
        non_zero_act = [(i, name, last_action[i]) for i, name in enumerate(ISAACLAB_JOINT_ORDER) if abs(last_action[i]) > 0.01]
        if non_zero_act:
            for i, name, val in non_zero_act:
                print(f"  [{i:2d}] {name}: {val:.4f}")
        else:
            print("  (all near zero)")
        
        # Target joint positions
        print(f"\n[120-156] Target Joint Positions - 37 DOF (MOTION TARGETS):")
        target_pos = obs[120:157]
        print("  Key arm joints:")
        arm_indices = {
            10: 'left_shoulder_pitch', 11: 'left_shoulder_roll', 12: 'left_shoulder_yaw',
            2: 'left_elbow_pitch', 3: 'left_elbow_roll',
            25: 'right_shoulder_pitch', 26: 'right_shoulder_roll', 27: 'right_shoulder_yaw',
            17: 'right_elbow_pitch', 18: 'right_elbow_roll'
        }
        for idx, name in arm_indices.items():
            default = ISAACLAB_DEFAULT_POS[idx]
            print(f"  [{idx:2d}] {name}: {target_pos[idx]:.4f} (default: {default:.4f}, diff: {target_pos[idx]-default:.4f})")
        
        # Target keybody positions
        print(f"\n[157-177] Target Keybody Positions - 7 bodies x 3D = 21:")
        target_keybody = obs[157:178]
        body_names = [
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_elbow_link", "right_elbow_link",
            "left_shoulder_yaw_link", "right_shoulder_yaw_link",
            "torso_link"
        ]
        for i, name in enumerate(body_names):
            x, y, z = target_keybody[i*3:(i+1)*3]
            print(f"  [{i}] {name}: ({x:.4f}, {y:.4f}, {z:.4f})")
        
        print("="*70 + "\n")
    
    def print_arm_debug(self, obs, action, state, step):
        """Print focused debug info for arm joints."""
        # Arm joint indices in Isaac Lab order
        arm_il = [
            (10, 'L_sh_pitch'), (11, 'L_sh_roll'), (12, 'L_sh_yaw'), (2, 'L_elbow'),
            (25, 'R_sh_pitch'), (26, 'R_sh_roll'), (27, 'R_sh_yaw'), (17, 'R_elbow')
        ]
        # Corresponding MuJoCo indices
        arm_mj = [15, 16, 17, 18, 22, 23, 24, 25]
        
        print(f"\n--- Step {step} ARM DEBUG ---")
        print(f"{'Joint':<12} | {'Target':>8} | {'Current':>8} | {'Action':>8} | {'Result':>8}")
        print("-" * 60)
        
        for (il_idx, name), mj_idx in zip(arm_il, arm_mj):
            # Target from observation [120-156]
            target = obs[120 + il_idx]
            # Current joint position from observation [9-45]
            current = obs[9 + il_idx] + ISAACLAB_DEFAULT_POS[il_idx]  # obs has relative pos
            # Action output
            act = action[il_idx]
            # What the action would produce: default + action * scale
            result = ISAACLAB_DEFAULT_POS[il_idx] + act * self.action_scale
            
            print(f"{name:<12} | {target:>8.4f} | {current:>8.4f} | {act:>8.4f} | {result:>8.4f}")
        
        print("-" * 60)
    
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
    
    def run(self, duration: float = 30.0, loop_motion: bool = True):
        """Run simulation with viewer.
        
        Args:
            duration: Simulation duration in seconds
            loop_motion: If True, loop motion when it ends; otherwise stop
        """
        motion_info = ""
        if self.motion:
            motion_info = f" (motion: {self.motion.duration:.2f}s, {'looping' if loop_motion else 'once'})"
            if not loop_motion:
                duration = min(duration, self.motion.duration + 2.0)  # Add 2s buffer after motion ends
        
        print(f"\nStarting sim2sim simulation for {duration}s{motion_info}...")
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
                    # Update motion targets if using motion file
                    if self.motion:
                        self.update_targets_from_motion()
                        # Advance motion time
                        self.motion_time += self.control_dt
                        
                        # Handle non-looping case
                        if not loop_motion and self.motion_time > self.motion.duration:
                            self.motion_time = self.motion.duration - 0.01  # Hold last frame
                    
                    obs = self.build_observation(state)
                    action = self.forward(obs)
                    self.last_action = action.copy()
                    policy_count += 1
                    
                    # Print status periodically
                    if policy_count % 50 == 0:
                        motion_progress = ""
                        if self.motion:
                            progress = (self.motion_time % self.motion.duration) / self.motion.duration * 100
                            motion_progress = f", motion: {self.motion_time:.1f}s ({progress:.0f}%)"
                        print(f"Step {policy_count}: height={state['base_height']:.3f}m, "
                              f"roll={np.degrees(quat_to_euler(state['quat'])[0]):.1f}°{motion_progress}")
                    
                    # Print full observation on first step
                    if policy_count == 1:
                        self.print_observation(obs)
                    
                    # Print arm joint debug info periodically
                    if policy_count <= 5 or policy_count % 25 == 0:
                        self.print_arm_debug(obs, action, state, policy_count)
                
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


def find_motion_file(motion_path: str) -> str:
    """Find motion file, searching in common locations."""
    if os.path.exists(motion_path):
        return os.path.abspath(motion_path)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_paths = [
        os.path.join(script_dir, motion_path),
        os.path.join(script_dir, "sample_motions", motion_path),
        os.path.join(script_dir, "../datasets/teleop_motions/stage3_upper_body", motion_path),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return None


def list_available_motions():
    """List available motion files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    motion_dirs = [
        ("sample_motions", os.path.join(script_dir, "sample_motions")),
        ("datasets/teleop_motions/stage3_upper_body", 
         os.path.join(script_dir, "../datasets/teleop_motions/stage3_upper_body")),
    ]
    
    print("\nAvailable motion files:")
    print("=" * 50)
    
    found_any = False
    for name, path in motion_dirs:
        if os.path.isdir(path):
            pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
            if pkl_files:
                print(f"\n{name}/")
                for pkl in sorted(pkl_files):
                    # Try to get motion info
                    try:
                        with open(os.path.join(path, pkl), 'rb') as f:
                            motion = pickle.load(f)
                        fps = motion.get("fps", 50.0)
                        num_frames = motion["dof_pos"].shape[0]
                        duration = num_frames / fps
                        print(f"  - {pkl} ({duration:.1f}s, {num_frames} frames)")
                    except Exception:
                        print(f"  - {pkl}")
                    found_any = True
    
    if not found_any:
        print("  No motion files found.")
    
    print("\nUsage example:")
    print("  python sim2sim_mujoco.py --model policy.onnx --motion open-doors.pkl")
    print("  python sim2sim_mujoco.py --model policy.onnx --motion sample_motions/open-doors.pkl")
    print()


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim: Test policy in MuJoCo")
    parser.add_argument("--model", type=str, default="policy_stage4_82000.onnx",
                        help="Path to ONNX policy model")
    parser.add_argument("--mujoco_model", type=str, default=None,
                        help="Path to MuJoCo XML model")
    parser.add_argument("--motion", type=str, default=None,
                        help="Path to motion pkl file (e.g., sample_motions/open-doors.pkl)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--loop", action="store_true", default=True,
                        help="Loop motion playback (default: True)")
    parser.add_argument("--no-loop", action="store_false", dest="loop",
                        help="Don't loop motion, play once then hold")
    parser.add_argument("--list-motions", action="store_true",
                        help="List available motion files and exit")
    
    args = parser.parse_args()
    
    # List motions and exit if requested
    if args.list_motions:
        list_available_motions()
        sys.exit(0)
    
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
    
    # Find motion file if specified
    motion_file = None
    if args.motion:
        motion_file = find_motion_file(args.motion)
        if motion_file is None:
            print(f"Error: Motion file not found: {args.motion}")
            print("Searched in:")
            print(f"  - {args.motion}")
            print(f"  - sample_motions/{args.motion}")
            print(f"  - ../datasets/teleop_motions/stage3_upper_body/{args.motion}")
            sys.exit(1)
        print(f"Using motion file: {motion_file}")
    
    sim = G1Sim2Sim(args.model, args.mujoco_model, motion_file=motion_file)
    sim.run(args.duration, loop_motion=args.loop)


if __name__ == "__main__":
    main()

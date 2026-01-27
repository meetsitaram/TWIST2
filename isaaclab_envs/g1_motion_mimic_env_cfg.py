# Copyright (c) 2025
# G1 Motion Imitation Environment Configuration
#
# Extends Isaac Lab's G1 locomotion environment with motion tracking
# for imitation learning from teleop motion data.

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Import base G1 locomotion config
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

# Import MDP functions
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Our custom motion tracking MDP
from . import motion_mdp


##############################################################################
# MOTION TRACKING REWARDS
##############################################################################

@configclass
class G1MotionMimicRewards(RewardsCfg):
    """Reward terms for motion imitation.
    
    Replaces velocity tracking rewards with motion tracking rewards.
    """
    
    # === MOTION TRACKING REWARDS ===
    
    # Joint position tracking (primary reward)
    tracking_joint_dof = RewTerm(
        func=motion_mdp.tracking_joint_dof,
        weight=2.0,
        params={"std": 0.5},
    )
    
    # Joint velocity tracking
    tracking_joint_vel = RewTerm(
        func=motion_mdp.tracking_joint_vel,
        weight=0.2,
        params={"std": 1.0},
    )
    
    # Key body position tracking (hands, elbows, feet)
    # NOTE: These must be Isaac Lab robot body names (not motion data names)
    tracking_keybody_pos = RewTerm(
        func=motion_mdp.tracking_keybody_pos,
        weight=3.0,
        params={
            "key_bodies": [
                "left_ankle_roll_link", "right_ankle_roll_link",  # Feet
                "left_elbow_pitch_link", "right_elbow_pitch_link",  # Elbows (Isaac Lab naming)
            ],
            "std": 0.2,
        },
    )
    
    # Root position tracking (height)
    tracking_root_height = RewTerm(
        func=motion_mdp.tracking_root_height,
        weight=1.0,
        params={"std": 0.1},
    )
    
    # Root XY position tracking (horizontal movement for walking)
    tracking_root_pos_xy = RewTerm(
        func=motion_mdp.tracking_root_pos_xy,
        weight=2.0,  # Important for locomotion
        params={"std": 0.25},
    )
    
    # Root orientation tracking
    tracking_root_orientation = RewTerm(
        func=motion_mdp.tracking_root_orientation,
        weight=1.0,
        params={"std": 0.5},
    )
    
    # === STABILITY REWARDS (from base locomotion) ===
    
    # Termination penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # Feet air time for walking motions
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    
    # Feet slide penalty
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    
    # Feet distance penalty - discourage legs spreading too far apart
    feet_distance = RewTerm(
        func=motion_mdp.feet_distance_penalty,
        weight=-5.0,  # Strong penalty for wide stance
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_dist": 0.1,   # Min 10cm between feet
            "max_dist": 0.6,   # Max 60cm between feet (normal walking width)
        },
    )
    
    # === SMOOTHNESS PENALTIES ===
    
    # Action rate penalty
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # Joint acceleration penalty
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Joint torque penalty
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Joint position limits penalty
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Flat orientation penalty
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)


##############################################################################
# MOTION TRACKING OBSERVATIONS
##############################################################################

@configclass
class G1MotionMimicObservations:
    """Observation specification for motion imitation.
    
    Includes proprioceptive state + motion targets.
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""
        
        # Proprioception (inherited from locomotion)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        
        # Motion targets (from MotionLib)
        target_joint_pos = ObsTerm(func=motion_mdp.target_joint_pos)
        target_keybody_pos = ObsTerm(func=motion_mdp.target_keybody_pos_local)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()


##############################################################################
# MOTION TRACKING TERMINATIONS
##############################################################################

@configclass
class G1MotionMimicTerminations:
    """Termination conditions for motion imitation."""
    
    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Base contact (falling) - detect contact on upper body parts
    # Using regex to match torso, pelvis, head, waist, shoulders
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*torso.*|.*pelvis.*|.*head.*|.*waist.*"), 
            "threshold": 0.5  # Lower threshold for more sensitive detection
        },
    )
    
    # Height-based termination - robot root Z too low means it fell
    bad_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Motion tracking failure (too far from target)
    # DISABLED: The motion data uses absolute world positions which don't match
    # the robot's spawned position. Need to implement relative motion tracking.
    # motion_tracking_failure = DoneTerm(
    #     func=motion_mdp.motion_tracking_failure,
    #     params={"threshold": 5.0},  # meters (relaxed for training)
    # )


##############################################################################
# MAIN ENVIRONMENT CONFIG
##############################################################################

@configclass
class G1MotionMimicEnvCfg(G1FlatEnvCfg):
    """G1 Motion Imitation Environment Configuration.
    
    Extends G1 flat locomotion with motion tracking from pkl files.
    Uses MotionLib to load and sample from motion dataset.
    """
    
    # Motion file path (relative to TWIST2 root)
    motion_file: str = "motion_data_configs/teleop_dataset.yaml"
    
    # Motion tracking parameters
    motion_dt: float = 0.02  # 50Hz motion data
    motion_curriculum: bool = True
    motion_curriculum_gamma: float = 0.01
    
    # Whether to reset robot to motion reference pose
    reset_to_motion: bool = False  # Start from default pose for stability
    
    # Key bodies for tracking
    key_bodies: list = None  # Set in __post_init__
    
    # Override rewards with motion tracking rewards
    rewards: G1MotionMimicRewards = G1MotionMimicRewards()
    
    # Override observations to include motion targets
    observations: G1MotionMimicObservations = G1MotionMimicObservations()
    
    # Override terminations to include motion tracking failure
    terminations: G1MotionMimicTerminations = G1MotionMimicTerminations()
    
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        # Key bodies for tracking
        # NOTE: Motion data and Isaac Lab robot use different naming!
        #
        # For MotionLib (motion pkl body names):
        self.key_bodies = [
            "left_ankle_roll_link", "right_ankle_roll_link",  # Feet (same name)
            "left_elbow_link", "right_elbow_link",  # Elbows (motion naming)
            "left_shoulder_pitch_link", "right_shoulder_pitch_link",  # Shoulders (same name)
            "torso_link",  # Torso (same name)
        ]
        
        # For robot queries (Isaac Lab body names) - must be same order as above
        self.robot_key_bodies = [
            "left_ankle_roll_link", "right_ankle_roll_link",  # Feet
            "left_elbow_pitch_link", "right_elbow_pitch_link",  # Elbows (Isaac Lab naming)
            "left_shoulder_pitch_link", "right_shoulder_pitch_link",  # Shoulders
            "torso_link",  # Torso
        ]
        
        # Scene settings
        self.scene.num_envs = 4096
        self.episode_length_s = 10.0  # Shorter episodes for motion tracking
        
        # Disable velocity commands (we use motion targets instead)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        # Remove velocity tracking rewards (replaced by motion tracking)
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        
        # Adjust domain randomization for motion imitation
        self.events.push_robot = None  # Disable pushing during initial training
        self.events.add_base_mass = None


@configclass
class G1MotionMimicEnvCfg_ROBUST(G1MotionMimicEnvCfg):
    """Stage 2: Robustness training with push disturbances.
    
    Use this config after the robot has learned basic motion skills.
    Enables random push disturbances to improve stability.
    
    Usage:
        python scripts/train_isaaclab.py --robust --resume logs/.../model_5000.pt
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # Re-enable push disturbances (parent disables them)
        from isaaclab.managers import EventTermCfg
        from isaaclab.envs.mdp import events as mdp_events
        
        self.events.push_robot = EventTermCfg(
            func=mdp_events.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 10.0),  # Push every 6-10 seconds (more frequent)
            params={
                "velocity_range": {
                    "x": (-1.2, 1.2),  # Stronger pushes for stable robot
                    "y": (-1.2, 1.2),
                }
            },
        )
        
        # Optionally enable mass randomization for extra robustness
        # self.events.add_base_mass = EventTermCfg(...)


##############################################################################
# STAGE 3: UPPER BODY JOINT TRACKING
##############################################################################

@configclass
class G1MotionMimicRewards_STAGE3(G1MotionMimicRewards):
    """Stage 3: Upper body manipulation via joint angle tracking.
    
    The teleop dataset contains the robot's joint angles directly (dof_pos),
    so we use joint tracking as the primary reward. This is more direct than
    end-effector position tracking and doesn't require FK computation.
    
    Key differences from Stage 1:
    - Higher weight on joint tracking (primary objective)
    - Separate arm joint tracking with tighter precision
    - Upper body stability reward (keep torso stable)
    - Reduced lower body tracking weight (allow balance adaptation)
    """
    
    # === PRIMARY: JOINT ANGLE TRACKING ===
    
    # Overall joint tracking (all 29 DOFs)
    tracking_joint_dof = RewTerm(
        func=motion_mdp.tracking_joint_dof,
        weight=3.0,  # Increased from 2.0 - primary objective
        params={"std": 0.4},  # Tighter than Stage 1
    )
    
    # Arm joints tracking with higher weight
    # (shoulders, elbows, wrists - indices 13-28 on G1)
    tracking_arm_joints = RewTerm(
        func=motion_mdp.tracking_arm_joints,
        weight=4.0,  # High weight for arms specifically
        params={
            "std": 0.3,  # Tighter precision for arms
        },
    )
    
    # === STABILITY: KEEP BALANCE WHILE ARMS MOVE ===
    
    # Upper body stability (keep torso stable during manipulation)
    upper_body_stability = RewTerm(
        func=motion_mdp.upper_body_stability,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Keep height tracking to prevent crouching/rising
    tracking_root_height = RewTerm(
        func=motion_mdp.tracking_root_height,
        weight=2.0,  # Important for manipulation
        params={"std": 0.1},
    )
    
    # === REDUCED: LOWER BODY TRACKING (let it adapt for balance) ===
    
    # Reduce weight on root XY tracking - upper body is priority
    tracking_root_pos_xy = RewTerm(
        func=motion_mdp.tracking_root_pos_xy,
        weight=0.5,  # Reduced from 2.0 - allow drift for balance
        params={"std": 0.5},  # More lenient
    )
    
    # Disable EE tracking (not working, use joint tracking instead)
    tracking_ee_pos = None
    ee_velocity_direction = None


@configclass
class G1MotionMimicEnvCfg_STAGE3(G1MotionMimicEnvCfg):
    """Stage 3: Upper body end-effector tracking for manipulation.
    
    Builds on Stage 1/2 with precise end-effector position tracking.
    Uses teleop motion data that focuses on upper body movements.
    
    Key features:
    - Primary reward on hand position accuracy
    - Temporal tolerance for motion matching (~50ms window)
    - Balance rewards maintained to prevent falling
    - Velocity direction reward for smoother learning
    
    Usage:
        python scripts/train_isaaclab.py \
            --motion_file motion_data_configs/upper_body_teleop.yaml \
            --checkpoint logs/isaaclab/motion_mimic/model_STAGE2.pt \
            --num_envs 4096 --max_iterations 30000 --headless
    """
    
    # Override rewards with Stage 3 rewards
    rewards: G1MotionMimicRewards_STAGE3 = G1MotionMimicRewards_STAGE3()
    
    def __post_init__(self):
        super().__post_init__()
        
        # Episode settings for manipulation training
        self.episode_length_s = 15.0  # Longer episodes for manipulation sequences
        
        # IMPORTANT: Do NOT override key_bodies here!
        # We must keep the same observation space as Stage 1 to load checkpoints.
        # The EE tracking rewards (tracking_ee_pos, ee_velocity_direction) use
        # robot body names directly, independent of key_bodies.
        #
        # Inherited key_bodies from Stage 1:
        # - feet (2), elbows (2), shoulders (2), torso (1) = 7 bodies
        
        # Disable push disturbances during Stage 3
        # (re-enable for Stage 4 robust manipulation if needed)
        self.events.push_robot = None


@configclass
class G1MotionMimicEnvCfg_PLAY(G1MotionMimicEnvCfg):
    """Play/evaluation configuration with visualization."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Smaller scene for visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Longer episodes for evaluation
        self.episode_length_s = 30.0
        
        # Disable randomization
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        
        # Disable motion tracking termination for visualization
        self.terminations.motion_tracking_failure = None

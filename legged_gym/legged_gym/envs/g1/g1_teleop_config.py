"""
G1 Teleop Training Config

Training config specifically for custom teleop motion data captured via
multi-camera MediaPipe system and IK retargeting.

Based on G1MimicStuFutureCfg but configured for teleop dataset.
"""

from legged_gym.envs.g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


# Future motion steps - single frame for simpler training
TAR_MOTION_STEPS_FUTURE = [0]


class G1TeleopCfg(G1MimicPrivCfg):
    """Config for training on custom teleop motion data.
    
    This config is optimized for upper-body focused motions captured
    via MediaPipe + IK retargeting, with fixed base (standing in place).
    """
    
    class env(G1MimicPrivCfg.env):
        obs_type = 'student_future'
        
        # Motion step configuration
        tar_motion_steps = [0]
        tar_motion_steps_future = TAR_MOTION_STEPS_FUTURE
        
        # Observation dimensions
        n_mimic_obs_single = 6 + 29
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = G1MimicPrivCfg.env.n_proprio

        n_future_obs_single = 6 + 29
        n_future_obs = len(tar_motion_steps_future) * n_future_obs_single
        
        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (G1MimicPrivCfg.env.history_len + 1) + n_future_obs
        
        # Disable force curriculum for initial training
        enable_force_curriculum = False
        
        # Episode settings - shorter episodes for faster iteration
        episode_length_s = 10
        
        # Termination - more lenient for teleop motions
        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.8  # Slightly more lenient
        root_tracking_termination_dist = 1.0  # More lenient for fixed-base motions
    
    class motion(G1MimicPrivCfg.motion):
        # Point to teleop motion dataset
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/../motion_data_configs/teleop_dataset.yaml"
        
        # Motion curriculum - start with easier motions
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        motion_decompose = False
        motion_smooth = True
        
        # Key bodies for tracking - focus on upper body for teleop
        key_bodies = [
            "left_rubber_hand", "right_rubber_hand",  # Hands (primary)
            "left_elbow_link", "right_elbow_link",    # Elbows
            "head_mocap",                              # Head
            "left_ankle_roll_link", "right_ankle_roll_link",  # Feet (for stability)
            "left_knee_link", "right_knee_link",      # Knees
        ]
        upper_key_bodies = [
            "left_rubber_hand", "right_rubber_hand",
            "left_elbow_link", "right_elbow_link",
            "head_mocap"
        ]
        
        # Disable motion domain randomization initially
        motion_dr_enabled = False
        
        # Error aware sampling disabled for initial training
        use_error_aware_sampling = False
    
    class rewards(G1MimicPrivCfg.rewards):
        class scales:
            # Joint tracking - primary reward
            tracking_joint_dof = 2.0
            tracking_joint_vel = 0.2
            
            # Root tracking - important for stability
            tracking_root_translation_z = 1.0  # Height tracking
            tracking_root_rotation = 1.0
            tracking_root_linear_vel = 1.0
            tracking_root_angular_vel = 1.0
            
            # Key body tracking - high weight for hands
            tracking_keybody_pos = 3.0  # Higher for teleop (hands important)
            tracking_keybody_pos_global = 2.0
            
            # Survival bonus
            alive = 0.5
            
            # Stability penalties
            feet_slip = -0.1
            feet_contact_forces = -5e-4
            feet_stumble = -1.25
            feet_air_time = 5.0
            
            # Joint limit penalties
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            
            # Smoothness penalties
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.05
            
            # Angular stability
            ang_vel_xy = -0.01
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
    
    class domain_rand(G1MimicPrivCfg.domain_rand):
        # Start with less randomization for initial training
        domain_rand_general = True
        
        randomize_gravity = True
        gravity_range = (-0.05, 0.05)  # Less aggressive
        
        randomize_friction = True
        friction_range = [0.3, 1.5]  # Narrower range
        
        randomize_base_mass = True
        added_mass_range = [-2., 2]  # Less mass variation
        
        randomize_base_com = True
        added_com_range = [-0.03, 0.03]  # Less COM variation
        
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.8  # Gentler pushes
        
        randomize_motor = True
        motor_strength_range = [0.85, 1.15]  # Less motor variation
        
        action_delay = True
        action_buf_len = 6  # Slightly less delay


class G1TeleopCfgPPO(HumanoidMimicCfgPPO):
    """PPO training config for teleop motions."""
    
    seed = 1
    
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticFuture'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunner'
        max_iterations = 20001  # 20k iterations for initial training
        
        # Logging
        save_interval = 500
        experiment_name = 'teleop'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
    
    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
    
    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
        
        # Future motion encoder
        future_encoder_dims = [256, 256, 128]
        future_attention_heads = 4
        future_dropout = 0.1
        temporal_embedding_dim = 64
        future_latent_dim = 128
        num_future_steps = len(TAR_MOTION_STEPS_FUTURE)
        
        num_future_observations = G1TeleopCfg.env.n_future_obs
        
        # MoE parameters
        num_experts = 4
        expert_hidden_dims = [256, 128]
        gating_hidden_dim = 128
        moe_temperature = 1.0
        moe_topk = None
        load_balancing_loss_weight = 0.01

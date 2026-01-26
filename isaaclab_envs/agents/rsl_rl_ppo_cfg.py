# Copyright (c) 2025
# RSL-RL PPO Configuration for G1 Motion Imitation
#
# Based on TWIST2's training configuration adapted for Isaac Lab.

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1MotionMimicPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO Runner configuration for G1 motion imitation.
    
    Uses larger network and more training steps compared to velocity locomotion
    since motion imitation is a more complex task.
    """
    
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "g1_motion_mimic"
    empirical_normalization = False
    
    # Logging - options: "tensorboard", "wandb", "neptune"
    logger = "wandb"
    wandb_project = "twist2-isaaclab"
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1MotionMimicPPORunnerCfg_Play(G1MotionMimicPPORunnerCfg):
    """Configuration for playing/evaluating trained policy."""
    
    def __post_init__(self):
        super().__post_init__()
        self.resume = True

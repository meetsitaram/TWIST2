# Copyright (c) 2025
# G1 Motion Imitation Environment for Isaac Lab
# 
# This package provides motion imitation training environments for the
# Unitree G1 robot using Isaac Lab framework.

import gymnasium as gym

from .g1_motion_mimic_env_cfg import (
    G1MotionMimicEnvCfg, 
    G1MotionMimicEnvCfg_PLAY,
    G1MotionMimicEnvCfg_ROBUST,
    G1MotionMimicEnvCfg_STAGE3,
)
from .g1_motion_mimic_env import G1MotionMimicEnv

# Agent configs
from . import agents

##############################################################################
# Register Gym environments
##############################################################################

gym.register(
    id="Isaac-Motion-Mimic-G1-v0",
    entry_point="isaaclab_envs.g1_motion_mimic_env:G1MotionMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_envs.g1_motion_mimic_env_cfg:G1MotionMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionMimicPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Motion-Mimic-G1-Play-v0",
    entry_point="isaaclab_envs.g1_motion_mimic_env:G1MotionMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_envs.g1_motion_mimic_env_cfg:G1MotionMimicEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionMimicPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Motion-Mimic-G1-Robust-v0",
    entry_point="isaaclab_envs.g1_motion_mimic_env:G1MotionMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_envs.g1_motion_mimic_env_cfg:G1MotionMimicEnvCfg_ROBUST",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionMimicPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Motion-Mimic-G1-Stage3-v0",
    entry_point="isaaclab_envs.g1_motion_mimic_env:G1MotionMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_envs.g1_motion_mimic_env_cfg:G1MotionMimicEnvCfg_STAGE3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MotionMimicPPORunnerCfg",
    },
)

__all__ = [
    "G1MotionMimicEnvCfg", 
    "G1MotionMimicEnvCfg_PLAY",
    "G1MotionMimicEnvCfg_ROBUST",
    "G1MotionMimicEnvCfg_STAGE3",
    "G1MotionMimicEnv",
]

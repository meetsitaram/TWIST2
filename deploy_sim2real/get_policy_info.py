#!/usr/bin/env python3
"""
Extract and print policy information from a checkpoint.

This script analyzes a trained checkpoint and prints:
- Observation/action dimensions
- Network architecture
- Weight statistics

Usage:
    python get_policy_info.py --checkpoint path/to/model.pt
"""

import argparse
import os
import sys
import torch
import json


def analyze_checkpoint(checkpoint_path: str):
    """Analyze checkpoint and print policy information."""
    
    print("="*60)
    print("G1 Teleop Policy - Checkpoint Analysis")
    print("="*60)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"File size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB\n")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Checkpoint keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: Tensor {checkpoint[key].shape}")
        elif isinstance(checkpoint[key], dict):
            print(f"  {key}: Dict with {len(checkpoint[key])} items")
        else:
            print(f"  {key}: {type(checkpoint[key]).__name__}")
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'actor_critic' in checkpoint:
        state_dict = checkpoint['actor_critic']
    else:
        state_dict = checkpoint
    
    print(f"\n{'='*60}")
    print("Network Architecture")
    print("="*60)
    
    # Analyze actor
    actor_layers = {}
    critic_layers = {}
    other_layers = {}
    
    for name, param in state_dict.items():
        if 'actor' in name.lower():
            actor_layers[name] = param.shape
        elif 'critic' in name.lower():
            critic_layers[name] = param.shape
        else:
            other_layers[name] = param.shape
    
    print("\nActor layers:")
    for name, shape in sorted(actor_layers.items()):
        print(f"  {name}: {list(shape)}")
    
    # Infer dimensions
    obs_dim = None
    act_dim = None
    hidden_dims = []
    
    for name, shape in sorted(actor_layers.items()):
        if 'weight' in name:
            if obs_dim is None:
                obs_dim = shape[1]
            act_dim = shape[0]  # Last layer output
            if len(shape) == 2:
                hidden_dims.append(shape[0])
    
    # Remove last (output) dim from hidden dims
    if hidden_dims:
        hidden_dims = hidden_dims[:-1]
    
    print(f"\n{'='*60}")
    print("Inferred Dimensions")
    print("="*60)
    print(f"\n  Observation dim: {obs_dim}")
    print(f"  Action dim:      {act_dim}")
    print(f"  Hidden dims:     {hidden_dims}")
    
    # Check for std (exploration noise)
    std_key = None
    for name in state_dict.keys():
        if 'std' in name.lower() or 'log_std' in name.lower():
            std_key = name
            break
    
    if std_key:
        std_value = state_dict[std_key]
        if 'log' in std_key.lower():
            std_value = torch.exp(std_value)
        print(f"  Action std:      {std_value.mean().item():.4f} (mean)")
    
    print(f"\n{'='*60}")
    print("Deployment Information")
    print("="*60)
    
    print("""
Forward Pass (Python):
----------------------
```python
import torch

# Load model
checkpoint = torch.load("model.pt", map_location='cpu')
state_dict = checkpoint.get('model_state_dict', checkpoint)

# Create actor network
actor = torch.nn.Sequential(
    torch.nn.Linear({obs_dim}, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 256),
    torch.nn.ELU(),
    torch.nn.Linear(256, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, {act_dim}),
)

# Load weights (filter actor weights)
actor_state = {{k.replace('actor_backbone.', ''): v 
               for k, v in state_dict.items() 
               if 'actor_backbone' in k}}
actor.load_state_dict(actor_state)
actor.eval()

# Inference
observation = torch.randn(1, {obs_dim})
with torch.no_grad():
    action = actor(observation)  # Shape: (1, {act_dim})

# Convert to joint positions
action_scale = 0.5
default_pos = ...  # Default joint positions
target_pos = default_pos + action.numpy() * action_scale
```
""".format(obs_dim=obs_dim, act_dim=act_dim))
    
    print(f"\n{'='*60}")
    print("Action Interpretation")
    print("="*60)
    print("""
Actions are DELTAS from default joint positions:
  
  target_joint_pos = default_pos + (action * action_scale)
  
Where:
  - action_scale = 0.5 (from training config)
  - default_pos = robot's standing pose
  
The policy outputs continuous values roughly in [-1, 1] range.
Multiply by action_scale to get actual joint position deltas.
""")
    
    # Save info to JSON
    info = {
        "checkpoint_path": checkpoint_path,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden_dims": hidden_dims,
        "action_scale": 0.5,
        "actor_layers": {k: list(v) for k, v in actor_layers.items()},
    }
    
    json_path = checkpoint_path.replace('.pt', '_info.json')
    with open(json_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"\nInfo saved to: {json_path}")
    
    return info


def main():
    parser = argparse.ArgumentParser(description="Analyze policy checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    analyze_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()

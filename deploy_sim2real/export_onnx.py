#!/usr/bin/env python3
"""
Export trained PyTorch policy to ONNX format for deployment.

Usage:
    python export_onnx.py --checkpoint path/to/model.pt
    python export_onnx.py --checkpoint path/to/model.pt --output policy.onnx
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PolicyWrapper(nn.Module):
    """Wrapper that extracts just the actor forward pass for ONNX export."""
    
    def __init__(self, actor_critic, obs_dim, act_dim):
        super().__init__()
        self.actor = actor_critic.actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
    
    def forward(self, observation):
        """
        Forward pass for inference.
        
        Args:
            observation: (batch, obs_dim) observation tensor
            
        Returns:
            action: (batch, act_dim) action tensor (joint position deltas)
        """
        return self.actor(observation)


def load_checkpoint(checkpoint_path: str):
    """Load checkpoint and extract model configuration."""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'actor_critic' in checkpoint:
        state_dict = checkpoint['actor_critic']
    else:
        # Assume the checkpoint is the state dict directly
        state_dict = checkpoint
    
    # Infer dimensions from state dict
    # Actor backbone first layer: actor_backbone.0.weight has shape (hidden, input)
    # Actor backbone last layer: actor_backbone.N.weight has shape (output, hidden)
    
    actor_keys = [k for k in state_dict.keys() if 'actor' in k and 'weight' in k]
    
    if actor_keys:
        # Find first and last linear layers
        first_layer = sorted([k for k in actor_keys if 'backbone.0' in k or 'actor.0' in k])[0]
        last_layer = sorted([k for k in actor_keys])[-1]
        
        # Input dim from first layer
        obs_dim = state_dict[first_layer].shape[1]
        
        # Output dim from last layer  
        act_dim = state_dict[last_layer].shape[0]
        
        print(f"Inferred obs_dim: {obs_dim}, act_dim: {act_dim}")
    else:
        # Fallback to typical values
        obs_dim = 195
        act_dim = 37
        print(f"Using default obs_dim: {obs_dim}, act_dim: {act_dim}")
    
    return state_dict, obs_dim, act_dim


def create_actor_model(obs_dim: int, act_dim: int, hidden_dims=[512, 256, 128]):
    """Create a simple MLP actor matching the training architecture."""
    
    layers = []
    prev_dim = obs_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ELU())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, act_dim))
    
    return nn.Sequential(*layers)


def export_onnx(checkpoint_path: str, output_path: str, opset_version: int = 14):
    """Export model to ONNX format."""
    
    state_dict, obs_dim, act_dim = load_checkpoint(checkpoint_path)
    
    # Create actor model
    actor = create_actor_model(obs_dim, act_dim)
    
    # Filter state dict for actor weights only
    actor_state = {}
    for key, value in state_dict.items():
        # Handle different naming conventions
        if 'actor_backbone' in key:
            new_key = key.replace('actor_backbone.', '')
            actor_state[new_key] = value
        elif 'actor.' in key and 'backbone' not in key:
            new_key = key.replace('actor.', '')
            actor_state[new_key] = value
    
    if actor_state:
        try:
            actor.load_state_dict(actor_state)
            print("Loaded actor weights successfully")
        except Exception as e:
            print(f"Warning: Could not load weights directly: {e}")
            print("Attempting to load with strict=False...")
            actor.load_state_dict(actor_state, strict=False)
    else:
        print("Warning: No actor weights found in checkpoint")
        print("Available keys:", list(state_dict.keys())[:10])
    
    actor.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    
    torch.onnx.export(
        actor,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    # Verify the export
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation: PASSED")
    except ImportError:
        print("Note: Install 'onnx' package to validate the exported model")
    except Exception as e:
        print(f"ONNX validation warning: {e}")
    
    # Test with onnxruntime
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(output_path)
        
        # Test inference
        test_input = np.random.randn(1, obs_dim).astype(np.float32)
        outputs = session.run(None, {'observation': test_input})
        
        print(f"ONNX inference test: PASSED")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Output sample: {outputs[0][0, :5]}...")
        
    except ImportError:
        print("Note: Install 'onnxruntime' package to test inference")
    except Exception as e:
        print(f"ONNX runtime test warning: {e}")
    
    # Save metadata
    metadata_path = output_path.replace('.onnx', '_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Source checkpoint: {checkpoint_path}\n")
        f.write(f"Observation dim: {obs_dim}\n")
        f.write(f"Action dim: {act_dim}\n")
        f.write(f"Hidden dims: [512, 256, 128]\n")
        f.write(f"Activation: ELU\n")
        f.write(f"ONNX opset: {opset_version}\n")
        f.write(f"\nAction interpretation:\n")
        f.write(f"  Actions are joint position DELTAS\n")
        f.write(f"  target_pos = default_pos + (action * action_scale)\n")
        f.write(f"  action_scale = 0.5\n")
    
    print(f"Metadata saved to: {metadata_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export policy to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PyTorch checkpoint (.pt file)")
    parser.add_argument("--output", type=str, default="policy.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset version")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    export_onnx(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect motion PKL file contents"""

import argparse
from pose.utils.motion_lib_pkl import MotionLib
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use: cpu or cuda (default: cpu)")
    args = parser.parse_args()
    
    print(f"Loading: {args.motion_file}")
    print(f"Using device: {args.device}")
    motion_lib = MotionLib(args.motion_file, device=args.device)
    
    print(f"\n{'='*60}")
    print(f"Motion Library Information")
    print(f"{'='*60}")
    
    print(f"\nNumber of motions: {motion_lib.num_motions()}")
    
    for i in range(motion_lib.num_motions()):
        length = motion_lib.get_motion_length(i)
        num_frames = motion_lib.get_motion_num_frames(i)
        fps = num_frames / length if length > 0 else 0
        
        print(f"\nMotion {i}:")
        print(f"  Duration: {length:.2f} seconds")
        print(f"  Frames: {num_frames}")
        print(f"  FPS: {fps:.1f}")
        
        # Get first frame
        motion_id = torch.tensor([i], device=args.device)
        motion_times = torch.tensor([0.0], device=args.device)
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, _, _, _ = \
            motion_lib.calc_motion_frame(motion_id, motion_times)
        
        print(f"\n  First frame:")
        print(f"    Root position: {root_pos.cpu().numpy().squeeze()}")
        print(f"    Root velocity: {root_vel.cpu().numpy().squeeze()}")
        print(f"    Joint angles (first 5): {dof_pos.cpu().numpy().squeeze()[:5]}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()


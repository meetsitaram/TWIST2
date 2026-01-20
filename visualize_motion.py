#!/usr/bin/env python3
"""Visualize motion from PKL file without controller"""

import argparse
import numpy as np
import mujoco
from mujoco.viewer import launch_passive
from pose.utils.motion_lib_pkl import MotionLib
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", required=True, 
                       help="Path to PKL motion file")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Playback speed (1.0 = normal)")
    parser.add_argument("--loop", action="store_true",
                       help="Loop the motion")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use: cpu or cuda (default: cuda, auto-fallback to cpu)")
    args = parser.parse_args()
    
    # Load motion with device fallback
    print(f"Loading motion: {args.motion_file}")
    device = args.device
    if device == "cuda":
        try:
            import torch
            test_tensor = torch.randn(10, device="cuda")
            del test_tensor
            print(f"Using device: cuda")
        except Exception as e:
            print(f"CUDA failed ({e}), falling back to CPU")
            device = "cpu"
    if device == "cpu":
        print(f"Using device: cpu")
    
    motion_lib = MotionLib(args.motion_file, device=device)
    
    # Load MuJoCo model
    xml_file = "assets/g1/g1_mocap_29dof.xml"
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)
    
    # Open viewer
    viewer = launch_passive(model, data)
    
    print(f"Motion info:")
    print(f"  Number of motions: {motion_lib.num_motions()}")
    print(f"  Duration: {motion_lib.get_motion_length(0):.2f} seconds")
    print(f"  Playback speed: {args.speed}x")
    print(f"  Loop: {args.loop}")
    print(f"\nPress Ctrl+C to stop")
    
    control_dt = 0.01
    t = 0
    
    try:
        while viewer.is_running():
            t_start = time.time()
            
            # Get motion frame
            motion_time = t * control_dt
            motion_length = motion_lib.get_motion_length(0)
            
            if motion_time >= motion_length:
                if args.loop:
                    t = 0
                    motion_time = 0
                else:
                    print("Motion finished!")
                    break
            
            # Get pose at this time
            motion_id = torch.tensor([0], device=device)
            motion_times = torch.tensor([motion_time], device=device)
            
            root_pos, root_rot, _, _, dof_pos, _, _, _, _ = \
                motion_lib.calc_motion_frame(motion_id, motion_times)
            
            # Apply to MuJoCo
            root_pos_np = root_pos.cpu().numpy().squeeze()
            root_rot_np = root_rot.cpu().numpy().squeeze()
            dof_pos_np = dof_pos.cpu().numpy().squeeze()
            
            data.qpos[:3] = root_pos_np
            data.qpos[3:7] = root_rot_np[[3,0,1,2]]  # Flip quaternion format
            data.qpos[7:] = dof_pos_np
            
            mujoco.mj_forward(model, data)
            
            # Update camera
            pelvis_id = model.body("pelvis").id
            pelvis_pos = data.xpos[pelvis_id]
            viewer.cam.lookat = pelvis_pos
            viewer.cam.distance = 2.5
            viewer.sync()
            
            t += 1
            
            # Maintain playback speed
            elapsed = time.time() - t_start
            target_dt = control_dt / args.speed
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        viewer.close()
        print("Done!")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Teleop QC Comparison Script

Reads both teleop input (from isaac_lab_teleop_publisher.py) and 
Isaac Lab robot output (from play_isaaclab_teleop.py --publish_state)
and compares them to identify tracking discrepancies.

Usage:
    # Terminal 1: Run teleop publisher
    python isaac_lab_teleop_publisher.py --display --mujoco_viz
    
    # Terminal 2: Run Isaac Lab with state publishing
    python scripts/play_isaaclab_teleop.py \
        --checkpoint ... \
        --teleop redis --direct_override --publish_state
    
    # Terminal 3: Run this QC script
    python deploy_real/teleop_qc_compare.py
"""

import argparse
import os
import sys
import time
import json
import numpy as np

# Add TWIST2 root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWIST2_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, TWIST2_ROOT)

from robot_config import G1RobotConfig


def main():
    parser = argparse.ArgumentParser(description="QC comparison of teleop input vs Isaac Lab output")
    parser.add_argument("--redis_host", type=str, default="localhost")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument("--input_key", type=str, default="teleop:mimic_obs",
                       help="Redis key for teleop input (MuJoCo order)")
    parser.add_argument("--output_key", type=str, default="isaaclab:robot_state",
                       help="Redis key for Isaac Lab robot state (Isaac Lab order)")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Save comparison data to CSV file")
    parser.add_argument("--upper_only", action="store_true", default=True,
                       help="Only compare upper body joints (default: True)")
    args = parser.parse_args()
    
    try:
        import redis
    except ImportError:
        print("ERROR: redis-py not installed. Install with: pip install redis")
        return
    
    # Connect to Redis
    r = redis.Redis(host=args.redis_host, port=args.redis_port)
    try:
        r.ping()
        print(f"[Redis] Connected to {args.redis_host}:{args.redis_port}")
    except redis.ConnectionError as e:
        print(f"ERROR: Could not connect to Redis: {e}")
        return
    
    # Upper body joint indices (MuJoCo order)
    # We compare in MuJoCo index space - teleop input is in MuJoCo order
    # Isaac Lab output will be remapped back to MuJoCo order for comparison
    upper_body_mj_indices = []
    upper_body_names = []
    for mj_idx, name in enumerate(G1RobotConfig.MUJOCO_JOINT_ORDER):
        if any(p in name for p in ['shoulder', 'elbow', 'wrist']):
            upper_body_mj_indices.append(mj_idx)
            upper_body_names.append(name)
    
    # We need to know Isaac Lab joint names to remap - we'll get this from first state message
    isaaclab_joint_names = None
    mj_to_il = None
    
    print(f"\n[QC] Monitoring {len(upper_body_mj_indices)} upper body joints:")
    for i, (mj_idx, name) in enumerate(zip(upper_body_mj_indices, upper_body_names)):
        # Get expected Isaac Lab joint name from mapping
        il_name = G1RobotConfig.MUJOCO_TO_ISAACLAB_NAMES.get(name, None)
        print(f"  {name}: MJ[{mj_idx}] -> {il_name or 'NO EQUIV'}")
    
    print(f"\n[QC] Input key: {args.input_key}")
    print(f"[QC] Output key: {args.output_key}")
    print("\n" + "=" * 80)
    print("  TELEOP QC COMPARISON")
    print("  Comparing: Input (teleop) vs Output (Isaac Lab robot)")
    print("=" * 80)
    print("\nWaiting for data...\n")
    
    # CSV logging
    csv_file = None
    if args.log_file:
        csv_file = open(args.log_file, 'w')
        header = "timestamp,joint_name,input_rad,output_rad,error_rad,error_deg\n"
        csv_file.write(header)
        print(f"[QC] Logging to: {args.log_file}")
    
    # Comparison loop
    sample_count = 0
    error_history = {name: [] for name in upper_body_names}
    last_input = None
    last_output = None
    
    try:
        while True:
            # Read input (teleop) - MuJoCo order, raw bytes
            input_data = r.get(args.input_key)
            if input_data:
                input_dof = np.frombuffer(input_data, dtype=np.float32)
                last_input = input_dof
            
            # Read output (Isaac Lab) - JSON with joint_pos in Isaac Lab order
            output_data = r.get(args.output_key)
            if output_data:
                try:
                    output_state = json.loads(output_data)
                    output_dof_il = np.array(output_state['joint_pos'])
                    
                    # Get joint names from state if available (for remapping)
                    if 'joint_names' in output_state and isaaclab_joint_names is None:
                        isaaclab_joint_names = output_state['joint_names']
                        mj_to_il = G1RobotConfig.build_mujoco_to_isaaclab_mapping(isaaclab_joint_names)
                        print(f"[QC] Detected {len(isaaclab_joint_names)} Isaac Lab joints")
                    
                    # If we don't have joint names, assume standard G1 order
                    # and store raw output (we'll compare by known indices)
                    last_output = output_dof_il
                except (json.JSONDecodeError, KeyError) as e:
                    pass
            
            # Compare if both available
            if last_input is not None and last_output is not None:
                sample_count += 1
                timestamp = time.time()
                
                # Build mapping on first comparison (using joint names from state)
                if mj_to_il is None and isaaclab_joint_names is not None:
                    mj_to_il = G1RobotConfig.build_mujoco_to_isaaclab_mapping(isaaclab_joint_names)
                    print(f"[QC] Built joint mapping: {sum(1 for v in mj_to_il.values() if v is not None)} mapped joints")
                
                # If still no mapping, use the name-based lookup
                if mj_to_il is None and isaaclab_joint_names is None:
                    # First time - print a note
                    if sample_count == 1:
                        print("[QC] Note: No joint names from Isaac Lab state, using name-based lookup")
                    
                    # Build mapping using MUJOCO_TO_ISAACLAB_NAMES
                    # We need to find the Isaac Lab indices somehow
                    # For now, skip comparison until we have joint names
                    continue
                
                # Compare upper body joints
                errors = []
                for mj_idx, name in zip(upper_body_mj_indices, upper_body_names):
                    il_idx = mj_to_il.get(mj_idx) if mj_to_il else None
                    
                    if il_idx is None:
                        continue
                    
                    if mj_idx >= len(last_input) or il_idx >= len(last_output):
                        continue
                    
                    input_val = last_input[mj_idx]
                    output_val = last_output[il_idx]
                    error = abs(input_val - output_val)
                    errors.append(error)
                    
                    # Track history
                    error_history[name].append(error)
                    if len(error_history[name]) > 100:
                        error_history[name].pop(0)
                    
                    # Log to CSV
                    if csv_file:
                        csv_file.write(f"{timestamp},{name},{input_val:.4f},{output_val:.4f},{error:.4f},{np.degrees(error):.2f}\n")
                
                # Print summary every 50 samples
                if sample_count % 50 == 0 and mj_to_il is not None:
                    print(f"\n[QC] Sample {sample_count} - Joint Error Analysis:")
                    print("-" * 70)
                    print(f"{'Joint':<30} {'Input':>10} {'Output':>10} {'Error':>10} {'Err(deg)':>10}")
                    print("-" * 70)
                    
                    total_error = 0
                    for mj_idx, name in zip(upper_body_mj_indices, upper_body_names):
                        il_idx = mj_to_il.get(mj_idx) if mj_to_il else None
                        if il_idx is None or mj_idx >= len(last_input) or il_idx >= len(last_output):
                            continue
                        
                        input_val = last_input[mj_idx]
                        output_val = last_output[il_idx]
                        error = input_val - output_val  # Signed error
                        avg_err = np.mean(error_history[name]) if error_history[name] else 0
                        
                        # Color-code large errors
                        err_deg = np.degrees(abs(error))
                        if err_deg > 20:
                            status = " *** LARGE ***"
                        elif err_deg > 10:
                            status = " ** "
                        else:
                            status = ""
                        
                        print(f"{name:<30} {input_val:>10.3f} {output_val:>10.3f} {error:>10.3f} {err_deg:>10.1f}{status}")
                        total_error += abs(error)
                    
                    print("-" * 70)
                    print(f"{'TOTAL ABS ERROR':<30} {'':<10} {'':<10} {total_error:>10.3f} {np.degrees(total_error):>10.1f}")
                    
                    # Summary stats
                    if errors:
                        print(f"\n[QC] Error Stats: mean={np.mean(errors):.3f} rad ({np.degrees(np.mean(errors)):.1f}°), "
                              f"max={np.max(errors):.3f} rad ({np.degrees(np.max(errors)):.1f}°)")
            
            time.sleep(0.02)  # 50 Hz
    
    except KeyboardInterrupt:
        print("\n\n[QC] Stopped by user")
    
    finally:
        if csv_file:
            csv_file.close()
            print(f"[QC] Log saved to: {args.log_file}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("  FINAL ERROR SUMMARY")
    print("=" * 80)
    for name in upper_body_names:
        if error_history[name]:
            avg = np.mean(error_history[name])
            max_err = np.max(error_history[name])
            print(f"  {name:<30}: avg={np.degrees(avg):>6.1f}°, max={np.degrees(max_err):>6.1f}°")


if __name__ == "__main__":
    main()

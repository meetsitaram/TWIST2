#!/usr/bin/env python3
"""Debug script to check Redis data format for robot control."""

import redis
import json
import numpy as np
import time

def main():
    print("=" * 60)
    print("Redis Data Debug Tool")
    print("=" * 60)
    
    # Connect to Redis
    try:
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("Connected to Redis")
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        return
    
    # Check current values
    keys = [
        "action_body_unitree_g1_with_hands",
        "action_hand_left_unitree_g1_with_hands",
        "action_hand_right_unitree_g1_with_hands",
        "action_neck_unitree_g1_with_hands"
    ]
    
    print("\n--- Current Redis Values ---")
    for key in keys:
        data = r.get(key)
        if data:
            try:
                action = json.loads(data)
                arr = np.array(action)
                print(f"\n{key}:")
                print(f"  Length: {len(action)}")
                print(f"  Min: {arr.min():.4f}, Max: {arr.max():.4f}")
                print(f"  Values: {arr}")
            except Exception as e:
                print(f"\n{key}: Error parsing - {e}")
        else:
            print(f"\n{key}: No data")
    
    # Expected format for action_body
    print("\n--- Expected Format ---")
    print("action_body_unitree_g1_with_hands: 35 dimensions")
    print("  [0] root_vel_x")
    print("  [1] root_vel_y") 
    print("  [2] root_z (height)")
    print("  [3] roll")
    print("  [4] pitch")
    print("  [5] yaw_vel")
    print("  [6:35] dof_pos (29 joint angles)")
    
    # Monitor for changes
    print("\n--- Monitoring for Changes (press Ctrl+C to stop) ---")
    last_data = None
    try:
        while True:
            data = r.get("action_body_unitree_g1_with_hands")
            if data:
                action = json.loads(data)
                if action != last_data:
                    arr = np.array(action)
                    print(f"\nNew data at {time.strftime('%H:%M:%S')}:")
                    print(f"  Root: vel_x={arr[0]:.3f}, vel_y={arr[1]:.3f}, z={arr[2]:.3f}")
                    print(f"  Orient: roll={arr[3]:.3f}, pitch={arr[4]:.3f}, yaw_vel={arr[5]:.3f}")
                    print(f"  Joints (first 10): {arr[6:16]}")
                    last_data = action
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped")

if __name__ == "__main__":
    main()

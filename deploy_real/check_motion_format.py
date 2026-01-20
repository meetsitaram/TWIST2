#!/usr/bin/env python3
"""Check the format of example motion files."""

import pickle
import numpy as np

def main():
    motion_file = "../assets/example_motions/0807_yanjie_walk_001.pkl"
    
    print(f"Loading {motion_file}...")
    with open(motion_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nType: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if value.ndim == 2 and value.shape[1] <= 50:
                    print(f"    First frame: {value[0]}")
            else:
                print(f"  {key}: {type(value)}")
    elif isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        print(f"First frame: {data[0]}")
    else:
        print(f"Data: {data}")

if __name__ == "__main__":
    main()

# TWIST2 Cloud Training Guide

Train G1 teleop motion imitation policies on cloud GPU instances.

## Prerequisites

- Cloud instance with NVIDIA GPU (A100/H100/L40 recommended)
- CUDA 12.1+ drivers installed
- Isaac Gym downloaded (requires NVIDIA developer account)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/g1-pick-n-place.git
cd g1-pick-n-place/TWIST2
```

### 2. Create Environment

```bash
conda create -n twist2 python=3.8 -y
conda activate twist2

# PyTorch with CUDA 12.1
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Isaac Gym

Download from [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) and upload to your instance.

```bash
cd /path/to/isaacgym/python
pip install -e .

# Verify installation
python -c "import isaacgym; print('Isaac Gym OK')"
```

### 4. Install TWIST2 Dependencies

```bash
cd ~/g1-pick-n-place/TWIST2

# Install legged_gym
cd legged_gym && pip install -e . && cd ..

# Install pose utilities
cd pose && pip install -e . && cd ..

# Additional dependencies
pip install tensorboard rich pyyaml
```

### 5. Train

```bash
cd ~/g1-pick-n-place/TWIST2

# Train on teleop motions
bash train_teleop.sh my_experiment cuda:0

# Or manually:
python legged_gym/legged_gym/scripts/train.py \
    --task g1_teleop \
    --run_name my_experiment \
    --rl_device cuda:0 \
    --sim_device cuda:0 \
    --headless
```

## Training Configuration

### Motion Dataset

The teleop motion data is included in the repo:

```
datasets/teleop_motions/     # 20 converted motion files (~18MB)
motion_data_configs/teleop_dataset.yaml  # Dataset config
```

### Task Config

Edit `legged_gym/legged_gym/envs/g1/g1_teleop_config.py` to adjust:

- `num_envs` - Number of parallel environments (default: 4096)
- `max_iterations` - Training iterations (default: 20000)
- Reward scales for different tracking objectives
- Domain randomization parameters

## Monitoring

### TensorBoard

```bash
# In a separate terminal
tensorboard --logdir logs/ --bind_all

# Access at http://<instance-ip>:6006
```

### Training Output

Checkpoints saved to: `logs/g1_teleop/<run_name>/`

```
logs/g1_teleop/my_experiment/
├── model_*.pt          # Checkpoints
├── events.out.*        # TensorBoard logs
└── config.yaml         # Training config snapshot
```

## Export for Deployment

After training, export to ONNX for sim2sim/sim2real:

```bash
python legged_gym/legged_gym/scripts/export_onnx.py \
    --task g1_teleop \
    --load_run my_experiment \
    --checkpoint model_20000.pt
```

Output: `logs/g1_teleop/my_experiment/exported/policy.onnx`

## Cloud Provider Notes

### Lambda Labs

```bash
# A100 instances work well
# Pre-installed CUDA drivers
# Just install conda and follow steps above
```

### Brev.dev

```bash
# Use PyTorch template for faster setup
# GPU already configured
```

### RunPod

```bash
# Select PyTorch 2.x template
# SSH access for file transfer
```

## Troubleshooting

### CUDA Out of Memory

Reduce `num_envs` in config:

```python
class G1TeleopCfg:
    class env:
        num_envs = 2048  # Reduce from 4096
```

### Isaac Gym Import Error

Ensure isaacgym is installed in the correct environment:

```bash
conda activate twist2
pip show isaacgym  # Should show location
```

### Motion File Not Found

Check paths in `motion_data_configs/teleop_dataset.yaml` - they should be absolute or relative to the TWIST2 directory.

## Downloading Results

After training:

```bash
# From your local machine
scp -r user@cloud-ip:~/g1-pick-n-place/TWIST2/logs/g1_teleop/my_experiment ./

# Or use rsync for large runs
rsync -avz user@cloud-ip:~/g1-pick-n-place/TWIST2/logs/ ./logs/
```

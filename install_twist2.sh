#!/bin/bash
# TWIST2 Installation Script - Resumable Version
# Automatically saves progress and can resume from failures

set +e  # Don't exit on error - we'll handle them

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Progress tracking file
PROGRESS_FILE=".install_progress"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   TWIST2 Installation Script${NC}"
echo -e "${BLUE}   (Resumable)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "rsl_rl" ]; then
    echo -e "${RED}Error: This script must be run from TWIST2 directory${NC}"
    exit 1
fi

# Function to mark step as complete
mark_complete() {
    echo "$1" >> "$PROGRESS_FILE"
    echo -e "${GREEN}✓ Step $1 completed${NC}"
}

# Function to check if step is complete
is_complete() {
    if [ -f "$PROGRESS_FILE" ]; then
        grep -q "^$1$" "$PROGRESS_FILE"
        return $?
    fi
    return 1
}

# Function to reset progress (for fresh install)
reset_progress() {
    rm -f "$PROGRESS_FILE"
    echo -e "${YELLOW}Progress reset. Starting fresh installation.${NC}"
}

# Check for existing progress
if [ -f "$PROGRESS_FILE" ]; then
    echo -e "${CYAN}Found previous installation progress.${NC}"
    echo -e "${CYAN}Completed steps:${NC}"
    cat "$PROGRESS_FILE" | while read line; do
        echo -e "  ${GREEN}✓${NC} Step $line"
    done
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  [c] Continue from where it left off"
    echo "  [r] Reset and start fresh"
    echo "  [q] Quit"
    read -p "Choose option [c/r/q]: " choice
    case $choice in
        r|R)
            reset_progress
            ;;
        q|Q)
            echo "Exiting."
            exit 0
            ;;
        *)
            echo -e "${GREEN}Continuing installation...${NC}"
            ;;
    esac
    echo ""
fi

# Function to handle step execution with error handling
run_step() {
    local step_num=$1
    local step_name=$2
    shift 2
    local step_command="$@"
    
    if is_complete "$step_num"; then
        echo -e "${CYAN}[${step_num}/8] ${step_name} - Already completed, skipping${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}[${step_num}/8] ${step_name}...${NC}"
    
    # Execute the command
    eval "$step_command"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        mark_complete "$step_num"
        return 0
    else
        echo -e "${RED}✗ Step ${step_num} failed with exit code ${exit_code}${NC}"
        echo -e "${YELLOW}Options:${NC}"
        echo "  [r] Retry this step"
        echo "  [s] Skip this step (not recommended)"
        echo "  [q] Quit installation"
        read -p "Choose option [r/s/q]: " choice
        case $choice in
            r|R)
                echo -e "${YELLOW}Retrying step ${step_num}...${NC}"
                run_step "$step_num" "$step_name" "$step_command"
                return $?
                ;;
            s|S)
                echo -e "${YELLOW}Skipping step ${step_num}${NC}"
                mark_complete "$step_num"
                return 0
                ;;
            *)
                echo "Exiting. Run script again to resume from here."
                exit 1
                ;;
        esac
    fi
}

# Initialize conda in this shell
initialize_conda() {
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        fi
    fi
}

# Step 1: Check for conda
step1_check_conda() {
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}Error: conda not found.${NC}"
        echo "Please install Miniconda or Anaconda:"
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  bash Miniconda3-latest-Linux-x86_64.sh"
        return 1
    fi
    echo -e "${GREEN}✓ Conda found: $(conda --version)${NC}"
    initialize_conda
    return 0
}

# Step 2: Create conda environment
step2_create_env() {
    # Remove old environment if it exists and we're starting fresh
    if ! is_complete "2"; then
        conda env remove -n twist2 -y 2>/dev/null || true
    fi
    
    # Create environment
    conda create -n twist2 python=3.8 -y
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Activate it
    conda activate twist2
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to activate environment${NC}"
        return 1
    fi
    
    # Install libpython-static for IsaacGym compatibility
    echo "  Installing libpython-static for IsaacGym..."
    conda install -c conda-forge libpython-static -y
    
    # Set up LD_LIBRARY_PATH for IsaacGym (requires libpython3.8.so)
    echo "  Setting up LD_LIBRARY_PATH for IsaacGym..."
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
    
    # Create activation script
    cat > $CONDA_PREFIX/etc/conda/activate.d/isaacgym_env.sh << 'EOF'
#!/bin/bash
# Save original LD_LIBRARY_PATH
export _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
# Add conda lib to LD_LIBRARY_PATH for IsaacGym
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF
    
    # Create deactivation script to restore original
    cat > $CONDA_PREFIX/etc/conda/deactivate.d/isaacgym_env.sh << 'EOF'
#!/bin/bash
# Restore original LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
unset _OLD_LD_LIBRARY_PATH
EOF
    
    echo -e "${GREEN}✓ Environment 'twist2' created and activated${NC}"
    echo -e "${GREEN}✓ LD_LIBRARY_PATH configured for IsaacGym${NC}"
    return 0
}

# Step 3: Check for IsaacGym
step3_check_isaacgym() {
    conda activate twist2
    
    # Ensure LD_LIBRARY_PATH is set (in case activation script hasn't run yet)
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    
    if python -c "import isaacgym" 2>/dev/null; then
        echo -e "${GREEN}✓ IsaacGym already installed${NC}"
        return 0
    else
        echo -e "${RED}✗ IsaacGym not found${NC}"
        echo -e "${YELLOW}IsaacGym must be installed manually:${NC}"
        echo "  1. Download from: https://developer.nvidia.com/isaac-gym"
        echo "  2. Extract: tar -xf Isaac_Gym_Preview_4_Package.tar.gz"
        echo "  3. Install (with twist2 activated):"
        echo "     conda activate twist2"
        echo "     cd isaacgym/python && pip install -e ."
        echo ""
        echo -e "${YELLOW}Options:${NC}"
        echo "  [i] I've installed it, check again"
        echo "  [s] Skip for now (you can install later)"
        echo "  [q] Quit"
        read -p "Choose option [i/s/q]: " choice
        case $choice in
            i|I)
                if python -c "import isaacgym" 2>/dev/null; then
                    echo -e "${GREEN}✓ IsaacGym now detected!${NC}"
                    return 0
                else
                    echo -e "${RED}Still not found.${NC}"
                    echo -e "${YELLOW}Troubleshooting:${NC}"
                    echo "  - Make sure you ran 'pip install -e .' in isaacgym/python"
                    echo "  - Try: python -c 'import isaacgym' to see the error"
                    return 1
                fi
                ;;
            s|S)
                echo -e "${YELLOW}Skipping IsaacGym check. Install it later before running TWIST2.${NC}"
                return 0
                ;;
            *)
                return 1
                ;;
        esac
    fi
}

# Step 4: Install TWIST2 packages
step4_install_packages() {
    conda activate twist2
    
    echo "  Installing rsl_rl..."
    cd rsl_rl && pip install -e . && cd ..
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install rsl_rl${NC}"
        return 1
    fi
    echo -e "${GREEN}  ✓ rsl_rl installed${NC}"
    
    echo "  Installing legged_gym..."
    cd legged_gym && pip install -e . && cd ..
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install legged_gym${NC}"
        return 1
    fi
    echo -e "${GREEN}  ✓ legged_gym installed${NC}"
    
    echo "  Installing pose..."
    cd pose && pip install -e . && cd ..
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install pose${NC}"
        return 1
    fi
    echo -e "${GREEN}  ✓ pose installed${NC}"
    
    return 0
}

# Step 5: Install core dependencies
step5_install_core_deps() {
    conda activate twist2
    
    pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor zmq
    return $?
}

# Step 6: Install additional packages
step6_install_extras() {
    conda activate twist2
    
    pip install redis[hiredis] pyttsx3 onnx onnxruntime-gpu customtkinter
    return $?
}

# Step 7: Setup Redis
step7_setup_redis() {
    if command -v redis-server &> /dev/null; then
        echo -e "${GREEN}✓ Redis already installed${NC}"
        
        # Check if running
        if redis-cli ping &> /dev/null; then
            echo -e "${GREEN}✓ Redis is running${NC}"
            return 0
        else
            echo -e "${YELLOW}Starting Redis...${NC}"
            sudo systemctl start redis-server 2>/dev/null || redis-server --daemonize yes
            sleep 2
            if redis-cli ping &> /dev/null; then
                echo -e "${GREEN}✓ Redis started${NC}"
                return 0
            else
                echo -e "${YELLOW}Warning: Could not start Redis. You may need to start it manually.${NC}"
                return 0  # Don't fail the installation
            fi
        fi
    else
        echo -e "${YELLOW}Installing Redis...${NC}"
        sudo apt update
        sudo apt install -y redis-server
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Warning: Could not install Redis. You may need to install it manually.${NC}"
            return 0  # Don't fail the installation
        fi
        sudo systemctl enable redis-server
        sudo systemctl start redis-server
        echo -e "${GREEN}✓ Redis installed${NC}"
        return 0
    fi
}

# Step 8: Verify installation
step8_verify() {
    # Run verification with proper conda activation
    bash -c "
    source $CONDA_PREFIX/../etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
    conda activate twist2
    python << 'EOF'
import sys

packages = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'mujoco': 'MuJoCo',
    'onnxruntime': 'ONNX Runtime',
    'redis': 'Redis',
    'cv2': 'OpenCV',
    'rich': 'Rich',
}

# IsaacGym verification
try:
    import isaacgym
    print('  ✓ IsaacGym')
    packages['isaacgym'] = 'IsaacGym'
except ImportError:
    print('  ⚠ IsaacGym - Failed to import in verification (may still work normally)')

all_ok = True
for module, name in packages.items():
    if module == 'isaacgym':
        continue  # Already checked above
    try:
        __import__(module)
        print(f'  ✓ {name}')
    except ImportError:
        print(f'  ✗ {name} - MISSING!')
        all_ok = False

if not all_ok:
    sys.exit(1)
EOF
    "
    
    return $?
}

# Run all steps
run_step 1 "Checking for conda" "step1_check_conda"
run_step 2 "Creating conda environment" "step2_create_env"
run_step 3 "Checking for IsaacGym" "step3_check_isaacgym"
run_step 4 "Installing TWIST2 packages" "step4_install_packages"
run_step 5 "Installing core dependencies" "step5_install_core_deps"
run_step 6 "Installing additional packages" "step6_install_extras"
run_step 7 "Setting up Redis" "step7_setup_redis"
run_step 8 "Verifying installation" "step8_verify"

# Final summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}   Installation Complete! 🎉${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Environment: twist2 (Python 3.8)"
echo ""
echo "Next steps:"
echo "  1. Activate environment:"
echo -e "     ${CYAN}conda activate twist2${NC}"
echo ""
echo "  2. Test simulation:"
echo -e "     ${CYAN}bash sim2sim.sh${NC}"
echo ""
echo "  3. (Optional) Configure Redis for network access:"
echo -e "     ${CYAN}sudo nano /etc/redis/redis.conf${NC}"
echo "     Change: bind 0.0.0.0"
echo "     Change: protected-mode no"
echo -e "     Then: ${CYAN}sudo systemctl restart redis-server${NC}"
echo ""

# Check if IsaacGym was installed (with proper LD_LIBRARY_PATH)
conda activate twist2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
if ! python -c "import isaacgym" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Important: IsaacGym is not installed yet!${NC}"
    echo "Download and install from: https://developer.nvidia.com/isaac-gym"
    echo ""
else
    echo -e "${GREEN}✓ IsaacGym is installed and working${NC}"
fi

# Clean up progress file on successful completion
if is_complete "8"; then
    echo -e "${GREEN}Removing progress tracking file...${NC}"
    rm -f "$PROGRESS_FILE"
fi

echo -e "${GREEN}Installation script completed successfully!${NC}"
echo ""

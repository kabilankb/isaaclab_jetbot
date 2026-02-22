#!/usr/bin/env bash
# ==============================================================================
# IsaacLabTutorial - Launch Script
# ==============================================================================
# Usage: ./launch.sh <command> [options]
#
# Prerequisites:
#   - NVIDIA Isaac Sim 4.5.0+ installed
#   - Isaac Lab framework installed
#   - NVIDIA GPU with CUDA support
# ==============================================================================

set -euo pipefail

# ---- Configuration -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_NAME="Isaac-Lab-Tutorial-SphereFollow-Direct-v0"
DEFAULT_NUM_ENVS=100
DEFAULT_ALGORITHM="PPO"
DEFAULT_DEVICE="cuda:0"
DEFAULT_ML_FRAMEWORK="torch"

# ---- Colors ------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ---- Helper Functions --------------------------------------------------------
print_header() {
    echo -e "${CYAN}"
    echo "=================================================================="
    echo "  IsaacLabTutorial - JetBot Navigation RL Environment"
    echo "=================================================================="
    echo -e "${NC}"
}

print_usage() {
    print_header
    echo -e "${GREEN}Usage:${NC} ./launch.sh <command> [options]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo "  install            Install the isaac_lab_tutorial package (editable mode)"
    echo "  list-envs          List all registered environments"
    echo "  random-agent       Run environment with random actions"
    echo "  zero-agent         Run environment with zero actions"
    echo "  train              Train RL agent (PPO/AMP) using skrl"
    echo "  play               Evaluate a trained agent checkpoint"
    echo "  help               Show this help message"
    echo ""
    echo -e "${YELLOW}Options (for random-agent / zero-agent):${NC}"
    echo "  --num_envs N       Number of parallel environments (default: ${DEFAULT_NUM_ENVS})"
    echo "  --device DEVICE    Compute device (default: ${DEFAULT_DEVICE})"
    echo ""
    echo -e "${YELLOW}Options (for train):${NC}"
    echo "  --algorithm ALG    RL algorithm: PPO, AMP, IPPO, MAPPO (default: ${DEFAULT_ALGORITHM})"
    echo "  --num_envs N       Number of parallel environments (default: ${DEFAULT_NUM_ENVS})"
    echo "  --max_iterations N Maximum training iterations"
    echo "  --seed N           Random seed"
    echo "  --checkpoint PATH  Resume training from checkpoint"
    echo "  --ml_framework FW  ML framework: torch, jax, jax-numpy (default: ${DEFAULT_ML_FRAMEWORK})"
    echo "  --distributed      Enable multi-GPU training"
    echo "  --video            Record training videos"
    echo ""
    echo -e "${YELLOW}Options (for play):${NC}"
    echo "  --checkpoint PATH  Path to trained model checkpoint"
    echo "  --algorithm ALG    Algorithm used during training (default: ${DEFAULT_ALGORITHM})"
    echo "  --num_envs N       Number of environments (default: ${DEFAULT_NUM_ENVS})"
    echo "  --real-time        Run evaluation in real-time"
    echo "  --video            Record evaluation video"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./launch.sh install"
    echo "  ./launch.sh train --algorithm PPO --num_envs 100"
    echo "  ./launch.sh train --algorithm AMP --num_envs 50 --max_iterations 1000"
    echo "  ./launch.sh play --checkpoint logs/skrl/.../checkpoints/best_agent.pt"
    echo "  ./launch.sh random-agent --num_envs 64"
    echo ""
}

check_prerequisites() {
    echo -e "${CYAN}[CHECK]${NC} Verifying prerequisites..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}[ERROR]${NC} Python 3 is not installed."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "  Python version: ${GREEN}${PYTHON_VERSION}${NC}"

    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo -e "  GPU: ${GREEN}${GPU_NAME}${NC}"
    else
        echo -e "  ${YELLOW}[WARN]${NC} nvidia-smi not found. NVIDIA GPU required for Isaac Sim."
    fi

    # Check if Isaac Sim / Isaac Lab are importable
    if python3 -c "import isaaclab" 2>/dev/null; then
        echo -e "  Isaac Lab: ${GREEN}available${NC}"
    else
        echo -e "  Isaac Lab: ${RED}not found${NC} - install Isaac Lab first"
        echo -e "  See: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html"
    fi

    echo ""
}

# ---- Commands ----------------------------------------------------------------

cmd_install() {
    echo -e "${CYAN}[INSTALL]${NC} Installing isaac_lab_tutorial package..."
    check_prerequisites
    cd "${SCRIPT_DIR}/source/isaac_lab_tutorial"
    pip install -e .
    echo -e "${GREEN}[DONE]${NC} Package installed successfully."
    echo -e "  Run '${YELLOW}./launch.sh list-envs${NC}' to verify."
}

cmd_list_envs() {
    echo -e "${CYAN}[LIST]${NC} Listing registered environments..."
    python3 "${SCRIPT_DIR}/scripts/list_envs.py"
}

cmd_random_agent() {
    local extra_args=("$@")
    echo -e "${CYAN}[RUN]${NC} Running random action agent..."
    echo -e "  Task: ${YELLOW}${TASK_NAME}${NC}"
    python3 "${SCRIPT_DIR}/scripts/random_agent.py" \
        --task "${TASK_NAME}" \
        --num_envs "${DEFAULT_NUM_ENVS}" \
        "${extra_args[@]}"
}

cmd_zero_agent() {
    local extra_args=("$@")
    echo -e "${CYAN}[RUN]${NC} Running zero action agent..."
    echo -e "  Task: ${YELLOW}${TASK_NAME}${NC}"
    python3 "${SCRIPT_DIR}/scripts/zero_agent.py" \
        --task "${TASK_NAME}" \
        --num_envs "${DEFAULT_NUM_ENVS}" \
        "${extra_args[@]}"
}

cmd_train() {
    local extra_args=("$@")
    echo -e "${CYAN}[TRAIN]${NC} Starting RL training..."
    echo -e "  Task: ${YELLOW}${TASK_NAME}${NC}"
    python3 "${SCRIPT_DIR}/scripts/skrl/train.py" \
        --task "${TASK_NAME}" \
        "${extra_args[@]}"
}

cmd_play() {
    local extra_args=("$@")
    echo -e "${CYAN}[PLAY]${NC} Evaluating trained agent..."
    echo -e "  Task: ${YELLOW}${TASK_NAME}${NC}"
    python3 "${SCRIPT_DIR}/scripts/skrl/play.py" \
        --task "${TASK_NAME}" \
        "${extra_args[@]}"
}

# ---- Main Entry Point --------------------------------------------------------

if [[ $# -eq 0 ]]; then
    print_usage
    exit 0
fi

COMMAND="$1"
shift

case "${COMMAND}" in
    install)
        cmd_install
        ;;
    list-envs|list_envs)
        cmd_list_envs
        ;;
    random-agent|random_agent)
        cmd_random_agent "$@"
        ;;
    zero-agent|zero_agent)
        cmd_zero_agent "$@"
        ;;
    train)
        cmd_train "$@"
        ;;
    play|eval|evaluate)
        cmd_play "$@"
        ;;
    check)
        check_prerequisites
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}[ERROR]${NC} Unknown command: ${COMMAND}"
        echo ""
        print_usage
        exit 1
        ;;
esac

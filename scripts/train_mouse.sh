#!/bin/bash
# Mouse Training Script
# Created: 2025-11-24
# Purpose: Simplified training execution for mouse dataset

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV="3danimals"
PROJECT_DIR="/home/joon/dev/3DAnimals"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check data
check_data() {
    print_info "Checking dataset..."
    DATA_DIR="${PROJECT_DIR}/data/fauna_mouse/large_scale/mouse_dannce_6view/train"

    if [ ! -d "$DATA_DIR" ]; then
        print_error "Data directory not found: $DATA_DIR"
        exit 1
    fi

    # Count sequences (directories)
    NUM_SEQS=$(ls -d ${DATA_DIR}/*/ 2>/dev/null | wc -l)
    print_info "Found $NUM_SEQS sequences in dataset"

    if [ "$NUM_SEQS" -eq 0 ]; then
        print_error "No sequences found in $DATA_DIR"
        exit 1
    fi

    # Check first sequence files
    FIRST_SEQ=$(ls -d ${DATA_DIR}/*/ 2>/dev/null | head -1)
    NUM_RGB=$(ls ${FIRST_SEQ}*_rgb.png 2>/dev/null | wc -l)
    NUM_MASK=$(ls ${FIRST_SEQ}*_mask.png 2>/dev/null | wc -l)
    NUM_BOX=$(ls ${FIRST_SEQ}*_box.txt 2>/dev/null | wc -l)
    NUM_META=$(ls ${FIRST_SEQ}*_metadata.json 2>/dev/null | wc -l)

    print_info "First sequence files: RGB=$NUM_RGB, Mask=$NUM_MASK, Box=$NUM_BOX, Metadata=$NUM_META"

    if [ "$NUM_RGB" -eq 0 ] || [ "$NUM_MASK" -eq 0 ]; then
        print_error "Missing required files (RGB or mask) in $FIRST_SEQ"
        exit 1
    fi

    print_info "✓ Dataset check passed!"
}

# Function to check GPU
check_gpu() {
    print_info "Checking GPU..."
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found! GPU required for training."
        exit 1
    fi

    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    print_info "✓ GPU check passed!"
}

# Function to run debug training
run_debug() {
    print_info "Starting DEBUG training (5K iterations, ~15-20 minutes)..."
    print_warning "This is a quick validation run. Use 'full' mode for actual training."

    cd "$PROJECT_DIR"
    conda run -n "$CONDA_ENV" python run.py --config-name train_mouse_debug

    print_info "✓ Debug training completed!"
}

# Function to run full training
run_full() {
    print_info "Starting FULL training (50K iterations, ~2-3 hours)..."
    print_warning "This will take 2-3 hours. Consider using 'background' mode."

    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Training cancelled."
        exit 0
    fi

    cd "$PROJECT_DIR"
    conda run -n "$CONDA_ENV" python run.py --config-name train_mouse

    print_info "✓ Full training completed!"
}

# Function to run background training
run_background() {
    print_info "Starting BACKGROUND training (50K iterations, ~2-3 hours)..."

    LOG_FILE="/tmp/mouse_training_$(date +%Y%m%d_%H%M%S).log"

    cd "$PROJECT_DIR"
    nohup conda run -n "$CONDA_ENV" python run.py --config-name train_mouse > "$LOG_FILE" 2>&1 &

    PID=$!
    print_info "✓ Training started in background!"
    print_info "   PID: $PID"
    print_info "   Log: $LOG_FILE"
    print_info ""
    print_info "Monitor with: tail -f $LOG_FILE"
    print_info "Stop with: kill $PID"
    print_info "Check GPU: nvidia-smi"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [mode]"
    echo ""
    echo "Modes:"
    echo "  debug       - Quick validation (5K iters, 15-20 min)"
    echo "  full        - Full training (50K iters, 2-3 hours)"
    echo "  background  - Background training (50K iters, logs to /tmp)"
    echo ""
    echo "Examples:"
    echo "  $0 debug       # Always run this first!"
    echo "  $0 full        # Interactive full training"
    echo "  $0 background  # Long training without blocking terminal"
}

# Main script
main() {
    print_info "=== Mouse Training Script ==="
    print_info "Dataset: DANNCE 6-view at data/fauna_mouse/large_scale/mouse_dannce_6view/"
    print_info "Config: config/train_mouse*.yaml"
    print_info ""

    # Check prerequisites
    check_data
    check_gpu
    echo ""

    # Parse mode
    MODE="${1:-}"

    case "$MODE" in
        debug)
            run_debug
            ;;
        full)
            run_full
            ;;
        background)
            run_background
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"

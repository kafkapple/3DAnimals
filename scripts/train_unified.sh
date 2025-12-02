#!/bin/bash
# Unified Training Script for MagicPony, Ponymation, Fauna
# Created: 2025-12-02
# Purpose: 세 모델(Fauna, MagicPony, Ponymation)을 일관된 방식으로 학습
#
# Usage:
#   ./scripts/train_unified.sh <model> <mode>
#
# Models:
#   fauna         - Fauna model (기본)
#   magicpony     - MagicPony model
#   ponymation-s1 - Ponymation Stage 1 (관절 학습)
#   ponymation-s2 - Ponymation Stage 2 (Motion VAE)
#
# Modes:
#   debug      - 빠른 검증 (2-5K iters, 10-20분)
#   full       - 전체 학습
#   background - 백그라운드 학습

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
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

# Function to check data for each model
check_data() {
    local MODEL=$1
    print_info "Checking data for $MODEL..."

    case "$MODEL" in
        fauna)
            DATA_DIR="${PROJECT_DIR}/data/fauna_mouse/large_scale/mouse_dannce_6view/train"
            ;;
        magicpony)
            DATA_DIR="${PROJECT_DIR}/data/magicpony/mouse/train"
            ;;
        ponymation-s1|ponymation-s2)
            DATA_DIR="${PROJECT_DIR}/data/ponymation/mouse/train"
            ;;
        *)
            print_error "Unknown model: $MODEL"
            exit 1
            ;;
    esac

    if [ ! -d "$DATA_DIR" ]; then
        print_error "Data directory not found: $DATA_DIR"
        print_info "Run data conversion scripts first:"
        print_info "  python scripts/convert_fauna_to_magicpony.py --source ... --target ..."
        print_info "  python scripts/convert_fauna_to_ponymation.py --source ... --target ..."
        exit 1
    fi

    # Count samples
    NUM_SAMPLES=$(ls -d ${DATA_DIR}/*/ 2>/dev/null | wc -l)
    print_info "Found $NUM_SAMPLES samples/sequences in $DATA_DIR"

    if [ "$NUM_SAMPLES" -eq 0 ]; then
        print_error "No samples found in $DATA_DIR"
        exit 1
    fi

    print_info "✓ Data check passed for $MODEL!"
}

# Get config name based on model and mode
get_config_name() {
    local MODEL=$1
    local MODE=$2

    case "$MODEL" in
        fauna)
            case "$MODE" in
                debug) echo "train_fauna_mouse_6view_debug" ;;
                full|background) echo "train_fauna_mouse_large" ;;
            esac
            ;;
        magicpony)
            case "$MODE" in
                debug) echo "train_magicpony_mouse_debug" ;;
                full|background) echo "train_magicpony_mouse" ;;
            esac
            ;;
        ponymation-s1)
            case "$MODE" in
                debug) echo "train_ponymation_mouse_stage1_debug" ;;
                full|background) echo "train_ponymation_mouse_stage1" ;;
            esac
            ;;
        ponymation-s2)
            # Stage 2 doesn't have debug (requires Stage 1 checkpoint)
            echo "train_ponymation_mouse_stage2"
            ;;
        *)
            print_error "Unknown model: $MODEL"
            exit 1
            ;;
    esac
}

# Run training
run_training() {
    local MODEL=$1
    local MODE=$2
    local CONFIG=$3

    print_header "Starting $MODEL training ($MODE mode)"
    print_info "Config: $CONFIG"

    cd "$PROJECT_DIR"

    case "$MODE" in
        debug)
            print_info "Debug mode: 빠른 검증 (10-20분)"
            conda run -n "$CONDA_ENV" python run.py --config-name "$CONFIG"
            print_info "✓ Debug training completed!"
            ;;
        full)
            print_warning "Full mode: 전체 학습 시작"
            read -p "Continue? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_info "Training cancelled."
                exit 0
            fi
            conda run -n "$CONDA_ENV" python run.py --config-name "$CONFIG"
            print_info "✓ Full training completed!"
            ;;
        background)
            LOG_FILE="/tmp/${MODEL}_training_$(date +%Y%m%d_%H%M%S).log"
            print_info "Background mode: 로그 파일 -> $LOG_FILE"

            nohup conda run -n "$CONDA_ENV" python run.py --config-name "$CONFIG" > "$LOG_FILE" 2>&1 &

            PID=$!
            print_info "✓ Training started in background!"
            print_info "   PID: $PID"
            print_info "   Log: $LOG_FILE"
            print_info ""
            print_info "Monitor with: tail -f $LOG_FILE"
            print_info "Stop with: kill $PID"
            print_info "Check GPU: nvidia-smi"
            ;;
    esac
}

# Show usage
show_usage() {
    echo "Unified Training Script for 3DAnimals"
    echo ""
    echo "Usage: $0 <model> <mode>"
    echo ""
    echo "Models:"
    echo "  fauna         - Fauna model (범용 3D 동물 재구성)"
    echo "  magicpony     - MagicPony model (단일 이미지 3D 재구성)"
    echo "  ponymation-s1 - Ponymation Stage 1 (관절 학습)"
    echo "  ponymation-s2 - Ponymation Stage 2 (Motion VAE)"
    echo ""
    echo "Modes:"
    echo "  debug      - 빠른 검증 (2-5K iters, 10-20분)"
    echo "  full       - 전체 학습 (대화형)"
    echo "  background - 백그라운드 학습 (로그 /tmp)"
    echo ""
    echo "Examples:"
    echo "  $0 fauna debug           # Fauna 빠른 테스트"
    echo "  $0 magicpony debug       # MagicPony 빠른 테스트"
    echo "  $0 ponymation-s1 debug   # Ponymation Stage 1 테스트"
    echo "  $0 fauna full            # Fauna 전체 학습"
    echo "  $0 magicpony background  # MagicPony 백그라운드 학습"
    echo ""
    echo "Training Dependencies:"
    echo "  Fauna       -> 독립적 (바로 학습 가능)"
    echo "  MagicPony   -> 독립적 (바로 학습 가능)"
    echo "  Ponymation Stage 1 -> MagicPony 체크포인트 필요"
    echo "  Ponymation Stage 2 -> Stage 1 체크포인트 필요"
    echo ""
    echo "Data Directories:"
    echo "  Fauna:      data/fauna_mouse/large_scale/mouse_dannce_6view/"
    echo "  MagicPony:  data/magicpony/mouse/"
    echo "  Ponymation: data/ponymation/mouse/"
}

# Main script
main() {
    # Parse arguments
    MODEL="${1:-}"
    MODE="${2:-debug}"

    if [ -z "$MODEL" ]; then
        show_usage
        exit 1
    fi

    # Validate model
    case "$MODEL" in
        fauna|magicpony|ponymation-s1|ponymation-s2)
            ;;
        ponymation)
            print_warning "Ponymation requires 2-stage training:"
            print_info "  Stage 1: $0 ponymation-s1 $MODE"
            print_info "  Stage 2: $0 ponymation-s2 $MODE"
            exit 0
            ;;
        *)
            print_error "Unknown model: $MODEL"
            show_usage
            exit 1
            ;;
    esac

    # Validate mode
    case "$MODE" in
        debug|full|background)
            ;;
        *)
            print_error "Unknown mode: $MODE"
            show_usage
            exit 1
            ;;
    esac

    print_header "3DAnimals Unified Training"
    print_info "Model: $MODEL"
    print_info "Mode: $MODE"
    echo ""

    # Check prerequisites
    check_gpu
    check_data "$MODEL"
    echo ""

    # Get config name
    CONFIG=$(get_config_name "$MODEL" "$MODE")

    # Run training
    run_training "$MODEL" "$MODE" "$CONFIG"
}

# Run main
main "$@"

#!/bin/bash
# Unified Training Script for MagicPony, Ponymation, Fauna
# Created: 2025-12-02 (Updated)
# Purpose: 세 모델(Fauna, MagicPony, Ponymation)을 일관된 방식으로 학습
#
# Usage:
#   ./scripts/train_unified.sh <model> <mode> [training_type]
#
# Models:
#   fauna              - Fauna model
#   magicpony          - MagicPony model
#   ponymation-s1      - Ponymation Stage 1
#   ponymation-s2      - Ponymation Stage 2
#
# Modes:
#   debug      - 빠른 검증 (2-5K iters, 10-20분)
#   full       - 전체 학습
#   background - 백그라운드 학습
#
# Training Types:
#   scratch    - From scratch (default for fauna)
#   finetune   - From pretrained model (default for magicpony/ponymation)

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV="3danimals"
# Auto-detect project directory (works on both local and remote machines)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

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

print_section() {
    echo -e "${CYAN}--- $1 ---${NC}"
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

# Function to check pretrained model for finetune
check_pretrained() {
    local MODEL=$1
    local TRAINING_TYPE=$2

    if [ "$TRAINING_TYPE" != "finetune" ]; then
        return 0
    fi

    print_info "Checking pretrained model for $MODEL..."

    case "$MODEL" in
        magicpony)
            PRETRAINED="${PROJECT_DIR}/results/magicpony/pretrained_horse/pretrained_horse.pth"
            if [ ! -f "$PRETRAINED" ]; then
                print_error "Pretrained horse model not found: $PRETRAINED"
                print_info "Download with: cd results/magicpony && bash download_pretrained_magicpony.sh"
                exit 1
            fi
            ;;
        ponymation-s1)
            # Check for MagicPony checkpoint (either finetuned or pretrained horse)
            PRETRAINED_FINETUNED="${PROJECT_DIR}/results/magicpony/mouse_finetune/checkpoint.pth"
            PRETRAINED_HORSE="${PROJECT_DIR}/results/magicpony/pretrained_horse/pretrained_horse.pth"
            if [ -f "$PRETRAINED_FINETUNED" ]; then
                print_info "Using finetuned MagicPony mouse: $PRETRAINED_FINETUNED"
            elif [ -f "$PRETRAINED_HORSE" ]; then
                print_info "Using pretrained MagicPony horse: $PRETRAINED_HORSE"
            else
                print_error "No MagicPony checkpoint found!"
                print_info "Either finetune MagicPony first or download pretrained horse."
                exit 1
            fi
            ;;
        ponymation-s2)
            # Check for Stage 1 checkpoint (auto-detect latest checkpoint*.pth)
            # Search order: finetune full → finetune debug → scratch full → scratch debug
            STAGE1_FINETUNE_FULL="${PROJECT_DIR}/results/ponymation/mouse_finetune_stage1"
            STAGE1_FINETUNE_DEBUG="${PROJECT_DIR}/results/ponymation/mouse_finetune_stage1_debug"
            STAGE1_SCRATCH_FULL="${PROJECT_DIR}/results/ponymation/mouse_stage1"
            STAGE1_SCRATCH_DEBUG="${PROJECT_DIR}/results/ponymation/mouse_stage1_debug"

            # Find latest checkpoint in directory
            find_latest_checkpoint() {
                local dir=$1
                ls -t "$dir"/checkpoint*.pth 2>/dev/null | head -1
            }

            # Try finetune first, then scratch
            STAGE1_CKPT=$(find_latest_checkpoint "$STAGE1_FINETUNE_FULL")
            if [ -n "$STAGE1_CKPT" ]; then
                print_info "Using Stage 1 FINETUNE checkpoint: $STAGE1_CKPT"
            else
                STAGE1_CKPT=$(find_latest_checkpoint "$STAGE1_FINETUNE_DEBUG")
                if [ -n "$STAGE1_CKPT" ]; then
                    print_warning "Using Stage 1 FINETUNE DEBUG checkpoint: $STAGE1_CKPT"
                else
                    STAGE1_CKPT=$(find_latest_checkpoint "$STAGE1_SCRATCH_FULL")
                    if [ -n "$STAGE1_CKPT" ]; then
                        print_info "Using Stage 1 SCRATCH checkpoint: $STAGE1_CKPT"
                    else
                        STAGE1_CKPT=$(find_latest_checkpoint "$STAGE1_SCRATCH_DEBUG")
                        if [ -n "$STAGE1_CKPT" ]; then
                            print_warning "Using Stage 1 SCRATCH DEBUG checkpoint: $STAGE1_CKPT"
                        fi
                    fi
                fi
            fi

            if [ -z "$STAGE1_CKPT" ]; then
                print_error "Stage 1 checkpoint not found!"
                print_info "Run Stage 1 first:"
                print_info "  Finetune: $0 ponymation-s1 debug finetune"
                print_info "  Scratch:  $0 ponymation-s1 debug scratch"
                exit 1
            fi

            # Export for use in run command
            export STAGE1_CHECKPOINT="$STAGE1_CKPT"
            ;;
        fauna)
            PRETRAINED="${PROJECT_DIR}/results/fauna/pretrained_fauna/pretrained_fauna.pth"
            if [ ! -f "$PRETRAINED" ]; then
                print_error "Pretrained Fauna model not found: $PRETRAINED"
                print_info "Download with: cd results/fauna && bash download_pretrained_fauna.sh"
                exit 1
            fi
            ;;
    esac

    print_info "✓ Pretrained model check passed!"
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

# Get config name based on model, mode, and training type
get_config_name() {
    local MODEL=$1
    local MODE=$2
    local TRAINING_TYPE=$3

    case "$MODEL" in
        fauna)
            if [ "$TRAINING_TYPE" == "finetune" ]; then
                case "$MODE" in
                    debug) echo "train_fauna_mouse_6view_finetune" ;;  # No debug finetune yet
                    full|background) echo "train_fauna_mouse_large_finetune" ;;
                esac
            else
                case "$MODE" in
                    debug) echo "train_fauna_mouse_6view_debug" ;;
                    full|background) echo "train_fauna_mouse_large" ;;
                esac
            fi
            ;;
        magicpony)
            if [ "$TRAINING_TYPE" == "finetune" ]; then
                case "$MODE" in
                    debug) echo "finetune_magicpony_mouse_debug" ;;
                    full|background) echo "finetune_magicpony_mouse" ;;
                esac
            else
                case "$MODE" in
                    debug) echo "train_magicpony_mouse_debug" ;;
                    full|background) echo "train_magicpony_mouse" ;;
                esac
            fi
            ;;
        ponymation-s1)
            if [ "$TRAINING_TYPE" == "finetune" ]; then
                case "$MODE" in
                    debug) echo "finetune_ponymation_mouse_stage1_debug" ;;
                    full|background) echo "finetune_ponymation_mouse_stage1" ;;
                esac
            else
                case "$MODE" in
                    debug) echo "train_ponymation_mouse_stage1_debug" ;;
                    full|background) echo "train_ponymation_mouse_stage1" ;;
                esac
            fi
            ;;
        ponymation-s2)
            if [ "$TRAINING_TYPE" == "finetune" ]; then
                case "$MODE" in
                    debug) echo "finetune_ponymation_mouse_stage2_debug" ;;
                    full|background) echo "finetune_ponymation_mouse_stage2" ;;
                esac
            else
                case "$MODE" in
                    debug) echo "train_ponymation_mouse_stage2" ;;  # No scratch debug
                    full|background) echo "train_ponymation_mouse_stage2" ;;
                esac
            fi
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
    local TRAINING_TYPE=$4

    print_header "Starting $MODEL training"
    print_info "Config: $CONFIG"
    print_info "Mode: $MODE"
    print_info "Type: $TRAINING_TYPE"

    cd "$PROJECT_DIR"

    # Build extra args for Stage 2 (checkpoint_path override)
    local EXTRA_ARGS=""
    if [ -n "$STAGE1_CHECKPOINT" ]; then
        print_info "Stage 1 checkpoint: $STAGE1_CHECKPOINT"
        # Use ++ prefix to override existing key in Hydra config
        EXTRA_ARGS="++checkpoint_path=$STAGE1_CHECKPOINT"
    fi

    case "$MODE" in
        debug)
            print_info "Debug mode: 빠른 검증 (10-20분)"
            if [ -n "$EXTRA_ARGS" ]; then
                conda run --no-capture-output -n "$CONDA_ENV" python run.py --config-name "$CONFIG" $EXTRA_ARGS
            else
                conda run --no-capture-output -n "$CONDA_ENV" python run.py --config-name "$CONFIG"
            fi
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
            if [ -n "$EXTRA_ARGS" ]; then
                conda run --no-capture-output -n "$CONDA_ENV" python run.py --config-name "$CONFIG" $EXTRA_ARGS
            else
                conda run --no-capture-output -n "$CONDA_ENV" python run.py --config-name "$CONFIG"
            fi
            print_info "✓ Full training completed!"
            ;;
        background)
            LOG_FILE="/tmp/${MODEL}_${TRAINING_TYPE}_$(date +%Y%m%d_%H%M%S).log"
            print_info "Background mode: 로그 파일 -> $LOG_FILE"

            if [ -n "$EXTRA_ARGS" ]; then
                nohup conda run -n "$CONDA_ENV" python run.py --config-name "$CONFIG" $EXTRA_ARGS > "$LOG_FILE" 2>&1 &
            else
                nohup conda run -n "$CONDA_ENV" python run.py --config-name "$CONFIG" > "$LOG_FILE" 2>&1 &
            fi

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
    echo "Usage: $0 <model> <mode> [training_type]"
    echo ""
    echo "Models:"
    echo "  fauna              - Fauna model (범용 3D 동물 재구성)"
    echo "  magicpony          - MagicPony model (단일 이미지 3D 재구성)"
    echo "  ponymation-s1      - Ponymation Stage 1 (관절 학습)"
    echo "  ponymation-s2      - Ponymation Stage 2 (Motion VAE)"
    echo ""
    echo "Modes:"
    echo "  debug      - 빠른 검증 (2-5K iters, 10-20분)"
    echo "  full       - 전체 학습 (대화형)"
    echo "  background - 백그라운드 학습 (로그 /tmp)"
    echo ""
    echo "Training Types:"
    echo "  scratch    - From scratch (처음부터 학습)"
    echo "  finetune   - From pretrained (사전학습 모델에서 시작) [권장]"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # From Scratch (처음부터)"
    echo "  $0 fauna debug scratch           # Fauna scratch 테스트"
    echo "  $0 magicpony debug scratch       # MagicPony scratch 테스트"
    echo ""
    echo "  # Fine-tuning (사전학습에서 시작) [권장]"
    echo "  $0 magicpony debug finetune      # MagicPony finetune (horse→mouse)"
    echo "  $0 magicpony full finetune       # MagicPony finetune 전체"
    echo "  $0 ponymation-s1 debug finetune  # Ponymation S1 finetune"
    echo "  $0 ponymation-s2 debug finetune  # Ponymation S2 finetune"
    echo ""
    echo "Training Flow (Fine-tuning 권장 순서):"
    echo "  1. MagicPony horse pretrained → MagicPony mouse finetune"
    echo "  2. MagicPony mouse → Ponymation Stage 1"
    echo "  3. Ponymation Stage 1 → Ponymation Stage 2"
    echo ""
    echo "Prerequisites:"
    echo "  - Pretrained models: bash results/{model}/download_pretrained_{model}.sh"
    echo "  - Data conversion:   python scripts/convert_fauna_to_{model}.py ..."
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
    TRAINING_TYPE="${3:-}"

    if [ -z "$MODEL" ]; then
        show_usage
        exit 1
    fi

    # Set default training type based on model
    if [ -z "$TRAINING_TYPE" ]; then
        case "$MODEL" in
            fauna)
                TRAINING_TYPE="scratch"  # Fauna default: scratch (pretrained already exists)
                ;;
            magicpony|ponymation-s1|ponymation-s2)
                TRAINING_TYPE="finetune"  # MagicPony/Ponymation default: finetune
                ;;
        esac
    fi

    # Validate model
    case "$MODEL" in
        fauna|magicpony|ponymation-s1|ponymation-s2)
            ;;
        ponymation)
            print_warning "Ponymation requires 2-stage training:"
            print_info "  Stage 1: $0 ponymation-s1 $MODE $TRAINING_TYPE"
            print_info "  Stage 2: $0 ponymation-s2 $MODE $TRAINING_TYPE"
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

    # Validate training type
    case "$TRAINING_TYPE" in
        scratch|finetune)
            ;;
        *)
            print_error "Unknown training type: $TRAINING_TYPE"
            show_usage
            exit 1
            ;;
    esac

    print_header "3DAnimals Unified Training"
    print_info "Model: $MODEL"
    print_info "Mode: $MODE"
    print_info "Type: $TRAINING_TYPE"
    echo ""

    # Check prerequisites
    check_gpu
    check_data "$MODEL"
    check_pretrained "$MODEL" "$TRAINING_TYPE"
    echo ""

    # Get config name
    CONFIG=$(get_config_name "$MODEL" "$MODE" "$TRAINING_TYPE")

    # Verify config exists
    CONFIG_FILE="${PROJECT_DIR}/config/${CONFIG}.yaml"
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    print_info "Config file: $CONFIG_FILE"

    # Run training
    run_training "$MODEL" "$MODE" "$CONFIG" "$TRAINING_TYPE"
}

# Run main
main "$@"

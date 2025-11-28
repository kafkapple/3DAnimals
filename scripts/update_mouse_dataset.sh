#!/bin/bash
# Update Mouse Dataset Pipeline
# Replaces old mouse data with new SAM3D processed data
#
# Usage:
#   ./scripts/update_mouse_dataset.sh --source /path/to/sam3d/output --animal mouse
#   ./scripts/update_mouse_dataset.sh -s /path/to/sam3d/output -a mouse -r 0.8,0.1,0.1
#   ./scripts/update_mouse_dataset.sh --help

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Determine project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
SAM3D_SOURCE=""
ANIMAL="mouse"
SPLIT_RATIO="0.7,0.15,0.15"
SPLIT_MODE="frame"
USE_COPY="--copy"
SKIP_TEST=false

# Help message
show_help() {
    cat << EOF
Usage: $(basename $0) [OPTIONS]

Update dataset pipeline: SAM3D GUI output → Preprocessing → Train/Val/Test split → Ready to train

OPTIONS:
    -s, --source PATH     SAM3D GUI output directory (REQUIRED)
                          Example: /home/joon/dev/sam3d_gui/outputs/fauna_datasets/mouse

    -a, --animal NAME     Animal name (default: mouse)
                          Used for directory naming and config generation

    -r, --ratio RATIO     Train/Val/Test split ratio (default: 0.7,0.15,0.15)
                          Format: train,val,test (must sum to 1.0)

    -m, --mode MODE       Split mode: 'frame' or 'sequence' (default: frame)
                          frame: split individual frames
                          sequence: keep sequences together

    --symlink             Use symlinks instead of copying files (saves disk space)

    --skip-test           Skip dataset loading test at the end

    -h, --help            Show this help message

EXAMPLES:
    # Basic usage (mouse dataset)
    $(basename $0) --source /path/to/sam3d_output/mouse

    # Custom animal with different split
    $(basename $0) --source /path/to/sam3d_output/cat --animal cat --ratio 0.8,0.1,0.1

    # Use symlinks to save space
    $(basename $0) --source /path/to/sam3d_output/mouse --symlink

    # Sequence-based split (keeps temporal coherence)
    $(basename $0) --source /path/to/sam3d_output/mouse --mode sequence

OUTPUT:
    Processed data:  data/fauna_processed/{animal}/
    Final dataset:   data/fauna/large_scale/{animal}/
    Generated configs: config/train_{animal}.yaml, config/train_{animal}_debug.yaml

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--source)
            SAM3D_SOURCE="$2"
            shift 2
            ;;
        -a|--animal)
            ANIMAL="$2"
            shift 2
            ;;
        -r|--ratio)
            SPLIT_RATIO="$2"
            shift 2
            ;;
        -m|--mode)
            SPLIT_MODE="$2"
            shift 2
            ;;
        --symlink)
            USE_COPY=""
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$SAM3D_SOURCE" ]; then
    print_error "Missing required argument: --source"
    echo
    echo "Usage: $(basename $0) --source /path/to/sam3d/output [OPTIONS]"
    echo "Use --help for more information"
    exit 1
fi

# Set derived paths
PROCESSED_OUTPUT="${PROJECT_ROOT}/data/fauna_processed"
FAUNA_TARGET="${PROJECT_ROOT}/data/fauna/large_scale"

# Print configuration
echo
print_info "=== Dataset Update Pipeline ==="
echo
echo "Configuration:"
echo "  Source:      $SAM3D_SOURCE"
echo "  Animal:      $ANIMAL"
echo "  Split ratio: $SPLIT_RATIO"
echo "  Split mode:  $SPLIT_MODE"
echo "  Copy mode:   ${USE_COPY:-symlink}"
echo "  Output:      $FAUNA_TARGET/$ANIMAL"
echo

# Step 1: Check source
if [ ! -d "$SAM3D_SOURCE" ]; then
    print_error "SAM3D source not found: $SAM3D_SOURCE"
    exit 1
fi

# Count source files
SOURCE_FRAMES=$(find "$SAM3D_SOURCE" -name "*_rgb.png" 2>/dev/null | wc -l)
print_info "Found $SOURCE_FRAMES RGB frames in source"
echo

# Step 2: Preprocess SAM3D data
print_step "Step 1/4: Preprocessing SAM3D dataset..."
python3 ${PROJECT_ROOT}/scripts/preprocess_sam3d_dataset.py \
    --source "$SAM3D_SOURCE" \
    --animal "$ANIMAL" \
    --target "$PROCESSED_OUTPUT" \
    $USE_COPY

if [ $? -ne 0 ]; then
    print_error "Preprocessing failed"
    exit 1
fi

print_info "✓ Preprocessing complete"
echo

# Step 3: Backup old data (if exists)
if [ -L "${FAUNA_TARGET}/${ANIMAL}" ] || [ -d "${FAUNA_TARGET}/${ANIMAL}" ]; then
    BACKUP_DIR="${FAUNA_TARGET}/${ANIMAL}_backup_$(date +%Y%m%d_%H%M%S)"
    print_warning "Backing up existing ${ANIMAL} data to: $BACKUP_DIR"
    mv "${FAUNA_TARGET}/${ANIMAL}" "$BACKUP_DIR"
    print_info "✓ Backup complete"
else
    print_info "No existing ${ANIMAL} data to backup"
fi
echo

# Step 4: Prepare for 3DAnimals (split train/val/test)
print_step "Step 2/4: Splitting train/val/test..."
python3 ${PROJECT_ROOT}/scripts/prepare_fauna_dataset.py \
    --source "${PROCESSED_OUTPUT}/${ANIMAL}/train" \
    --animal "$ANIMAL" \
    --split-mode "$SPLIT_MODE" \
    --ratio "$SPLIT_RATIO"

if [ $? -ne 0 ]; then
    print_error "Dataset preparation failed"
    exit 1
fi

print_info "✓ Dataset prepared"
echo

# Step 5: Verify
print_step "Step 3/4: Verifying dataset..."
if [ ! -d "${FAUNA_TARGET}/${ANIMAL}/train" ]; then
    print_error "Train directory not created"
    exit 1
fi

TRAIN_COUNT=$(find ${FAUNA_TARGET}/${ANIMAL}/train -name "*_rgb.png" 2>/dev/null | wc -l)
VAL_COUNT=$(find ${FAUNA_TARGET}/${ANIMAL}/val -name "*_rgb.png" 2>/dev/null | wc -l)
TEST_COUNT=$(find ${FAUNA_TARGET}/${ANIMAL}/test -name "*_rgb.png" 2>/dev/null | wc -l)
TOTAL_COUNT=$((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))

print_info "Train frames: $TRAIN_COUNT"
print_info "Val frames:   $VAL_COUNT"
print_info "Test frames:  $TEST_COUNT"
print_info "Total:        $TOTAL_COUNT"
print_info "✓ Verification complete"
echo

# Step 6: Test loading (optional)
if [ "$SKIP_TEST" = false ]; then
    print_step "Step 4/4: Testing dataset loading..."

    CONFIG_NAME="train_${ANIMAL}_debug"
    if [ ! -f "${PROJECT_ROOT}/config/${CONFIG_NAME}.yaml" ]; then
        CONFIG_NAME="train_${ANIMAL}"
    fi

    if [ -f "${PROJECT_ROOT}/config/${CONFIG_NAME}.yaml" ]; then
        timeout 60 conda run -n 3danimals python ${PROJECT_ROOT}/run.py \
            --config-name "$CONFIG_NAME" 2>&1 | \
            grep -E "(Loading|using.*categories|large_scale_${ANIMAL})" | head -5 || true

        print_info "✓ Dataset loading test complete"
    else
        print_warning "Config not found: ${CONFIG_NAME}.yaml (skipping load test)"
    fi
else
    print_info "Step 4/4: Skipped (--skip-test)"
fi
echo

# Summary
print_info "=== Update Complete ==="
echo
echo "📁 Dataset location:"
echo "   ${FAUNA_TARGET}/${ANIMAL}/"
echo
echo "📊 Split summary:"
echo "   Train: $TRAIN_COUNT frames ($(echo "scale=0; $TRAIN_COUNT * 100 / $TOTAL_COUNT" | bc)%)"
echo "   Val:   $VAL_COUNT frames ($(echo "scale=0; $VAL_COUNT * 100 / $TOTAL_COUNT" | bc)%)"
echo "   Test:  $TEST_COUNT frames ($(echo "scale=0; $TEST_COUNT * 100 / $TOTAL_COUNT" | bc)%)"
echo
if [ -n "$BACKUP_DIR" ]; then
    echo "🗂️  Old data backed up to:"
    echo "   $BACKUP_DIR"
    echo
fi
echo "✅ Next steps:"
echo "   1. Review: ls ${FAUNA_TARGET}/${ANIMAL}/"
echo "   2. Debug:  conda run -n 3danimals python run.py --config-name train_${ANIMAL}_debug"
echo "   3. Train:  conda run -n 3danimals python run.py --config-name train_${ANIMAL}"
echo

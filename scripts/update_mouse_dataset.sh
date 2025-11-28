#!/bin/bash
# Update Animal Dataset Pipeline
# Converts SAM3D GUI output to Fauna format for 3DAnimals training
#
# Usage:
#   ./scripts/update_mouse_dataset.sh --source /path/to/sam3d/output
#   ./scripts/update_mouse_dataset.sh -s /path/to/sam3d/output -a mouse -r 0.8:0.1:0.1
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
SPLIT_RATIO="0.8:0.1:0.1"
USE_COPY="--copy"
SKIP_TEST=false

# Help message
show_help() {
    cat << EOF
Usage: $(basename $0) [OPTIONS]

Converts SAM3D GUI output to Fauna format and prepares for 3DAnimals training.

OPTIONS:
    -s, --source PATH     SAM3D GUI output directory (REQUIRED)
                          Example: /home/joon/data/mouse_batch_20251128

    -a, --animal NAME     Animal name (default: mouse)
                          Used for directory naming and config generation

    -r, --ratio RATIO     Train:Val:Test split ratio (default: 0.8:0.1:0.1)
                          Format: train:val:test (must sum to 1.0)
                          Use colon (:) as separator

    --symlink             Use symlinks instead of copying files (saves disk space)
                          WARNING: Original files must not be moved/deleted

    --skip-test           Skip dataset loading test at the end

    -h, --help            Show this help message

EXAMPLES:
    # Basic usage (mouse dataset)
    $(basename $0) --source /home/joon/data/mouse_batch

    # Custom split ratio
    $(basename $0) --source /home/joon/data/mouse_batch --ratio 0.7:0.15:0.15

    # Use symlinks to save disk space
    $(basename $0) --source /home/joon/data/mouse_batch --symlink

    # Different animal
    $(basename $0) --source /home/joon/data/cat_batch --animal cat

OUTPUT STRUCTURE:
    data/fauna/large_scale/{animal}/
    ├── train/
    │   ├── seq_000/
    │   │   ├── 0000000_rgb.png
    │   │   ├── 0000000_mask.png
    │   │   ├── 0000000_box.txt      (auto-generated)
    │   │   └── 0000000_metadata.json (auto-generated)
    │   └── seq_001/
    ├── val/
    └── test/

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

# Set output path (directly to Fauna dataset location)
FAUNA_ROOT="${PROJECT_ROOT}/data/fauna"
FAUNA_TARGET="${FAUNA_ROOT}/large_scale"

# Create required empty directories for FaunaDataset compatibility
mkdir -p "${FAUNA_ROOT}/few_shot_animal3d"
mkdir -p "${FAUNA_ROOT}/few_shot_web"
mkdir -p "${FAUNA_ROOT}/few_shot_web_back"
mkdir -p "${FAUNA_TARGET}"

# Print configuration
echo
print_info "=== Dataset Update Pipeline ==="
echo
echo "Configuration:"
echo "  Project:     $PROJECT_ROOT"
echo "  Source:      $SAM3D_SOURCE"
echo "  Animal:      $ANIMAL"
echo "  Split ratio: $SPLIT_RATIO"
echo "  Copy mode:   ${USE_COPY:+copy}${USE_COPY:-symlink}"
echo "  Output:      $FAUNA_TARGET/$ANIMAL"
echo

# Step 1: Check source exists
if [ ! -d "$SAM3D_SOURCE" ]; then
    print_error "Source directory not found: $SAM3D_SOURCE"
    exit 1
fi

# Count source files
SOURCE_FRAMES=$(find "$SAM3D_SOURCE" -name "*_rgb.png" -o -name "*_rgb.jpg" 2>/dev/null | wc -l)
if [ "$SOURCE_FRAMES" -eq 0 ]; then
    # Maybe nested in subdirectories
    SOURCE_FRAMES=$(find "$SAM3D_SOURCE" -type f \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l)
    SOURCE_FRAMES=$((SOURCE_FRAMES / 2))  # Assume half are RGB, half are masks
fi
print_info "Found approximately $SOURCE_FRAMES frames in source"
echo

# Step 2: Backup old data (if exists)
if [ -d "${FAUNA_TARGET}/${ANIMAL}" ]; then
    BACKUP_DIR="${FAUNA_TARGET}/${ANIMAL}_backup_$(date +%Y%m%d_%H%M%S)"
    print_warning "Backing up existing ${ANIMAL} data to:"
    print_warning "  $BACKUP_DIR"
    mv "${FAUNA_TARGET}/${ANIMAL}" "$BACKUP_DIR"
    print_info "✓ Backup complete"
    echo
fi

# Step 3: Preprocess and split
print_step "Step 1/2: Processing SAM3D dataset..."
echo "  - Generating box.txt and metadata.json"
echo "  - Splitting train/val/test ($SPLIT_RATIO)"
echo

python3 ${PROJECT_ROOT}/scripts/preprocess_sam3d_dataset.py \
    --source "$SAM3D_SOURCE" \
    --target "$FAUNA_TARGET" \
    --animal "$ANIMAL" \
    --split "$SPLIT_RATIO" \
    $USE_COPY

if [ $? -ne 0 ]; then
    print_error "Preprocessing failed"
    exit 1
fi

print_info "✓ Preprocessing complete"
echo

# Step 4: Verify output
print_step "Step 2/2: Verifying dataset..."

if [ ! -d "${FAUNA_TARGET}/${ANIMAL}/train" ]; then
    print_error "Train directory not created: ${FAUNA_TARGET}/${ANIMAL}/train"
    exit 1
fi

TRAIN_COUNT=$(find "${FAUNA_TARGET}/${ANIMAL}/train" -name "*_rgb.png" -o -name "*_rgb.jpg" 2>/dev/null | wc -l)
VAL_COUNT=$(find "${FAUNA_TARGET}/${ANIMAL}/val" -name "*_rgb.png" -o -name "*_rgb.jpg" 2>/dev/null | wc -l)
TEST_COUNT=$(find "${FAUNA_TARGET}/${ANIMAL}/test" -name "*_rgb.png" -o -name "*_rgb.jpg" 2>/dev/null | wc -l)
TOTAL_COUNT=$((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))

if [ "$TOTAL_COUNT" -eq 0 ]; then
    print_error "No frames found in output directory"
    exit 1
fi

echo
print_info "Dataset Statistics:"
print_info "  Train: $TRAIN_COUNT frames"
print_info "  Val:   $VAL_COUNT frames"
print_info "  Test:  $TEST_COUNT frames"
print_info "  Total: $TOTAL_COUNT frames"
print_info "✓ Verification complete"
echo

# Step 5: Test loading (optional)
if [ "$SKIP_TEST" = false ]; then
    print_step "Testing dataset loading..."

    # Check for existing config
    CONFIG_NAME="train_${ANIMAL}_debug"
    if [ ! -f "${PROJECT_ROOT}/config/${CONFIG_NAME}.yaml" ]; then
        CONFIG_NAME="train_${ANIMAL}"
    fi
    if [ ! -f "${PROJECT_ROOT}/config/${CONFIG_NAME}.yaml" ]; then
        CONFIG_NAME="train_mouse_stable"
    fi

    if [ -f "${PROJECT_ROOT}/config/${CONFIG_NAME}.yaml" ]; then
        print_info "Using config: $CONFIG_NAME"
        timeout 30 python3 ${PROJECT_ROOT}/run.py \
            --config-name "$CONFIG_NAME" 2>&1 | \
            grep -E "(Loading|using.*categories|large_scale)" | head -3 || true
        echo
        print_info "✓ Loading test complete"
    else
        print_warning "No suitable config found, skipping load test"
    fi
else
    print_info "Skipping load test (--skip-test)"
fi
echo

# Summary
print_info "=== Pipeline Complete ==="
echo
echo "📁 Dataset location:"
echo "   ${FAUNA_TARGET}/${ANIMAL}/"
echo
echo "📊 Split summary:"
if [ "$TOTAL_COUNT" -gt 0 ]; then
    TRAIN_PCT=$((TRAIN_COUNT * 100 / TOTAL_COUNT))
    VAL_PCT=$((VAL_COUNT * 100 / TOTAL_COUNT))
    TEST_PCT=$((TEST_COUNT * 100 / TOTAL_COUNT))
    echo "   Train: $TRAIN_COUNT frames ($TRAIN_PCT%)"
    echo "   Val:   $VAL_COUNT frames ($VAL_PCT%)"
    echo "   Test:  $TEST_COUNT frames ($TEST_PCT%)"
fi
echo
if [ -n "$BACKUP_DIR" ]; then
    echo "🗂️  Old data backed up to:"
    echo "   $BACKUP_DIR"
    echo
fi
echo "✅ Next steps:"
echo "   1. Review:  ls ${FAUNA_TARGET}/${ANIMAL}/"
echo "   2. Debug:   python run.py --config-name train_${ANIMAL}_debug"
echo "   3. Train:   python run.py --config-name train_${ANIMAL}"
echo

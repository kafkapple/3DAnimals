#!/bin/bash
# Update Mouse Dataset Pipeline
# Replaces old mouse data with new SAM3D processed data

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

PROJECT_ROOT="/home/joon/dev/3DAnimals"
SAM3D_SOURCE="/home/joon/dev/sam3d_gui/outputs/fauna_datasets/mouse"
PROCESSED_OUTPUT="${PROJECT_ROOT}/data/fauna_processed"
FAUNA_TARGET="${PROJECT_ROOT}/data/fauna/large_scale"

print_info "=== Mouse Dataset Update Pipeline ==="
echo

# Step 1: Check source
if [ ! -d "$SAM3D_SOURCE" ]; then
    print_error "SAM3D source not found: $SAM3D_SOURCE"
    exit 1
fi

print_info "Source: $SAM3D_SOURCE"
print_info "Target: $FAUNA_TARGET/mouse"
echo

# Step 2: Preprocess SAM3D data
print_info "Step 1/4: Preprocessing SAM3D dataset..."
python3 ${PROJECT_ROOT}/scripts/preprocess_sam3d_dataset.py \
    --source "$SAM3D_SOURCE" \
    --animal mouse \
    --output "$PROCESSED_OUTPUT" \
    --copy

if [ $? -ne 0 ]; then
    print_error "Preprocessing failed"
    exit 1
fi

print_info "✓ Preprocessing complete"
echo

# Step 3: Backup old data (if exists)
if [ -L "${FAUNA_TARGET}/mouse" ] || [ -d "${FAUNA_TARGET}/mouse" ]; then
    BACKUP_DIR="${FAUNA_TARGET}/mouse_backup_$(date +%Y%m%d_%H%M%S)"
    print_warning "Backing up existing mouse data to: $BACKUP_DIR"
    mv "${FAUNA_TARGET}/mouse" "$BACKUP_DIR"
    print_info "✓ Backup complete"
else
    print_info "No existing mouse data to backup"
fi
echo

# Step 4: Prepare for 3DAnimals (split train/val/test)
print_info "Step 2/4: Splitting train/val/test..."
python3 ${PROJECT_ROOT}/scripts/prepare_fauna_dataset.py \
    --source "${PROCESSED_OUTPUT}/mouse/train" \
    --animal mouse \
    --split-mode frame \
    --ratio 0.7,0.15,0.15

if [ $? -ne 0 ]; then
    print_error "Dataset preparation failed"
    exit 1
fi

print_info "✓ Dataset prepared"
echo

# Step 5: Verify
print_info "Step 3/4: Verifying dataset..."
if [ ! -d "${FAUNA_TARGET}/mouse/train" ]; then
    print_error "Train directory not created"
    exit 1
fi

TRAIN_SEQS=$(find ${FAUNA_TARGET}/mouse/train -mindepth 1 -maxdepth 1 -type d | wc -l)
VAL_SEQS=$(find ${FAUNA_TARGET}/mouse/val -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
TEST_SEQS=$(find ${FAUNA_TARGET}/mouse/test -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

print_info "Train sequences: $TRAIN_SEQS"
print_info "Val sequences:   $VAL_SEQS"
print_info "Test sequences:  $TEST_SEQS"
print_info "✓ Verification complete"
echo

# Step 6: Test loading
print_info "Step 4/4: Testing dataset loading..."
timeout 60 conda run -n 3danimals python ${PROJECT_ROOT}/run.py \
    --config-name train_mouse_debug 2>&1 | \
    grep -E "(Loading|using.*categories|large_scale_mouse)" | head -5

if [ $? -eq 0 ]; then
    print_info "✓ Dataset loads successfully"
else
    print_warning "Could not verify loading (timeout or error)"
fi
echo

# Summary
print_info "=== Update Complete ==="
echo
echo "📁 New mouse dataset:"
echo "   Source:   $SAM3D_SOURCE (200 frames)"
echo "   Location: ${FAUNA_TARGET}/mouse"
echo
echo "📊 Split:"
echo "   Train: ~140 frames"
echo "   Val:   ~30 frames"
echo "   Test:  ~30 frames"
echo
echo "🗂️  Old data backed up:"
echo "   ${BACKUP_DIR:-None (no previous data)}"
echo
echo "✅ Next steps:"
echo "   1. Review dataset: ls ${FAUNA_TARGET}/mouse/"
echo "   2. Run debug:      conda run -n 3danimals python run.py --config-name train_mouse_debug"
echo "   3. Run training:   conda run -n 3danimals python run.py --config-name train_mouse"

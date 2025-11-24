#!/bin/bash
# Multi-Animal Training Setup Script
# Enables training with all available Fauna animals

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

PROJECT_ROOT="/home/joon/dev/3DAnimals"
FAUNA_ROOT="${PROJECT_ROOT}/data/fauna"
TARGET_DIR="${FAUNA_ROOT}/large_scale"
SOURCE_DIR="${FAUNA_ROOT}/Fauna_dataset/large_scale"

print_info "=== Multi-Animal Training Setup ==="
echo

# Check source
if [ ! -d "$SOURCE_DIR" ]; then
    print_error "Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Count available animals
AVAILABLE=$(ls -d ${SOURCE_DIR}/*/ 2>/dev/null | wc -l)
print_info "Found $AVAILABLE animals in Fauna dataset"
echo

# Interactive mode
echo "Setup options:"
echo "  [1] All animals (8 animals, ~63K frames)"
echo "  [2] Select specific animals"
echo "  [3] Mouse + Large animals (mouse, elephant, horse, giraffe)"
echo "  [4] Small animals only (mouse, sheep)"
echo
read -p "Choose option [1]: " OPTION
OPTION=${OPTION:-1}

case "$OPTION" in
    1)
        print_info "Setting up ALL animals..."
        ANIMALS=("bear_comb_dinov2_new" "cow_comb_dinov2_new" "elephant_comb_dinov2_new"
                 "giraffe_comb_dinov2_new" "horse_comb_dinov2_new" "sheep_comb_dinov2_new"
                 "zebra_comb_dinov2_new" "mouse_markerless_6view")
        ;;
    2)
        print_info "Available animals:"
        ls -d ${SOURCE_DIR}/*/ | xargs -n1 basename
        echo
        read -p "Enter animal names (space-separated): " ANIMALS_INPUT
        ANIMALS=($ANIMALS_INPUT)
        ;;
    3)
        print_info "Setting up Mouse + Large animals..."
        ANIMALS=("elephant_comb_dinov2_new" "horse_comb_dinov2_new"
                 "giraffe_comb_dinov2_new" "mouse_markerless_6view")
        ;;
    4)
        print_info "Setting up Small animals..."
        ANIMALS=("mouse_markerless_6view" "sheep_comb_dinov2_new")
        ;;
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

# Create symlinks or copy
echo
echo "Link method:"
echo "  [1] Symlink (fast, saves space, requires source availability)"
echo "  [2] Copy (slow, uses space, standalone)"
echo
read -p "Choose method [1]: " METHOD
METHOD=${METHOD:-1}

echo
print_info "Processing animals..."

for animal in "${ANIMALS[@]}"; do
    SOURCE_ANIMAL="${SOURCE_DIR}/${animal}"

    # Extract animal name (first part before _)
    ANIMAL_NAME=$(echo "$animal" | cut -d'_' -f1)
    TARGET_ANIMAL="${TARGET_DIR}/${ANIMAL_NAME}"

    if [ ! -d "$SOURCE_ANIMAL" ]; then
        print_warning "Skipping $animal (not found)"
        continue
    fi

    # Remove if exists
    if [ -L "$TARGET_ANIMAL" ] || [ -d "$TARGET_ANIMAL" ]; then
        print_info "Removing existing: $ANIMAL_NAME"
        rm -rf "$TARGET_ANIMAL"
    fi

    if [ "$METHOD" = "1" ]; then
        # Symlink
        print_info "Linking: $ANIMAL_NAME"
        ln -sf "$SOURCE_ANIMAL" "$TARGET_ANIMAL"
    else
        # Copy
        print_info "Copying: $ANIMAL_NAME (this may take a while...)"
        mkdir -p "$TARGET_ANIMAL"
        cp -r "${SOURCE_ANIMAL}"/* "$TARGET_ANIMAL"/
    fi
done

# Summary
echo
print_info "=== Setup Complete ==="
echo
echo "📁 Active animals in data/fauna/large_scale/:"
ls -d ${TARGET_DIR}/*/ 2>/dev/null | xargs -n1 basename | while read animal; do
    TRAIN_DIR="${TARGET_DIR}/${animal}/train"
    if [ -d "$TRAIN_DIR" ]; then
        SEQS=$(ls -d ${TRAIN_DIR}/*/ 2>/dev/null | wc -l)
        echo "   ✓ ${animal}: ${SEQS} sequences"
    fi
done

echo
print_info "Next steps:"
echo "  1. Review config: config/train_mouse.yaml"
echo "  2. Run debug:     conda run -n 3danimals python run.py --config-name train_mouse_debug"
echo "  3. Check loading: Should see 'using N categories, contains: [...]'"
echo
print_warning "Note: Multi-animal training will take longer (~10-20x)"
print_warning "Recommendation: Start with debug mode (5K iters) to verify"

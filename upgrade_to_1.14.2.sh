~#!/bin/bash
# =============================================================================
# upgrade_to_1.14.2.sh
# Safely upgrades PhysiCell core to v1.14.2 without requiring sample_projects/
# or documentation/ directories.
#
# Run this from INSIDE your ProstateCancer-ABM directory:
#   cd /path/to/ProstateCancer-ABM
#   bash upgrade_to_1.14.2.sh
# =============================================================================

set -e  # exit on any error

PROJECT_DIR="$(pwd)"
ZIP="PhysiCell_V.1.14.2.zip"
URL="https://github.com/MathCancer/PhysiCell/releases/download/1.14.2/PhysiCell_V.1.14.2.zip"

echo "=============================================="
echo "  PhysiCell upgrade: 1.14.0 → 1.14.2"
echo "  Project dir: $PROJECT_DIR"
echo "=============================================="

# Safety check: make sure we're in the right place
if [ ! -f "custom_modules/custom.cpp" ]; then
    echo "ERROR: Cannot find custom_modules/custom.cpp"
    echo "Make sure you run this from inside ProstateCancer-ABM/"
    exit 1
fi

echo ""
echo "Step 1: Downloading PhysiCell v1.14.2..."
curl -L "$URL" --output "$ZIP"

echo ""
echo "Step 2: Backing up current core files..."
mkdir -p backup_1.14.0
cp -r core/   backup_1.14.0/core
cp -r modules/ backup_1.14.0/modules
cp -r BioFVM/  backup_1.14.0/BioFVM
cp VERSION.txt backup_1.14.0/VERSION.txt
echo "  Backup saved to backup_1.14.0/"

echo ""
echo "Step 3: Extracting and installing new core files..."
unzip -o "$ZIP" "PhysiCell/VERSION.txt"         && mv -f PhysiCell/VERSION.txt .
unzip -o "$ZIP" "PhysiCell/core/*"              && cp -r PhysiCell/core/* core/
unzip -o "$ZIP" "PhysiCell/modules/*"           && cp -r PhysiCell/modules/* modules/
unzip -o "$ZIP" "PhysiCell/BioFVM/*"            && cp -r PhysiCell/BioFVM/* BioFVM/

echo ""
echo "Step 4: Cleaning up..."
rm -rf PhysiCell "$ZIP"

echo ""
echo "Step 5: Verifying version..."
cat VERSION.txt

echo ""
echo "Step 6: Recompiling..."
make clean
make -j4

echo ""
echo "=============================================="
echo "  Upgrade complete! Now on PhysiCell v1.14.2"
echo "  Your custom_modules/ and config/ are untouched."
echo "  Old core backed up in backup_1.14.0/"
echo "=============================================="~

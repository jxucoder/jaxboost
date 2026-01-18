#!/usr/bin/env bash
# Setup external repositories for benchmarking and comparison.
#
# Usage:
#   ./scripts/setup_repos.sh
#
# This script clones external repos needed for:
#   - SLACE: Original PyTorch implementation for fair comparison benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPOS_DIR="$PROJECT_ROOT/repos"

mkdir -p "$REPOS_DIR"

# =============================================================================
# SLACE: Soft Labels Accumulating Cross Entropy (AAAI 2025)
# =============================================================================
# Paper: https://ojs.aaai.org/index.php/AAAI/article/view/34158
# Used for: ordinal regression benchmark comparisons

SLACE_DIR="$REPOS_DIR/SLACE"
SLACE_REPO="https://github.com/inbarnachmani/SLACE.git"

if [ -d "$SLACE_DIR" ]; then
    echo "SLACE repo already exists at $SLACE_DIR"
    echo "Pulling latest changes..."
    cd "$SLACE_DIR" && git pull
else
    echo "Cloning SLACE repo..."
    git clone "$SLACE_REPO" "$SLACE_DIR"
fi

echo ""
echo "Setup complete! External repos are in: $REPOS_DIR"
echo ""
echo "Note: repos/ is gitignored. Re-run this script after fresh clone."

#!/bin/bash
set -e

echo "=== FlashInfer Bench Starter Kit - Environment Setup ==="
echo ""

# -------------------------------------------------------
# 1. Install Miniforge (conda for ARM64 Mac)
# -------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "[1/7] Installing Miniforge via Homebrew..."
    brew install miniforge
    conda init "$(basename "$SHELL")"
    echo ""
    echo ">>> conda installed. Please restart your terminal, then re-run this script."
    exit 0
else
    echo "[1/7] conda already installed, skipping."
fi

# -------------------------------------------------------
# 2. Create conda environment
# -------------------------------------------------------
if conda env list | grep -q "fi-bench"; then
    echo "[2/7] conda env 'fi-bench' already exists, skipping."
else
    echo "[2/7] Creating conda environment 'fi-bench' (Python 3.12)..."
    conda create -n fi-bench python=3.12 -y
fi

# -------------------------------------------------------
# 3. Install dependencies
# -------------------------------------------------------
echo "[3/7] Installing Python dependencies..."
eval "$(conda shell.bash hook)"
conda activate fi-bench
pip install flashinfer-bench modal

# -------------------------------------------------------
# 4. Modal setup
# -------------------------------------------------------
echo ""
echo "[4/7] Setting up Modal (will open browser for login/signup)..."
echo "    If you don't have a Modal account, one will be created."
modal setup

# -------------------------------------------------------
# 5. Download dataset from HuggingFace
# -------------------------------------------------------
DATASET_DIR="$(dirname "$PWD")/mlsys26-contest"

if [ -d "$DATASET_DIR" ]; then
    echo "[5/7] Dataset already exists at $DATASET_DIR, skipping."
else
    echo "[5/7] Downloading competition dataset from HuggingFace..."
    if ! command -v git-lfs &> /dev/null; then
        brew install git-lfs
        git lfs install
    fi
    git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest "$DATASET_DIR"
fi

# -------------------------------------------------------
# 6. Set environment variable
# -------------------------------------------------------
SHELL_RC="$HOME/.zshrc"
if grep -q "FIB_DATASET_PATH" "$SHELL_RC" 2>/dev/null; then
    echo "[6/7] FIB_DATASET_PATH already in $SHELL_RC, skipping."
else
    echo "[6/7] Adding FIB_DATASET_PATH to $SHELL_RC..."
    echo "" >> "$SHELL_RC"
    echo "# FlashInfer Bench dataset path" >> "$SHELL_RC"
    echo "export FIB_DATASET_PATH=\"$DATASET_DIR\"" >> "$SHELL_RC"
fi
export FIB_DATASET_PATH="$DATASET_DIR"

# -------------------------------------------------------
# 7. Upload dataset to Modal Volume
# -------------------------------------------------------
echo "[7/7] Creating Modal volume and uploading dataset..."
modal volume create flashinfer-trace 2>/dev/null || true
modal volume put flashinfer-trace "$DATASET_DIR"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Restart your terminal (or run: source ~/.zshrc)"
echo "  2. conda activate fi-bench"
echo "  3. python scripts/pack_solution.py    # Pack your solution"
echo "  4. modal run scripts/run_modal.py     # Run on B200 GPU"

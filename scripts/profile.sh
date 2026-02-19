#!/usr/bin/env bash
# ForgeFlow: Run NCU profiling on Modal B200
# Usage: ./scripts/profile.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Ensure conda env is active
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "fi-bench" ]]; then
    echo "[profile] WARNING: conda env 'fi-bench' not active. Activate with: conda activate fi-bench"
fi

echo "[profile] Running NCU profiling via Modal B200..."
echo "[profile] This may take several minutes."
echo ""

python -m forgeflow profile

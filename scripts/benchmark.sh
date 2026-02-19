#!/usr/bin/env bash
# ForgeFlow: Run benchmark pipeline via Modal B200
# Usage: ./scripts/benchmark.sh [baseline|trial|loop N|status]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Ensure conda env is active
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "fi-bench" ]]; then
    echo "[benchmark] WARNING: conda env 'fi-bench' not active. Activate with: conda activate fi-bench"
fi

COMMAND="${1:-baseline}"

case "$COMMAND" in
    baseline)
        echo "[benchmark] Running baseline measurement..."
        python -m forgeflow baseline
        ;;
    trial)
        MESSAGE="${2:-manual trial}"
        echo "[benchmark] Running trial: $MESSAGE"
        python -m forgeflow trial -m "$MESSAGE"
        ;;
    loop)
        N="${2:-5}"
        echo "[benchmark] Running $N optimization trials..."
        python -m forgeflow loop "$N"
        ;;
    status)
        python -m forgeflow status
        ;;
    *)
        echo "Usage: $0 [baseline|trial|loop N|status]"
        echo ""
        echo "Commands:"
        echo "  baseline    Run initial baseline measurement"
        echo "  trial [msg] Run a single optimization trial"
        echo "  loop [N]    Run N trials (default: 5)"
        echo "  status      Print optimization status"
        exit 1
        ;;
esac

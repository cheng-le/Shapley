#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-$ROOT/pile_10b/config.json}"
WORK_DIR="${WORK_DIR:-$ROOT/runs/pile_10b}"
GPUS="${GPUS:-4}"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv-pile10b}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$WORK_DIR"
# A second launcher must never mutate a running extraction or checkpoint.
exec 9>"$WORK_DIR/launcher.lock"
flock -n 9 || { echo 'Another launcher owns this WORK_DIR.' >&2; exit 1; }
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  "$PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13), "Use Python 3.10-3.12"'
  "$PYTHON" -m venv "$VENV"
  PYTHON="$VENV/bin/python"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install torch==2.6.0 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
  "$PYTHON" -m pip install -r pile_10b/requirements.txt
fi
PIPE=("$PYTHON" -u -m pile_10b.pipeline --config "$CONFIG" --work-dir "$WORK_DIR")
"${PIPE[@]}" preflight --gpus "$GPUS"
"$PYTHON" -m pip freeze > "$WORK_DIR/environment.txt"
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
"${PIPE[@]}" prepare 2>&1 | tee -a "$WORK_DIR/prepare.log"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
  -m pile_10b.pipeline --config "$CONFIG" --work-dir "$WORK_DIR" features 2>&1 | tee -a "$WORK_DIR/features.log"
"${PIPE[@]}" graph 2>&1 | tee -a "$WORK_DIR/graph.log"
"${PIPE[@]}" verify
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
  -m pile_10b.pipeline --config "$CONFIG" --work-dir "$WORK_DIR" train 2>&1 | tee -a "$WORK_DIR/train.log"
"${PIPE[@]}" report
echo "Completed: $WORK_DIR/report.json"

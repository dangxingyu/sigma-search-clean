#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_REPO="$(cd "$REPO/.." && pwd)"
cd "$REPO"

if [[ -f "$PARENT_REPO/nanochat/.venv/bin/activate" ]]; then
  # Prefer the live project venv on this cluster; the vendored nanochat tree is
  # source-only in the standalone handoff and may contain no torch install.
  # shellcheck disable=SC1091
  source "$PARENT_REPO/nanochat/.venv/bin/activate"
elif [[ -f "$REPO/nanochat/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/nanochat/.venv/bin/activate"
elif [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[error] no nanochat venv found; run scripts/setup_env.sh first" >&2
  exit 1
fi

export PYTHONPATH="$REPO:$REPO/nanochat"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-sigma}"

STAMP="${STAMP:-20260501}"
TOKENS="${TOKENS:-1073741824}"
DEPTH="${DEPTH:-8}"
NPROC="${NPROC:-8}"
LRS="${LRS:-0.005 0.01 0.02 0.04}"

V29_OUT="${V29_OUT:-$PARENT_REPO/search_evals/v29_topaware_k1_256k_alpha_sweep_20260430}"
V29_LOG="${V29_LOG:-$PARENT_REPO/logs/v29_topaware_k1_256k_alpha_sweep_20260430}"
V31_OUT="${V31_OUT:-$PARENT_REPO/search_evals/v31_topaware_k1_512k_alpha_sweep_${STAMP}}"
V31_LOG="${V31_LOG:-$PARENT_REPO/logs/v31_topaware_k1_512k_alpha_sweep_${STAMP}}"

echo "v31 queued alpha sweep"
echo "repo=$REPO"
echo "v29_out=$V29_OUT"
echo "v31_out=$V31_OUT"

while pgrep -af "run_v28_v29_topaware_k1_alpha_sweeps_on_node.sh|v29_topaware_k1_256k_alpha_sweep_20260430" >/dev/null; do
  echo "$(date) waiting for active v29 sweep to finish..."
  sleep 120
done

while read -r pid cmd; do
  [[ -z "${pid:-}" ]] && continue
  if [[ "$pid" != "$$" ]]; then
    echo "keeping v26b supervisor stopped pid=$pid: $cmd"
    kill -STOP "$pid" || true
  fi
done < <(pgrep -af "run_v26b_large_then_maybe_v27_on_node.sh" || true)

echo
echo "=== fill/verify 256K v29 grid ==="
python run_top_aware_muon_sweep.py \
  --out-root "$V29_OUT" \
  --log-root "$V29_LOG" \
  --nanochat-dir nanochat \
  --methods "streaming_identity native_muon top_aware_muon" \
  --batches "262144" \
  --alphas "0.25 0.5 0.75 0.875" \
  --lrs "$LRS" \
  --seeds "42" \
  --tokens "$TOKENS" \
  --depth "$DEPTH" \
  --nproc-per-node "$NPROC" \
  --max-device-batch-size 16 \
  --pure-qr \
  --streaming-num-iters 2 \
  --fallback-ortho-tol 0.01

echo
echo "=== run 512K alpha grid with native Muon baseline ==="
python run_top_aware_muon_sweep.py \
  --out-root "$V31_OUT" \
  --log-root "$V31_LOG" \
  --nanochat-dir nanochat \
  --methods "${METHODS_512K:-streaming_identity native_muon top_aware_muon}" \
  --batches "524288" \
  --alphas "${ALPHAS_512K:-0.25 0.5 0.75 0.875}" \
  --lrs "$LRS" \
  --seeds "42" \
  --tokens "$TOKENS" \
  --depth "$DEPTH" \
  --nproc-per-node "$NPROC" \
  --max-device-batch-size 16 \
  --pure-qr \
  --streaming-num-iters 2 \
  --fallback-ortho-tol 0.01

MPLCONFIGDIR="$MPLCONFIGDIR" python make_topaware_alpha_dashboard.py || true

#!/usr/bin/env bash
set -eu
umask 077

ROOT=${1:?data root required}
RUN=${2:?run root required}
RELEASE=${3:?release directory required}
LOG="$RUN/rna-foundation-stage2-dispatch-v1.log"
set -C
exec >"$LOG" 2>&1
set +C
ulimit -v 12582912
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$RELEASE:$RELEASE/src:$RUN"
exec nice -n 19 ionice -c 2 -n 7 "$ROOT/c3-vcell/.venv/bin/python" -u \
    "$RELEASE/wait-rna-foundation-stage2-v1.py" \
    --root "$ROOT" --run-root "$RUN" --release "$RELEASE" \
    --preflight "$RUN/rna-foundation-stage2-preflight-v1/preflight.json" \
    --output "$RUN/rna-foundation-stage2-v1" \
    --status "$RUN/rna-foundation-stage2-dispatch-v1.json"

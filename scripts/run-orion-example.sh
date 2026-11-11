#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: bash scripts/run-orion-example.sh SUBSET.h5ad NEW_RUN_DIRECTORY" >&2
    exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
subset="$1"
run="$2"
max_per_target="${C3_MAX_PER_TARGET:-120}"
max_controls="${C3_MAX_CONTROLS:-2000}"
prediction_cells="${C3_PREDICTION_CELLS:-100}"
sampling="${C3_SAMPLING:-negative-binomial}"
if [ ! -f "$subset" ] || [ -e "$run" ]; then
    echo "The subset must exist and the run directory must not already exist." >&2
    exit 1
fi
export PATH="$project_root/.venv/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
mkdir -p "$run"

stage() {
    local name="$1"
    shift
    echo "Starting $name"
    nice -n 10 "$@" 2>&1 | tee "$run/$name.log"
}

stage prepare c3-vcell prepare --input "$subset" --outdir "$run/split" \
    --max-targets 20 --max-per-target "$max_per_target" --max-controls "$max_controls" \
    --max-genes 4096 --holdout-fraction 0.25 --seed 42
stage train c3-vcell train --train "$run/split/train.h5ad" --outdir "$run/model" \
    --device cpu --latent-dim 96 --hidden-dim 192 --batch-size 48 \
    --pretrain-steps 200 --flow-steps 300 --seed 42
stage predict c3-vcell predict --checkpoint "$run/model/model.pt" \
    --reference "$run/split/reference.h5ad" --controls "$run/split/input_controls.h5ad" \
    --output "$run/prediction.h5ad" --cells-per-target "$prediction_cells" \
    --sampling "$sampling" --device cpu --seed 42
stage control-baseline c3-vcell control-baseline --reference "$run/split/reference.h5ad" \
    --controls "$run/split/input_controls.h5ad" --output "$run/control-baseline.h5ad" \
    --cells-per-target "$prediction_cells" --seed 42
stage score c3-vcell score --prediction "$run/prediction.h5ad" \
    --reference "$run/split/reference.h5ad" --outdir "$run/score" --threads 2
stage score-control c3-vcell score --prediction "$run/control-baseline.h5ad" \
    --reference "$run/split/reference.h5ad" --outdir "$run/control-score" --threads 2
stage calibrate c3-vcell score --prediction "$run/prediction.h5ad" \
    --reference "$run/split/reference.h5ad" --outdir "$run/local-calibration" \
    --threads 2 --calibrate
echo "Completed local Orion run: $run"

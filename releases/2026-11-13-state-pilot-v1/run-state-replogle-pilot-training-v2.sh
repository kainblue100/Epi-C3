#!/usr/bin/env bash
set -euo pipefail

D="${D:?Set D to the parent run directory}"
P="${P:-$D/state-replogle-pilot-v1}"
PY="${PY:-$D/state-benchmark-env-v2/bin/python}"
STATE="${STATE:-$D/state-benchmark-env-v2/bin/state}"
RUN="${RUN:-$P/runs-v2}"

export P
"$PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil

p = Path(os.environ["P"])
inputs = p / "inputs-cellload-v2"
audit = json.loads((p / "split-audit-cellload-v2.json").read_text())
manifest = json.loads((inputs / "manifest.json").read_text())

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

checks = {
    p / "frozen-plan.json": manifest["plan_sha256"],
    inputs / "data" / "replogle.h5ad": manifest["dataset_sha256"],
    inputs / "independent-reference-controls.h5ad": manifest["reference_controls_sha256"],
    inputs / "split.toml": manifest["split_sha256"],
    inputs / "manifest.json": audit["inputs_manifest_sha256"],
}
if audit["status"] != "complete" or audit["hvg_count"] != 2000:
    raise ValueError("Frozen-input split audit has not passed")
if audit["splits"]["test"]["perturbation_count"] != 28:
    raise ValueError("Unexpected number of held-out test perturbations")
for path, expected in checks.items():
    if digest(path) != expected:
        raise ValueError(f"Frozen input hash mismatch: {path.name}")
if shutil.disk_usage(p).free < 201 * 1024**3:
    raise OSError("Insufficient disk space above the 200 GiB reserve")
PY

if [ -e "$RUN" ]; then
    echo "Refusing to overwrite existing run directory: $RUN" >&2
    exit 1
fi
umask 077
mkdir "$RUN"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
ulimit -c 0
ulimit -v 12582912

common=(
    "output_dir=$RUN"
    "use_wandb=false"
    "overwrite=false"
    "data.kwargs.toml_config_path=$P/inputs-cellload-v2/split.toml"
    "data.kwargs.embed_key=X_hvg"
    "data.kwargs.output_space=embedding"
    "data.kwargs.num_workers=0"
    "data.kwargs.pin_memory=false"
    "data.kwargs.pert_col=target_gene"
    "data.kwargs.cell_type_key=cell_type"
    "data.kwargs.batch_col=batch_var"
    "data.kwargs.control_pert=non-targeting"
    "data.kwargs.barcode=true"
    "data.kwargs.downsample_cells=null"
    "model.device=cpu"
    "model.kwargs.cell_set_len=8"
    "model.kwargs.hidden_dim=64"
    "model.kwargs.n_encoder_layers=1"
    "model.kwargs.n_decoder_layers=1"
    "training.batch_size=2"
    "training.train_seed=42"
    "training.wandb_track=false"
    "training.max_steps=512"
    "training.val_freq=128"
    "training.ckpt_every_n_steps=256"
)

for model in perturb_mean context_mean state; do
    echo "Starting $model"
    case "$model" in
        state)
            extra=(
                "model.kwargs.transformer_backbone_kwargs.intermediate_size=128"
                "model.kwargs.transformer_backbone_kwargs.num_hidden_layers=1"
                "model.kwargs.transformer_backbone_kwargs.num_attention_heads=4"
                "model.kwargs.transformer_backbone_kwargs.num_key_value_heads=4"
                "model.kwargs.transformer_backbone_kwargs.head_dim=16"
            )
            ;;
        *)
            extra=(
                "model.kwargs.cell_sentence_len=8"
                "model.kwargs.transformer_backbone_kwargs.n_embd=64"
                "model.kwargs.transformer_backbone_kwargs.d_inner=128"
                "model.kwargs.transformer_backbone_kwargs.n_layer=1"
                "model.kwargs.transformer_backbone_kwargs.n_head=4"
            )
            ;;
    esac
    nice -n 19 ionice -c2 -n7 timeout 5400 \
        "$STATE" tx train "model=$model" "name=$model" \
        "${common[@]}" "${extra[@]}" >"$RUN/$model.log" 2>&1
    test -s "$RUN/$model/checkpoints/final.ckpt"
    echo "Finished $model"
done
echo "All three bounded pilot training runs completed."

"""Train a resumable shared RNA encoder on protected Arc, Tabula and Orion splits."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
import pyarrow
import scipy
import torch

import c3vcell.model as model_module
from c3vcell.model import C3VCell, ModelConfig
import epi_c3_foundation_sources_v1 as foundation_sources
import epi_c3_foundation_stage1_sources_v1 as stage_sources
import epi_c3_foundation_training_v1 as training
from epi_c3_shared_foundation_v1 import FORMAT, file_digest, gene_axis_digest


CODE_ROOT = Path(__file__).resolve().parent
SEED = 20261116
BATCH_SIZE = 48
RESERVE = 200 * 1024**3
HEADROOM = 4 * 1024**3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def atomic_torch(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".part")
    with tmp.open("wb") as file:
        torch.save(obj, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def check_space(directory: Path) -> None:
    available = shutil.disk_usage(directory).free
    if available < RESERVE + HEADROOM:
        raise OSError(f"Insufficient checkpoint space above the 200 GiB reserve: {available} bytes free")


def load_script(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load source helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_stat(path: Path, provenance: dict) -> None:
    stat = path.stat()
    if stat.st_size != provenance["source_size_bytes"] or stat.st_mtime_ns != provenance["source_mtime_ns"]:
        raise ValueError(f"Previously verified input has changed: {path}")


@torch.no_grad()
def diagnostics(model: C3VCell, streams: dict) -> dict:
    model.eval()
    output = {}
    for index, (name, source) in enumerate(sorted(streams.items())):
        counts = torch.as_tensor(source.validation.toarray(), dtype=torch.float32)
        measured = torch.from_numpy(source.measured).expand_as(counts)
        generator = torch.Generator().manual_seed(SEED + index)
        hidden = (torch.rand(counts.shape, generator=generator) < 0.2) & measured
        visible = measured & ~hidden
        latent = model.encoder(counts, visible.float())
        predicted = torch.log1p(model.probabilities(latent, measured.float()) * 10_000.0)
        observed = torch.log1p(counts / counts.sum(dim=1, keepdim=True) * 10_000.0)
        positive = hidden & (counts > 0)
        unmasked_latent = model.encoder(counts, measured.float())
        standard_deviation = unmasked_latent.std(dim=0, unbiased=False)
        if not torch.any(hidden) or not torch.any(positive) or not torch.isfinite(unmasked_latent).all():
            raise ValueError(f"Invalid held-out representation diagnostics for {name}")
        values = {
            "cells": len(counts),
            "masked_measured_entries": int(hidden.sum()),
            "masked_detected_entries": int(positive.sum()),
            "masked_all_mse": float((predicted[hidden] - observed[hidden]).square().mean()),
            "masked_detected_mse": float((predicted[positive] - observed[positive]).square().mean()),
            "latent_std_mean": float(standard_deviation.mean()),
            "latent_dimensions_std_gt_0_1": int((standard_deviation > 0.1).sum()),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"Nonfinite validation diagnostic for {name}")
        output[name] = values
    model.train()
    return output


def build_sources(root: Path):
    root = root.resolve()
    run_root = root / "c3-vcell/runs"
    runs = run_root / "continuation-20261111-v1"
    arc = load_script(CODE_ROOT / "stream-arc-training-v1.py", "foundation_stage1_arc")
    orion = load_script(CODE_ROOT / "stream-orion-representation-v1.py", "foundation_stage1_orion")
    arc_path = root / "arc-institute-virtual-cell-dataset/adata_Training.h5ad"
    tabula_path = root / "Tabula-Sapiens-v2-CELLxGENE/Tabula_Sapiens_v2_All_Cells.h5ad"
    orion_root, tahoe_root = root / "X-Atlas-Orion", root / "Tahoe-100M"
    arc_plan_path, tabula_plan_path = runs / "arc-stream-full-v1/plan.json", runs / "tabula-stream-full-hct-axis-v1/plan.json"
    arc_plan, tabula_plan = (json.loads(path.read_text()) for path in (arc_plan_path, tabula_plan_path))
    verify_stat(arc_path, arc_plan["provenance"])
    verify_stat(tabula_path, tabula_plan["provenance"])
    orion_meta_path, tahoe_meta_path = runs / "orion-full-metadata-v1/metadata-plan.json", runs / "tahoe-full-metadata-v2/metadata-plan.json"
    orion_meta, tahoe_meta = (json.loads(path.read_text()) for path in (orion_meta_path, tahoe_meta_path))
    if orion_meta["state"] != "complete" or tahoe_meta["state"] != "complete":
        raise ValueError("Both verified gene annotation manifests must be complete")
    if orion_meta["provenance"]["script_sha256"] != file_digest(CODE_ROOT / "scan-orion-stream-v1.py"):
        raise ValueError("Orion selection code no longer matches its frozen scan")
    for root, metadata in ((orion_root, orion_meta), (tahoe_root, tahoe_meta)):
        if metadata["provenance"]["gene_metadata_sha256"] != file_digest(root / "metadata/gene_metadata.parquet"):
            raise ValueError(f"Gene metadata changed for {root.name}")
    with h5py.File(tabula_path, "r") as file:
        tabula_ids = foundation_sources.h5_text(file["raw/var/_index"])
        tabula_symbols = foundation_sources.h5_text(file["raw/var/feature_name"])
    with h5py.File(arc_path, "r") as file:
        arc_symbols = foundation_sources.h5_text(file["var/_index"])
    orion_frame = pd.read_parquet(orion_root / "metadata/gene_metadata.parquet")
    tahoe_frame = pd.read_parquet(tahoe_root / "metadata/gene_metadata.parquet")
    alignment = foundation_sources.align_human_genes(
        tabula_ids, tabula_symbols, tahoe_frame, orion_frame, arc_symbols,
    )
    graph_path = runs / "replogle-gwps-graph-plan-v1/plan.json"
    graph = json.loads(graph_path.read_text())
    heldout = set(graph["target_split"]["validation"] + graph["target_split"]["test"])
    heldout.update(orion_meta["heldout_targets"])
    split_files = sorted(set(
        list(run_root.glob("arc-*/split/split.json")) + list(runs.glob("arc-*/split/split.json"))
    ))
    protected = sorted(set(
        path
        for base in (run_root, runs)
        for pattern in ("arc-*/split/input_controls.h5ad", "arc-*/split/reference.h5ad")
        for path in base.glob(pattern)
    ))
    if not split_files or not protected:
        raise ValueError("Existing Arc holdout artifacts are required")
    for path in split_files:
        heldout.update(json.loads(path.read_text())["test_targets"])
    old_split = run_root / "arc-input-controls-nested-v1/split/split.json"
    streams = {}
    try:
        streams["arc"] = stage_sources.ArcStream(
            arc_path, arc, alignment, old_split, protected, heldout, batch_size=BATCH_SIZE, seed=SEED,
        )
        streams["tabula"] = stage_sources.TabulaStream(
            tabula_path, alignment, batch_size=BATCH_SIZE, seed=SEED,
        )
        for index, context in enumerate(("HCT116", "HEK293T")):
            streams[f"orion-{context}"] = stage_sources.OrionStream(
                orion_root, orion_meta, orion_frame, alignment, orion, context, heldout,
                batch_size=BATCH_SIZE, seed=SEED + index,
            )
    except Exception:
        for stream in streams.values():
            stream.close()
        raise
    provenance = {
        "arc_source_sha256_from_prior_verified_scan": arc_plan["provenance"]["source_sha256"],
        "tabula_source_sha256_from_prior_verified_scan": tabula_plan["provenance"]["source_sha256"],
        "orion_metadata_plan_sha256": file_digest(orion_meta_path),
        "tahoe_metadata_plan_sha256_for_gene_axis_only": file_digest(tahoe_meta_path),
        "graph_split_sha256": file_digest(graph_path),
        "arc_split_files_sha256": {str(path.relative_to(root)): file_digest(path) for path in split_files},
        "arc_protected_paths": [str(path.relative_to(root)) for path in protected],
        "source_code_sha256": {
            str(Path(path).name): file_digest(Path(path)) for path in (
                __file__, foundation_sources.__file__, stage_sources.__file__, training.__file__,
                model_module.__file__, CODE_ROOT / "stream-arc-training-v1.py",
                CODE_ROOT / "stream-orion-representation-v1.py", CODE_ROOT / "scan-orion-stream-v1.py",
            )
        },
        "software_versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "h5py": h5py.__version__, "pyarrow": pyarrow.__version__,
        },
    }
    return alignment, streams, heldout, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing datasets and verified run plans")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--stop-after-updates", type=int, default=0,
                        help="Gracefully pause after this many updates in this invocation; 0 means no limit")
    args = parser.parse_args()
    if args.checkpoint_every < 1 or args.stop_after_updates < 0:
        raise ValueError("Checkpoint frequency must be positive and the update limit nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    check_space(args.output)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        alignment, streams, heldout, provenance = build_sources(args.data_root)
        try:
            config = ModelConfig(
                len(alignment.genes), latent_dim=64, hidden_dim=128,
                variance_weight=5.0, covariance_weight=0.1, regularization_view="unmasked",
            )
            model = C3VCell(config)
            trainer = training.InterleavedFoundationTrainer(model, streams, learning_rate=0.001, seed=SEED)
            plan = {
                "format": "epi-c3-shared-rna-stage1-plan-v1",
                "scope": "incomplete-stage1-not-a-full-foundation",
                "gene_axis_sha256": gene_axis_digest(alignment.genes), "genes": list(alignment.genes),
                "gene_alignment": alignment.notes,
                "heldout_target_count": len(heldout), "heldout_targets_sha256": stage_sources.digest_json(sorted(heldout)),
                "batch_size": BATCH_SIZE, "seed": SEED, "learning_rate": 0.001,
                "model_config": model.configuration(),
                "sampling": "draw sources in proportion to remaining eligible cell counts; shuffle contiguous windows or Orion shard order",
                "one_epoch": True,
                "sources": {
                    name: {
                        "expected_training_cells": int(stream.expected_cells),
                        "measured_genes": int(stream.measured.sum()), "details": stream.notes,
                    } for name, stream in sorted(streams.items())
                },
                "provenance": provenance,
                "limitations": [
                    "This stage excludes Tahoe training cells and other eligible expression datasets; it is not the final foundation.",
                    "Tabula is the CELLxGENE curated copy, not a byte-verified Figshare original.",
                    "Dataset-level feature masks do not prove that every cell measured every annotated feature.",
                    "Only Tabula's validation is donor-disjoint; Orion and Arc validation is cell-disjoint.",
                    "Cross-dataset biological duplication and all benchmark holdouts are not yet exhaustively verified.",
                    "CPU-only single-epoch training does not establish representation quality or a successful downstream model.",
                ],
            }
            plan_sha = stage_sources.digest_json(plan)
            plan_path, checkpoint_path = args.output / "plan.json", args.output / "recovery.pt"
            if plan_path.exists():
                if stage_sources.digest_json(json.loads(plan_path.read_text())) != plan_sha:
                    raise ValueError("Current inputs, code, software or source splits differ from the frozen plan")
                if not checkpoint_path.exists():
                    raise FileNotFoundError("An existing run requires a recovery checkpoint")
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if saved["plan_sha256"] != plan_sha:
                    raise ValueError("Recovery checkpoint and frozen plan disagree")
                trainer.restore(saved["trainer"])
                before = saved["diagnostics_before"]
            else:
                if any(path.name != ".lock" for path in args.output.iterdir()):
                    raise FileExistsError("Refusing to overwrite an existing output directory")
                before = diagnostics(model, streams)
                atomic_json(plan_path, plan)
                atomic_torch(checkpoint_path, {
                    "format": "epi-c3-shared-rna-stage1-recovery-v1", "plan_sha256": plan_sha,
                    "trainer": trainer.snapshot(), "diagnostics_before": before,
                })
            stop = {"requested": False}
            def request_stop(signum, frame):
                stop["requested"] = True
            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)

            def save_state(status: str, final_diagnostics=None):
                check_space(args.output)
                atomic_torch(checkpoint_path, {
                    "format": "epi-c3-shared-rna-stage1-recovery-v1", "plan_sha256": plan_sha,
                    "trainer": trainer.snapshot(), "diagnostics_before": before,
                })
                state = {
                    "status": status, "updated_at_utc": now(), "plan_sha256": plan_sha,
                    "trained_cells": trainer.trained, "expected_cells": {
                        name: stream.expected_cells for name, stream in streams.items()
                    },
                    "updates": trainer.updates, "singleton_updates": trainer.singleton_updates,
                    "diagnostics_before": before, "diagnostics_after": final_diagnostics,
                    "scope": "incomplete-stage1-not-a-full-foundation",
                }
                atomic_json(args.output / "state.json", state)
                print(json.dumps({key: state[key] for key in ("status", "updates", "trained_cells")}), flush=True)

            updates_here = 0
            while not stop["requested"]:
                result = trainer.step()
                if result is None:
                    break
                updates_here += 1
                if trainer.updates % args.checkpoint_every == 0:
                    save_state("running")
                if args.stop_after_updates and updates_here >= args.stop_after_updates:
                    stop["requested"] = True
            if stop["requested"]:
                save_state("paused")
                return
            trainer.verify_exhausted()
            after = diagnostics(model, streams)
            final = {
                "format": FORMAT, "status": "complete", "scope": "stage1-incomplete",
                "genes": list(alignment.genes), "gene_axis_sha256": gene_axis_digest(alignment.genes),
                "model_config": model.configuration(), "model_state": model.state_dict(),
                "training": {"plan_sha256": plan_sha, "cells": trainer.trained, "updates": trainer.updates},
            }
            atomic_torch(args.output / "foundation-stage1.pt", final)
            restored = torch.load(args.output / "foundation-stage1.pt", map_location="cpu", weights_only=True)
            if any(not torch.equal(restored["model_state"][key], value) for key, value in model.state_dict().items()):
                raise ValueError("Stage-one checkpoint verification failed")
            save_state("stage1-complete", after)
        finally:
            for stream in streams.values():
                stream.close()


if __name__ == "__main__":
    main()

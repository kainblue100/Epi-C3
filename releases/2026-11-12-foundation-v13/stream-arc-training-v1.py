"""Memory-bounded, resumable Arc genetic-perturbation training."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from c3vcell.data import CONTROL, validate_counts
from c3vcell.identity import check_cell_id_schemes
from c3vcell.integrity import file_sha256
from c3vcell.model import C3VCell, ModelConfig
from c3vcell.reference import load_reference, reference_metadata
from c3vcell.training import representation_diagnostics


VERSION = 1
RESERVE = 200 * 1024**3
NEED_HEADROOM = 12 * 1024**3


class TrainingPaused(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest_json(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_torch(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def check_disk(path: Path, *, headroom: int = 0) -> None:
    free = shutil.disk_usage(path).free
    if free < RESERVE + headroom:
        raise OSError(
            f"Disk reserve requires {RESERVE + headroom} bytes, but only {free} remain on {path}"
        )


def text_values(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        return node.asstr()[:]
    if set(node.keys()) != {"categories", "codes"}:
        raise ValueError(f"Unsupported string encoding: {node.name}")
    categories = node["categories"].asstr()[:]
    codes = node["codes"][:]
    if np.any(codes < 0) or np.any(codes >= len(categories)):
        raise ValueError(f"Missing or invalid categorical metadata: {node.name}")
    return categories[codes]


def index_values(group: h5py.Group) -> np.ndarray:
    name = group.attrs.get("_index", "_index")
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return text_values(group[name])


class ArcSource:
    def __init__(self, path: Path):
        self.path = path
        self.file = h5py.File(path, "r")
        try:
            matrix = self.file["X"]
            if matrix.attrs.get("encoding-type") != "csr_matrix":
                raise ValueError("Arc X must be a CSR count matrix")
            self.n_cells, self.n_genes = map(int, matrix.attrs["shape"])
            self.ptr = matrix["indptr"][:].astype(np.int64)
            self.indices = matrix["indices"]
            self.values = matrix["data"]
            if (
                len(self.ptr) != self.n_cells + 1
                or self.ptr[0] != 0
                or self.ptr[-1] != len(self.indices)
                or self.ptr[-1] != len(self.values)
                or np.any(np.diff(self.ptr) < 0)
            ):
                raise ValueError("Arc CSR pointers are inconsistent")
            obs = self.file["obs"]
            self.ids = index_values(obs).astype(str)
            self.labels = text_values(obs["target_gene"]).astype(str)
            self.batches = text_values(obs["batch"]).astype(str)
            self.contexts = (
                text_values(obs["context"]).astype(str)
                if "context" in obs else np.full(self.n_cells, "local")
            )
            self.genes = index_values(self.file["var"]).astype(str).tolist()
            if (
                any(len(a) != self.n_cells for a in (self.ids, self.labels, self.batches, self.contexts))
                or len(self.genes) != self.n_genes
                or len(set(self.ids)) != self.n_cells
                or len(set(self.genes)) != self.n_genes
                or any(not item for item in self.genes)
                or any(not item for item in self.ids)
                or any(not item for item in self.labels)
                or any(not item for item in self.batches)
                or any(not item for item in self.contexts)
            ):
                raise ValueError("Arc cell, gene or treatment metadata is invalid")
            targets = set(self.labels) - {CONTROL}
            if not targets or not targets <= set(self.genes):
                raise ValueError("Every Arc perturbation target must occur on the gene axis")
        except Exception:
            self.file.close()
            raise

    def close(self) -> None:
        self.file.close()

    def matrix(self, rows: np.ndarray) -> sp.csr_matrix:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or not len(rows) or np.any(rows < 0) or np.any(rows >= self.n_cells):
            raise ValueError("Source row indices must be a nonempty in-range vector")
        unique, inverse = np.unique(rows, return_inverse=True)
        data_parts = []
        index_parts = []
        ptr = [0]
        for row in unique:
            start, end = int(self.ptr[row]), int(self.ptr[row + 1])
            indices = self.indices[start:end]
            values = self.values[start:end]
            if (
                not len(indices)
                or np.any(indices < 0)
                or np.any(indices >= self.n_genes)
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or not np.equal(values, np.floor(values)).all()
            ):
                raise ValueError(f"Invalid raw Arc count values or indices in source row {row}")
            index_parts.append(indices)
            data_parts.append(values.astype(np.float32, copy=False))
            ptr.append(ptr[-1] + len(indices))
        matrix = sp.csr_matrix(
            (np.concatenate(data_parts), np.concatenate(index_parts), np.asarray(ptr, dtype=np.int64)),
            shape=(len(unique), self.n_genes),
            dtype=np.float32,
        )
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0):
            raise ValueError("A source cell has no nonzero counts on its gene axis")
        return matrix[inverse]

    def counts(self, rows: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.matrix(rows).toarray(), dtype=torch.float32)

    def annotation(self, rows: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "target_gene": self.labels[rows],
            "context": self.contexts[rows],
            "batch": self.batches[rows],
        }, index=pd.Index(self.ids[rows], name="cell_id"))


def make_split(source: ArcSource, heldout: list[str], seed: int) -> dict:
    targets = set(source.labels) - {CONTROL}
    if len(heldout) < 2 or len(set(heldout)) != len(heldout) or not set(heldout) < targets:
        raise ValueError("Held-out targets must be unique, available, and leave training targets")
    rng = np.random.default_rng(seed)
    control_rows = np.flatnonzero(source.labels == CONTROL)
    controls = defaultdict(list)
    for row in control_rows:
        controls[(source.contexts[row], source.batches[row])].append(int(row))
    perturb_groups = {
        (source.contexts[row], source.batches[row])
        for row in np.flatnonzero(source.labels != CONTROL)
    }
    if perturb_groups - set(controls):
        raise ValueError(f"Perturbation batches have no matched controls: {sorted(perturb_groups-set(controls))}")
    train_control, reference_control, input_control = [], [], []
    for key in sorted(controls):
        group = rng.permutation(np.asarray(controls[key], dtype=np.int64))
        if len(group) < 5:
            raise ValueError(f"Need at least five independent controls in batch {key}")
        n_each = min(max(1, round(len(group) * 0.125)), (len(group) - 3) // 2)
        input_control.extend(group[:n_each])
        reference_control.extend(group[n_each:2 * n_each])
        train_control.extend(group[2 * n_each:])
    holdout_rows = np.flatnonzero(np.isin(source.labels, heldout))
    perturbed_train = np.flatnonzero(
        (source.labels != CONTROL) & ~np.isin(source.labels, heldout)
    )
    train = np.sort(np.concatenate((perturbed_train, np.asarray(train_control, dtype=np.int64))))
    if len(train) < 2 or len(holdout_rows) < 2:
        raise ValueError("Training and heldout partitions need at least two cells")
    parts = {
        "train": train,
        "heldout_targets": np.sort(holdout_rows),
        "reference_controls": np.sort(np.asarray(reference_control, dtype=np.int64)),
        "input_controls": np.sort(np.asarray(input_control, dtype=np.int64)),
    }
    if sum(map(len, parts.values())) != source.n_cells:
        raise ValueError("The Arc split does not partition all source cells")
    all_rows = np.concatenate(list(parts.values()))
    if len(set(all_rows.tolist())) != source.n_cells:
        raise ValueError("The Arc split contains overlapping cells")
    return parts


def eval_rows(source: ArcSource, split: dict, heldout: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 11)
    selected_targets = []
    for target in sorted(heldout):
        candidates = split["heldout_targets"][source.labels[split["heldout_targets"]] == target]
        selected_targets.extend(rng.choice(candidates, min(100, len(candidates)), replace=False))
    chosen_controls = {}
    for role in ("reference_controls", "input_controls"):
        groups = defaultdict(list)
        for row in split[role]:
            groups[(source.contexts[row], source.batches[row])].append(int(row))
        chosen = []
        for key in sorted(groups):
            available = np.asarray(groups[key], dtype=np.int64)
            chosen.extend(rng.choice(available, min(20, len(available)), replace=False))
        chosen_controls[role] = chosen
    reference = np.sort(np.asarray(selected_targets + chosen_controls["reference_controls"], dtype=np.int64))
    inputs = np.sort(np.asarray(chosen_controls["input_controls"], dtype=np.int64))
    if not len(reference) or not len(inputs):
        raise ValueError("The bounded evaluation has no reference or input-control cells")
    return reference, inputs


def write_split_h5ad(path: Path, source: ArcSource, rows: np.ndarray) -> str:
    if path.exists() or path.with_name(path.name + ".part").exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    result = ad.AnnData(
        X=source.matrix(rows),
        obs=source.annotation(rows),
        var=pd.DataFrame(index=pd.Index(source.genes, name="gene")),
    )
    validate_counts(result)
    partial = path.with_name(path.name + ".part")
    result.write_h5ad(partial, compression="gzip", compression_opts=1)
    with h5py.File(partial, "r") as file:
        if tuple(file["X"].attrs["shape"]) != result.shape:
            raise RuntimeError(f"Written split has an unexpected shape: {path}")
    checksum = file_sha256(partial)
    os.link(partial, path)
    partial.unlink()
    return checksum


def batch_ranges(size: int, batch_size: int, seed: int, epoch: int) -> list[tuple[int, int]]:
    if size < 2 or batch_size < 2:
        raise ValueError("At least two cells and a batch size of at least two are required")
    ranges = [(start, min(size, start + batch_size)) for start in range(0, size, batch_size)]
    if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] == 1:
        ranges[-2] = (ranges[-2][0], ranges[-1][1])
        ranges.pop()
    order = np.random.default_rng(seed + epoch + 100_003).permutation(len(ranges))
    return [ranges[int(i)] for i in order]


def shifts_for_training(
    source: ArcSource, train_rows: np.ndarray, model: C3VCell, *,
    batch_size: int, min_cells: int, monitor=None,
) -> tuple[dict, dict]:
    latent = np.empty((len(train_rows), model.config.latent_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(train_rows), batch_size):
            end = min(start + batch_size, len(train_rows))
            latent[start:end] = model.encode(source.counts(train_rows[start:end])).cpu().numpy()
            if (start // batch_size) % 250 == 0:
                if monitor is not None:
                    monitor()
                print(f"Encoded population means: {start}/{len(train_rows)}", flush=True)
    if not np.isfinite(latent).all():
        raise RuntimeError("The streamed latent population contains non-finite values")
    controls, perturbed = defaultdict(list), defaultdict(list)
    for position, row in enumerate(train_rows):
        key = (source.contexts[row], source.batches[row])
        if source.labels[row] == CONTROL:
            controls[key].append(position)
        else:
            perturbed[(source.labels[row], *key)].append(position)
    controls = {key: np.asarray(value, dtype=np.int64) for key, value in controls.items()}
    if any((context, batch) not in controls for _, context, batch in perturbed):
        raise ValueError("A streamed perturbation group has no matched training controls")
    means = {}
    groups = defaultdict(list)
    excluded = 0
    for (target, context, batch), positions in sorted(perturbed.items()):
        key = (context, batch)
        control_positions = controls[key]
        if len(positions) < min_cells or len(control_positions) < min_cells:
            excluded += 1
            continue
        if key not in means:
            means[key] = latent[control_positions].mean(axis=0)
        shift = latent[np.asarray(positions, dtype=np.int64)].mean(axis=0) - means[key]
        groups[target].append((control_positions, shift.astype(np.float32)))
    if not groups:
        raise ValueError("No training group has enough matched cells to estimate a population shift")
    all_targets = set(source.labels[train_rows]) - {CONTROL}
    details = {
        "eligible_groups": sum(len(value) for value in groups.values()),
        "excluded_groups": excluded,
        "targets_without_eligible_groups": sorted(map(str, all_targets - set(groups))),
        "min_group_cells": min_cells,
        "population_cells_encoded": len(train_rows),
    }
    return dict(groups), details


def verify_final(path: Path, plan: dict, expected_state=None) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    provenance = plan["provenance"]
    training_ids = checkpoint.get("training_cell_ids")
    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("control") != CONTROL
        or checkpoint.get("training_cells") != provenance["train_cells"]
        or not isinstance(training_ids, list)
        or len(training_ids) != provenance["train_cells"]
        or len(set(training_ids)) != len(training_ids)
        or digest_json(training_ids) != provenance["train_ids_sha256"]
        or digest_json(checkpoint.get("genes")) != provenance["genes_sha256"]
        or checkpoint.get("training_source_sha256") != provenance["source_sha256"]
        or checkpoint.get("streaming_provenance", {}).get("plan_sha256") != digest_json(plan)
        or checkpoint.get("reference", {}).get("source_identity") != provenance["reference_source_identity"]
        or set(checkpoint.get("training_targets", [])) & set(provenance["heldout_targets"])
    ):
        raise ValueError("The streamed Arc model failed its provenance verification")
    for name, value in checkpoint["model_state"].items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid parameter in streamed Arc model: {name}")
        if expected_state is not None and (name not in expected_state or not torch.equal(value, expected_state[name])):
            raise ValueError(f"Existing model differs from its final recovery state: {name}")
    if expected_state is not None and set(checkpoint["model_state"]) != set(expected_state):
        raise ValueError("Existing model has a different parameter set from recovery")
    return checkpoint


def config_from_args(args) -> dict:
    return {
        "source": str(args.source.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "reference": str(args.reference.resolve()),
        "reference_manifest": str(args.reference_manifest.resolve()),
        "holdout_split": str(args.holdout_split.resolve()),
        "seed": args.seed,
        "pretrain_epochs": args.pretrain_epochs,
        "flow_steps": args.flow_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "variance_weight": args.variance_weight,
        "covariance_weight": args.covariance_weight,
        "min_group_cells": args.min_group_cells,
        "reference_max_cells": args.reference_max_cells,
        "reference_seed": args.reference_seed,
        "reference_min_coverage": args.reference_min_coverage,
    }


def inspect_inputs(args, *, rehash_source: bool) -> tuple[ArcSource, dict, dict, list[str], object]:
    for path in (args.source, args.source_manifest, args.reference, args.reference_manifest, args.holdout_split):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        args.pretrain_epochs < 1 or args.flow_steps < 1 or args.batch_size < 2
        or args.latent_dim < 1 or args.hidden_dim < 1 or args.min_group_cells < 1
        or args.reference_max_cells < 2 or not 0 < args.reference_min_coverage <= 1
        or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
        or not math.isfinite(args.variance_weight) or args.variance_weight < 0
        or not math.isfinite(args.covariance_weight) or args.covariance_weight < 0
    ):
        raise ValueError("Invalid training dimensions, steps, hyperparameters or reference settings")
    check_disk(args.outdir.parent, headroom=NEED_HEADROOM)
    manifest = json.loads(args.source_manifest.read_text())
    matches = [item for item in manifest["files"] if item.get("rfilename") == args.source.name]
    if len(matches) != 1:
        raise ValueError("The source manifest does not uniquely identify the Arc H5AD")
    expected = matches[0].get("lfs", {}).get("sha256")
    stat = args.source.stat()
    if (
        not isinstance(expected, str) or len(expected) != 64
        or matches[0]["size"] != stat.st_size
        or matches[0].get("lfs", {}).get("size") != stat.st_size
    ):
        raise ValueError("The Arc source size or LFS checksum is invalid")
    if rehash_source and file_sha256(args.source) != expected:
        raise ValueError("Arc H5AD SHA-256 does not match the fixed-version manifest")
    reference_manifest = json.loads(args.reference_manifest.read_text())
    if (
        reference_manifest.get("source_matrix") != "raw/X"
        or file_sha256(args.reference) != reference_manifest.get("reference_sha256")
    ):
        raise ValueError("The Tabula raw-count reference failed verification")
    heldout = sorted(json.loads(args.holdout_split.read_text())["test_targets"])
    source = ArcSource(args.source)
    try:
        split = make_split(source, heldout, args.seed)
        reference = load_reference(
            args.reference, source.genes, max_cells=args.reference_max_cells,
            seed=args.reference_seed, min_coverage=args.reference_min_coverage,
        )
        check_cell_id_schemes(source.ids, reference.cell_ids)
        if set(source.ids) & set(reference.cell_ids):
            raise ValueError("Tabula reference cells overlap the Arc source")
        metadata = {
            "source_sha256": expected,
            "source_size_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "source_revision": manifest.get("revision"),
            "reference_sha256": reference_manifest["reference_sha256"],
            "reference_source_identity": reference.source_identity,
            "holdout_split_sha256": file_sha256(args.holdout_split),
            "code_sha256": file_sha256(Path(__file__)),
            "genes_sha256": digest_json(source.genes),
            "train_ids_sha256": digest_json(source.ids[split["train"]].tolist()),
            "heldout_targets": heldout,
            "source_cells": source.n_cells,
            "train_cells": len(split["train"]),
            "heldout_target_cells": len(split["heldout_targets"]),
            "reference_control_cells": len(split["reference_controls"]),
            "input_control_cells": len(split["input_controls"]),
            "train_targets": len(set(source.labels[split["train"]]) - {CONTROL}),
            "model_genes": source.n_genes,
            "reference_cells": len(reference.cell_ids),
            "reference_coverage": reference.shared_genes / source.n_genes,
            "holdout_previously_examined": True,
            "official_leaderboard_score": False,
        }
        return source, split, metadata, heldout, reference
    except Exception:
        source.close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "source-manifest", "reference", "reference-manifest", "holdout-split", "outdir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--flow-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--variance-weight", type=float, default=5.0)
    parser.add_argument("--covariance-weight", type=float, default=0.1)
    parser.add_argument("--min-group-cells", type=int, default=3)
    parser.add_argument("--reference-max-cells", type=int, default=12000)
    parser.add_argument("--reference-seed", type=int, default=20261111)
    parser.add_argument("--reference-min-coverage", type=float, default=0.9)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-batches", type=int, default=0)
    parser.add_argument("--stop-after-flow-step", type=int, default=0)
    args = parser.parse_args()
    if (
        args.checkpoint_every < 1 or args.stop_after_batches < 0
        or args.stop_after_flow_step < 0 or args.stop_after_flow_step > args.flow_steps
    ):
        raise ValueError("Checkpoint interval must be positive and stop count nonnegative")
    if args.resume and args.check_only:
        raise ValueError("Choose either --resume or --check-only")
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {args.outdir}")
    if args.resume and not args.outdir.is_dir():
        raise FileNotFoundError(f"Resume requires an existing run: {args.outdir}")
    torch.set_num_threads(1)
    source, split, metadata, heldout, reference = inspect_inputs(args, rehash_source=not args.resume)
    try:
        config = config_from_args(args)
        ranges_per_epoch = len(batch_ranges(len(split["train"]), args.batch_size, args.seed, 0))
        metadata["representation_batches_per_epoch"] = ranges_per_epoch
        metadata["representation_updates"] = 2 * ranges_per_epoch * args.pretrain_epochs
        if args.check_only:
            print(json.dumps({"config": config, "provenance": metadata}, indent=2, allow_nan=False))
            return
        if not args.resume:
            args.outdir.mkdir()
        lock_path = args.outdir / ".lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            state_path = args.outdir / "state.json"
            plan_path = args.outdir / "plan.json"
            recovery_path = args.outdir / "recovery.pt"
            state = {"state": "running", "started_at_utc": now()}
            if args.resume:
                existing = json.loads(plan_path.read_text())
                if existing.get("config") != config or existing.get("provenance") != metadata:
                    raise ValueError("Source, code, reference, split or settings changed since checkpoint")
                for role, expected in existing["evaluation_sha256"].items():
                    if file_sha256(args.outdir / "split" / f"{role}.h5ad") != expected:
                        raise ValueError(f"The {role} evaluation split changed since checkpoint")
                plan = existing
                snapshot = torch.load(recovery_path, map_location="cpu", weights_only=True)
                if snapshot.get("plan_sha256") != digest_json(plan):
                    raise ValueError("Recovery checkpoint does not match its immutable plan")
                if snapshot.get("phase") == "complete":
                    final_path = args.outdir / "model.pt"
                    completed = verify_final(final_path, plan, snapshot["model_state"])
                    state.update(
                        state="complete", stage="complete", finished_at_utc=now(),
                        checkpoint=str(final_path), checkpoint_sha256=file_sha256(final_path),
                        population_shift_groups=completed["population_shift_groups"],
                        representation_diagnostics=completed["representation_diagnostics"],
                    )
                    write_json(state_path, state)
                    print(f"Verified completed streamed Arc model: {final_path}", flush=True)
                    return
            else:
                state.update(stage="initializing", provenance=metadata)
                write_json(state_path, state)
                try:
                    split_dir = args.outdir / "split"
                    split_dir.mkdir()
                    reference_rows, input_rows = eval_rows(source, split, heldout, args.seed)
                    check_disk(args.outdir, headroom=NEED_HEADROOM)
                    checksums = {
                        "reference": write_split_h5ad(split_dir / "reference.h5ad", source, reference_rows),
                        "input_controls": write_split_h5ad(split_dir / "input_controls.h5ad", source, input_rows),
                    }
                    plan = {
                        "format": "arc-stream-training-v1",
                        "config": config,
                        "provenance": metadata,
                        "evaluation_sha256": checksums,
                        "evaluation_cells": {"reference": len(reference_rows), "input_controls": len(input_rows)},
                        "notes": [
                            "A single representation epoch visits every training cell exactly once.",
                            "For each Arc batch, one bounded Tabula reference batch is also trained.",
                            "Heldout targets and separate control cells are excluded from all training phases.",
                            "Evaluation targets were examined previously; metrics are not official scores.",
                        ],
                    }
                    write_json(plan_path, plan)
                except Exception as exc:
                    state.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_at_utc=now())
                    write_json(state_path, state)
                    raise
                snapshot = None

            if snapshot is None:
                torch.manual_seed(args.seed)
            model = C3VCell(ModelConfig(
                source.n_genes, args.latent_dim, args.hidden_dim,
                variance_weight=args.variance_weight, covariance_weight=args.covariance_weight,
                regularization_view="unmasked",
            ))
            rng = np.random.default_rng(args.seed)
            if snapshot is None:
                phase, epoch, batch_position, flow_step = "representation", 0, 0, 0
            else:
                model.load_state_dict(snapshot["model_state"])
                rng.bit_generator.state = snapshot["numpy_rng_state"]
                torch.set_rng_state(snapshot["torch_rng_state"])
                phase = snapshot["phase"]
                epoch = snapshot["epoch"]
                batch_position = snapshot["batch_position"]
                flow_step = snapshot["flow_step"]
            diagnostics = None if snapshot is None else snapshot.get("representation_diagnostics")
            groups_details = None if snapshot is None else snapshot.get("population_shift_groups")
            stopping = False

            def request_stop(signum, frame):
                nonlocal stopping
                stopping = True

            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)

            def save(optimizer) -> None:
                check_disk(args.outdir)
                save_torch(recovery_path, {
                    "format": "arc-stream-recovery-v1",
                    "plan_sha256": digest_json(plan),
                    "phase": phase,
                    "epoch": epoch,
                    "batch_position": batch_position,
                    "flow_step": flow_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "representation_diagnostics": diagnostics,
                    "population_shift_groups": groups_details,
                })
                state.update(
                    state="running", stage=phase, epoch=epoch, next_batch=batch_position,
                    flow_step=flow_step, updated_at_utc=now(),
                )
                write_json(state_path, state)

            try:
                if phase == "representation":
                    optimizer = torch.optim.AdamW(
                        list(model.encoder.parameters()) + list(model.predictor.parameters())
                        + list(model.decoder.parameters()) + [model.log_theta],
                        lr=args.learning_rate,
                    )
                    if snapshot is not None:
                        optimizer.load_state_dict(snapshot["optimizer_state"])
                    else:
                        save(optimizer)
                    model.train()
                    batches_this_invocation = 0
                    while epoch < args.pretrain_epochs:
                        batches = batch_ranges(len(split["train"]), args.batch_size, args.seed, epoch)
                        while batch_position < len(batches):
                            left, right = batches[batch_position]
                            counts = source.counts(split["train"][left:right])
                            ref_rows = rng.integers(0, reference.matrix.shape[0], size=len(counts))
                            ref_counts = torch.as_tensor(reference.matrix[ref_rows].toarray(), dtype=torch.float32)
                            ref_mask = torch.as_tensor(
                                np.broadcast_to(reference.measured, ref_counts.shape).copy(),
                                dtype=torch.float32,
                            )
                            for current, measured, label in ((counts, None, "arc"), (ref_counts, ref_mask, "reference")):
                                optimizer.zero_grad(set_to_none=True)
                                loss, metrics = model.representation_loss(current, measured=measured)
                                if not torch.isfinite(loss):
                                    raise RuntimeError(f"Non-finite representation loss at batch {batch_position}")
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                                optimizer.step()
                                model.update_teacher()
                            batch_position += 1
                            batches_this_invocation += 1
                            if batch_position == 1 or batch_position % 25 == 0 or batch_position == len(batches):
                                print(json.dumps({
                                    "phase": phase, "epoch": epoch, "batch": batch_position,
                                    "batches": len(batches), "last_source": label, **metrics,
                                }), flush=True)
                            if batch_position % args.checkpoint_every == 0 or stopping:
                                save(optimizer)
                            if stopping or (args.stop_after_batches and batches_this_invocation >= args.stop_after_batches):
                                save(optimizer)
                                state.update(state="paused", updated_at_utc=now())
                                write_json(state_path, state)
                                return
                        epoch += 1
                        batch_position = 0
                        save(optimizer)
                    diag_rng = np.random.default_rng(args.seed + 8)
                    rows = np.sort(diag_rng.choice(
                        split["train"], min(512, len(split["train"])), replace=False,
                    ))
                    diagnostic_matrix = source.matrix(rows)
                    diagnostics = representation_diagnostics(
                        model, diagnostic_matrix, device=torch.device("cpu"),
                        batch_size=args.batch_size, seed=args.seed, max_cells=len(rows),
                    )
                    if diagnostics["collapse_warning"]:
                        warnings.warn(
                            "Streamed representation has a low-variation or low-rank warning; "
                            "conditioned predictions may be unreliable.",
                            RuntimeWarning, stacklevel=2,
                        )
                    phase = "group-preparation"
                    save(None)
                if phase in ("group-preparation", "flow"):
                    model.encoder.requires_grad_(False)
                    model.decoder.requires_grad_(False)
                    model.predictor.requires_grad_(False)
                    def group_monitor():
                        if stopping:
                            raise TrainingPaused("Pause requested during group preparation")
                        check_disk(args.outdir)

                    groups, groups_details = shifts_for_training(
                        source, split["train"], model, batch_size=args.batch_size,
                        min_cells=args.min_group_cells, monitor=group_monitor,
                    )
                    optimizer = torch.optim.AdamW(model.velocity.parameters(), lr=args.learning_rate)
                    if phase == "flow":
                        optimizer.load_state_dict(snapshot["optimizer_state"])
                    else:
                        phase = "flow"
                        save(optimizer)
                    model.train()
                    targets = np.asarray(sorted(groups), dtype=object)
                    gene_index = {name: index for index, name in enumerate(source.genes)}
                    while flow_step < args.flow_steps:
                        chosen = rng.choice(targets, size=args.batch_size)
                        picked = [groups[name][int(rng.integers(len(groups[name])))] for name in chosen]
                        positions = np.asarray([rng.choice(group[0]) for group in picked], dtype=np.int64)
                        counts = source.counts(split["train"][positions])
                        shift = torch.as_tensor(np.stack([group[1] for group in picked]), dtype=torch.float32)
                        indices = torch.tensor([gene_index[name] for name in chosen], dtype=torch.long)
                        optimizer.zero_grad(set_to_none=True)
                        loss, metrics = model.flow_loss_shift(counts, shift, indices)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite flow loss at step {flow_step}")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.velocity.parameters(), 5.0)
                        optimizer.step()
                        flow_step += 1
                        if flow_step == 1 or flow_step % 25 == 0 or flow_step == args.flow_steps:
                            print(json.dumps({"phase": phase, "step": flow_step, **metrics}), flush=True)
                        pause_at_step = args.stop_after_flow_step and flow_step >= args.stop_after_flow_step
                        if flow_step % args.checkpoint_every == 0 or stopping or pause_at_step:
                            save(optimizer)
                        if stopping or pause_at_step:
                            state.update(state="paused", updated_at_utc=now())
                            write_json(state_path, state)
                            return
                    save(optimizer)
                    checkpoint = {
                        "format_version": 1,
                        "model_config": model.configuration(),
                        "model_state": model.state_dict(),
                        "genes": source.genes,
                        "control": CONTROL,
                        "graph": None,
                        "epigenetics": None,
                        "seed": args.seed,
                        "training_targets": sorted(map(str, set(source.labels[split["train"]]) - {CONTROL})),
                        "training_cells": len(split["train"]),
                        "training_cell_ids": source.ids[split["train"]].tolist(),
                        "training_source_sha256": metadata["source_sha256"],
                        "representation_diagnostics": diagnostics,
                        "training_parameters": {
                            "pretrain_steps": metadata["representation_updates"],
                            "flow_steps": args.flow_steps,
                            "batch_size": args.batch_size,
                            "learning_rate": args.learning_rate,
                            "variance_weight": args.variance_weight,
                            "covariance_weight": args.covariance_weight,
                            "regularization_view": "unmasked",
                            "allow_unmatched_batch": False,
                            "flow_target": "population-shift",
                            "min_group_cells": args.min_group_cells,
                            "pretrain_epochs": args.pretrain_epochs,
                            "streaming": True,
                        },
                        "population_shift_groups": groups_details,
                        "reference": reference_metadata(
                            reference, args.reference, fraction=0.5,
                            max_cells=args.reference_max_cells, seed=args.reference_seed,
                            min_coverage=args.reference_min_coverage, model_genes=source.n_genes,
                        ),
                        "unmatched_batch_groups": [],
                        "streaming_provenance": {
                            "plan_sha256": digest_json(plan),
                            "heldout_previously_examined": True,
                            "official_leaderboard_score": False,
                        },
                    }
                    for tensor in checkpoint["model_state"].values():
                        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                            raise ValueError("The final model contains non-finite parameters")
                    source_stat = args.source.stat()
                    if source_stat.st_size != metadata["source_size_bytes"] or source_stat.st_mtime_ns != metadata["source_mtime_ns"]:
                        raise RuntimeError("The Arc source changed while training")
                    check_disk(args.outdir)
                    final_path = args.outdir / "model.pt"
                    if not final_path.exists():
                        save_torch(final_path, checkpoint)
                    verify_final(final_path, plan, model.state_dict())
                    phase = "complete"
                    save(None)
                    state.update(
                        state="complete", stage="complete", finished_at_utc=now(),
                        checkpoint=str(final_path), checkpoint_sha256=file_sha256(final_path),
                        population_shift_groups=groups_details,
                        representation_diagnostics=diagnostics,
                    )
                    write_json(state_path, state)
                    print(f"Completed streamed Arc model: {final_path}", flush=True)
            except TrainingPaused:
                state.update(state="paused", updated_at_utc=now())
                write_json(state_path, state)
            except Exception as exc:
                state.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_at_utc=now())
                write_json(state_path, state)
                raise
    finally:
        source.close()


if __name__ == "__main__":
    main()

"""Resumable, target-disjoint representation training on Orion parquet shards."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch

from c3vcell.features import gene_token_mapping
from c3vcell.model import C3VCell, ModelConfig
from c3vcell.training import representation_diagnostics


RESERVE = 200 * 1024**3
SCANNER_PATH = Path(__file__).with_name("scan-orion-stream-v1.py")
spec = importlib.util.spec_from_file_location("orion_scanner_support", SCANNER_PATH)
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)
CONTROL = scanner.CONTROL
COLUMNS = [
    "gene_token_id", "gene_expression", "cell_barcode", "sample",
    "guide_target", "gene_target", "pass_guide_filter",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_torch(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def check_disk(path: Path, headroom: int = 0) -> None:
    if shutil.disk_usage(path).free < RESERVE + headroom:
        raise OSError("Free disk space would fall below the required 200 GiB reserve")


def select_training(batch: pa.RecordBatch, context: str, heldout: set[str], known: set[str], seed: int):
    labels = batch.column("gene_target").to_pylist()
    samples = batch.column("sample").to_pylist()
    barcodes = batch.column("cell_barcode").to_pylist()
    guides = batch.column("guide_target").to_pylist()
    qc = batch.column("pass_guide_filter").to_pylist()
    selected = []
    for index, (label, sample, barcode, guide, passed) in enumerate(zip(
        labels, samples, barcodes, guides, qc, strict=True,
    )):
        if passed is None or passed is False or passed == 0:
            continue
        if passed is not True and passed != 1:
            raise ValueError(f"Invalid Orion QC value: {passed!r}")
        if sample is None or barcode is None or guide is None:
            raise ValueError("A passing Orion cell lacks sample, barcode, or guide")
        if label == CONTROL:
            keep = scanner.control_role(context, str(sample), str(barcode), seed) == "training"
        else:
            keep = isinstance(label, str) and label in known and label not in heldout
        if keep:
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def counts_from_batch(batch: pa.RecordBatch, selected: np.ndarray, token_map: np.ndarray, n_genes: int):
    if not len(selected):
        return sp.csr_matrix((0, n_genes), dtype=np.float32)
    picked = batch.take(pa.array(selected, type=pa.int64()))
    tokens = picked.column("gene_token_id")
    values = picked.column("gene_expression")
    if (
        not pa.types.is_list(tokens.type) or not pa.types.is_list(values.type)
        or tokens.null_count or values.null_count
    ):
        raise ValueError("Orion gene and expression fields must be complete list arrays")
    token_lengths = np.diff(tokens.offsets.to_numpy())
    value_lengths = np.diff(values.offsets.to_numpy())
    if not np.array_equal(token_lengths, value_lengths):
        raise ValueError("Orion gene and expression arrays have different lengths")
    flat_tokens = tokens.flatten()
    flat_values = values.flatten()
    if flat_tokens.null_count or flat_values.null_count:
        raise ValueError("Orion gene and expression arrays contain null elements")
    raw_ids = flat_tokens.to_numpy(zero_copy_only=False)
    expression = flat_values.to_numpy(zero_copy_only=False)
    if (
        np.any(raw_ids < 0) or np.any(raw_ids >= len(token_map))
        or not np.isfinite(expression).all() or np.any(expression < 0)
        or not np.equal(expression, np.floor(expression)).all()
        or np.any(expression > np.iinfo(np.int32).max)
    ):
        raise ValueError("Invalid Orion gene tokens or raw integer counts")
    columns = token_map[raw_ids]
    if np.any(columns < 0):
        raise ValueError("An Orion gene token is absent from the source annotation")
    rows = np.repeat(np.arange(len(selected), dtype=np.int32), token_lengths)
    matrix = sp.coo_matrix(
        (expression, (rows, columns)), shape=(len(selected), n_genes), dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    if (
        not np.isfinite(matrix.data).all() or np.any(matrix.data > np.iinfo(np.int32).max)
        or np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0)
    ):
        raise ValueError("Aggregated Orion counts are invalid or an eligible cell has no counts")
    return matrix


def create_token_map(tokens: dict[int, int]) -> np.ndarray:
    if not tokens or max(tokens) > 10_000_000:
        raise ValueError("Orion gene token space is unexpectedly large")
    mapping = np.full(max(tokens) + 1, -1, dtype=np.int32)
    for token, index in tokens.items():
        mapping[token] = index
    return mapping


def split_training_sizes(metadata: dict, context: str):
    heldout = set(metadata["heldout_targets"])
    targets = metadata["target_counts"][context]
    target_cells = sum(count for target, count in targets.items() if target not in heldout)
    controls = sum(
        item["training_controls"] for item in metadata["samples"].values() if item["context"] == context
    )
    training_targets = sorted(target for target in targets if target not in heldout)
    return int(target_cells + controls), training_targets


def check_files(root: Path, entries: list[dict]) -> None:
    for item in entries:
        stat = (root / item["path"]).stat()
        if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"Orion source shard changed after its metadata scan: {item['path']}")


def diagnostic_sample(root, entries, model, token_map, known, heldout, context, split_seed, diagnostic_seed, batch_size):
    pieces = []
    count = 0
    for item in entries:
        for batch in pq.ParquetFile(root / item["path"]).iter_batches(batch_size=128, columns=COLUMNS):
            selected = select_training(batch, context, heldout, known, split_seed)
            if len(selected):
                selected = selected[:512 - count]
                pieces.append(counts_from_batch(batch, selected, token_map, model.config.n_genes))
                count += len(selected)
            if count >= 512:
                break
        if count >= 512:
            break
    if count < 2:
        raise ValueError("Insufficient training cells for representation diagnostics")
    result = representation_diagnostics(
        model, sp.vstack(pieces, format="csr"), device=torch.device("cpu"),
        batch_size=batch_size, seed=diagnostic_seed, max_cells=count,
    )
    result["sampling"] = "first 512 eligible training cells in sorted source shards"
    return result


def verify_final(path: Path, plan: dict, expected_state: dict):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format") != "orion-stream-representation-v1"
        or checkpoint.get("plan_sha256") != digest_json(plan)
        or checkpoint.get("training_cells") != plan["eligible_training_cells"]
        or checkpoint.get("genes") != plan["genes"]
    ):
        raise ValueError("Orion representation checkpoint does not match the immutable plan")
    if set(checkpoint["model_state"]) != set(expected_state):
        raise ValueError("Orion final model parameter names differ")
    for name, value in checkpoint["model_state"].items():
        if not torch.isfinite(value).all() or not torch.equal(value, expected_state[name]):
            raise ValueError(f"Invalid or changed Orion model parameter: {name}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata-plan", type=Path, required=True)
    parser.add_argument("--context", choices=scanner.CONTEXTS, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--variance-weight", type=float, default=5.0)
    parser.add_argument("--covariance-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-batches", type=int, default=0)
    args = parser.parse_args()
    if (
        args.epochs < 1 or args.batch_size < 2 or args.checkpoint_every < 1
        or args.stop_after_batches < 0 or args.learning_rate <= 0
        or args.resume and args.check_only
    ):
        raise ValueError("Invalid representation training settings")
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and args.outdir.exists():
        raise FileExistsError("Refusing to overwrite an existing Orion training run")
    if args.resume and not args.outdir.is_dir():
        raise FileNotFoundError("Cannot resume a missing Orion training run")
    check_disk(args.outdir.parent, 3 * 1024**3)
    torch.set_num_threads(1)
    metadata = json.loads(args.metadata_plan.read_text())
    if metadata.get("state") != "complete" or metadata["config"]["seed"] != args.seed:
        raise ValueError("A completed Orion metadata plan with the same split seed is required")
    if metadata["provenance"]["script_sha256"] != sha256(SCANNER_PATH):
        raise ValueError("The Orion split algorithm changed since metadata planning")
    gene_metadata = args.root / "metadata/gene_metadata.parquet"
    if metadata["provenance"]["gene_metadata_sha256"] != sha256(gene_metadata):
        raise ValueError("Orion gene metadata changed after metadata planning")
    genes, tokens, ensembls = gene_token_mapping(
        pd.read_parquet(gene_metadata), token_column="gene_token_id", symbol_column="gene_name",
    )
    mapping = create_token_map(tokens)
    known, heldout = set(genes), set(metadata["heldout_targets"])
    entries = [
        item for item in metadata["provenance"]["files"]
        if Path(item["path"]).name.startswith(args.context + "_")
    ]
    if not entries or not heldout.issubset(known):
        raise ValueError("The metadata plan lacks shards or valid held-out genes")
    check_files(args.root, entries)
    expected_cells, training_targets = split_training_sizes(metadata, args.context)
    if expected_cells < 2 or not training_targets:
        raise ValueError("There are too few eligible Orion training cells or targets")
    config = {
        "context": args.context, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim, "variance_weight": args.variance_weight,
        "covariance_weight": args.covariance_weight, "seed": args.seed,
        "checkpoint_every": args.checkpoint_every,
    }
    plan = {
        "format": "orion-stream-representation-plan-v1", "config": config,
        "source_repo": metadata["provenance"]["source_repo"],
        "source_revision": metadata["provenance"]["revision"],
        "metadata_plan_sha256": sha256(args.metadata_plan),
        "script_sha256": sha256(Path(__file__)), "scanner_support_sha256": sha256(SCANNER_PATH),
        "gene_metadata_sha256": sha256(gene_metadata), "files": entries,
        "genes": genes, "ensembl_ids": [str(ensembls.loc[name]) for name in genes],
        "heldout_targets": sorted(heldout), "training_targets": training_targets,
        "eligible_training_cells": expected_cells,
        "notes": [
            "Each epoch visits every eligible target/control training cell once.",
            "Held-out perturbations and distinct inference/reference controls never enter training.",
            "If one cell remains at an epoch end, its update omits the undefined covariance term.",
            "This is representation only; its population-flow parameters are untrained.",
            "Orion is licensed CC BY-NC-SA 4.0; this is not an official challenge result.",
        ],
    }
    if args.check_only:
        first = entries[0]
        batch = next(pq.ParquetFile(args.root / first["path"]).iter_batches(batch_size=128, columns=COLUMNS))
        rows = select_training(batch, args.context, heldout, known, args.seed)
        counts_from_batch(batch, rows, mapping, len(genes))
        print(json.dumps({
            "context": args.context, "shards": len(entries), "model_genes": len(genes),
            "eligible_training_cells": expected_cells, "training_targets": len(training_targets),
            "heldout_targets": sorted(heldout), "probe_rows": len(rows),
        }, indent=2))
        return
    if not args.resume:
        args.outdir.mkdir()
    with (args.outdir / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path, state_path = args.outdir / "plan.json", args.outdir / "state.json"
        recovery_path = args.outdir / "recovery.pt"
        state = {"state": "running", "started_at_utc": now()}
        if args.resume:
            previous = json.loads(plan_path.read_text())
            if previous != plan:
                raise ValueError("Orion source, code or training settings changed since checkpoint")
            snapshot = torch.load(recovery_path, map_location="cpu", weights_only=True)
            if snapshot.get("plan_sha256") != digest_json(plan):
                raise ValueError("Orion recovery checkpoint does not match its plan")
            if snapshot.get("phase") == "complete":
                final = args.outdir / "representation.pt"
                done = verify_final(final, plan, snapshot["model_state"])
                state.update(
                    state="complete", stage="complete", finished_at_utc=now(),
                    checkpoint=str(final), checkpoint_sha256=sha256(final),
                    training_cells=done["training_cells"], representation_diagnostics=done["representation_diagnostics"],
                )
                write_json(state_path, state)
                return
        else:
            write_json(plan_path, plan)
            snapshot = None
            torch.manual_seed(args.seed)
        model = C3VCell(ModelConfig(
            len(genes), args.latent_dim, args.hidden_dim, variance_weight=args.variance_weight,
            covariance_weight=args.covariance_weight, regularization_view="unmasked",
        ))
        optimizer = torch.optim.AdamW(
            list(model.encoder.parameters()) + list(model.predictor.parameters())
            + list(model.decoder.parameters()) + [model.log_theta], lr=args.learning_rate,
        )
        if snapshot is None:
            epoch, file_position, batch_position, updates, trained, singleton_updates = 0, 0, 0, 0, 0, 0
            pending = None
        else:
            model.load_state_dict(snapshot["model_state"])
            optimizer.load_state_dict(snapshot["optimizer_state"])
            torch.set_rng_state(snapshot["torch_rng_state"])
            epoch, file_position, batch_position = snapshot["epoch"], snapshot["file_position"], snapshot["batch_position"]
            updates, trained, pending = snapshot["updates"], snapshot["trained"], snapshot["pending"]
            singleton_updates = snapshot["singleton_updates"]
        stopping = False

        def request_stop(signum, frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        def save(phase: str):
            check_disk(args.outdir)
            save_torch(recovery_path, {
                "format": "orion-stream-recovery-v1", "plan_sha256": digest_json(plan),
                "phase": phase, "epoch": epoch, "file_position": file_position,
                "batch_position": batch_position, "updates": updates, "trained": trained,
                "singleton_updates": singleton_updates,
                "pending": pending, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(),
            })
            state.update(
                state="running", stage=phase, epoch=epoch, file_position=file_position,
                files=len(entries), next_batch=batch_position, updates=updates, trained_cells=trained,
                expected_cells_per_epoch=expected_cells, pending_cells=0 if pending is None else len(pending),
                singleton_updates=singleton_updates,
                updated_at_utc=now(),
            )
            write_json(state_path, state)

        def update_model(counts: torch.Tensor):
            nonlocal updates, trained, singleton_updates
            optimizer.zero_grad(set_to_none=True)
            original_config = model.config
            if len(counts) == 1 and model.config.covariance_weight > 0:
                model.config = replace(model.config, covariance_weight=0.0)
                singleton_updates += 1
            try:
                loss, metrics = model.representation_loss(counts)
            finally:
                model.config = original_config
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Orion representation loss at update {updates}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.update_teacher()
            updates += 1
            trained += len(counts)
            if updates == 1 or updates % 100 == 0:
                print(json.dumps({
                    "phase": "representation", "context": args.context, "epoch": epoch,
                    "file": file_position, "files": len(entries), "updates": updates,
                    "trained_cells": trained, **metrics,
                }), flush=True)

        try:
            if snapshot is None:
                save("representation")
            model.train()
            invocation_updates = 0
            while epoch < args.epochs:
                while file_position < len(entries):
                    item = entries[file_position]
                    check_files(args.root, [item])
                    for index, batch in enumerate(pq.ParquetFile(args.root / item["path"]).iter_batches(
                        batch_size=args.batch_size, columns=COLUMNS,
                    )):
                        if index < batch_position:
                            continue
                        rows = select_training(batch, args.context, heldout, known, args.seed)
                        matrix = counts_from_batch(batch, rows, mapping, len(genes))
                        if len(rows):
                            current = torch.as_tensor(matrix.toarray(), dtype=torch.float32)
                            if pending is not None:
                                current = torch.cat((pending, current))
                                pending = None
                            if len(current) == 1:
                                pending = current
                            else:
                                update_model(current)
                                invocation_updates += 1
                        batch_position = index + 1
                        if stopping or (len(rows) and updates and updates % args.checkpoint_every == 0):
                            save("representation")
                        if stopping or (args.stop_after_batches and invocation_updates >= args.stop_after_batches):
                            save("representation")
                            state.update(state="paused", updated_at_utc=now())
                            write_json(state_path, state)
                            return
                    file_position += 1
                    batch_position = 0
                    save("representation")
                if pending is not None:
                    update_model(pending)
                    pending = None
                if trained != expected_cells * (epoch + 1):
                    raise ValueError(f"Eligible Orion cell count mismatch: {trained} != {expected_cells * (epoch + 1)}")
                epoch += 1
                file_position = batch_position = 0
                save("representation")
            diagnostics = diagnostic_sample(
                args.root, entries, model, mapping, known, heldout, args.context,
                args.seed, args.seed + 8, args.batch_size,
            )
            if diagnostics["collapse_warning"]:
                warnings.warn("Orion representation diagnostics indicate possible collapse", RuntimeWarning)
            check_files(args.root, entries)
            check_disk(args.outdir)
            final = args.outdir / "representation.pt"
            if not final.exists():
                save_torch(final, {
                    "format": "orion-stream-representation-v1", "plan_sha256": digest_json(plan),
                    "model_config": model.configuration(), "model_state": model.state_dict(),
                    "genes": genes, "ensembl_ids": plan["ensembl_ids"],
                    "context": args.context, "training_cells": expected_cells, "epochs": args.epochs,
                    "training_targets": training_targets, "heldout_targets": sorted(heldout),
                    "singleton_updates_without_covariance": singleton_updates,
                    "representation_diagnostics": diagnostics,
                    "note": "Representation only; the population-flow parameters are untrained.",
                })
            verify_final(final, plan, model.state_dict())
            save("complete")
            state.update(
                state="complete", stage="complete", finished_at_utc=now(),
                checkpoint=str(final), checkpoint_sha256=sha256(final),
                training_cells=expected_cells, representation_diagnostics=diagnostics,
            )
            write_json(state_path, state)
            print(f"Verified completed Orion representation: {final}", flush=True)
        except Exception as exc:
            state.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_at_utc=now())
            write_json(state_path, state)
            raise


if __name__ == "__main__":
    main()

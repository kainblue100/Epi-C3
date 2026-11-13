"""Resumable, compound-disjoint representation training on Tahoe train shards."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import signal
import warnings
from dataclasses import replace
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


HERE = Path(__file__).parent
SCANNER_PATH = HERE / "scan-tahoe-stream-v1.py"
HELPER_PATH = HERE / "stream-orion-representation-v1.py"
spec = importlib.util.spec_from_file_location("tahoe_scan_support", SCANNER_PATH)
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)
spec = importlib.util.spec_from_file_location("stream_training_support", HELPER_PATH)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
COLUMNS = [*scanner.COLUMNS, "genes", "expressions"]


def treatment_split(metadata: dict, drug_metadata: pd.DataFrame):
    structures = {}
    for row in drug_metadata.itertuples(index=False):
        drug = str(row.drug)
        if drug in structures:
            raise ValueError(f"Duplicate Tahoe drug metadata: {drug!r}")
        structures[drug] = scanner.normalized_smiles(row.canonical_smiles, drug)
    heldout = set(metadata["heldout_compounds"])
    groups = metadata["groups"]
    for group in groups.values():
        drug, structure = group["drug"], group["canonical_smiles"]
        if drug == scanner.TAHOE_CONTROL or structure is None:
            continue
        previous = structures.get(drug)
        if previous is not None and previous != structure:
            raise ValueError(f"Inconsistent source SMILES for Tahoe drug {drug!r}")
        structures[drug] = structure
    aliases = {drug for drug, structure in structures.items() if structure in heldout}
    if not heldout.issubset({structures[drug] for drug in aliases}):
        raise ValueError("A held-out Tahoe structure has no identifiable drug alias")
    if sum(group["cells"] for group in groups.values()) != metadata["rows"]:
        raise ValueError("Tahoe group counts do not account for every scanned row")
    excluded_unknown = sum(
        group["cells"] for group in groups.values()
        if group["drug"] != scanner.TAHOE_CONTROL and structures.get(group["drug"]) is None
    )
    excluded_heldout = sum(group["cells"] for group in groups.values() if group["drug"] in aliases)
    controls = metadata["control_counts"]
    expected = metadata["rows"] - excluded_unknown - excluded_heldout - controls["reference"] - controls["input_controls"]
    if expected < 2:
        raise ValueError("Not enough eligible Tahoe training rows")
    return structures, aliases, {
        "eligible_rows": expected, "excluded_unknown_structure_rows": excluded_unknown,
        "excluded_heldout_compound_rows": excluded_heldout,
        "excluded_reference_controls": controls["reference"],
        "excluded_inference_controls": controls["input_controls"],
    }


def select_training(batch: pa.RecordBatch, metadata: dict, structures: dict, aliases: set[str], seed: int):
    columns = {name: batch.column(name).to_pylist() for name in scanner.COLUMNS}
    rows = []
    for index, (drug, sample, plate, line, barcode, smiles) in enumerate(zip(
        *(columns[name] for name in scanner.COLUMNS), strict=True,
    )):
        if any(value is None or not str(value) for value in (drug, sample, plate, line, barcode)):
            raise ValueError("Tahoe cell has missing identity or treatment metadata")
        key = f"{sample}\t{line}"
        group = metadata["groups"].get(key)
        if group is None or (drug, plate) != (group["drug"], group["plate"]):
            raise ValueError(f"Tahoe source no longer agrees with the scanned group {key!r}")
        if drug == scanner.TAHOE_CONTROL:
            if scanner.control_role(str(sample), str(barcode), seed) == "training":
                rows.append(index)
            continue
        canonical = scanner.normalized_smiles(smiles, str(drug))
        if canonical != group["canonical_smiles"]:
            raise ValueError(f"Tahoe structure no longer agrees with the scanned group {key!r}")
        structure = structures.get(drug)
        if structure is not None and drug not in aliases:
            rows.append(index)
    return np.asarray(rows, dtype=np.int64)


def counts_from_batch(batch: pa.RecordBatch, selected: np.ndarray, token_map: np.ndarray, n_genes: int):
    if not len(selected):
        return sp.csr_matrix((0, n_genes), dtype=np.float32)
    picked = batch.take(pa.array(selected, type=pa.int64()))
    tokens, values = picked.column("genes"), picked.column("expressions")
    if (
        not pa.types.is_list(tokens.type) or not pa.types.is_list(values.type)
        or tokens.null_count or values.null_count
    ):
        raise ValueError("Tahoe tokens and counts must be complete list arrays")
    lengths = np.diff(tokens.offsets.to_numpy())
    if not np.array_equal(lengths, np.diff(values.offsets.to_numpy())) or np.any(lengths < 2):
        raise ValueError("Tahoe gene and expression arrays need matching nonempty lengths")
    flat_tokens, flat_values = tokens.flatten(), values.flatten()
    if flat_tokens.null_count or flat_values.null_count:
        raise ValueError("Tahoe token or count arrays contain null elements")
    ids = flat_tokens.to_numpy(zero_copy_only=False)
    expression = flat_values.to_numpy(zero_copy_only=False)
    starts = np.concatenate(([0], np.cumsum(lengths[:-1], dtype=np.int64)))
    if np.any(ids[starts] != 1) or np.any(expression[starts] != -2):
        raise ValueError("Tahoe rows require the CLS token 1 and expression -2")
    keep = np.ones(len(ids), dtype=bool)
    keep[starts] = False
    ids, expression = ids[keep], expression[keep]
    if (
        np.any(ids < 0) or np.any(ids >= len(token_map))
        or not np.isfinite(expression).all() or np.any(expression < 0)
        or not np.equal(expression, np.floor(expression)).all()
        or np.any(expression > np.iinfo(np.int32).max)
    ):
        raise ValueError("Invalid Tahoe gene tokens or raw integer counts")
    columns = token_map[ids]
    if np.any(columns < 0):
        raise ValueError("Tahoe gene token is absent from the source annotation")
    row_indices = np.repeat(np.arange(len(selected), dtype=np.int32), lengths - 1)
    matrix = sp.coo_matrix(
        (expression, (row_indices, columns)), shape=(len(selected), n_genes), dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    if (
        not np.isfinite(matrix.data).all() or np.any(matrix.data > np.iinfo(np.int32).max)
        or np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0)
    ):
        raise ValueError("Aggregated Tahoe counts are invalid or an eligible row has no counts")
    return matrix


def verify_final(path: Path, plan: dict, expected_state: dict):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format") != "tahoe-stream-representation-v1"
        or checkpoint.get("plan_sha256") != support.digest_json(plan)
        or checkpoint.get("training_rows") != plan["split"]["eligible_rows"]
        or checkpoint.get("genes") != plan["genes"]
        or set(checkpoint["model_state"]) != set(expected_state)
    ):
        raise ValueError("Tahoe representation checkpoint does not match the immutable plan")
    for name, value in checkpoint["model_state"].items():
        if not torch.isfinite(value).all() or not torch.equal(value, expected_state[name]):
            raise ValueError(f"Invalid or changed Tahoe model parameter: {name}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata-plan", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--variance-weight", type=float, default=5.0)
    parser.add_argument("--covariance-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-batches", type=int, default=0)
    args = parser.parse_args()
    if (
        args.epochs < 1 or args.batch_size < 2 or args.checkpoint_every < 1
        or args.stop_after_batches < 0 or args.latent_dim < 1 or args.hidden_dim < 1
        or not np.isfinite(args.learning_rate) or args.learning_rate <= 0
        or not np.isfinite(args.variance_weight) or args.variance_weight < 0
        or not np.isfinite(args.covariance_weight) or args.covariance_weight < 0
        or args.resume and args.check_only
    ):
        raise ValueError("Invalid Tahoe representation training settings")
    args.outdir.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and args.outdir.exists():
        raise FileExistsError("Refusing to overwrite an existing Tahoe training run")
    if args.resume and not args.outdir.is_dir():
        raise FileNotFoundError("Cannot resume a missing Tahoe training run")
    support.check_disk(args.outdir.parent, 3 * 1024**3)
    torch.set_num_threads(1)
    metadata = json.loads(args.metadata_plan.read_text())
    if metadata.get("state") != "complete" or metadata["config"]["seed"] != args.seed:
        raise ValueError("A completed Tahoe metadata plan with the same split seed is required")
    provenance = metadata["provenance"]
    if provenance["script_sha256"] != support.sha256(SCANNER_PATH):
        raise ValueError("The Tahoe split algorithm changed since metadata planning")
    for name in ("gene", "sample", "drug"):
        path = args.root / f"metadata/{name}_metadata.parquet"
        if provenance[f"{name}_metadata_sha256"] != support.sha256(path):
            raise ValueError(f"Tahoe {name} metadata changed since metadata planning")
    gene_metadata = args.root / "metadata/gene_metadata.parquet"
    genes, tokens, ensembls = gene_token_mapping(
        pd.read_parquet(gene_metadata), token_column="token_id", symbol_column="gene_symbol",
    )
    mapping = support.create_token_map(tokens)
    structures, aliases, split = treatment_split(
        metadata, pd.read_parquet(args.root / "metadata/drug_metadata.parquet"),
    )
    entries = provenance["files"]
    support.check_files(args.root, entries)
    config = {
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim,
        "variance_weight": args.variance_weight, "covariance_weight": args.covariance_weight,
        "seed": args.seed, "checkpoint_every": args.checkpoint_every,
    }
    plan = {
        "format": "tahoe-stream-representation-plan-v1", "config": config,
        "source_repo": provenance["source_repo"], "source_revision": provenance["revision"],
        "metadata_plan_sha256": support.sha256(args.metadata_plan),
        "script_sha256": support.sha256(Path(__file__)),
        "scanner_support_sha256": support.sha256(SCANNER_PATH),
        "training_support_sha256": support.sha256(HELPER_PATH),
        "gene_metadata_sha256": support.sha256(gene_metadata), "files": entries,
        "genes": genes, "ensembl_ids": [str(ensembls.loc[name]) for name in genes],
        "heldout_compounds": sorted(metadata["heldout_compounds"]),
        "heldout_drug_aliases": sorted(aliases), "split": split,
        "notes": [
            "Each epoch visits every eligible expression-shard row once.",
            "The split groups dataset-provided canonical SMILES strings after trimming outer whitespace.",
            "All known held-out drug aliases and their doses are excluded, including rows missing a structure.",
            "Unknown-structure drugs are excluded because compound-disjointness cannot be established.",
            "Cells without a structure are not used to fit drug conditions.",
            "Controls are independently split by stable sample and barcode before training.",
            "QC categories from obs_metadata are not joined and no additional QC filtering is claimed.",
            "A singleton epoch-end update omits the undefined covariance term.",
            "Representation only; the population-flow and drug-conditioning parameters are untrained.",
        ],
    }
    if args.check_only:
        batch = next(pq.ParquetFile(args.root / entries[0]["path"]).iter_batches(
            batch_size=128, columns=COLUMNS,
        ))
        rows = select_training(batch, metadata, structures, aliases, args.seed)
        counts_from_batch(batch, rows, mapping, len(genes))
        print(json.dumps({
            "shards": len(entries), "model_genes": len(genes), "split": split,
            "heldout_compounds": plan["heldout_compounds"], "probe_rows": len(rows),
        }, indent=2))
        return
    if not args.resume:
        args.outdir.mkdir()
    with (args.outdir / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path, state_path = args.outdir / "plan.json", args.outdir / "state.json"
        recovery_path = args.outdir / "recovery.pt"
        state = {"state": "running", "started_at_utc": support.now()}
        if args.resume:
            previous = json.loads(plan_path.read_text())
            if previous != plan:
                raise ValueError("Tahoe source, code or training settings changed since checkpoint")
            snapshot = torch.load(recovery_path, map_location="cpu", weights_only=True)
            if snapshot.get("plan_sha256") != support.digest_json(plan):
                raise ValueError("Tahoe recovery checkpoint does not match its plan")
            if snapshot.get("phase") == "complete":
                final = args.outdir / "representation.pt"
                done = verify_final(final, plan, snapshot["model_state"])
                state.update(
                    state="complete", stage="complete", finished_at_utc=support.now(),
                    checkpoint=str(final), checkpoint_sha256=support.sha256(final),
                    training_rows=done["training_rows"], representation_diagnostics=done["representation_diagnostics"],
                )
                support.write_json(state_path, state)
                return
        else:
            support.write_json(plan_path, plan)
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
            support.check_disk(args.outdir, 3 * 1024**3)
            support.save_torch(recovery_path, {
                "format": "tahoe-stream-recovery-v1", "plan_sha256": support.digest_json(plan),
                "phase": phase, "epoch": epoch, "file_position": file_position,
                "batch_position": batch_position, "updates": updates, "trained": trained,
                "singleton_updates": singleton_updates, "pending": pending,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            })
            state.update(
                state="running", stage=phase, epoch=epoch, file_position=file_position,
                files=len(entries), next_batch=batch_position, updates=updates, trained_rows=trained,
                expected_rows_per_epoch=split["eligible_rows"],
                pending_rows=0 if pending is None else len(pending), singleton_updates=singleton_updates,
                updated_at_utc=support.now(),
            )
            support.write_json(state_path, state)

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
                raise RuntimeError(f"Non-finite Tahoe representation loss at update {updates}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.update_teacher()
            updates += 1
            trained += len(counts)
            if updates == 1 or updates % 100 == 0:
                print(json.dumps({
                    "phase": "representation", "epoch": epoch, "file": file_position,
                    "files": len(entries), "updates": updates, "trained_rows": trained, **metrics,
                }), flush=True)

        try:
            if snapshot is None:
                save("representation")
            model.train()
            invocation_updates = 0
            while epoch < args.epochs:
                while file_position < len(entries):
                    item = entries[file_position]
                    support.check_files(args.root, [item])
                    for index, batch in enumerate(pq.ParquetFile(args.root / item["path"]).iter_batches(
                        batch_size=args.batch_size, columns=COLUMNS,
                    )):
                        if index < batch_position:
                            continue
                        rows = select_training(batch, metadata, structures, aliases, args.seed)
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
                            state.update(state="paused", updated_at_utc=support.now())
                            support.write_json(state_path, state)
                            return
                    file_position += 1
                    batch_position = 0
                if pending is not None:
                    update_model(pending)
                    pending = None
                if trained != split["eligible_rows"] * (epoch + 1):
                    raise ValueError(f"Eligible Tahoe row count mismatch: {trained}")
                epoch += 1
                file_position = batch_position = 0
                save("representation")
            pieces, count = [], 0
            for item in entries:
                for batch in pq.ParquetFile(args.root / item["path"]).iter_batches(batch_size=128, columns=COLUMNS):
                    rows = select_training(batch, metadata, structures, aliases, args.seed)[:512 - count]
                    if len(rows):
                        pieces.append(counts_from_batch(batch, rows, mapping, len(genes)))
                        count += len(rows)
                    if count >= 512:
                        break
                if count >= 512:
                    break
            if count < 2:
                raise ValueError("Insufficient Tahoe rows for representation diagnostics")
            diagnostics = representation_diagnostics(
                model, sp.vstack(pieces, format="csr"), device=torch.device("cpu"),
                batch_size=args.batch_size, seed=args.seed + 8, max_cells=count,
            )
            diagnostics["sampling"] = "first 512 eligible training rows in sorted source shards"
            if diagnostics["collapse_warning"]:
                warnings.warn("Tahoe representation diagnostics indicate possible collapse", RuntimeWarning)
            support.check_files(args.root, entries)
            support.check_disk(args.outdir, 3 * 1024**3)
            final = args.outdir / "representation.pt"
            if not final.exists():
                support.save_torch(final, {
                    "format": "tahoe-stream-representation-v1", "plan_sha256": support.digest_json(plan),
                    "model_config": model.configuration(), "model_state": model.state_dict(),
                    "genes": genes, "ensembl_ids": plan["ensembl_ids"],
                    "training_rows": split["eligible_rows"], "epochs": args.epochs,
                    "heldout_compounds": plan["heldout_compounds"],
                    "heldout_drug_aliases": plan["heldout_drug_aliases"],
                    "singleton_updates_without_covariance": singleton_updates,
                    "representation_diagnostics": diagnostics,
                    "note": "Representation only; population-flow and drug-conditioning parameters are untrained.",
                })
            verify_final(final, plan, model.state_dict())
            save("complete")
            state.update(
                state="complete", stage="complete", finished_at_utc=support.now(),
                checkpoint=str(final), checkpoint_sha256=support.sha256(final),
                training_rows=split["eligible_rows"], representation_diagnostics=diagnostics,
            )
            support.write_json(state_path, state)
            print(f"Verified completed Tahoe representation: {final}", flush=True)
        except Exception as exc:
            state.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_at_utc=support.now())
            support.write_json(state_path, state)
            raise


if __name__ == "__main__":
    main()

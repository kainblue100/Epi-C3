"""Verify protected control streams and a read-only foundation compatibility smoke."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch

import c3vcell.model as model_module
from c3vcell.model import C3VCell, ModelConfig
import epi_c3_scperturbbench_foundation_v1 as bench
from epi_c3_foundation_stage1_sources_v1 import digest_json
from epi_c3_shared_foundation_v1 import file_digest
from khp_transfer_common_v1 import read_plan


SNAPSHOT_SHA256 = "f28885d6a7cb5d424c9588c80ebff2d9a1a87f1917a8aebc8b0d2b451577c34c"


def decode(group: h5py.Group, name: str) -> np.ndarray:
    node = group[name]
    return node["categories"].asstr()[:][node["codes"][:]]


def independent_rows(node: h5py.Group, rows: list[int], mapping: np.ndarray, genes: int) -> np.ndarray:
    pointer = node["indptr"][:]
    encoding = node.attrs["encoding-type"]
    output = np.zeros((len(rows), genes), dtype=np.float32)
    if encoding == "csr_matrix":
        for index, row in enumerate(rows):
            start, end = int(pointer[row]), int(pointer[row + 1])
            columns, values = node["indices"][start:end], node["data"][start:end]
            mapped = mapping[columns]
            keep = mapped >= 0
            output[index, mapped[keep]] = values[keep]
    elif encoding == "csc_matrix":
        indices = node["indices"][:]
        data = node["data"][:]
        selected = np.asarray(rows)
        for column, target in enumerate(mapping):
            if target < 0:
                continue
            start, end = int(pointer[column]), int(pointer[column + 1])
            column_rows = indices[start:end]
            locations = np.searchsorted(column_rows, selected)
            valid = locations < len(column_rows)
            present = np.zeros(len(rows), dtype=bool)
            present[valid] = column_rows[locations[valid]] == selected[valid]
            output[present, target] = data[start + locations[present]]
    else:
        raise ValueError("Unexpected count encoding")
    return output


def verify(root: Path, run: Path, audit_dir: Path, snapshot: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    public = audit_dir / "audit.json"
    private_path = audit_dir / "sources.private.json"
    report = json.loads(public.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    if file_digest(private_path) != report["private_sha256"]:
        raise ValueError("The private source manifest changed")
    plan_path = run / "rna-foundation-stage1-v1/plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if file_digest(plan_path) != report["foundation_plan_sha256"]:
        raise ValueError("The foundation gene-axis plan changed")
    khp_path = run / "khp-cd8-transfer-v1/plan.json"
    if file_digest(khp_path) != report["khp_evaluation_plan_sha256"]:
        raise ValueError("The frozen KHP evaluation split changed")
    selection = read_plan(khp_path)["selection"]
    protected = set(selection["target_input_control"]) | set(selection["target_reference_control"])
    for rows in selection["target_reference_treatments"].values():
        protected.update(rows)
    checkpoint_path = snapshot / "recovery.pt"
    if file_digest(checkpoint_path) != SNAPSHOT_SHA256:
        raise ValueError("The frozen early foundation checkpoint changed")
    if file_digest(Path(model_module.__file__)) != plan["provenance"]["source_code_sha256"]["model.py"]:
        raise ValueError("The foundation model implementation changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["plan_sha256"] != digest_json(plan):
        raise ValueError("The early checkpoint does not match the foundation plan")
    torch.set_num_threads(1)
    torch.manual_seed(20261117)
    model = C3VCell(ModelConfig(**plan["model_config"]))
    model.load_state_dict(checkpoint["trainer"]["model_state"], strict=True)
    model.eval()
    sources = {}
    source_root = root / "benchmarks/scPerturbBench-zenodo-20823660"
    for name in ("KHP", "CMO"):
        source_plan = private["sources"][name]
        path = source_root / report["sources"][name]["file"]
        rows = np.asarray(source_plan["training_rows"], dtype=np.int64)
        validation = np.asarray(source_plan["validation_rows"], dtype=np.int64)
        selected = np.concatenate((rows, validation))
        with h5py.File(path, "r") as file:
            conditions = decode(file["obs"], "condition")
            types = decode(file["obs"], "cell_type")
            if np.any(conditions[selected] != "ctrl"):
                raise ValueError("A candidate benchmark source includes treated cells")
            if name == "KHP":
                if np.any(types[selected] == "CD8T") or set(selected.tolist()).intersection(protected):
                    raise ValueError("KHP candidate controls overlap target evaluation rows")
            else:
                donors = decode(file["obs"], "donor_id")
                withheld = set(donors.tolist()) - set(donors[selected].tolist())
                if len(withheld) != 1 or np.any(np.isin(donors[selected], list(withheld))):
                    raise ValueError("CMO candidate controls do not hold out one donor")
            chosen = sorted(set((rows[0], rows[len(rows) // 2], rows[-1], validation[0], validation[-1])))
            direct = independent_rows(
                file["layers/counts"], chosen, np.asarray(source_plan["mapping"], dtype=np.int64), len(plan["genes"]),
            )
        stream = bench.BenchmarkControlStream(path, public, private_path, name, plan["genes"])
        try:
            projected = stream._read(np.asarray(chosen, dtype=np.int64)).toarray()
            if not np.array_equal(projected, direct):
                raise ValueError("Stream projection differs from independently reconstructed source rows")
            first = stream.next_batch()
            first_copy = first.copy()
            stream.set_cursor({"position": 0})
            if (stream.next_batch() != first_copy).nnz:
                raise ValueError("A source cursor did not reproduce its first batch")
            seen, batches, total = first.shape[0], 1, float(first.sum())
            while True:
                batch = stream.next_batch()
                if batch is None:
                    break
                if np.any(~stream.measured[batch.indices]):
                    raise ValueError("A streamed value lies outside the measured-gene mask")
                seen += batch.shape[0]
                batches += 1
                total += float(batch.sum())
            if seen != len(rows) or not math.isfinite(total) or total <= 0:
                raise ValueError("The candidate stream did not exhaust its frozen rows")
            counts = torch.from_numpy(stream.validation.toarray())
            measured = torch.from_numpy(stream.measured).expand_as(counts).float()
            with torch.no_grad():
                loss, metrics = model.representation_loss(counts, measured)
            if not torch.isfinite(loss) or any(not math.isfinite(value) for value in metrics.values()):
                raise ValueError("The foundation produced nonfinite values for a candidate source")
            sources[name] = {
                "training_controls_streamed": seen,
                "validation_controls": len(validation),
                "batches_streamed": batches,
                "mapped_training_count_sum": total,
                "directly_reconstructed_rows": len(chosen),
                "measured_gene_count": int(stream.measured.sum()),
                "cursor_reproduced_first_batch": True,
                "readonly_representation_loss_finite": True,
            }
            print(f"{name}: verified {seen} candidate training controls and {len(validation)} validation controls", flush=True)
        finally:
            stream.close()
    output.mkdir(mode=0o700)
    result = {
        "format": "epi-c3-scperturbbench-foundation-verification-v1",
        "state": "complete",
        "source_audit_sha256": file_digest(public),
        "private_sha256": file_digest(private_path),
        "foundation_snapshot_sha256": SNAPSHOT_SHA256,
        "verification_script_sha256": file_digest(Path(__file__)),
        "sources": sources,
        "training_updates": 0,
        "limitations": [
            "The same frozen mapping and row plan are used; five source rows per dataset were independently reconstructed.",
            "A finite read-only loss is an input-compatibility check, not an estimate of prediction accuracy.",
            "Within-source protected-row checks do not prove cross-dataset identity or donor independence.",
        ],
    }
    with (output / "verification.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps({"state": result["state"], "output": str(output), "training_updates": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.run_root, args.audit, args.snapshot, args.output)

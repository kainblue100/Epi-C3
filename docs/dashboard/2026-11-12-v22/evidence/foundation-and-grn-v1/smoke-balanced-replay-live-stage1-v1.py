"""Verify replay cursors against the frozen, real stage-one source plan."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch


def load_driver(path: Path):
    spec = importlib.util.spec_from_file_location("live_stage1_driver", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the frozen stage-one driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_batch(matrix: sp.csr_matrix, measured: np.ndarray, genes: int) -> None:
    if (
        not sp.isspmatrix_csr(matrix) or matrix.shape[1] != genes or matrix.shape[0] < 1
        or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
        or np.any(matrix.data != np.floor(matrix.data)) or np.any(matrix.data > 2**24)
        or np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0)
        or np.any(~measured[matrix.indices])
    ):
        raise ValueError("A real source returned an invalid measured-count batch")


def equal(first: sp.csr_matrix, second: sp.csr_matrix) -> bool:
    return first.shape == second.shape and (first != second).nnz == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing real-source smoke report")
    sys.path.insert(0, str(args.run_root))
    stage1 = load_driver(args.run_root / "train-rna-foundation-stage1-v1.py")
    plan_path = args.run_root / "rna-foundation-stage1-v1/plan.json"
    frozen = json.loads(plan_path.read_text(encoding="utf-8"))
    if frozen.get("format") != "epi-c3-shared-rna-stage1-plan-v1":
        raise ValueError("Unexpected frozen stage-one plan")
    if stage1.ROOT.resolve() != args.data_root.resolve() or stage1.RUNS.resolve() != args.run_root.resolve():
        raise ValueError("The live stage-one driver is bound to different source or run roots")
    torch.set_num_threads(1)
    alignment, streams, heldout, provenance = stage1.build_sources()
    try:
        if (
            provenance != frozen["provenance"]
            or stage1.gene_axis_digest(alignment.genes) != frozen["gene_axis_sha256"]
            or stage1.stage_sources.digest_json(sorted(heldout)) != frozen["heldout_targets_sha256"]
            or set(streams) != set(frozen["sources"])
        ):
            raise ValueError("The live stage-one sources differ from the frozen plan")
        result = {}
        for name, stream in sorted(streams.items()):
            expected = frozen["sources"][name]
            if (
                stream.expected_cells != expected["expected_training_cells"]
                or int(stream.measured.sum()) != expected["measured_genes"]
                or stream.notes != expected["details"]
            ):
                raise ValueError(f"Frozen source details changed for {name}")
            initial = stream.cursor()
            first = stream.next_batch()
            if first is None:
                raise ValueError(f"Source {name} has no initial training batch")
            check_batch(first, stream.measured, len(alignment.genes))
            after_first = stream.cursor()
            stream.set_cursor(initial)
            replayed_first = stream.next_batch()
            if replayed_first is None or not equal(first, replayed_first):
                raise ValueError(f"Initial replay changed the {name} batch")
            stream.set_cursor(after_first)
            second = stream.next_batch()
            if second is None:
                raise ValueError(f"Source {name} has no second training batch")
            check_batch(second, stream.measured, len(alignment.genes))
            stream.set_cursor(after_first)
            replayed_second = stream.next_batch()
            if replayed_second is None or not equal(second, replayed_second):
                raise ValueError(f"Checkpoint replay changed the {name} batch")
            result[name] = {
                "first_batch_rows": int(first.shape[0]),
                "second_batch_rows": int(second.shape[0]),
                "measured_genes": int(stream.measured.sum()),
                "initial_and_checkpoint_replay_exact": True,
            }
    finally:
        for stream in streams.values():
            stream.close()
    report = {
        "format": "epi-c3-balanced-real-source-smoke-v1",
        "state": "complete",
        "stage1_plan_sha256": stage1.stage_sources.digest_json(frozen),
        "gene_axis_sha256": frozen["gene_axis_sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": result,
        "limitations": [
            "Only two batches per stage-one source were compared; this is not a full replay epoch.",
            "This is a read-only source smoke test, not balanced model training or a quality evaluation.",
            "Tahoe is covered separately by its frozen isolation audit and real-stream replay smoke.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

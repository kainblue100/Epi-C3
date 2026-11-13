"""Exercise the actual stage-two source construction without optimizer updates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from epi_c3_shared_foundation_v1 import file_digest


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_stream(stream) -> dict:
    initial = stream.cursor()
    first = stream.next_batch()
    if (
        not sp.isspmatrix_csr(first)
        or first.shape[0] < 1
        or not np.isfinite(first.data).all()
        or np.any(first.data < 0)
        or np.any(~stream.measured[first.indices])
    ):
        raise ValueError("A stage-two source produced an invalid first batch")
    stream.set_cursor(initial)
    repeated = stream.next_batch()
    if first.shape != repeated.shape or (first != repeated).nnz:
        raise ValueError("A stage-two source did not reproduce its first batch")
    stream.set_cursor(initial)
    return {
        "expected_training_rows": stream.expected_cells,
        "first_batch_rows": first.shape[0],
        "measured_genes": int(stream.measured.sum()),
        "cursor_replay_exact": True,
    }


def run(root: Path, run_root: Path, release: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    driver = release / "train-rna-foundation-stage2-v1.py"
    stage2 = load_script(driver, "stage2_live_preflight")
    sys.path.insert(0, str(run_root))
    stage1_path = run_root / "train-rna-foundation-stage1-v1.py"
    stage1 = stage2.load_script(stage1_path, "stage1_live_preflight")
    plan_path = run_root / "rna-foundation-stage1-v1/plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(data_root=root, run_root=run_root)
    replay, fresh, audit_sha = stage2.build_streams(args, stage1, plan)
    try:
        sources = {name: check_stream(stream) for name, stream in sorted(replay.items())}
        sources["tahoe"] = check_stream(fresh)
    finally:
        fresh.close()
        for stream in replay.values():
            stream.close()
    output.mkdir(mode=0o700)
    report = {
        "format": "epi-c3-shared-stage2-live-preflight-v1",
        "state": "complete",
        "stage1_plan_sha256": stage2.canonical_digest(plan),
        "stage2_driver_sha256": file_digest(driver),
        "stage1_driver_sha256": file_digest(stage1_path),
        "tahoe_isolation_audit_sha256": audit_sha,
        "preflight_script_sha256": file_digest(Path(__file__)),
        "sources": sources,
        "optimizer_updates": 0,
        "limitations": [
            "The stage-one job is still running; this is not a completed-stage-one or stage-two training test.",
            "Only the first batch and cursor replay were inspected for each source.",
            "The Tahoe audit protects inventoried identities, not every possible benchmark or cross-source duplicate.",
        ],
    }
    with (output / "preflight.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps({"state": report["state"], "sources": sources, "optimizer_updates": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.run_root, args.release, args.output)

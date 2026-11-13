"""Safely queue stage two until the original shared RNA stage one completes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


RESERVE_WITH_HEADROOM = 204 * 1024**3


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_status(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def stage1_ready(directory: Path) -> bool:
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") not in {"running", "stage1-complete"}:
        raise RuntimeError(f"Stage one stopped or has an unexpected status: {state.get('status')!r}")
    lock_path = directory / ".lock"
    if not lock_path.is_file():
        raise FileNotFoundError("The stage-one trainer lock is missing")
    with lock_path.open("rb") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    latest = json.loads(state_path.read_text(encoding="utf-8"))
    if latest.get("status") == "stage1-complete":
        if not (directory / "foundation-stage1.pt").is_file():
            raise FileNotFoundError("Stage one claims completion but its checkpoint is missing")
        return True
    raise RuntimeError("Stage one reports running but no trainer holds its lock")


def verify_preflight(preflight: Path, run: Path, release: Path) -> dict:
    report = json.loads(preflight.read_text(encoding="utf-8"))
    state = json.loads((run / "rna-foundation-stage1-v1/state.json").read_text(encoding="utf-8"))
    if (
        report.get("format") != "epi-c3-shared-stage2-live-preflight-v1"
        or report.get("state") != "complete"
        or report.get("optimizer_updates") != 0
        or report.get("stage1_plan_sha256") != state.get("plan_sha256")
        or report.get("stage2_driver_sha256") != digest(release / "train-rna-foundation-stage2-v1.py")
        or report.get("stage1_driver_sha256") != digest(run / "train-rna-foundation-stage1-v1.py")
        or report.get("tahoe_isolation_audit_sha256") != digest(run / "tahoe-foundation-isolation-v1/audit.json")
        or set(report.get("sources", {})) != {"arc", "tabula", "orion-HCT116", "orion-HEK293T", "tahoe"}
        or any(
            item.get("cursor_replay_exact") is not True
            or not isinstance(item.get("expected_training_rows"), int)
            or item["expected_training_rows"] < 1
            or not isinstance(item.get("first_batch_rows"), int)
            or item["first_batch_rows"] < 1
            or not isinstance(item.get("measured_genes"), int)
            or item["measured_genes"] < 1
            for item in report["sources"].values()
        )
    ):
        raise ValueError("The verified stage-two source preflight no longer matches its inputs")
    return report


def guard_files(run: Path, release: Path, preflight: Path) -> dict[Path, str]:
    files = [
        Path(__file__), preflight,
        run / "rna-foundation-stage1-v1/plan.json",
        run / "train-rna-foundation-stage1-v1.py",
        run / "epi_c3_foundation_tahoe_v2.py",
        run / "epi_c3_tahoe_isolation_v1.py",
        run / "audit-tahoe-foundation-isolation-v1.py",
        run / "scan-tahoe-stream-v1.py",
        run / "stream-tahoe-representation-v1.py",
        run / "stream-orion-representation-v1.py",
        run / "stream-arc-training-v1.py",
        run / "tahoe-foundation-isolation-v1/audit.json",
        release / "train-rna-foundation-stage2-v1.py",
        release / "epi_c3_foundation_balanced_v1.py",
        release / "epi_c3_foundation_sources_v1.py",
        release / "epi_c3_foundation_stage1_sources_v1.py",
        release / "epi_c3_foundation_training_v1.py",
        release / "epi_c3_shared_foundation_v1.py",
        release / "src/c3vcell/model.py",
    ]
    return {path: digest(path) for path in files}


def dispatch(root: Path, run: Path, release: Path, preflight: Path, output: Path, status: Path, interval: int) -> None:
    if interval < 1:
        raise ValueError("The waiting interval must be positive")
    if status.exists() or output.exists():
        raise FileExistsError("Refusing to overwrite an existing dispatch status or stage-two output")
    report = verify_preflight(preflight, run, release)
    hashes = guard_files(run, release, preflight)
    lock_path = status.with_name(status.name + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.exists() or output.exists():
            raise FileExistsError("Another stage-two dispatch has already started")
        state = {
            "state": "running", "stage": "waiting-for-stage1", "started_at_utc": now(),
            "stage1_plan_sha256": report["stage1_plan_sha256"], "preflight_sha256": digest(preflight),
            "guarded_files_sha256": {str(path): value for path, value in hashes.items()},
            "note": "The original stage-one job is never modified. The stage-two driver performs its own completion and provenance checks.",
        }
        save_status(status, state)
        try:
            while True:
                if shutil.disk_usage(status.parent).free < RESERVE_WITH_HEADROOM:
                    raise OSError("Insufficient free space above the 200 GiB reserve and 4 GiB headroom")
                for path, expected in hashes.items():
                    if digest(path) != expected:
                        raise ValueError(f"A guarded continuation input changed: {path.name}")
                if stage1_ready(run / "rna-foundation-stage1-v1"):
                    break
                time.sleep(interval)
            if output.exists():
                raise FileExistsError("The stage-two output appeared while the dispatcher was waiting")
            verify_preflight(preflight, run, release)
            state.update(stage="stage2-training", updated_at_utc=now())
            save_status(status, state)
            subprocess.run([
                sys.executable, str(release / "train-rna-foundation-stage2-v1.py"),
                "--data-root", str(root), "--run-root", str(run), "--output", str(output),
                "--checkpoint-every", "250",
            ], check=True)
            result = json.loads((output / "state.json").read_text(encoding="utf-8"))
            if result.get("status") != "stage2-complete":
                raise RuntimeError("Stage two exited without a completed checkpoint")
            checkpoint = output / "foundation-stage2.pt"
            state.update(
                state="complete", stage="complete", checkpoint_sha256=digest(checkpoint),
                finished_at_utc=now(),
            )
            save_status(status, state)
        except Exception as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}", finished_at_utc=now())
            save_status(status, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    dispatch(args.root, args.run_root, args.release, args.preflight, args.output, args.status, args.interval)

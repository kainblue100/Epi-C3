"""Continue a completed stage-one RNA model with audited, balanced Tahoe replay."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys

import pandas as pd
import torch

from c3vcell.model import C3VCell, ModelConfig
from epi_c3_foundation_balanced_v1 import BalancedContinualTrainer
from epi_c3_shared_foundation_v1 import FORMAT, file_digest, gene_axis_digest


def canonical_digest(value: dict) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_stage1(run_root: Path) -> tuple[dict, dict, dict, str, str]:
    directory = run_root / "rna-foundation-stage1-v1"
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if state.get("status") != "stage1-complete":
        raise ValueError("The shared RNA stage-one run has not completed")
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if plan.get("format") != "epi-c3-shared-rna-stage1-plan-v1":
        raise ValueError("The frozen stage-one plan has an unexpected format")
    plan_sha = canonical_digest(plan)
    genes = plan["genes"]
    expected = {name: item["expected_training_cells"] for name, item in plan["sources"].items()}
    if (
        state.get("plan_sha256") != plan_sha
        or state.get("expected_cells") != expected or state.get("trained_cells") != expected
        or gene_axis_digest(genes) != plan["gene_axis_sha256"]
        or plan["model_config"]["n_genes"] != len(genes)
    ):
        raise ValueError("The completed stage-one state does not match its frozen plan")
    diagnostics = state.get("diagnostics_after")
    if not isinstance(diagnostics, dict) or set(diagnostics) != set(expected):
        raise ValueError("Completed stage-one diagnostics are missing")
    for name, result in diagnostics.items():
        if (
            not isinstance(result, dict)
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in result.values())
            or result.get("latent_dimensions_std_gt_0_1", 0) < 1
        ):
            raise ValueError(f"Stage-one representation diagnostics are invalid or collapsed for {name}")
    path = directory / "foundation-stage1.pt"
    before = path.stat()
    checkpoint_sha = file_digest(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("The stage-one checkpoint changed while it was being verified")
    if (
        checkpoint.get("format") != FORMAT or checkpoint.get("status") != "complete"
        or checkpoint.get("scope") != "stage1-incomplete"
        or checkpoint.get("genes") != genes
        or checkpoint.get("gene_axis_sha256") != plan["gene_axis_sha256"]
        or checkpoint.get("model_config") != plan["model_config"]
        or checkpoint.get("training", {}).get("plan_sha256") != plan_sha
        or checkpoint["training"].get("cells") != expected
        or checkpoint["training"].get("updates") != state.get("updates")
        or any(not torch.isfinite(value).all() for value in checkpoint["model_state"].values())
    ):
        raise ValueError("The stage-one checkpoint does not match its completed state")
    return plan, state, checkpoint, plan_sha, checkpoint_sha


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load required source helper {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_streams(args, stage1, plan):
    if hasattr(stage1, "ROOT"):
        if stage1.ROOT.resolve() != args.data_root.resolve() or stage1.RUNS.resolve() != args.run_root.resolve():
            raise ValueError("The stage-one driver points to different input roots")
        alignment, streams, heldout, provenance = stage1.build_sources()
    else:
        alignment, streams, heldout, provenance = stage1.build_sources(args.data_root)
    fresh = None
    try:
        if (
            provenance != plan["provenance"]
            or gene_axis_digest(alignment.genes) != plan["gene_axis_sha256"]
            or stage1.stage_sources.digest_json(sorted(heldout)) != plan["heldout_targets_sha256"]
            or set(streams) != set(plan["sources"])
        ):
            raise ValueError("The replay sources no longer match the frozen stage-one plan")
        for name, stream in streams.items():
            expected = plan["sources"][name]
            if (
                stream.expected_cells != expected["expected_training_cells"]
                or int(stream.measured.sum()) != expected["measured_genes"]
                or stream.notes != expected["details"]
            ):
                raise ValueError(f"The frozen replay source changed: {name}")
        import epi_c3_foundation_tahoe_v2 as adapter

        audit_path = args.run_root / "tahoe-foundation-isolation-v1/audit.json"
        audit_sha = file_digest(audit_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        helper = load_script(args.run_root / "audit-tahoe-foundation-isolation-v1.py", "stage2_tahoe_audit")
        tahoe = helper.load_tahoe(args.run_root)
        tahoe_root = args.data_root / "Tahoe-100M"
        fresh = adapter.TahoeStream(
            tahoe_root, args.run_root, args.run_root / "tahoe-full-metadata-v2/metadata-plan.json",
            args.run_root / "tahoe-protected-identities-v1", audit_path.parent,
            pd.read_parquet(tahoe_root / "metadata/gene_metadata.parquet"), plan["genes"],
            tahoe.support, tahoe, batch_size=audit["batch_size"], seed=audit["order_seed"],
        )
        if file_digest(audit_path) != audit_sha:
            raise ValueError("The Tahoe isolation audit changed while constructing the stream")
        return streams, fresh, audit_sha
    except Exception:
        if fresh is not None:
            fresh.close()
        for stream in streams.values():
            stream.close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--stop-after-updates", type=int, default=0)
    args = parser.parse_args()
    if args.checkpoint_every < 1 or args.stop_after_updates < 0:
        raise ValueError("Checkpoint frequency must be positive and the update limit nonnegative")
    plan1, state1, initial, plan1_sha, initial_sha = load_stage1(args.run_root)
    sys.path.insert(0, str(args.run_root))
    stage1 = load_script(args.run_root / "train-rna-foundation-stage1-v1.py", "stage2_source_driver")
    torch.set_num_threads(1)
    torch.manual_seed(stage1.SEED + 1)
    streams, fresh, audit_sha = build_streams(args, stage1, plan1)
    try:
        if file_digest(args.run_root / "rna-foundation-stage1-v1/foundation-stage1.pt") != initial_sha:
            raise ValueError("The verified stage-one checkpoint changed before continuation")
        model = C3VCell(ModelConfig(**initial["model_config"]))
        model.load_state_dict(initial["model_state"], strict=True)
        source_hashes = {name: plan1_sha for name in streams}
        source_hashes["fresh"] = audit_sha
        trainer = BalancedContinualTrainer(
            model, fresh, streams, learning_rate=plan1["learning_rate"],
            source_manifest_sha256=source_hashes, gene_axis_sha256=plan1["gene_axis_sha256"],
            initial_checkpoint_sha256=initial_sha,
        )
        plan = {
            "format": "epi-c3-shared-rna-stage2-plan-v1",
            "scope": "stage2-incomplete-not-a-full-foundation",
            "stage1_plan_sha256": plan1_sha,
            "stage1_checkpoint_sha256": initial_sha,
            "stage1_diagnostics_after": state1["diagnostics_after"],
            "tahoe_isolation_audit_sha256": audit_sha,
            "continuation_plan": trainer.plan,
            "continuation_plan_sha256": trainer.plan_sha256,
            "driver_sha256": file_digest(Path(__file__)),
            "trainer_sha256": file_digest(Path(sys.modules[trainer.__module__].__file__)),
            "sampling": "one pass over audited Tahoe rows with approximately equal prior-source replay exposure",
            "limitations": [
                "Replay exposures are repeated rows, not new unique cells.",
                "Row balance is not equal gradient or optimizer-step balance when source batch sizes differ.",
                "Other downloaded expression datasets and independent quality evaluations remain incomplete.",
                "The continued checkpoint must not be presented as a completed full foundation model.",
            ],
        }
        plan_sha = canonical_digest(plan)
        args.output.mkdir(parents=True, exist_ok=True)
        stage1.check_space(args.output)
        with (args.output / ".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan_path, recovery_path = args.output / "plan.json", args.output / "recovery.pt"
            if plan_path.exists():
                if canonical_digest(json.loads(plan_path.read_text(encoding="utf-8"))) != plan_sha:
                    raise ValueError("The current continuation inputs differ from the frozen plan")
                saved = torch.load(recovery_path, map_location="cpu", weights_only=True)
                if saved["plan_sha256"] != plan_sha:
                    raise ValueError("The recovery checkpoint and continuation plan disagree")
                trainer.restore(saved["trainer"])
                before = saved["diagnostics_before"]
            else:
                if any(path.name != ".lock" for path in args.output.iterdir()):
                    raise FileExistsError("Refusing to overwrite an existing output directory")
                before = stage1.diagnostics(model, {**streams, "tahoe": fresh})
                stage1.atomic_json(plan_path, plan)
                stage1.atomic_torch(recovery_path, {
                    "plan_sha256": plan_sha, "trainer": trainer.snapshot(), "diagnostics_before": before,
                })
            stop = {"requested": False}

            def request_stop(signum, frame):
                stop["requested"] = True

            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)

            def save_state(status: str, final_diagnostics=None):
                stage1.check_space(args.output)
                stage1.atomic_torch(recovery_path, {
                    "plan_sha256": plan_sha, "trainer": trainer.snapshot(), "diagnostics_before": before,
                })
                state = {
                    "status": status, "updated_at_utc": stage1.now(), "plan_sha256": plan_sha,
                    "scope": plan["scope"], "fresh_rows": trainer.fresh_rows,
                    "expected_fresh_rows": fresh.expected_cells, "replay_rows": trainer.replay_rows,
                    "replay_epochs": trainer.replay_epochs, "updates": trainer.updates,
                    "source_updates": trainer.source_updates,
                    "singleton_updates": trainer.singleton_updates,
                    "diagnostics_before": before, "diagnostics_after": final_diagnostics,
                }
                stage1.atomic_json(args.output / "state.json", state)
                print(json.dumps({key: state[key] for key in ("status", "updates", "fresh_rows", "replay_rows")}),
                      flush=True)

            updates_here = 0
            while not stop["requested"]:
                if trainer.step() is None:
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
            after = stage1.diagnostics(model, {**streams, "tahoe": fresh})
            final = {
                "format": FORMAT, "status": "complete", "scope": "stage2-incomplete",
                "genes": plan1["genes"], "gene_axis_sha256": plan1["gene_axis_sha256"],
                "model_config": model.configuration(), "model_state": model.state_dict(),
                "training": {
                    "plan_sha256": plan_sha, "fresh_rows": trainer.fresh_rows,
                    "replay_rows": trainer.replay_rows, "replay_epochs": trainer.replay_epochs,
                    "source_updates": trainer.source_updates, "updates": trainer.updates,
                },
            }
            final_path = args.output / "foundation-stage2.pt"
            stage1.atomic_torch(final_path, final)
            restored = torch.load(final_path, map_location="cpu", weights_only=True)
            if any(not torch.equal(restored["model_state"][key], value) for key, value in model.state_dict().items()):
                raise ValueError("Stage-two checkpoint verification failed")
            save_state("stage2-complete", after)
    finally:
        fresh.close()
        for stream in streams.values():
            stream.close()


if __name__ == "__main__":
    main()

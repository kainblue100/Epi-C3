import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


path = Path(__file__).with_name("wait-rna-foundation-stage2-v1.py")
if not path.exists():
    path = Path(__file__).parent.parent / path.name
spec = importlib.util.spec_from_file_location("wait_stage2", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StageOneReadinessTests(unittest.TestCase):
    def test_preflight_must_match_live_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            run = directory / "run"
            release = directory / "release"
            (run / "rna-foundation-stage1-v1").mkdir(parents=True)
            (run / "tahoe-foundation-isolation-v1").mkdir()
            release.mkdir()
            (run / "rna-foundation-stage1-v1/state.json").write_text(
                json.dumps({"plan_sha256": "plan"}), encoding="utf-8"
            )
            stage1 = run / "train-rna-foundation-stage1-v1.py"
            stage2 = release / "train-rna-foundation-stage2-v1.py"
            audit = run / "tahoe-foundation-isolation-v1/audit.json"
            for path in (stage1, stage2, audit):
                path.write_text(path.name, encoding="utf-8")
            source = {
                "cursor_replay_exact": True, "expected_training_rows": 2,
                "first_batch_rows": 1, "measured_genes": 3,
            }
            report = {
                "format": "epi-c3-shared-stage2-live-preflight-v1", "state": "complete",
                "optimizer_updates": 0, "stage1_plan_sha256": "plan",
                "stage1_driver_sha256": module.digest(stage1),
                "stage2_driver_sha256": module.digest(stage2),
                "tahoe_isolation_audit_sha256": module.digest(audit),
                "sources": {name: dict(source) for name in (
                    "arc", "tabula", "orion-HCT116", "orion-HEK293T", "tahoe"
                )},
            }
            preflight = directory / "preflight.json"
            preflight.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(module.verify_preflight(preflight, run, release)["state"], "complete")
            report["sources"]["tahoe"]["cursor_replay_exact"] = False
            preflight.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                module.verify_preflight(preflight, run, release)

    def test_running_requires_live_lock_and_completion_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            state = directory / "state.json"
            lock_path = directory / ".lock"
            state.write_text(json.dumps({"status": "running"}), encoding="utf-8")
            lock_path.touch()
            with lock_path.open("rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(module.stage1_ready(directory))
                fcntl.flock(lock, fcntl.LOCK_UN)
            with self.assertRaisesRegex(RuntimeError, "no trainer"):
                module.stage1_ready(directory)
            state.write_text(json.dumps({"status": "stage1-complete"}), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
                module.stage1_ready(directory)
            (directory / "foundation-stage1.pt").touch()
            with lock_path.open("rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(module.stage1_ready(directory))
                fcntl.flock(lock, fcntl.LOCK_UN)
            self.assertTrue(module.stage1_ready(directory))

    def test_paused_or_unknown_stage_one_never_starts_continuation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for value in ("paused", "failed", "unknown", None):
                (directory / "state.json").write_text(json.dumps({"status": value}), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "unexpected status"):
                    module.stage1_ready(directory)

    def test_existing_status_or_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            status = directory / "status.json"
            status.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                module.dispatch(directory, directory, directory, directory / "preflight", directory / "out", status, 1)
            self.assertEqual(status.read_text(encoding="utf-8"), "original")

    def test_changed_guard_stops_before_launch_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            guard = directory / "guard"
            guard.write_text("original", encoding="utf-8")
            status = directory / "status.json"

            def change_guard(_):
                guard.write_text("changed", encoding="utf-8")
                return False

            with (
                mock.patch.object(module, "verify_preflight", return_value={"stage1_plan_sha256": "a"}),
                mock.patch.object(module, "guard_files", return_value={guard: module.digest(guard)}),
                mock.patch.object(module, "stage1_ready", side_effect=change_guard),
                mock.patch.object(module.time, "sleep"),
                mock.patch.object(module.subprocess, "run") as launch,
                mock.patch.object(module.shutil, "disk_usage", return_value=mock.Mock(free=module.RESERVE_WITH_HEADROOM)),
            ):
                with self.assertRaisesRegex(ValueError, "guarded continuation input changed"):
                    module.dispatch(directory, directory, directory, guard, directory / "out", status, 1)
            launch.assert_not_called()
            self.assertEqual(json.loads(status.read_text(encoding="utf-8"))["state"], "failed")

    def test_completed_stage_starts_once_and_records_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            guard = directory / "guard"
            guard.write_text("original", encoding="utf-8")
            output = directory / "out"
            status = directory / "status.json"

            def complete(command, check):
                self.assertTrue(check)
                self.assertIn(str(output), command)
                output.mkdir()
                (output / "state.json").write_text(json.dumps({"status": "stage2-complete"}), encoding="utf-8")
                (output / "foundation-stage2.pt").write_bytes(b"checkpoint")

            with (
                mock.patch.object(module, "verify_preflight", return_value={"stage1_plan_sha256": "a"}),
                mock.patch.object(module, "guard_files", return_value={guard: module.digest(guard)}),
                mock.patch.object(module, "stage1_ready", return_value=True),
                mock.patch.object(module.subprocess, "run", side_effect=complete) as launch,
                mock.patch.object(module.shutil, "disk_usage", return_value=mock.Mock(free=module.RESERVE_WITH_HEADROOM)),
            ):
                module.dispatch(directory, directory, directory, guard, output, status, 1)
            launch.assert_called_once()
            result = json.loads(status.read_text(encoding="utf-8"))
            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["checkpoint_sha256"], module.digest(output / "foundation-stage2.pt"))

    def test_disk_reserve_prevents_launch_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            guard = directory / "guard"
            guard.write_text("original", encoding="utf-8")
            status = directory / "status.json"
            with (
                mock.patch.object(module, "verify_preflight", return_value={"stage1_plan_sha256": "a"}),
                mock.patch.object(module, "guard_files", return_value={guard: module.digest(guard)}),
                mock.patch.object(module.subprocess, "run") as launch,
                mock.patch.object(module.shutil, "disk_usage", return_value=mock.Mock(free=module.RESERVE_WITH_HEADROOM - 1)),
            ):
                with self.assertRaisesRegex(OSError, "Insufficient free space"):
                    module.dispatch(directory, directory, directory, guard, directory / "out", status, 1)
            launch.assert_not_called()
            self.assertEqual(json.loads(status.read_text(encoding="utf-8"))["state"], "failed")


if __name__ == "__main__":
    unittest.main()

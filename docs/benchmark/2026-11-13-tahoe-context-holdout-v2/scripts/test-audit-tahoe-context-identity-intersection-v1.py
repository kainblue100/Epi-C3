import importlib.util
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


spec = importlib.util.spec_from_file_location(
    "identity_intersection", Path(__file__).with_name("audit-tahoe-context-identity-intersection-v1.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def protected(sample, barcode, source):
    return {
        "cell_id": module.identity(sample, barcode), "sample": sample, "barcode": barcode,
        "legacy_source_files": [source] if source else [],
    }


def validation(sample, barcode, source):
    return {
        "cell_id": module.identity(sample, barcode), "sample": sample, "barcode": barcode,
        "source_file": source, "source_row": 1,
    }


def setup_fixture(root):
    (root / "data").mkdir()
    path = root / "data" / "part.parquet"
    pq.write_table(pa.table({
        "sample": ["a", "b", "c", "z"],
        "BARCODE_SUB_LIB_ID": ["x", "y", "v", "unused"],
        "cell_line_id": ["held", "other", "held", "other"],
    }), path, row_group_size=1)
    stat = path.stat()
    source = "data/part.parquet"
    entries = {source: {"path": source, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}}
    p = [protected("a", "x", source), protected("b", "y", source)]
    v = [validation("c", "v", source)]
    progress = {
        "protected_eligible": {item["cell_id"]: 1 for item in p},
        "protected_seen": {item["cell_id"]: 1 for item in p},
        "validation_eligible": {item["cell_id"]: 1 for item in v},
        "validation_seen": {item["cell_id"]: 1 for item in v},
        "legacy_seen": {item["cell_id"]: [source] for item in p},
    }
    groups = {
        "a\theld": {"sample": "a", "cell_line_id": "held"},
        "b\tother": {"sample": "b", "cell_line_id": "other"},
        "c\theld": {"sample": "c", "cell_line_id": "held"},
    }
    return entries, p, v, progress, groups


class IdentityIntersectionTests(unittest.TestCase):
    def test_scans_only_metadata_and_finds_context_intersections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries, protected_rows, validation_rows, progress, groups = setup_fixture(root)
            shards, coverage = module.candidate_sources(protected_rows, validation_rows, progress, entries)
            result = module.scan(root, entries, groups, shards, {"held"})
            self.assertEqual(coverage["protected_located"], 2)
            self.assertEqual(coverage["validation_located"], 1)
            self.assertEqual(result["protected_heldout"], 1)
            self.assertEqual(result["protected_retained"], 1)
            self.assertEqual(result["validation_heldout"], 1)
            self.assertEqual(result["row_groups_pruned"], 1)
            self.assertEqual(result["row_groups_scanned"], 3)

    def test_missing_or_changed_shard_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries, protected_rows, validation_rows, progress, groups = setup_fixture(root)
            shards, _ = module.candidate_sources(protected_rows, validation_rows, progress, entries)
            del groups["a\theld"]
            with self.assertRaisesRegex(ValueError, "frozen source groups"):
                module.scan(root, entries, groups, shards, {"held"})
            path = root / "data" / "part.parquet"
            with path.open("ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "shard changed"):
                module.scan(root, entries, groups, shards, {"held"})

    def test_numeric_sample_statistics_cannot_prune(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numeric.parquet"
            pq.write_table(pa.table({"sample": [1]}), path)
            parquet = pq.ParquetFile(path)
            try:
                with self.assertRaisesRegex(ValueError, "sample field"):
                    module.group_may_contain(parquet, 0, ["1"])
            finally:
                parquet.close()

    def test_requires_verified_unique_source_and_exposes_only_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            entries, protected_rows, validation_rows, progress, _ = setup_fixture(Path(directory))
            protected_rows[0]["legacy_source_files"] = []
            shards, coverage = module.candidate_sources(protected_rows, validation_rows, progress, entries)
            self.assertEqual(coverage["protected_unresolved"], 1)
            self.assertNotIn(protected_rows[0]["cell_id"], str(coverage))
            self.assertTrue(all(type(value) is int for value in coverage.values()))
            self.assertEqual(sum(map(len, shards.values())), 2)
            progress["protected_seen"][protected_rows[1]["cell_id"]] = 2
            with self.assertRaisesRegex(ValueError, "exactly one"):
                module.candidate_sources(protected_rows, validation_rows, progress, entries)

    def test_rejects_tampered_identity_and_unverified_legacy_source(self):
        with tempfile.TemporaryDirectory() as directory:
            entries, protected_rows, validation_rows, progress, _ = setup_fixture(Path(directory))
            protected_rows[0]["barcode"] = "altered"
            with self.assertRaisesRegex(ValueError, "invalid or duplicated"):
                module.candidate_sources(protected_rows, validation_rows, progress, entries)
            protected_rows[0]["barcode"] = "x"
            del progress["legacy_seen"][protected_rows[0]["cell_id"]]
            with self.assertRaisesRegex(ValueError, "not verified"):
                module.candidate_sources(protected_rows, validation_rows, progress, entries)


if __name__ == "__main__":
    unittest.main()

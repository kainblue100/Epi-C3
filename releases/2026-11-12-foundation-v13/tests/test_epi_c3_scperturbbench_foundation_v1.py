import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import scipy.sparse as sp

import epi_c3_scperturbbench_foundation_v1 as module
from epi_c3_shared_foundation_v1 import file_digest, gene_axis_digest


def write_matrix(path, values, encoding):
    matrix = sp.csr_matrix(values) if encoding == "csr_matrix" else sp.csc_matrix(values)
    with h5py.File(path, "w") as file:
        group = file.create_group("layers/counts")
        group.attrs["encoding-type"] = encoding
        group.attrs["shape"] = matrix.shape
        for key in ("data", "indices", "indptr"):
            group.create_dataset(key, data=getattr(matrix, key))


class MappingTests(unittest.TestCase):
    def test_ambiguous_and_colliding_symbols_are_not_projected(self):
        axis = ["ENSG1", "ENSG2", "HUMAN_SYMBOL:amb", "HUMAN_SYMBOL:orphan"]
        symbols = {
            "a": {"ENSG1"},
            "amb": {"ENSG1", "ENSG2"},
            "collision1": {"ENSG2"},
            "collision2": {"ENSG2"},
        }
        mapping, report = module.resolve_mapping(
            ["a", "amb", "orphan", "collision1", "collision2"], axis, symbols
        )
        self.assertEqual(mapping.tolist(), [0, -1, 3, -1, -1])
        self.assertEqual(report["status_counts"]["ambiguous_symbol"], 1)
        self.assertEqual(report["status_counts"]["mapping_collision"], 2)

    def test_known_missing_direct_id_does_not_fall_back_to_symbol(self):
        mapping, _ = module.resolve_mapping(
            ["a", "b"], ["ENSG1"], {"a": {"ENSG1"}}, ["ENSGX", "ENSG1"]
        )
        self.assertEqual(mapping.tolist(), [-1, 0])


class SparseAuditTests(unittest.TestCase):
    def test_csr_and_csc_count_and_project_identically_across_chunks(self):
        values = np.asarray([[1, 2, 0, 3], [0, 1, 2, 0], [4, 0, 1, 1]], dtype=float)
        mapping = np.asarray([0, 2, -1, 3])
        with tempfile.TemporaryDirectory() as temporary:
            for encoding in ("csr_matrix", "csc_matrix"):
                path = Path(temporary) / encoding
                write_matrix(path, values, encoding)
                with h5py.File(path, "r") as file:
                    report, mapped = module.audit_sparse_counts(file["layers/counts"], mapping, chunk_size=2)
                np.testing.assert_array_equal(mapped, [6, 1, 5])
                self.assertEqual(report["total_counts"], 15)
                self.assertEqual(report["mapped_counts"], 12)

    def test_fractional_counts_and_duplicate_indices_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.h5"
            write_matrix(path, np.asarray([[1.5, 2], [3, 0]]), "csr_matrix")
            with h5py.File(path, "r") as file:
                with self.assertRaisesRegex(ValueError, "Count values"):
                    module.audit_sparse_counts(file["layers/counts"], np.asarray([0, 1]))
            with h5py.File(path, "r+") as file:
                file["layers/counts/data"][0] = 1
                file["layers/counts/indices"][1] = 0
            with h5py.File(path, "r") as file:
                with self.assertRaisesRegex(ValueError, "duplicated"):
                    module.audit_sparse_counts(file["layers/counts"], np.asarray([0, 1]), chunk_size=1)

    def test_unique_but_unsorted_sparse_indices_are_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsorted.h5"
            write_matrix(path, np.asarray([[1, 2, 0], [3, 0, 4]]), "csr_matrix")
            with h5py.File(path, "r+") as file:
                indices = file["layers/counts/indices"]
                data = file["layers/counts/data"]
                indices[:2] = indices[:2][::-1]
                data[:2] = data[:2][::-1]
            with h5py.File(path, "r") as file:
                report, mapped = module.audit_sparse_counts(
                    file["layers/counts"], np.asarray([0, 1, 2]), chunk_size=1
                )
            self.assertEqual(report["unsorted_major_vectors"], 1)
            np.testing.assert_array_equal(mapped, [3, 7])


class PartitionTests(unittest.TestCase):
    def test_khp_excludes_target_evaluation_and_all_treated_cells(self):
        condition = np.asarray(["ctrl", "drug", "ctrl", "ctrl", "ctrl", "ctrl", "ctrl", "ctrl"])
        cell_type = np.asarray(["B", "B", "CD8T", "CD8T", "B", "B", "B", "B"])
        positive = np.asarray([True, True, True, True, True, True, False, True])
        selected, summary = module.partition_controls(
            "KHP", condition, cell_type, positive, {2, 3}, validation_size=1, seed=7
        )
        all_rows = set(selected["training_rows"]) | set(selected["validation_rows"])
        self.assertEqual(all_rows, {0, 4, 5, 7})
        self.assertEqual(summary["known_protected_rows"], 2)
        self.assertEqual(selected, module.partition_controls(
            "KHP", condition, cell_type, positive, {2, 3}, validation_size=1, seed=7
        )[0])
        with self.assertRaisesRegex(ValueError, "CD8T"):
            module.partition_controls("KHP", condition, cell_type, positive, {0}, validation_size=1)

    def test_cmo_holds_out_one_entire_donor(self):
        conditions = np.asarray(["ctrl", "ctrl", "drug"] * 3)
        cell_types = np.asarray(["B"] * 9)
        donors = np.asarray(["a"] * 3 + ["b"] * 3 + ["c"] * 3)
        selected, summary = module.partition_controls(
            "CMO", conditions, cell_types, np.ones(9, dtype=bool), set(),
            donors=donors, validation_size=1, seed=5,
        )
        rows = selected["training_rows"] + selected["validation_rows"]
        self.assertEqual(len(set(donors[rows])), 2)
        self.assertTrue(all(conditions[row] == "ctrl" for row in rows))
        self.assertEqual(summary["heldout_donor_rows_excluded"], 3)


class StreamTests(unittest.TestCase):
    def test_both_encodings_preserve_mask_projection_and_recovery(self):
        values = np.asarray([
            [1, 2, 0, 3], [0, 1, 2, 0], [4, 0, 1, 1],
            [2, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 2],
        ], dtype=float)
        axis = ["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"]
        mapping = [0, 3, -1, 4]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for encoding in ("csr_matrix", "csc_matrix"):
                data = root / f"{encoding}.h5"
                write_matrix(data, values, encoding)
                if encoding == "csr_matrix":
                    with h5py.File(data, "r+") as file:
                        node = file["layers/counts"]
                        node["indices"][:3] = node["indices"][:3][::-1]
                        node["data"][:3] = node["data"][:3][::-1]
                private_path = root / f"{encoding}.private.json"
                public_path = root / f"{encoding}.audit.json"
                digest = file_digest(data)
                private = {
                    "format": module.PRIVATE_FORMAT, "gene_axis_sha256": gene_axis_digest(axis),
                    "sources": {"KHP": {
                        "source_sha256": digest, "shape": list(values.shape), "encoding": encoding,
                        "mapping": mapping, "training_rows": [0, 1, 2, 3], "validation_rows": [4, 5],
                    }},
                }
                private_path.write_text(json.dumps(private), encoding="utf-8")
                public = {
                    "format": module.PUBLIC_FORMAT, "state": "complete",
                    "gene_axis_sha256": gene_axis_digest(axis), "private_sha256": file_digest(private_path),
                    "source_module_sha256": file_digest(Path(module.__file__)),
                    "sources": {"KHP": {"source_sha256": digest}},
                }
                public_path.write_text(json.dumps(public), encoding="utf-8")
                stream = module.BenchmarkControlStream(data, public_path, private_path, "KHP", axis, batch_size=2)
                try:
                    self.assertEqual(stream.measured.tolist(), [True, False, False, True, True])
                    self.assertEqual(stream.validation.shape, (2, 5))
                    first = stream.next_batch()
                    np.testing.assert_array_equal(first.toarray(), [[1, 0, 0, 2, 3], [0, 0, 0, 1, 0]])
                    checkpoint = stream.cursor()
                    second = stream.next_batch().toarray()
                    stream.set_cursor(checkpoint)
                    np.testing.assert_array_equal(stream.next_batch().toarray(), second)
                    self.assertIsNone(stream.next_batch())
                    with self.assertRaisesRegex(ValueError, "cursor"):
                        stream.set_cursor({"position": True})
                finally:
                    stream.close()
                private_path.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source plan"):
                    module.BenchmarkControlStream(data, public_path, private_path, "KHP", axis, batch_size=2)


if __name__ == "__main__":
    unittest.main()

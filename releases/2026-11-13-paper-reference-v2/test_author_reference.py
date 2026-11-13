import copy
import hashlib
import importlib.util
import statistics
import tempfile
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("render_author_reference.py")
SPEC = importlib.util.spec_from_file_location("author_reference", PATH)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def fixture():
    contexts, aggregates, models = [], [], []
    for dataset, names in reference.CONTEXTS.items():
        for representation in reference.REPRESENTATIONS:
            available = list(names)
            if dataset == "Parse" and representation == "X_hvg":
                available.append("CD4_Memory")
            models.append({
                "dataset": dataset, "regime": "fewshot", "representation": representation,
                "origin": "original fewshot",
                "evaluations": [{"context": name} for name in available],
                "missing_contexts": ["CD4_Memory"] if dataset == "Parse" and representation == "X_state" else [],
            })
        for metric, _ in reference.METRICS:
            rows = []
            for i, name in enumerate(names):
                row = {
                    "dataset": dataset, "regime": "fewshot", "context": name,
                    "metric": metric, "n_pairs": str(i + 1),
                    "X_hvg": str(0.1 + i * 0.03), "X_state": str(0.2 + i * 0.04),
                }
                rows.append(row)
            contexts.extend(rows)
            aggregates.append({
                "dataset": dataset, "regime": "fewshot", "metric": metric,
                "n_contexts": str(len(names)),
                "n_pairs": str(sum(int(row["n_pairs"]) for row in rows)),
                "X_hvg": str(statistics.fmean(float(row["X_hvg"]) for row in rows)),
                "X_state": str(statistics.fmean(float(row["X_state"]) for row in rows)),
            })
    return contexts, aggregates, {"updated": "fixture", "models": models}


class AuthorReferenceTests(unittest.TestCase):
    def test_equal_weight_context_means_and_population_sd(self):
        contexts, aggregates, index = fixture()
        result = reference.build_summary(contexts, aggregates, index)
        self.assertFalse(result["project_same_protocol_evaluation"])
        self.assertFalse(result["article_figure_2_reproduction"])
        self.assertEqual([len(dataset["panels"]) for dataset in result["datasets"]], [6, 6, 6])
        parse = result["datasets"][1]["panels"][0]
        self.assertEqual(parse["n_contexts"], 3)
        self.assertEqual(parse["n_pairs_total"], 6)
        self.assertAlmostEqual(parse["scores"]["X_hvg"]["mean"], 0.13)
        self.assertAlmostEqual(parse["scores"]["X_hvg"]["population_sd"], statistics.pstdev([0.1, 0.13, 0.16]))

    def test_rejects_missing_or_duplicated_context(self):
        contexts, aggregates, index = fixture()
        with self.assertRaisesRegex(ValueError, "Unexpected contexts"):
            reference.build_summary(contexts[1:], aggregates, index)
        with self.assertRaisesRegex(ValueError, "Unexpected contexts"):
            reference.build_summary(contexts + [contexts[0]], aggregates, index)

    def test_rejects_incorrect_aggregate_or_pair_count(self):
        contexts, aggregates, index = fixture()
        changed = copy.deepcopy(aggregates)
        changed[0]["X_hvg"] = "0.9"
        with self.assertRaisesRegex(ValueError, "Aggregate mismatch"):
            reference.build_summary(contexts, changed, index)
        changed = copy.deepcopy(aggregates)
        changed[0]["n_pairs"] = "999"
        with self.assertRaisesRegex(ValueError, "Paired count mismatch"):
            reference.build_summary(contexts, changed, index)

    def test_rejects_nonfinite_and_changed_source(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            reference._finite("nan", "test")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.csv"
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Source SHA-256 mismatch"):
                reference.verify_source(path, hashlib.sha256(b"original").hexdigest())

    def test_rejects_undocumented_parse_gap(self):
        contexts, aggregates, index = fixture()
        parse_se = next(
            model for model in index["models"]
            if model["dataset"] == "Parse" and model["representation"] == "X_state"
        )
        parse_se["missing_contexts"] = []
        with self.assertRaisesRegex(ValueError, "CD4 Memory gap"):
            reference.build_summary(contexts, aggregates, index)


if __name__ == "__main__":
    unittest.main()

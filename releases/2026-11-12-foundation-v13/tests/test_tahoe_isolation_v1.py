import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import epi_c3_tahoe_isolation_v1 as isolation


spec = importlib.util.spec_from_file_location(
    "freeze_tahoe", Path(__file__).parent.parent / "freeze-tahoe-protected-identities-v1.py",
)
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)

spec = importlib.util.spec_from_file_location(
    "verify_tahoe", Path(__file__).parent.parent / "verify-tahoe-foundation-isolation-v1.py",
)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def test_row_group_pruning_uses_bounds_and_scans_unknown_statistics(tmp_path):
    table = pa.table({"sample": ["smp_010", "smp_011", "smp_090"], "BARCODE_SUB_LIB_ID": ["a", "b", "c"]})
    path = tmp_path / "with-stats.parquet"
    pq.write_table(table, path, row_group_size=2)
    parquet = pq.ParquetFile(path)
    assert isolation.row_group_may_contain(parquet, 0, ["smp_011"])
    assert not isolation.row_group_may_contain(parquet, 0, ["smp_090"])
    assert isolation.row_group_may_contain(parquet, 1, ["smp_090"])
    without = tmp_path / "without-stats.parquet"
    pq.write_table(table, without, row_group_size=2, write_statistics=False)
    assert isolation.row_group_may_contain(pq.ParquetFile(without), 0, ["smp_090"])


def test_protected_pairs_reject_mismatched_hashes():
    pairs = {}
    cell_id = isolation.identity("smp_1", "barcode")
    isolation.add_pair(pairs, cell_id, "smp_1", "barcode")
    assert isolation.grouped_barcodes(pairs) == {"smp_1": {"barcode"}}
    with pytest.raises(ValueError, match="does not match"):
        isolation.add_pair(pairs, cell_id, "smp_2", "barcode")


def test_interim_condition_and_control_samples_are_bound_to_identity():
    treatment_id = isolation.identity("smp_t", "a")
    control_id = isolation.identity("smp_c", "b")
    obs = pd.DataFrame({
        "target_gene": ["compound-000", "non-targeting"],
        "context": ["line", "line"], "batch": ["plate", "plate"],
    }, index=[treatment_id, control_id])
    conditions = {"compound-000": {"sample": "smp_t", "cell_line_id": "line", "plate": "plate"}}
    expected = freeze.modern_expectations(obs, "interim", conditions, {("line", "plate"): {"smp_c"}})
    assert expected == {treatment_id: {"smp_t"}, control_id: {"smp_c"}}
    obs.loc[treatment_id, "batch"] = "other"
    with pytest.raises(ValueError, match="frozen condition"):
        freeze.modern_expectations(obs, "interim", conditions, {("line", "plate"): {"smp_c"}})


def test_independent_eligibility_handles_control_buckets_and_heldout_compounds():
    metadata = {
        "config": {"seed": 17},
        "groups": {
            "control\tline": {"drug": "DMSO_TF", "plate": "plate"},
            "treated\tline": {"drug": "drug", "plate": "plate", "canonical_smiles": "molecule"},
        },
    }
    training = next(f"b{i}" for i in range(100) if verify.control_is_training("control", f"b{i}", 17))
    heldout = next(f"b{i}" for i in range(100) if not verify.control_is_training("control", f"b{i}", 17))
    row = {
        "drug": "DMSO_TF", "sample": "control", "plate": "plate", "cell_line_id": "line",
        "BARCODE_SUB_LIB_ID": training, "canonical_smiles": "",
    }
    assert verify.eligible(row, metadata, {}, set(), lambda value, drug: value)
    row["BARCODE_SUB_LIB_ID"] = heldout
    assert not verify.eligible(row, metadata, {}, set(), lambda value, drug: value)
    row.update(drug="drug", sample="treated", canonical_smiles="molecule")
    assert not verify.eligible(row, metadata, {"drug": "molecule"}, {"drug"}, lambda value, drug: value)
    assert verify.eligible(row, metadata, {"drug": "molecule"}, set(), lambda value, drug: value)

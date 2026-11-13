import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from c3vcell.identity import source_cell_id
from epi_c3_foundation_tahoe_v1 import TahoeStream


class FakeHelper:
    @staticmethod
    def create_token_map(tokens):
        result = np.full(max(tokens) + 1, -1, dtype=np.int32)
        for token, value in tokens.items():
            result[token] = value
        return result

    @staticmethod
    def check_files(root, entries):
        for entry in entries:
            if not (root / entry["path"]).is_file():
                raise ValueError("Missing source")


class FakeTahoe:
    COLUMNS = ["sample", "BARCODE_SUB_LIB_ID", "value"]

    @staticmethod
    def treatment_split(metadata, frame):
        return {}, set(), {"eligible_rows": metadata["rows"]}

    @staticmethod
    def select_training(batch, metadata, structures, aliases, seed):
        return np.arange(batch.num_rows, dtype=np.int64)

    @staticmethod
    def counts_from_batch(batch, selected, mapping, n_genes):
        values = np.asarray(batch.column("value").to_pylist())[selected]
        result = np.zeros((len(selected), n_genes), dtype=np.float32)
        result[:, mapping[5]] = values
        return sp.csr_matrix(result)


def build(tmp_path, protected):
    (tmp_path / "metadata").mkdir(exist_ok=True)
    pd.DataFrame({"drug": ["dummy"]}).to_parquet(tmp_path / "metadata/drug_metadata.parquet")
    pq.write_table(pa.table({
        "sample": ["sample1"] * 40,
        "BARCODE_SUB_LIB_ID": [f"cell-{index}" for index in range(40)],
        "value": list(range(1, 41)),
    }), tmp_path / "part.parquet")
    metadata = {
        "state": "complete", "rows": 40, "config": {"seed": 17},
        "provenance": {"files": [{"path": "part.parquet"}]},
    }
    frame = pd.DataFrame({"ensembl_id": ["ENSG1"], "token_id": [5]})
    return TahoeStream(
        tmp_path, metadata, frame, ["ENSG1", "ENSG2"], FakeHelper, FakeTahoe,
        protected, batch_size=12, seed=5,
    )


def test_tahoe_validation_exclusion_and_cursor_replay(tmp_path):
    stream = build(tmp_path, set())
    assert stream.expected_cells == 8
    assert stream.validation.shape == (32, 2)
    first = stream.next_batch()
    cursor = stream.cursor()
    following = stream.next_batch()
    stream.set_cursor(cursor)
    replay = stream.next_batch()
    assert first.shape[0] == 4
    assert (following != replay).nnz == 0
    assert first.shape[0] + following.shape[0] == stream.expected_cells
    assert stream.next_batch() is None


def test_tahoe_rejects_an_existing_evaluation_cell(tmp_path):
    protected = {source_cell_id("tahoe", "sample1", "cell-4")}
    with pytest.raises(ValueError, match="evaluation identity"):
        build(tmp_path, protected)

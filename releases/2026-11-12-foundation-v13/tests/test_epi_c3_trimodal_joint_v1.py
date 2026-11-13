from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from c3vcell.model import C3VCell, ModelConfig
from epi_c3_shared_foundation_v1 import FORMAT, file_digest, gene_axis_digest, load_foundation
from epi_c3_trimodal_joint_v1 import (
    CellKey,
    ModalityIdentity,
    PairedCell,
    TriModalJointEncoder,
    masked_rna_loss,
    same_sample_contrastive_loss,
    validate_paired_cells,
)


def cell(line: str, barcode: str, split: str = "train") -> PairedCell:
    key = CellKey("source", "experiment", "sample", barcode, line, "human", "GRCh38")
    return PairedCell(
        ModalityIdentity("rna", key, "a" * 64, "RNA-source"),
        ModalityIdentity("atac", key, "a" * 64, "ATAC-source"),
        ModalityIdentity("contacts", key, "a" * 64, "contact-source"),
        split,
    )


def empty_contacts(batch: int, width: int) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.long), torch.empty(0), (batch, width),
    ).coalesce()


def test_three_lines_and_exact_source_matched_identity():
    cells = [cell("K562", "a"), cell("GM12878", "b"), cell("H1", "c")]
    result = validate_paired_cells(cells, species="human", assembly="GRCh38")
    assert result["training_lines"] == {"GM12878": 1, "H1": 1, "K562": 1}
    for field in ("barcode", "cell_line", "species", "assembly", "sample"):
        changed_key = replace(cells[0].contacts.key, **{field: "different"})
        bad = replace(cells[0], contacts=replace(cells[0].contacts, key=changed_key))
        with pytest.raises(ValueError):
            validate_paired_cells([bad, *cells[1:]], species="human", assembly="GRCh38")
    with pytest.raises(ValueError):
        validate_paired_cells(cells[:2], species="human", assembly="GRCh38")


def test_duplicate_cell_across_splits_and_unverified_mapping_rejected():
    cells = [cell("K562", "a"), cell("GM12878", "b"), cell("H1", "c")]
    with pytest.raises(ValueError):
        validate_paired_cells(cells + [replace(cells[0], split="test")], species="human", assembly="GRCh38")
    bad = replace(cells[0], atac=replace(cells[0].atac, pairing_map_sha256="b" * 64))
    with pytest.raises(ValueError):
        validate_paired_cells([bad, *cells[1:]], species="human", assembly="GRCh38")


def test_joint_encoding_and_gradients_with_sparse_positive_contacts():
    torch.manual_seed(7)
    model = TriModalJointEncoder(4, 3, 5, hidden_dim=8, latent_dim=6, allow_untrained_rna=True)
    rna = torch.tensor([[1.0, 0, 2, 0], [0, 1, 0, 3]])
    atac = torch.tensor([[0.0, 1, 0], [1, 0, 2]])
    contacts = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1], [1, 4, 2]]), torch.tensor([1.0, 2, 3]), (2, 5),
    )
    output = model(rna, torch.ones_like(rna, dtype=torch.bool), atac,
                   torch.ones_like(atac, dtype=torch.bool), contacts, torch.ones(2, dtype=torch.bool))
    assert output["fused"].shape == (2, 6)
    assert torch.allclose(output["modality_weights"].sum(dim=1), torch.ones(2))
    assert output["modality_present"].all()
    output["fused"].square().sum().backward()
    assert model.contact_encoder.edge_embedding.grad is not None
    assert torch.isfinite(model.contact_encoder.edge_embedding.grad).all()
    assert torch.any(model.contact_encoder.edge_embedding.grad != 0)


def test_rna_only_pretraining_explicitly_masks_unmeasured_modalities():
    torch.manual_seed(8)
    model = TriModalJointEncoder(4, 3, 5, hidden_dim=8, latent_dim=6, allow_untrained_rna=True)
    rna = torch.tensor([[1.0, 0, 2, 0], [0, 1, 0, 3]])
    output = model(rna, torch.ones_like(rna, dtype=torch.bool), torch.zeros(2, 3),
                   torch.zeros(2, 3, dtype=torch.bool), empty_contacts(2, 5), torch.zeros(2, dtype=torch.bool))
    assert output["modality_present"].tolist() == [[True, False, False], [True, False, False]]
    assert output["modality_weights"].tolist() == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    output["rna_reconstruction"].sum().backward()
    assert model.rna_encoder.network[0].weight.grad is not None


def test_missing_assay_cannot_have_contact_or_unmeasured_value():
    model = TriModalJointEncoder(2, 2, 3, hidden_dim=4, latent_dim=2, allow_untrained_rna=True)
    dense = torch.ones((1, 2))
    measured = torch.ones((1, 2), dtype=torch.bool)
    bad_contacts = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), (1, 3))
    with pytest.raises(ValueError):
        model(dense, measured, dense, measured, bad_contacts, torch.tensor([False]))
    with pytest.raises(ValueError):
        model(dense, torch.tensor([[True, False]]), dense, measured, empty_contacts(1, 3), torch.tensor([False]))
    zero_contacts = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([0.0]), (1, 3))
    with pytest.raises(ValueError):
        model(dense, measured, dense, measured, zero_contacts, torch.tensor([True]))


def test_rna_loss_only_uses_measured_heldout_targets():
    target = torch.tensor([[2.0, 5.0, 9.0]])
    predicted = torch.tensor([[1.0, 100.0, 0.0]])
    measured = torch.tensor([[True, True, False]])
    heldout = torch.tensor([[True, False, False]])
    assert masked_rna_loss(predicted, target, measured, heldout).item() == 1.0
    with pytest.raises(ValueError):
        masked_rna_loss(predicted, target, measured, torch.tensor([[False, False, True]]))


def test_contrastive_alignment_requires_distinct_same_sample_train_cells():
    cells = [cell("K562", "a"), cell("K562", "b"), cell("H1", "c")]
    left = torch.tensor([[1.0, 0], [0, 1], [0.5, 0.5]], requires_grad=True)
    right = left.detach().clone().requires_grad_()
    loss = same_sample_contrastive_loss(left, right, cells)
    loss.backward()
    assert torch.isfinite(loss) and left.grad is not None and right.grad is not None
    with pytest.raises(ValueError):
        same_sample_contrastive_loss(left, right, [cell("K562", "a"), cell("H1", "b"), cell("GM", "c")])
    with pytest.raises(ValueError):
        same_sample_contrastive_loss(left, right, [cells[0], replace(cells[1], split="test"), cells[2]])


def test_joint_model_requires_verified_foundation_unless_explicitly_exploring(tmp_path):
    with pytest.raises(ValueError, match="foundation"):
        TriModalJointEncoder(2, 2, 3, hidden_dim=4, latent_dim=3)
    base = C3VCell(ModelConfig(4, latent_dim=3, hidden_dim=5))
    genes = ["a", "b", "c", "d"]
    path = tmp_path / "foundation.pt"
    torch.save({
        "format": FORMAT, "status": "complete", "scope": "exploratory",
        "genes": genes, "gene_axis_sha256": gene_axis_digest(genes),
        "model_config": base.configuration(), "model_state": base.state_dict(),
    }, path)
    foundation = load_foundation(path, file_digest(path), ["c", "a"], allow_exploratory=True)
    with pytest.raises(ValueError, match="full foundation"):
        TriModalJointEncoder(2, 2, 3, hidden_dim=4, latent_dim=3, rna_encoder=foundation)
    joint = TriModalJointEncoder(
        2, 2, 3, hidden_dim=4, latent_dim=3, rna_encoder=foundation,
        allow_exploratory_foundation=True,
    )
    assert joint.rna_encoder is foundation
    rna = torch.tensor([[2.0, 1.0], [1.0, 0.0]])
    result = joint(
        rna, torch.ones_like(rna, dtype=torch.bool), torch.zeros(2, 2),
        torch.zeros(2, 2, dtype=torch.bool), empty_contacts(2, 3), torch.zeros(2, dtype=torch.bool),
    )
    assert torch.isfinite(result["fused"]).all()
    assert result["modality_present"].tolist() == [[True, False, False], [True, False, False]]

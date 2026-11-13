from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from c3vcell.model import C3VCell, ModelConfig
from epi_c3_shared_foundation_v1 import FORMAT, file_digest, gene_axis_digest, load_foundation
from epi_c3_trimodal_joint_v1 import CellKey, ModalityIdentity, PairedCell, TriModalJointEncoder
from epi_c3_trimodal_response_v1 import TriModalPerturbationResponse, population_target_excluded_loss


def paired_cell(line: str, barcode: str) -> PairedCell:
    key = CellKey("source", "experiment", "sample", barcode, line, "human", "GRCh38")
    return PairedCell(
        ModalityIdentity("rna", key, "a" * 64, "RNA-source"),
        ModalityIdentity("atac", key, "a" * 64, "ATAC-source"),
        ModalityIdentity("contacts", key, "a" * 64, "contact-source"),
        "train",
    )


@pytest.fixture
def example(tmp_path):
    torch.manual_seed(31)
    genes = ["A", "B", "C", "D"]
    local_genes = ["C", "A", "B"]
    base = C3VCell(ModelConfig(4, latent_dim=4, hidden_dim=6))
    path = tmp_path / "foundation.pt"
    torch.save({
        "format": FORMAT, "status": "complete", "scope": "exploratory",
        "genes": genes, "gene_axis_sha256": gene_axis_digest(genes),
        "model_config": base.configuration(), "model_state": base.state_dict(),
    }, path)
    foundation = load_foundation(path, file_digest(path), local_genes, allow_exploratory=True)
    joint = TriModalJointEncoder(
        3, 2, 4, hidden_dim=6, latent_dim=4, rna_encoder=foundation,
        allow_exploratory_foundation=True,
    )
    model = TriModalPerturbationResponse(
        joint, local_genes, species="human", assembly="GRCh38", hidden_dim=5, allow_untrained_joint=True,
    )
    rna = torch.tensor([[1.0, 2, 3], [3, 0, 1], [1, 2, 0]])
    rna_mask = torch.tensor([[True, True, True], [True, False, True], [True, True, True]])
    atac = torch.tensor([[1.0, 3], [2, 1], [0, 2]])
    contacts = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1, 2], [0, 2, 1, 3]]), torch.tensor([1.0, 2, 3, 4]), (3, 4),
    )
    inputs = {
        "rna": rna, "rna_measured": rna_mask, "atac": atac,
        "atac_measured": torch.ones_like(atac, dtype=torch.bool), "contacts": contacts,
        "contact_present": torch.ones(3, dtype=torch.bool),
        "cells": [paired_cell("K562", "a"), paired_cell("GM12878", "b"), paired_cell("H1", "c")],
        "target_indices": torch.tensor([0, 1, 2]),
        "intervention_direction": torch.tensor([-1.0, 1.0, -1.0]),
    }
    return model, inputs


def test_untrained_joint_and_gene_axis_require_explicit_gates(example):
    model, _ = example
    with pytest.raises(ValueError, match="untrained"):
        TriModalPerturbationResponse(model.joint, ["C", "A", "B"], species="human", assembly="GRCh38")
    with pytest.raises(ValueError, match="gene axis"):
        TriModalPerturbationResponse(
            model.joint, ["A", "C", "B"], species="human", assembly="GRCh38", allow_untrained_joint=True,
        )
    with pytest.raises(ValueError, match="human axis"):
        TriModalPerturbationResponse(
            model.joint, ["C", "A", "B"], species="mouse", assembly="mm10", allow_untrained_joint=True,
        )
    untrained_rna = TriModalJointEncoder(3, 2, 4, hidden_dim=6, latent_dim=4, allow_untrained_rna=True)
    with pytest.raises(ValueError, match="foundation encoder"):
        TriModalPerturbationResponse(
            untrained_rna, ["C", "A", "B"], species="human", assembly="GRCh38", allow_untrained_joint=True,
        )


def test_zero_strength_is_control_and_preserves_measurement_mask(example):
    model, inputs = example
    result = model(**inputs, strength=0.0)
    expected = torch.log1p(inputs["rna"] / inputs["rna"].sum(dim=1, keepdim=True) * 10_000.0)
    assert torch.allclose(result["log_expression"], expected)
    assert torch.equal(result["measured"], inputs["rna_measured"])
    assert torch.allclose(result["modality_weights"].sum(dim=1), torch.ones(3))


def test_response_depends_on_condition_and_observed_contacts(example):
    model, inputs = example
    with torch.no_grad():
        torch.nn.init.normal_(model.response_query.weight, std=0.4)
        model.gene_bias.fill_(2.0)
    initial = model(**inputs)["log_expression"]
    other_target = model(**{**inputs, "target_indices": torch.tensor([1, 1, 2])})["log_expression"]
    other_direction = model(**{
        **inputs, "intervention_direction": torch.tensor([1.0, 1.0, -1.0]),
    })["log_expression"]
    changed_contacts = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1, 2], [0, 2, 1, 3]]), torch.tensor([9.0, 2, 3, 4]), (3, 4),
    )
    other_contact = model(**{**inputs, "contacts": changed_contacts})["log_expression"]
    changed_atac = inputs["atac"].clone()
    changed_atac[0] = torch.tensor([4.0, 1.0])
    other_atac = model(**{**inputs, "atac": changed_atac})["log_expression"]
    assert not torch.allclose(initial[0], other_target[0])
    assert not torch.allclose(initial[0], other_direction[0])
    assert not torch.allclose(initial[0], other_contact[0])
    assert not torch.allclose(initial[0], other_atac[0])
    initial.sum().backward()
    assert model.joint.contact_encoder.edge_embedding.grad is not None
    assert torch.any(model.joint.contact_encoder.edge_embedding.grad != 0)
    assert model.joint.atac_encoder.network[0].weight.grad is not None
    assert torch.any(model.joint.atac_encoder.network[0].weight.grad != 0)
    assert model.joint.rna_encoder.encoder.value.weight.grad is None


def test_response_rejects_mismatched_cells_or_missing_modalities(example):
    model, inputs = example
    first = inputs["cells"][0]
    changed_key = replace(first.contacts.key, barcode="different")
    bad_cell = replace(first, contacts=replace(first.contacts, key=changed_key))
    with pytest.raises(ValueError, match="same cell"):
        model(**{**inputs, "cells": [bad_cell, *inputs["cells"][1:]]})
    missing_atac = inputs["atac"].clone()
    missing_atac[0] = 0
    missing_mask = inputs["atac_measured"].clone()
    missing_mask[0] = False
    with pytest.raises(ValueError, match="all three"):
        model(**{**inputs, "atac": missing_atac, "atac_measured": missing_mask})
    empty_row_contacts = torch.sparse_coo_tensor(
        torch.tensor([[1, 2], [1, 3]]), torch.tensor([3.0, 4]), (3, 4),
    )
    with pytest.raises(ValueError, match="observed contact"):
        model(**{**inputs, "contacts": empty_row_contacts})
    with pytest.raises(ValueError, match="intervention"):
        model(**{**inputs, "intervention_direction": torch.zeros(3)})


def test_population_loss_handles_unequal_cells_and_missing_values_without_pairing():
    predicted = torch.tensor([[10.0, 2, float("nan"), 1], [20, 4, float("nan"), 1]],
                             requires_grad=True)
    observed = torch.tensor([[100.0, 1, float("nan"), 3], [200, 5, float("nan"), 5],
                             [300, 3, float("nan"), 4]])
    control_mask = torch.tensor([[True, True, False, True], [True, True, False, True]])
    observed_mask = torch.tensor([[True, True, False, True], [True, True, False, True],
                                  [True, True, False, True]])
    loss = population_target_excluded_loss(
        predicted, observed, control_measured=control_mask, observed_measured=observed_mask, target_index=0,
    )
    assert loss.item() == pytest.approx(4.5)
    reordered = population_target_excluded_loss(
        predicted.flip(0), observed.flip(0), control_measured=control_mask.flip(0),
        observed_measured=observed_mask.flip(0), target_index=0,
    )
    assert torch.equal(loss, reordered)
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0, 3].item() == pytest.approx(-1.5)
    assert predicted.grad[:, 0].abs().sum().item() == 0
    assert predicted.grad[:, 2].abs().sum().item() == 0


def test_population_loss_rejects_empty_non_target_observations():
    values = torch.ones((1, 3))
    with pytest.raises(ValueError, match="non-target gene"):
        population_target_excluded_loss(
            values, values, control_measured=torch.tensor([[True, False, False]]),
            observed_measured=torch.tensor([[True, True, False]]), target_index=0,
        )
    with pytest.raises(ValueError, match="same measured gene panel"):
        population_target_excluded_loss(
            values, values, control_measured=torch.tensor([[True, True, False]]),
            observed_measured=torch.tensor([[True, False, True]]), target_index=0,
        )


def test_synthetic_population_objective_can_train_without_cell_pairing(example):
    model, inputs = example
    inputs["cells"] = [paired_cell("K562", barcode) for barcode in ("a", "b", "c")]
    inputs["rna_measured"] = torch.ones_like(inputs["rna"], dtype=torch.bool)
    inputs["target_indices"] = torch.zeros(3, dtype=torch.long)
    inputs["intervention_direction"] = -torch.ones(3)
    baseline = model(**inputs)["log_expression"].detach()
    observed = baseline.mean(dim=0).repeat(5, 1) + torch.tensor([0.0, 0.4, 0.6])
    observed_mask = torch.ones_like(observed, dtype=torch.bool)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.01)

    def objective():
        return population_target_excluded_loss(
            model(**inputs)["log_expression"], observed, control_measured=inputs["rna_measured"],
            observed_measured=observed_mask, target_index=0,
        )

    initial = objective().item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        optimizer.step()
    assert objective().item() < initial * 0.1

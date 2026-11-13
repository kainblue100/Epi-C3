from __future__ import annotations

import pytest
import torch

from epi_c3_gene_grn_v1 import target_excluded_loss


def test_target_excluded_loss_intersects_control_and_treated_measurements():
    predicted = torch.tensor([[1.0, 3.0, 200.0, 7.0], [2.0, 4.0, 0.0, 9.0]], requires_grad=True)
    observed = torch.tensor([[float("nan"), 1.0, float("nan"), 5.0], [3.0, 4.0, 4.0, float("nan")]])
    control_measured = torch.tensor([[True, True, True, False], [True, True, True, False]])
    observed_measured = torch.tensor([[False, True, False, True], [True, True, True, False]])
    loss = target_excluded_loss(
        predicted, observed, torch.tensor([0, 1]),
        control_measured=control_measured, observed_measured=observed_measured,
    )
    assert loss.item() == pytest.approx(7.0)
    loss.backward()
    assert torch.allclose(predicted.grad, torch.tensor([[0.0, 4 / 3, 0.0, 0.0], [-2 / 3, 0.0, -8 / 3, 0.0]]))


def test_target_excluded_loss_matches_fully_measured_behavior():
    predicted = torch.tensor([[1.0, 3.0, 6.0], [2.0, 4.0, 0.0]])
    observed = torch.tensor([[0.0, 1.0, 2.0], [3.0, 2.0, 4.0]])
    targets = torch.tensor([0, 1])
    measured = torch.ones_like(predicted, dtype=torch.bool)
    expected = (predicted - observed).square().scatter(1, targets[:, None], 0).mean() * 3 / 2
    result = target_excluded_loss(
        predicted, observed, targets, control_measured=measured, observed_measured=measured,
    )
    assert torch.allclose(result, expected)


def test_target_excluded_loss_rejects_empty_intersections_and_invalid_masks():
    predicted = torch.ones((2, 3))
    observed = torch.ones_like(predicted)
    targets = torch.tensor([0, 1])
    measured = torch.ones_like(predicted, dtype=torch.bool)
    missing = torch.tensor([[True, False, False], [True, True, True]])
    with pytest.raises(ValueError, match="non-target gene"):
        target_excluded_loss(
            predicted, observed, targets, control_measured=missing, observed_measured=measured,
        )
    with pytest.raises(ValueError, match="supervised loss inputs"):
        target_excluded_loss(
            predicted, observed, targets, control_measured=measured.float(), observed_measured=measured,
        )
    observed[0, 1] = float("nan")
    with pytest.raises(ValueError, match="supervised loss inputs"):
        target_excluded_loss(
            predicted, observed, targets, control_measured=measured, observed_measured=measured,
        )

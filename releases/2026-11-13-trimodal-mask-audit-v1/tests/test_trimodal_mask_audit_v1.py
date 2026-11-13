from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


V13 = Path(__file__).resolve().parents[2] / "2026-11-12-foundation-v13"
sys.path[:0] = [str(V13), str(V13 / "src")]

from c3vcell.model import C3VCell, ModelConfig
from epi_c3_shared_foundation_v1 import FORMAT, file_digest, gene_axis_digest, load_foundation
from epi_c3_trimodal_joint_v1 import TriModalJointEncoder, masked_rna_loss


class TrimodalMaskAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        self.first = torch.tensor([[2.0, 4.0, 8.0, 0.0]])
        self.changed = torch.tensor([[2.0, 4.0, 80.0, 0.0]])
        self.measured = torch.tensor([[True, True, True, False]])
        self.hidden = torch.tensor([[False, False, True, False]])
        self.visible = self.measured & ~self.hidden
        self.atac = torch.tensor([[1.0, 2.0]])
        self.atac_measured = torch.ones_like(self.atac, dtype=torch.bool)
        self.contacts = torch.sparse_coo_tensor(
            torch.tensor([[0], [1]]), torch.tensor([1.0]), (1, 3), check_invariants=True,
        ).coalesce()
        self.contact_present = torch.tensor([True])

    def joint(self, kind: str) -> TriModalJointEncoder:
        torch.manual_seed(731)
        if kind == "standin":
            model = TriModalJointEncoder(4, 2, 3, hidden_dim=8, latent_dim=6, allow_untrained_rna=True)
        else:
            base = C3VCell(ModelConfig(4, latent_dim=6, hidden_dim=8))
            genes = ["A", "B", "C", "D"]
            directory = tempfile.TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            path = Path(directory.name) / "synthetic-exploratory.pt"
            torch.save({
                "format": FORMAT, "status": "complete", "scope": "exploratory",
                "genes": genes, "gene_axis_sha256": gene_axis_digest(genes),
                "model_config": base.configuration(), "model_state": base.state_dict(),
            }, path)
            foundation = load_foundation(path, file_digest(path), genes, allow_exploratory=True)
            model = TriModalJointEncoder(
                4, 2, 3, hidden_dim=8, latent_dim=6, rna_encoder=foundation,
                allow_exploratory_foundation=True,
            )
        return model.eval()

    def encode(self, model: TriModalJointEncoder, counts: torch.Tensor, mask: torch.Tensor) -> dict:
        return model(
            counts, mask, self.atac, self.atac_measured, self.contacts, self.contact_present,
        )

    def test_loss_accepts_target_that_was_visible_to_the_joint_encoder(self) -> None:
        for kind in ("standin", "foundation"):
            with self.subTest(kind=kind):
                model = self.joint(kind)
                first = self.encode(model, self.first, self.measured)
                changed = self.encode(model, self.changed, self.measured)
                self.assertFalse(torch.allclose(first["fused"], changed["fused"]))
                self.assertFalse(torch.allclose(first["rna_reconstruction"], changed["rna_reconstruction"]))
                for result, target in ((first, self.first), (changed, self.changed)):
                    loss = masked_rna_loss(result["rna_reconstruction"], target, self.measured, self.hidden)
                    self.assertTrue(torch.isfinite(loss))

    def test_visible_only_candidate_hides_input_but_preserves_label_loss(self) -> None:
        unmeasured = ~self.measured
        self.assertEqual(int(self.visible.sum()), 2)
        self.assertEqual(int(self.hidden.sum()), 1)
        self.assertEqual(int(unmeasured.sum()), 1)
        self.assertFalse(torch.any(self.hidden & unmeasured))
        for kind in ("standin", "foundation"):
            with self.subTest(kind=kind):
                model = self.joint(kind)
                with self.assertRaises(ValueError):
                    self.encode(model, self.first, self.visible)
                raw = self.first.clone().requires_grad_()
                first = self.encode(model, raw.masked_fill(~self.visible, 0), self.visible)
                changed = self.encode(model, self.changed.masked_fill(~self.visible, 0), self.visible)
                self.assertTrue(torch.equal(first["fused"], changed["fused"]))
                self.assertTrue(torch.equal(first["rna_reconstruction"], changed["rna_reconstruction"]))
                gradient = torch.autograd.grad(
                    first["rna_reconstruction"][0, 2], raw, retain_graph=True,
                )[0]
                self.assertEqual(gradient[0, 2].item(), 0.0)
                self.assertEqual(gradient[0, 3].item(), 0.0)
                self.assertGreater(gradient[0, :2].abs().sum().item(), 0.0)

                first_target = torch.log1p(self.first / 24.0 * 10000.0)
                changed_target = torch.log1p(self.changed / 96.0 * 10000.0)
                first_loss = masked_rna_loss(
                    first["rna_reconstruction"], first_target.detach(), self.measured, self.hidden,
                )
                changed_loss = masked_rna_loss(
                    changed["rna_reconstruction"], changed_target.detach(), self.measured, self.hidden,
                )
                self.assertTrue(torch.isfinite(first_loss))
                self.assertTrue(torch.isfinite(changed_loss))
                self.assertNotEqual(first_loss.item(), changed_loss.item())
                first_loss.backward()
                self.assertGreater(model.rna_decoder.weight.grad.abs().sum().item(), 0.0)

    def test_visibility_mask_alone_leaks_through_full_library_denominator(self) -> None:
        torch.manual_seed(731)
        encoder = C3VCell(ModelConfig(4, latent_dim=6, hidden_dim=8)).encoder.eval()
        mask = self.visible.to(self.first.dtype)
        first = encoder(self.first, mask)
        changed = encoder(self.changed, mask)
        self.assertFalse(torch.allclose(first, changed))
        first_hidden = encoder(self.first.masked_fill(~self.visible, 0), mask)
        changed_hidden = encoder(self.changed.masked_fill(~self.visible, 0), mask)
        self.assertTrue(torch.equal(first_hidden, changed_hidden))


if __name__ == "__main__":
    unittest.main()

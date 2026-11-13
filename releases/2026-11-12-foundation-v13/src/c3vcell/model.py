"""A small count-aware JEPA and conditional population-flow model."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_genes: int
    latent_dim: int = 128
    hidden_dim: int = 256
    mask_probability: float = 0.2
    teacher_decay: float = 0.99
    variance_weight: float = 0.1
    covariance_weight: float = 0.0
    regularization_view: Literal["masked", "unmasked"] = "masked"
    epi_feature_dim: int = 0
    epi_graph_propagation: bool = False

    def __post_init__(self) -> None:
        if self.n_genes < 1 or self.latent_dim < 1 or self.hidden_dim < 1:
            raise ValueError("Model dimensions must be positive")
        if not 0 <= self.mask_probability < 1:
            raise ValueError("mask_probability must be in [0, 1)")
        if not 0 <= self.teacher_decay < 1:
            raise ValueError("teacher_decay must be in [0, 1)")
        if not math.isfinite(self.variance_weight) or self.variance_weight < 0:
            raise ValueError("variance_weight must be finite and nonnegative")
        if not math.isfinite(self.covariance_weight) or self.covariance_weight < 0:
            raise ValueError("covariance_weight must be finite and nonnegative")
        if self.regularization_view not in {"masked", "unmasked"}:
            raise ValueError("regularization_view must be masked or unmasked")
        if self.epi_feature_dim < 0:
            raise ValueError("epi_feature_dim must be nonnegative")
        if self.epi_graph_propagation and not self.epi_feature_dim:
            raise ValueError("Epigenomic graph propagation requires epigenomic features")


class GeneEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.value = nn.Linear(config.n_genes, config.latent_dim)
        self.visible = nn.Linear(config.n_genes, config.latent_dim, bias=False)
        self.body = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )

    def forward(self, counts: Tensor, visible: Tensor) -> Tensor:
        library = counts.sum(dim=-1, keepdim=True)
        if torch.any(library <= 0):
            raise ValueError("Every cell must have at least one measured count")
        normalized = torch.log1p(counts / library * 10_000.0)
        encoded = self.value(normalized * visible) + self.visible(visible)
        return self.body(encoded)


def negative_binomial_loss(
    counts: Tensor, mean: Tensor, theta: Tensor, measured: Tensor | None = None
) -> Tensor:
    """Mean negative log likelihood over measured genes and cells."""
    theta = theta.clamp(min=1e-4, max=1e4)
    mean = mean.clamp(min=1e-8)
    total = theta + mean
    log_prob = (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1)
        + theta * (torch.log(theta) - torch.log(total))
        + counts * (torch.log(mean) - torch.log(total))
    )
    if measured is None:
        return -log_prob.mean()
    if measured.shape != counts.shape or not torch.any(measured):
        raise ValueError("A nonempty measurement mask must match the count matrix")
    return -(log_prob * measured).sum() / measured.sum()


class C3VCell(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = GeneEncoder(config)
        self.teacher = copy.deepcopy(self.encoder)
        self.teacher.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.n_genes),
        )
        self.log_theta = nn.Parameter(torch.zeros(config.n_genes))
        self.velocity = nn.Sequential(
            nn.Linear(config.latent_dim * 4 + 1, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.epi_adapter = None
        if config.epi_feature_dim:
            self.register_buffer("epi_values", torch.zeros(config.n_genes, config.epi_feature_dim))
            self.register_buffer("epi_masks", torch.zeros(config.n_genes, config.epi_feature_dim))
            with torch.random.fork_rng(devices=[]):
                self.epi_adapter = nn.Sequential(
                    nn.Linear(config.epi_feature_dim * 2, config.latent_dim),
                    nn.GELU(),
                    nn.Linear(config.latent_dim, config.latent_dim),
                )
                nn.init.zeros_(self.epi_adapter[-1].weight)
                nn.init.zeros_(self.epi_adapter[-1].bias)

    def set_epigenetic_features(self, values: Tensor, masks: Tensor) -> None:
        shape = (self.config.n_genes, self.config.epi_feature_dim)
        if (
            self.epi_adapter is None or values.shape != shape or masks.shape != shape
            or not torch.isfinite(values).all() or not torch.isfinite(masks).all()
            or torch.any((values < 0) | (values > 1))
            or torch.any((masks != 0) & (masks != 1))
            or torch.any(values[masks == 0] != 0)
        ):
            raise ValueError("Epigenomic features and binary availability masks are invalid")
        self.epi_values.copy_(values.to(device=self.epi_values.device, dtype=self.epi_values.dtype))
        self.epi_masks.copy_(masks.to(device=self.epi_masks.device, dtype=self.epi_masks.dtype))

    def configuration(self) -> dict:
        return asdict(self.config)

    def encode(self, counts: Tensor, measured: Tensor | None = None) -> Tensor:
        if measured is None:
            measured = torch.ones_like(counts)
        if measured.shape != counts.shape:
            raise ValueError("Measurement mask must match the count matrix")
        return self.encoder(counts * measured, measured)

    def probabilities(self, latent: Tensor, measured: Tensor | None = None) -> Tensor:
        logits = self.decoder(latent)
        if measured is not None:
            if measured.shape != logits.shape or torch.any(measured.sum(dim=-1) == 0):
                raise ValueError("Each cell needs at least one measured model gene")
            logits = logits.masked_fill(measured == 0, float("-inf"))
        return torch.softmax(logits, dim=-1)

    def representation_loss(
        self, counts: Tensor, measured: Tensor | None = None
    ) -> tuple[Tensor, dict[str, float]]:
        if counts.ndim != 2 or counts.shape[1] != self.config.n_genes:
            raise ValueError("Counts must have shape (cells, model genes)")
        if measured is None:
            measured = torch.ones_like(counts)
        if measured.shape != counts.shape or torch.any(measured.sum(dim=-1) == 0):
            raise ValueError("Each cell needs a nonempty measurement mask")
        counts = counts * measured
        visible_a = (torch.rand_like(counts) >= self.config.mask_probability).to(counts.dtype) * measured
        visible_b = (torch.rand_like(counts) >= self.config.mask_probability).to(counts.dtype) * measured
        latent = self.encoder(counts, visible_a)
        with torch.no_grad():
            target = self.teacher(counts, visible_b)
        prediction = self.predictor(latent)
        jepa = F.mse_loss(prediction, target)
        regularized = (
            latent if self.config.regularization_view == "masked"
            else self.encoder(counts, measured)
        )
        std = torch.sqrt(regularized.var(dim=0, unbiased=False) + 1e-4)
        variance = F.relu(1.0 - std).mean()
        covariance = regularized.new_zeros(())
        if self.config.covariance_weight > 0:
            if len(regularized) < 2:
                raise ValueError("Covariance regularization requires at least two cells per batch")
            centered = regularized - regularized.mean(dim=0)
            matrix = centered.T @ centered / (len(regularized) - 1)
            off_diagonal = matrix - torch.diag_embed(torch.diagonal(matrix))
            covariance = off_diagonal.square().sum() / regularized.shape[1]
        mean = self.probabilities(latent, measured) * counts.sum(dim=-1, keepdim=True)
        theta = F.softplus(self.log_theta) + 1e-4
        reconstruction = negative_binomial_loss(counts, mean, theta, measured)
        loss = (
            reconstruction + 0.1 * jepa + self.config.variance_weight * variance
            + self.config.covariance_weight * covariance
        )
        return loss, {
            "loss": float(loss.detach()),
            "nb": float(reconstruction.detach()),
            "jepa": float(jepa.detach()),
            "variance": float(variance.detach()),
            "covariance": float(covariance.detach()),
        }

    @torch.no_grad()
    def update_teacher(self) -> None:
        decay = self.config.teacher_decay
        for teacher, student in zip(self.teacher.parameters(), self.encoder.parameters()):
            teacher.mul_(decay).add_(student, alpha=1.0 - decay)
        for teacher, student in zip(self.teacher.buffers(), self.encoder.buffers()):
            teacher.copy_(student)

    def target_embedding(
        self, target_index: Tensor, graph: Tensor | None = None, *, detach_gene_features: bool = False,
    ) -> Tensor:
        if target_index.ndim != 1:
            raise ValueError("target_index must be a vector")
        if torch.any(target_index < 0) or torch.any(target_index >= self.config.n_genes):
            raise ValueError("Target gene is missing from the model gene axis")
        features = self.encoder.value.weight.transpose(0, 1)
        if detach_gene_features:
            features = features.detach()
        if self.epi_adapter is not None and self.config.epi_graph_propagation:
            availability = (self.epi_masks.sum(dim=-1, keepdim=True) > 0).to(features.dtype)
            encoded = self.epi_adapter(torch.cat((self.epi_values, self.epi_masks), dim=-1))
            features = features + availability * encoded
        if graph is not None:
            if not graph.is_sparse or graph.shape != (self.config.n_genes, self.config.n_genes):
                raise ValueError("Gene graph must be a sparse square matrix on the model gene axis")
            features = torch.sparse.mm(graph, features)
        selected = features[target_index]
        if self.epi_adapter is not None and not self.config.epi_graph_propagation:
            values = self.epi_values[target_index]
            masks = self.epi_masks[target_index]
            availability = (masks.sum(dim=-1, keepdim=True) > 0).to(selected.dtype)
            selected = selected + availability * self.epi_adapter(torch.cat((values, masks), dim=-1))
        return F.normalize(selected, dim=-1)

    def flow(
        self, latent: Tensor, time: Tensor, origin: Tensor, target: Tensor, context: Tensor
    ) -> Tensor:
        if time.ndim == 1:
            time = time[:, None]
        return self.velocity(torch.cat((latent, origin, target, context, time), dim=-1))

    def flow_loss(
        self, control: Tensor, perturbed: Tensor, target_index: Tensor, graph: Tensor | None = None
    ) -> tuple[Tensor, dict[str, float]]:
        if control.shape != perturbed.shape or control.shape[0] != target_index.shape[0]:
            raise ValueError("Flow batches must contain matched numbers of cells and targets")
        embedding = self.target_embedding(target_index, graph, detach_gene_features=True)
        return self.flow_loss_condition(control, perturbed, embedding)

    def flow_loss_condition(
        self, control: Tensor, perturbed: Tensor, embedding: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        if control.shape != perturbed.shape or embedding.shape != (
            control.shape[0], self.config.latent_dim
        ):
            raise ValueError("Flow batches must contain matched cells and condition embeddings")
        with torch.no_grad():
            origin = self.encode(control)
            destination = self.encode(perturbed)
        return self._flow_objective(origin, destination, embedding)

    def flow_loss_shift(
        self, control: Tensor, shift: Tensor, target_index: Tensor,
        graph: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        if (
            shift.shape != (control.shape[0], self.config.latent_dim)
            or target_index.shape != (control.shape[0],)
        ):
            raise ValueError("Population shifts and targets must match the cell batch")
        with torch.no_grad():
            origin = self.encode(control)
        embedding = self.target_embedding(target_index, graph, detach_gene_features=True)
        return self._flow_objective(origin, origin + shift, embedding)

    def _flow_objective(
        self, origin: Tensor, destination: Tensor, embedding: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        time = torch.rand(origin.shape[0], 1, device=origin.device)
        intermediate = (1 - time) * origin + time * destination
        predicted = self.flow(intermediate, time, origin, embedding, origin)
        matching = F.mse_loss(predicted, destination - origin)
        null_velocity = self.flow(origin, time, origin, torch.zeros_like(embedding), origin)
        identity = null_velocity.square().mean()
        loss = matching + 0.1 * identity
        return loss, {
            "loss": float(loss.detach()),
            "flow": float(matching.detach()),
            "identity": float(identity.detach()),
        }

    @torch.no_grad()
    def predict_counts(
        self,
        control: Tensor,
        target_index: Tensor,
        *,
        steps: int = 16,
        graph: Tensor | None = None,
        measured: Tensor | None = None,
        sampling: Literal["negative-binomial", "residual-rounding"] = "negative-binomial",
        effect_strength: float = 1.0,
    ) -> Tensor:
        target = self.target_embedding(target_index, graph)
        return self.predict_counts_condition(
            control, target, steps=steps, measured=measured, sampling=sampling,
            effect_strength=effect_strength,
        )

    @torch.no_grad()
    def predict_counts_condition(
        self,
        control: Tensor,
        target: Tensor,
        *,
        steps: int = 16,
        measured: Tensor | None = None,
        sampling: Literal["negative-binomial", "residual-rounding"] = "negative-binomial",
        effect_strength: float = 1.0,
    ) -> Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        if not math.isfinite(effect_strength) or not 0 <= effect_strength <= 1:
            raise ValueError("effect_strength must be finite and between zero and one")
        if target.shape != (control.shape[0], self.config.latent_dim):
            raise ValueError("Condition embeddings must match the cell batch and latent dimension")
        if measured is None:
            measured = torch.ones_like(control)
        control = control * measured
        origin = self.encode(control, measured)
        latent = origin.clone()
        for step in range(steps):
            time = torch.full((len(control), 1), step / steps, device=control.device)
            latent += self.flow(latent, time, origin, target, origin) / steps
        before = self.probabilities(origin, measured)
        after = self.probabilities(latent, measured)
        log_fold = (
            (torch.log(after + 1e-8) - torch.log(before + 1e-8)).clamp(-2, 2)
            * effect_strength
        )
        library = control.sum(dim=-1, keepdim=True)
        if sampling == "negative-binomial":
            baseline = 0.9 * control + 0.1 * before * library
            mean = baseline * torch.exp(log_fold)
            mean = mean / mean.sum(dim=-1, keepdim=True).clamp(min=1e-8) * library
            theta = (F.softplus(self.log_theta) + 1e-4).expand_as(mean)
            probs = (mean / (mean + theta)).clamp(min=0, max=1 - 1e-6)
            result = torch.distributions.NegativeBinomial(total_count=theta, probs=probs).sample()
        elif sampling == "residual-rounding":
            mean = control * torch.exp(log_fold)
            mean = mean * (library / mean.sum(dim=-1, keepdim=True).clamp(min=1e-8))
            integer = torch.floor(mean)
            result = integer + torch.bernoulli((mean - integer).clamp(0, 1))
        else:
            raise ValueError(f"Unknown count sampling strategy: {sampling!r}")
        if not torch.isfinite(result).all() or torch.any(result.sum(dim=-1) > 1_000_000):
            raise ValueError("Generated counts are invalid or exceed 1,000,000 per cell")
        return result.to(torch.int32)

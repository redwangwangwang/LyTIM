"""Clinical energy, trust gating, and stopping components for LyTIM.

The module is intentionally independent from the large vision/language backbones so it
can be unit-tested on CPU.  It consumes frozen TIM representations and supplies the
trainable control signals used by ``models/model_lytim.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_DISEASES = 14
NUM_STATES = 4
NUM_TRANSITIONS = 4

# Per-disease transition ids.
TRANSITION_NEW = 0
TRANSITION_RESOLVED = 1
TRANSITION_PERSISTENT = 2
TRANSITION_STABLE_NEGATIVE = 3


@dataclass
class EnergyBreakdown:
    """Per-sample clinical surrogate energy and its components."""

    total: torch.Tensor
    state: torch.Tensor
    change: torch.Tensor
    backward: torch.Tensor
    fact_errors: torch.Tensor


def _safe_probabilities(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = probabilities.clamp_min(eps)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)


def jensen_shannon_divergence(
    first: torch.Tensor,
    second: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return Jensen-Shannon divergence along the last dimension.

    The returned tensor has all leading dimensions preserved.  Natural logarithms are
    used, so the maximum divergence is ``log(2)``.
    """

    first = _safe_probabilities(first, eps)
    second = _safe_probabilities(second, eps)
    midpoint = 0.5 * (first + second)
    first_kl = (first * (first.log() - midpoint.log())).sum(dim=-1)
    second_kl = (second * (second.log() - midpoint.log())).sum(dim=-1)
    return 0.5 * (first_kl + second_kl)


def masked_mean(
    values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean that remains finite when every entry is masked."""

    if mask is None:
        return values.mean(dim=dim)
    mask = mask.to(device=values.device, dtype=values.dtype)
    numerator = (values * mask).sum(dim=dim)
    denominator = mask.sum(dim=dim).clamp_min(eps)
    return numerator / denominator



def monotonic_acceptance_mask(
    current_energy: torch.Tensor,
    candidate_energy: torch.Tensor,
    epsilon: float,
    active: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the per-sample LyTIM accept decision.

    A strict decrease of at least ``epsilon`` is required.  This small helper makes
    the inference invariant explicit and independently testable.
    """

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    if current_energy.shape != candidate_energy.shape:
        raise ValueError("Energy tensors must have identical shapes.")
    accept = candidate_energy.lt(current_energy - float(epsilon))
    if active is not None:
        if active.shape != accept.shape:
            raise ValueError("active mask must match the energy shape.")
        accept = accept & active.bool()
    return accept

def labels_to_probabilities(labels: torch.Tensor, num_states: int = NUM_STATES) -> torch.Tensor:
    """Convert integer CheXbert labels to one-hot probabilities."""

    labels = labels.long().clamp(min=0, max=num_states - 1)
    return F.one_hot(labels, num_classes=num_states).to(dtype=torch.float32)


def transition_targets_from_labels(
    previous_labels: torch.Tensor,
    current_labels: torch.Tensor,
    positive_index: int = 1,
    negative_index: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive temporal transition labels without changing the dataset.

    Blank/not-mentioned and uncertain labels are masked for the 13 pathology heads.
    The CheXbert ``no finding`` head is binary, so both 0 and 1 are treated as known
    states for the final disease slot.
    """

    if previous_labels.shape != current_labels.shape:
        raise ValueError("Previous and current label tensors must have identical shapes.")
    if previous_labels.ndim != 2:
        raise ValueError("Expected labels with shape [batch, diseases].")

    previous_labels = previous_labels.long()
    current_labels = current_labels.long()
    previous_positive = previous_labels.eq(positive_index)
    current_positive = current_labels.eq(positive_index)

    previous_known = previous_labels.eq(positive_index) | previous_labels.eq(negative_index)
    current_known = current_labels.eq(positive_index) | current_labels.eq(negative_index)

    # CheXbert uses a binary head for "no finding".  Preserve supervision for it.
    if previous_labels.shape[1] >= NUM_DISEASES:
        previous_known = previous_known.clone()
        current_known = current_known.clone()
        previous_known[:, -1] = previous_labels[:, -1].le(1)
        current_known[:, -1] = current_labels[:, -1].le(1)

    valid = previous_known & current_known
    targets = torch.full_like(previous_labels, fill_value=-100)
    targets[valid & ~previous_positive & current_positive] = TRANSITION_NEW
    targets[valid & previous_positive & ~current_positive] = TRANSITION_RESOLVED
    targets[valid & previous_positive & current_positive] = TRANSITION_PERSISTENT
    targets[valid & ~previous_positive & ~current_positive] = TRANSITION_STABLE_NEGATIVE
    return targets, valid


def safe_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross entropy that returns a differentiable zero for an empty valid set."""

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1).long()
    valid = flat_targets.ne(ignore_index)
    if not bool(valid.any()):
        return flat_logits.sum() * 0.0
    return F.cross_entropy(flat_logits[valid], flat_targets[valid])


class LyTIMController(nn.Module):
    """Trainable clinical controller placed on top of frozen TIM representations."""

    def __init__(
        self,
        hidden_size: int,
        controller_dim: int = 512,
        num_diseases: int = NUM_DISEASES,
        num_states: int = NUM_STATES,
        num_prompt_tokens: int = 16,
        state_weight: float = 1.0,
        change_weight: float = 1.0,
        backward_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or controller_dim <= 0:
            raise ValueError("hidden_size and controller_dim must be positive.")
        if num_diseases <= 0 or num_states <= 1 or num_prompt_tokens <= 0:
            raise ValueError("Invalid controller dimensions.")

        self.hidden_size = hidden_size
        self.controller_dim = controller_dim
        self.num_diseases = num_diseases
        self.num_states = num_states
        self.num_prompt_tokens = num_prompt_tokens
        self.state_weight = float(state_weight)
        self.change_weight = float(change_weight)
        self.backward_weight = float(backward_weight)

        self.image_reduce = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, controller_dim),
            nn.GELU(),
            nn.LayerNorm(controller_dim),
        )
        self.report_reduce = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, controller_dim),
            nn.GELU(),
            nn.LayerNorm(controller_dim),
        )

        self.image_state_head = nn.Linear(controller_dim, num_diseases * num_states)
        self.report_state_head = nn.Linear(controller_dim, num_diseases * num_states)

        self.progression_encoder = nn.Sequential(
            nn.Linear(controller_dim * 4, controller_dim),
            nn.GELU(),
            nn.LayerNorm(controller_dim),
            nn.Dropout(0.1),
        )
        self.progression_head = nn.Linear(controller_dim, num_diseases * NUM_TRANSITIONS)

        self.backward_head = nn.Sequential(
            nn.Linear(controller_dim * 2, controller_dim),
            nn.GELU(),
            nn.LayerNorm(controller_dim),
            nn.Linear(controller_dim, num_diseases * num_states),
        )

        self.fact_gate_head = nn.Sequential(
            nn.Linear(controller_dim + num_diseases, controller_dim),
            nn.GELU(),
            nn.Linear(controller_dim, num_diseases),
        )
        self.prompt_gate_head = nn.Linear(num_diseases, num_prompt_tokens)
        self.fact_embeddings = nn.Parameter(torch.empty(num_diseases, hidden_size))
        self.prompt_fact_queries = nn.Parameter(torch.empty(num_prompt_tokens, num_diseases))

        # current energy, candidate energy, improvement, descriptor change,
        # mean fact error, and maximum fact error.
        self.stop_head = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        nn.init.normal_(self.fact_embeddings, std=0.02)
        nn.init.normal_(self.prompt_fact_queries, std=0.02)

    @staticmethod
    def _pool(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden_states.ndim == 2:
            return hidden_states
        if hidden_states.ndim != 3:
            raise ValueError("Expected hidden states with shape [B, H] or [B, T, H].")
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.to(hidden_states.device, hidden_states.dtype).unsqueeze(-1)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def encode_images(self, previous: torch.Tensor, current: torch.Tensor) -> dict[str, torch.Tensor]:
        previous_vector = self.image_reduce(self._pool(previous))
        current_vector = self.image_reduce(self._pool(current))
        progression_input = torch.cat(
            [
                previous_vector,
                current_vector,
                torch.abs(current_vector - previous_vector),
                current_vector * previous_vector,
            ],
            dim=-1,
        )
        progression_vector = self.progression_encoder(progression_input)
        image_state_logits = self.image_state_head(current_vector).view(
            -1, self.num_diseases, self.num_states
        )
        progression_logits = self.progression_head(progression_vector).view(
            -1, self.num_diseases, NUM_TRANSITIONS
        )
        return {
            "previous_vector": previous_vector,
            "current_vector": current_vector,
            "progression_vector": progression_vector,
            "image_state_logits": image_state_logits,
            "progression_logits": progression_logits,
        }

    def encode_report(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        report_vector = self.report_reduce(self._pool(hidden_states, attention_mask))
        report_state_logits = self.report_state_head(report_vector).view(
            -1, self.num_diseases, self.num_states
        )
        return report_vector, report_state_logits

    def predict_backward_state(
        self,
        report_vector: torch.Tensor,
        progression_vector: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.backward_head(torch.cat([report_vector, progression_vector], dim=-1))
        return logits.view(-1, self.num_diseases, self.num_states)

    def compute_energy(
        self,
        report_probabilities: torch.Tensor,
        image_probabilities: torch.Tensor,
        progression_probabilities: torch.Tensor,
        previous_probabilities: torch.Tensor,
        backward_probabilities: Optional[torch.Tensor] = None,
        transition_mask: Optional[torch.Tensor] = None,
    ) -> EnergyBreakdown:
        """Compute the non-negative multi-view clinical inconsistency energy."""

        report_probabilities = _safe_probabilities(report_probabilities)
        image_probabilities = _safe_probabilities(image_probabilities)
        previous_probabilities = _safe_probabilities(previous_probabilities)
        progression_probabilities = _safe_probabilities(progression_probabilities)

        state_fact = jensen_shannon_divergence(report_probabilities, image_probabilities)

        previous_positive = previous_probabilities[..., 1]
        current_positive = report_probabilities[..., 1]
        implied_previous = (
            progression_probabilities[..., TRANSITION_RESOLVED]
            + progression_probabilities[..., TRANSITION_PERSISTENT]
        )
        implied_current = (
            progression_probabilities[..., TRANSITION_NEW]
            + progression_probabilities[..., TRANSITION_PERSISTENT]
        )
        change_fact = 0.5 * (
            torch.abs(previous_positive - implied_previous)
            + torch.abs(current_positive - implied_current)
        )
        if transition_mask is not None:
            change_fact_for_total = change_fact * transition_mask.to(change_fact.dtype)
        else:
            change_fact_for_total = change_fact

        if backward_probabilities is None:
            backward_fact = torch.abs(previous_positive - implied_previous)
        else:
            backward_probabilities = _safe_probabilities(backward_probabilities)
            backward_fact = jensen_shannon_divergence(
                backward_probabilities,
                previous_probabilities,
            )

        state_energy = state_fact.mean(dim=-1)
        change_energy = masked_mean(change_fact, transition_mask, dim=-1)
        backward_energy = backward_fact.mean(dim=-1)
        total = (
            self.state_weight * state_energy
            + self.change_weight * change_energy
            + self.backward_weight * backward_energy
        )
        fact_errors = (
            self.state_weight * state_fact
            + self.change_weight * change_fact_for_total
            + self.backward_weight * backward_fact
        )
        return EnergyBreakdown(
            total=total,
            state=state_energy,
            change=change_energy,
            backward=backward_energy,
            fact_errors=fact_errors,
        )

    def gate_refinement_prompt(
        self,
        difference_prompt: torch.Tensor,
        fact_errors: torch.Tensor,
        report_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a fact-level trust gate to TIM's error prompt embeddings."""

        if difference_prompt.ndim != 3:
            raise ValueError("difference_prompt must have shape [B, prompt_tokens, hidden].")
        if difference_prompt.shape[1] != self.num_prompt_tokens:
            raise ValueError(
                f"Expected {self.num_prompt_tokens} prompt tokens, got {difference_prompt.shape[1]}."
            )

        normalized_errors = fact_errors / fact_errors.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        fact_gate = torch.sigmoid(
            self.fact_gate_head(torch.cat([report_vector, normalized_errors], dim=-1))
        )
        weighted_facts = fact_gate * normalized_errors
        prompt_gate = torch.sigmoid(self.prompt_gate_head(weighted_facts)).unsqueeze(-1)

        disease_context = weighted_facts.unsqueeze(-1) * self.fact_embeddings.unsqueeze(0)
        query_weights = torch.softmax(self.prompt_fact_queries, dim=-1)
        fact_context = torch.einsum("pd,bdh->bph", query_weights, disease_context)

        # The 0.25 residual prevents complete information erasure early in training.
        gated_prompt = difference_prompt * (0.25 + prompt_gate) + fact_context
        return gated_prompt, fact_gate

    def stop_logits(
        self,
        current_energy: torch.Tensor,
        candidate_energy: torch.Tensor,
        descriptor_change: torch.Tensor,
        candidate_fact_errors: torch.Tensor,
    ) -> torch.Tensor:
        improvement = current_energy - candidate_energy
        features = torch.stack(
            [
                current_energy,
                candidate_energy,
                improvement,
                descriptor_change,
                candidate_fact_errors.mean(dim=-1),
                candidate_fact_errors.max(dim=-1).values,
            ],
            dim=-1,
        )
        return self.stop_head(features).squeeze(-1)

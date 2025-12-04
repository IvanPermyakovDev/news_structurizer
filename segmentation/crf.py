"""
Minimal CRF layer for sequence labeling (adapted for chunk boundary tagging).
Source: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html with minor tweaks.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        if reduction not in ("none", "sum", "mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        if emissions.dim() != 3:
            raise ValueError("emissions must have dimension of (seq_length, batch_size, num_tags) or (batch_size, seq_length, num_tags)")
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        log_denominator = self._compute_log_partition_function(emissions, mask)
        log_numerator = self._compute_joint_likelihood(emissions, tags, mask)
        loss = log_denominator - log_numerator
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()
        return loss

    def _compute_joint_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_length, batch_size = tags.shape
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            transition_score = self.transitions[tags[i - 1], tags[i]]
            emission_score = emissions[i, torch.arange(batch_size), tags[i]]
            score += transition_score * mask[i] + emission_score * mask[i]
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_log_partition_function(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_length, batch_size, num_tags = emissions.shape
        alphas = self.start_transitions + emissions[0]
        for i in range(1, seq_length):
            broadcast_alphas = alphas.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            inner = broadcast_alphas + self.transitions + broadcast_emissions
            new_alphas = torch.logsumexp(inner, dim=1)
            alphas = torch.where(mask[i].unsqueeze(1), new_alphas, alphas)
        alphas += self.end_transitions
        return torch.logsumexp(alphas, dim=1)

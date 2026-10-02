"""Tests for the relevance-weight regularizers in :mod:`shotcloud.legacy_pivot.regularizers`."""

from __future__ import annotations

import math

import torch

from shotcloud.legacy_pivot.regularizers import (
    entropy_regularizer,
    ess_regularizer,
    fixed_recency_prior,
    prior_anchor_regularizer,
)

# ---------------------------------------------------------------------------
# entropy_regularizer
# ---------------------------------------------------------------------------


def test_entropy_uniform_is_minus_log_n() -> None:
    """For uniform π over N positions, returned value is -log(N)."""
    n = 10
    pi = torch.full((1, n), 1.0 / n)
    val = entropy_regularizer(pi)
    assert abs(float(val) - (-math.log(n))) < 1e-5


def test_entropy_one_hot_is_zero() -> None:
    pi = torch.zeros(1, 5)
    pi[0, 2] = 1.0
    val = entropy_regularizer(pi)
    # -H = -0 = 0
    assert abs(float(val)) < 1e-5


def test_entropy_gradient_pushes_toward_uniform() -> None:
    """Optimizing -H (i.e., maximizing H) flattens π."""
    pi_logits = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
    pi = torch.softmax(pi_logits, dim=-1)
    loss = entropy_regularizer(pi)  # we add this to the loss with positive λ
    loss.backward()
    # Logit 0 (currently dominant) should have a gradient that decreases it.
    assert pi_logits.grad[0, 0] > 0  # increasing the dominant logit increases the loss


# ---------------------------------------------------------------------------
# ess_regularizer
# ---------------------------------------------------------------------------


def test_ess_uniform_equals_one_over_n() -> None:
    n = 8
    pi = torch.full((1, n), 1.0 / n)
    # 1/N_eff = Σπ² = N · (1/N)² = 1/N
    assert abs(float(ess_regularizer(pi)) - 1.0 / n) < 1e-5


def test_ess_one_hot_is_one() -> None:
    pi = torch.zeros(1, 5)
    pi[0, 2] = 1.0
    assert abs(float(ess_regularizer(pi)) - 1.0) < 1e-5


def test_ess_gradient_pushes_toward_uniform() -> None:
    pi_logits = torch.tensor([[3.0, 0.0, 0.0]], requires_grad=True)
    pi = torch.softmax(pi_logits, dim=-1)
    loss = ess_regularizer(pi)
    loss.backward()
    # Increasing the dominant logit further would concentrate π → larger Σπ² → larger loss.
    assert pi_logits.grad[0, 0] > 0


# ---------------------------------------------------------------------------
# prior_anchor_regularizer
# ---------------------------------------------------------------------------


def test_prior_anchor_zero_when_pi_equals_prior() -> None:
    pi = torch.tensor([[0.25, 0.5, 0.25]])
    pi_prior = torch.tensor([[0.25, 0.5, 0.25]])
    assert abs(float(prior_anchor_regularizer(pi, pi_prior))) < 1e-9


def test_prior_anchor_positive_when_pi_differs() -> None:
    pi = torch.tensor([[0.8, 0.1, 0.1]])
    pi_prior = torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    assert prior_anchor_regularizer(pi, pi_prior) > 0


def test_prior_anchor_respects_mask() -> None:
    """Padded positions don't contribute even if π and prior differ there."""
    pi = torch.tensor([[0.5, 0.5, 0.0]])
    pi_prior = torch.tensor([[0.5, 0.5, 1e-12]])  # padded prior arbitrary
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    val = float(prior_anchor_regularizer(pi, pi_prior, mask=mask))
    assert abs(val) < 1e-9


def test_prior_anchor_shape_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="pi"):
        prior_anchor_regularizer(torch.zeros(1, 3), torch.zeros(1, 4))


# ---------------------------------------------------------------------------
# fixed_recency_prior
# ---------------------------------------------------------------------------


def test_fixed_recency_prior_decays_with_games_ago() -> None:
    games_ago = torch.tensor([[0.0, 1.0, 2.0]])
    pi0 = fixed_recency_prior(games_ago, lam=1.0)
    assert pi0[0, 0] > pi0[0, 1] > pi0[0, 2]
    assert abs(float(pi0.sum()) - 1.0) < 1e-6


def test_fixed_recency_prior_lam_zero_is_uniform() -> None:
    games_ago = torch.tensor([[0.0, 5.0, 10.0]])
    pi0 = fixed_recency_prior(games_ago, lam=0.0)
    assert torch.allclose(pi0, torch.full_like(pi0, 1.0 / 3))


def test_fixed_recency_prior_respects_mask() -> None:
    games_ago = torch.tensor([[0.0, 1.0, 2.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    pi0 = fixed_recency_prior(games_ago, mask=mask, lam=1.0)
    assert pi0[0, 2] == 0.0
    assert abs(float(pi0[0, :2].sum()) - 1.0) < 1e-6

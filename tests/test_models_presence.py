"""Tests for :class:`shotcloud.models.PresenceModel`, the causal on-court presence model."""

from __future__ import annotations

import pytest
import torch

from shotcloud.models.presence import (
    PRESENCE_N_BINS,
    PRESENCE_N_POSITIONS,
    PRESENCE_N_STARTER,
    PresenceModel,
)


def _make_inputs(batch: int = 4, k_max: int = 5) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    prior_bins = torch.rand(batch, k_max, PRESENCE_N_BINS)
    prior_ages_days = torch.rand(batch, k_max) * 60.0
    prior_mask = (torch.rand(batch, k_max) > 0.3).to(torch.float32)
    position_idx = torch.randint(0, PRESENCE_N_POSITIONS, (batch,))
    starter_idx = torch.randint(0, PRESENCE_N_STARTER, (batch,))
    history_count = prior_mask.sum(dim=-1)
    return {
        "prior_bins": prior_bins,
        "prior_ages_days": prior_ages_days,
        "prior_mask": prior_mask,
        "position_idx": position_idx,
        "starter_idx": starter_idx,
        "history_count": history_count,
    }


def test_output_shape_and_range() -> None:
    m = PresenceModel()
    out = m(**_make_inputs())
    assert out.shape == (4, PRESENCE_N_BINS)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_parameter_count_is_small() -> None:
    """The parameters are ``ρ̃``, ``b0``, ``β̃_h`` and an
    (n_positions × n_starter × n_bins) pool table: 3 + 3·2·30 = 183 scalars.
    """
    m = PresenceModel()
    total = sum(p.numel() for p in m.parameters())
    expected = 3 + PRESENCE_N_POSITIONS * PRESENCE_N_STARTER * PRESENCE_N_BINS
    assert total == expected


def test_cold_start_returns_pool() -> None:
    """With ``history_count == 0`` the gate is :math:`\\sigma(b_0)` and the self
    curve is all zeros, so the output is :math:`(1-\\sigma(b_0))\\cdot q_\\mathrm{pool}`.

    Initialization uses :math:`b_0=-1`, so :math:`\\sigma(-1) \\approx 0.27`
    and the cold-start output is :math:`0.73 \\cdot q_\\mathrm{pool}`.
    """
    m = PresenceModel()
    batch = 2
    inputs = {
        "prior_bins": torch.zeros(batch, 3, PRESENCE_N_BINS),
        "prior_ages_days": torch.zeros(batch, 3),
        "prior_mask": torch.zeros(batch, 3),
        "position_idx": torch.zeros(batch, dtype=torch.long),
        "starter_idx": torch.zeros(batch, dtype=torch.long),
        "history_count": torch.zeros(batch),
    }
    out = m(**inputs)
    # Pool at init is σ(0) = 0.5 across bins.
    expected_pool = torch.full((batch, PRESENCE_N_BINS), 0.5)
    sigmoid_b0 = float(torch.sigmoid(m.b0).item())
    expected = (1.0 - sigmoid_b0) * expected_pool
    assert torch.allclose(out, expected, atol=1e-5), out


def test_pure_self_when_history_very_large() -> None:
    """At a very large ``history_count`` the gate saturates at 1 and the output equals
    the recency-weighted self curve (a large ``β_h`` makes the saturation tight).
    """
    m = PresenceModel(beta_h_init=5.0)  # softplus(5)=5.0067 → strong slope
    batch = 2
    k_max = 2
    prior_bins = torch.tensor(
        [
            [
                [0.9] * PRESENCE_N_BINS,
                [0.1] * PRESENCE_N_BINS,
            ],
            [
                [0.5] * PRESENCE_N_BINS,
                [0.5] * PRESENCE_N_BINS,
            ],
        ]
    )
    inputs = {
        "prior_bins": prior_bins,
        "prior_ages_days": torch.zeros(batch, k_max),  # zero age → equal weights
        "prior_mask": torch.ones(batch, k_max),
        "position_idx": torch.zeros(batch, dtype=torch.long),
        "starter_idx": torch.zeros(batch, dtype=torch.long),
        "history_count": torch.tensor([1e6, 1e6]),
    }
    out = m(**inputs)
    # With equal weights, self curve = mean over the two priors → 0.5 and 0.5.
    expected = torch.full((batch, PRESENCE_N_BINS), 0.5)
    assert torch.allclose(out, expected, atol=1e-4), out


def test_recent_games_dominate_old_games() -> None:
    """With ``ρ > 0`` and a large age gap, the recent prior dominates the self curve."""
    m = PresenceModel()  # default ρ ≈ 1/45
    batch = 1
    k_max = 2
    prior_bins = torch.zeros(batch, k_max, PRESENCE_N_BINS)
    # Recent prior (age 1 day) all-1; old prior (age 365 days) all-0.
    prior_bins[0, 0] = 1.0  # recent
    prior_bins[0, 1] = 0.0  # old
    prior_ages_days = torch.tensor([[1.0, 365.0]])
    prior_mask = torch.ones(batch, k_max)
    # Self only.
    out_self = m.self_curve(prior_bins, prior_ages_days, prior_mask)
    # With ρ ≈ 1/45, w(1) ≈ exp(-1/45) ≈ 0.978, w(365) ≈ exp(-365/45) ≈
    # 2.7e-4. The recent prior gets ~99.97% of the weight, so the self
    # curve should be very close to 1 across bins.
    assert (out_self > 0.99).all(), out_self


def test_gate_increases_with_history_count() -> None:
    """The gate :math:`\\lambda(N)` is non-decreasing in ``N``."""
    m = PresenceModel()
    n_values = torch.tensor([0.0, 1.0, 5.0, 50.0, 500.0])
    lam = m.history_gate(n_values)
    for i in range(1, n_values.shape[0]):
        assert lam[i] >= lam[i - 1] - 1e-8, lam


def test_per_bin_bce_loss_is_finite_under_training_step() -> None:
    """One backward pass of per-bin BCE leaves parameters and gradients finite."""
    torch.manual_seed(0)
    m = PresenceModel()
    inputs = _make_inputs(batch=8, k_max=4)
    target = torch.rand(8, PRESENCE_N_BINS)
    out = m(**inputs)
    loss = torch.nn.functional.binary_cross_entropy(out, target)
    loss.backward()
    for p in m.parameters():
        assert torch.isfinite(p).all()
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_shape_validation() -> None:
    m = PresenceModel()
    bad_bins = torch.zeros(4, 5, PRESENCE_N_BINS - 1)
    with pytest.raises(ValueError, match="prior_bins"):
        m(
            prior_bins=bad_bins,
            prior_ages_days=torch.zeros(4, 5),
            prior_mask=torch.zeros(4, 5),
            position_idx=torch.zeros(4, dtype=torch.long),
            starter_idx=torch.zeros(4, dtype=torch.long),
            history_count=torch.zeros(4),
        )

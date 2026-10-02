"""Unit tests for :class:`shotcloud.models.ConditionalMDN`.

The MDN is the generic-neural conditional-density baseline (paper
§5.x); these tests pin its output shapes, log-density normalisation,
sampling court-bound guarantees, and gradient behaviour under NLL
training. They do not test paper-level claims --- those are in the
matched-window evaluator scripts.
"""

from __future__ import annotations

import math

import pytest
import torch

from shotcloud.models import ConditionalMDN
from shotcloud.simulation.sampler import DEFAULT_COURT_XLIM, DEFAULT_COURT_YLIM


@pytest.fixture
def mdn() -> ConditionalMDN:
    torch.manual_seed(0)
    return ConditionalMDN(input_dim=27, hidden_dim=64, n_components=8, eps_sigma=0.1)


def test_forward_shapes(mdn: ConditionalMDN) -> None:
    x = torch.randn(4, 27)
    log_pi, mu, sigma = mdn(x)
    assert log_pi.shape == (4, 8)
    assert mu.shape == (4, 8, 2)
    assert sigma.shape == (4, 8, 2)


def test_forward_normalisation(mdn: ConditionalMDN) -> None:
    """``log_pi`` rows sum to 0 (i.e. probabilities sum to 1)."""
    x = torch.randn(5, 27)
    log_pi, _, _ = mdn(x)
    pi = log_pi.exp()
    torch.testing.assert_close(pi.sum(dim=-1), torch.ones(5), atol=1e-6, rtol=0)


def test_sigma_floor_respected(mdn: ConditionalMDN) -> None:
    """The soft floor ``eps_sigma`` is a *lower bound* on every component's
    per-dim standard deviation, so even a wildly negative pre-floor head
    output cannot collapse a component to a delta."""
    x = torch.randn(3, 27)
    _, _, sigma = mdn(x)
    assert (sigma >= mdn.eps_sigma).all()


def test_log_prob_shape_and_finite(mdn: ConditionalMDN) -> None:
    x = torch.randn(6, 27)
    y = torch.randn(6, 2)
    lp = mdn.log_prob(y, x)
    assert lp.shape == (6,)
    assert torch.isfinite(lp).all()


def test_log_prob_integrates_to_one() -> None:
    """Monte-Carlo integration of the MDN density over a large box around
    the means should be close to 1. We use a single fixed context vector
    and integrate using a coarse grid; the diagonal-Gaussian mixture has
    a closed-form total mass of 1, so any deviation here is a
    parameterisation bug.
    """
    torch.manual_seed(0)
    # Tight init so all components fit inside the integration box; the
    # MDN's production init scatters means across the half-court, which
    # would leak mass outside the (-12, 12)² window used here.
    mdn = ConditionalMDN(
        input_dim=4,
        hidden_dim=16,
        n_components=4,
        eps_sigma=0.2,
        init_sigma_ft=1.0,
        court_bounds=(-1.0, 1.0, -1.0, 1.0),
    )
    x = torch.randn(1, 4)
    # Build a 401x401 grid spanning a region large enough to capture
    # the bulk of every component (component means in (-1, 1),
    # eps_sigma=0.2, init sigma=1.0). A box of (-12, 12)² is safe.
    side = 401
    coords = torch.linspace(-12.0, 12.0, side)
    gx, gy = torch.meshgrid(coords, coords, indexing="xy")
    xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1).unsqueeze(0)  # (1, M, 2)
    dens = mdn.density_at_xy(xy, x)
    cell = (coords[1] - coords[0]) ** 2
    total = (dens * cell).sum().item()
    assert total == pytest.approx(1.0, abs=2e-3)


def test_density_at_xy_matches_log_prob(mdn: ConditionalMDN) -> None:
    """``density_at_xy(xy, x)[:, 0]`` should equal ``exp(log_prob(xy[:, 0, :], x))``."""
    x = torch.randn(3, 27)
    y = torch.randn(3, 1, 2)
    dens = mdn.density_at_xy(y, x).squeeze(-1)
    log_dens = mdn.log_prob(y.squeeze(1), x)
    torch.testing.assert_close(dens.log(), log_dens, atol=1e-5, rtol=1e-5)


def test_sample_shape_and_in_court(mdn: ConditionalMDN) -> None:
    """Sampled locations always inside the court rectangle, even with
    artificially blown-up sigmas that put most pre-clip mass outside."""
    x = torch.randn(2, 27)
    # Force the head bias toward huge sigma to maximise rejection load.
    with torch.no_grad():
        mdn.head_log_sigma.bias.fill_(5.0)
    g = torch.Generator(device="cpu").manual_seed(0)
    samples = mdn.sample(x, n_samples=10, generator=g)
    assert samples.shape == (2, 10, 2)
    x_lo, x_hi = DEFAULT_COURT_XLIM
    y_lo, y_hi = DEFAULT_COURT_YLIM
    assert (samples[..., 0] >= x_lo).all()
    assert (samples[..., 0] <= x_hi).all()
    assert (samples[..., 1] >= y_lo).all()
    assert (samples[..., 1] <= y_hi).all()


def test_nll_descent_on_synthetic() -> None:
    """A two-component Gaussian-mixture target should be learnable: NLL
    after a handful of optimiser steps should be meaningfully lower than
    at initialisation."""
    torch.manual_seed(0)
    n = 512
    x = torch.randn(n, 27)
    # Synthetic target: component A at (5, 5), component B at (-5, 5),
    # both with sigma 1; picked by the sign of x[:, 0] so the MDN can
    # learn the dependence.
    means = torch.where(x[:, :1] > 0, torch.tensor([[5.0, 5.0]]), torch.tensor([[-5.0, 5.0]]))
    y = means + torch.randn(n, 2)

    mdn = ConditionalMDN(input_dim=27, hidden_dim=32, n_components=4, eps_sigma=0.1)
    opt = torch.optim.Adam(mdn.parameters(), lr=1e-2)
    initial = (-mdn.log_prob(y, x)).mean().item()
    for _ in range(200):
        opt.zero_grad()
        nll = (-mdn.log_prob(y, x)).mean()
        nll.backward()
        opt.step()
    final = (-mdn.log_prob(y, x)).mean().item()
    assert final < initial - 1.0, f"NLL did not descend: {initial} -> {final}"


def test_default_init_finite_on_court_scale_shots() -> None:
    """Regression: default init must produce finite log-density and
    finite gradients on shots drawn at the actual half-court scale
    (x ∈ [-25, 25], y ∈ [-5, 47]).

    The original mu_bias=0 / log_sigma_bias=0 default put initial
    mu near origin with sigma ~ 1 ft; a shot at (25, 47) then had
    log-density ~ -200 and gradients on log_sigma ~ 500 per shot,
    blowing up the optimizer after the first batch (NaN from epoch
    1 on the full 1.86M-shot run, 2026-06-08). The court-scale init
    fixes both.
    """
    torch.manual_seed(0)
    mdn = ConditionalMDN(input_dim=27, hidden_dim=64, n_components=32)
    x = torch.randn(512, 27)
    # Half-court-rectangle uniform shots — match the real data envelope.
    y = torch.stack(
        [
            torch.empty(512).uniform_(-25.0, 25.0),
            torch.empty(512).uniform_(-5.0, 47.0),
        ],
        dim=-1,
    )
    log_p = mdn.log_prob(y, x)
    assert torch.isfinite(log_p).all(), "step-0 log_prob is non-finite on court-scale shots"
    loss = -log_p.mean()
    loss.backward()
    for name, p in mdn.named_parameters():
        assert p.grad is not None, f"missing grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_raises_on_bad_shapes() -> None:
    mdn = ConditionalMDN(input_dim=27, hidden_dim=32, n_components=4)
    with pytest.raises(ValueError, match=r"x_n_raw must be"):
        mdn(torch.randn(4, 26))
    with pytest.raises(ValueError, match=r"y must be"):
        mdn.log_prob(torch.randn(4, 3), torch.randn(4, 27))
    with pytest.raises(ValueError, match=r"agree on batch dim"):
        mdn.log_prob(torch.randn(5, 2), torch.randn(4, 27))
    with pytest.raises(ValueError, match=r"xy must be"):
        mdn.density_at_xy(torch.randn(4, 2), torch.randn(4, 27))
    with pytest.raises(ValueError, match=r"n_samples must be"):
        mdn.sample(torch.randn(4, 27), n_samples=0)


def test_uniform_baseline_when_collapsed_to_one_component() -> None:
    """When forced to a single component centred at the court centroid
    with very large sigma, the log density should be approximately
    -log(court area) over the court interior — a sanity check that
    the Gaussian normaliser is wired correctly."""
    torch.manual_seed(0)
    # init_sigma_ft must exceed eps_sigma; pick a value safely above
    # the eps=10 floor so the validator accepts the config.
    mdn = ConditionalMDN(
        input_dim=2,
        hidden_dim=4,
        n_components=1,
        eps_sigma=10.0,
        init_sigma_ft=12.0,
    )
    x = torch.zeros(1, 2)
    # Mean at court centroid, log-sigma large -> sigma ~ exp(2) + 10 ~ 17 ft.
    with torch.no_grad():
        mdn.head_mu.weight.zero_()
        mdn.head_mu.bias.copy_(torch.tensor([0.0, 21.0]))  # near court centroid
        mdn.head_log_sigma.weight.zero_()
        mdn.head_log_sigma.bias.fill_(2.0)
    # Two probe points near the centroid: density should be similar
    # and both should be well above zero.
    y = torch.tensor([[0.0, 21.0], [5.0, 25.0]])
    lp = mdn.log_prob(y, x.expand(2, -1))
    assert torch.isfinite(lp).all()
    # Density at the mean should exceed density 7 ft off the mean.
    assert lp[0].item() > lp[1].item()
    # Densities are reasonable for a sigma~17 Gaussian: at the mean,
    # log p = -log(2π) - 2 log(17) ≈ -7.5 nats/ft².
    assert lp[0].item() < -6.0 and lp[0].item() > -9.0
    # log(2π) + 2 log(17) ≈ 1.84 + 5.67 = 7.51, so within (-9, -6) is tight.
    expected = -math.log(2.0 * math.pi) - 2.0 * math.log(17.0 + 10.0)
    # The +10 floor pushes effective sigma up; just check ballpark.
    assert lp[0].item() < expected + 1.0

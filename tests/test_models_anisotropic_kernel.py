"""Tests for the Tier-2 anisotropic kernel modules.

Scope:

* Isotropic-collapse invariant (load-bearing): both kernels at
  ``σ_r = σ_t = σ_init`` (resp. ``σ_x = σ_y = σ_init, ρ = 0``) reduce
  bit-exactly (float32 noise) to the fixed-σ isotropic Gaussian
  ``-log(2π) − log(σ²) − ½ ‖δ‖²/σ²``.
* Parameter counts: RT = 2 × N_ZONES; FC = 3 × N_ZONES.
* Bounded σ / ρ stay in range under extreme raw logits.
* Per-zone divergence: tweaking one zone's σ shifts only that zone's
  log-kernel.
* Origin guard for RT (s_m ≈ 0): isotropic fallback uses σ_r at that
  zone; no NaN.
* Gradient flow to every parameter.
* State-dict round-trip preserves learned values.
* Input shape validation.
* The RT kernel's r̂_m / t̂_m frame is correctly basket-centered:
  a shot directly outward from a support shot toward the basket sees
  only σ_r; a shot perpendicular sees only σ_t.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from shotcloud.data.zones import N_ZONES
from shotcloud.models.anisotropic_kernel import (
    FullCovarianceZoneKernel,
    RadialTangentZoneKernel,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _iso_log_kernel(support_xy: torch.Tensor, shot_xy: torch.Tensor, sigma: float) -> torch.Tensor:
    """Reference fixed-σ isotropic Gaussian log-kernel."""
    delta = shot_xy.unsqueeze(1) - support_xy  # (B, M, 2)
    dist2 = (delta * delta).sum(dim=-1)  # (B, M)
    return -math.log(2.0 * math.pi) - math.log(sigma**2) - 0.5 * dist2 / (sigma**2)


def _random_batch(B: int = 3, M: int = 5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random support_xy / shot_xy, on-court coordinates."""
    g = torch.Generator().manual_seed(seed)
    # Bias support shots toward the court, away from the basket exactly,
    # so the radial frame is well-defined for all rows.
    support = torch.empty(B, M, 2).uniform_(-15.0, 15.0, generator=g)
    support[..., 1] = support[..., 1].abs() + 1.0  # y ≥ 1
    shot = torch.empty(B, 2).uniform_(-15.0, 15.0, generator=g)
    return support, shot


# --------------------------------------------------------------------------- #
# Parameter counts
# --------------------------------------------------------------------------- #


def test_rt_kernel_param_count() -> None:
    rt = RadialTangentZoneKernel()
    n_params = sum(p.numel() for p in rt.parameters())
    assert n_params == 2 * N_ZONES


def test_fc_kernel_param_count() -> None:
    fc = FullCovarianceZoneKernel()
    n_params = sum(p.numel() for p in fc.parameters())
    assert n_params == 3 * N_ZONES


# --------------------------------------------------------------------------- #
# Isotropic-collapse invariant
# --------------------------------------------------------------------------- #


def test_rt_at_init_matches_isotropic_kernel() -> None:
    """**Load-bearing invariant for Tier-2 Option 1.**

    At ``σ_r = σ_t = σ_init = 1.5``, the radial-tangential kernel must
    reduce bit-exactly (to float32 precision) to the fixed-σ=1.5
    isotropic Gaussian, regardless of shot position. This is the
    "anisotropic-collapses-to-isotropic-at-init" identity.
    """
    rt = RadialTangentZoneKernel(sigma_min=1.0, sigma_max=2.5, sigma_init=1.5)
    support, shot = _random_batch(B=4, M=11)
    out = rt(support_xy=support, shot_xy=shot)
    expected = _iso_log_kernel(support, shot, sigma=1.5)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=0)


def test_fc_at_init_matches_isotropic_kernel() -> None:
    """**Load-bearing invariant for Tier-2 Option 3.**

    At ``σ_x = σ_y = σ_init = 1.5, ρ = 0``, the bounded-correlation
    kernel must reduce bit-exactly to the fixed-σ=1.5 isotropic
    Gaussian.
    """
    fc = FullCovarianceZoneKernel(
        sigma_min=1.0, sigma_max=2.5, sigma_init=1.5, rho_max=0.8, rho_init=0.0
    )
    support, shot = _random_batch(B=4, M=11)
    out = fc(support_xy=support, shot_xy=shot)
    expected = _iso_log_kernel(support, shot, sigma=1.5)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=0)


# --------------------------------------------------------------------------- #
# Bounded outputs
# --------------------------------------------------------------------------- #


def test_rt_sigma_stays_in_bounds_under_extreme_logits() -> None:
    rt = RadialTangentZoneKernel(sigma_min=1.0, sigma_max=2.5)
    with torch.no_grad():
        rt.raw_r.fill_(100.0)
        rt.raw_t.fill_(-100.0)
    sr = rt.sigma_r()
    st = rt.sigma_t()
    assert torch.all(sr <= 2.5 + 1e-6)
    assert torch.all(sr >= 1.0 - 1e-6)
    assert torch.all(st <= 2.5 + 1e-6)
    assert torch.all(st >= 1.0 - 1e-6)


def test_fc_sigma_and_rho_stay_in_bounds_under_extreme_logits() -> None:
    fc = FullCovarianceZoneKernel(sigma_min=1.0, sigma_max=2.5, rho_max=0.8)
    with torch.no_grad():
        fc.raw_sx.fill_(100.0)
        fc.raw_sy.fill_(-100.0)
        fc.raw_rho.fill_(100.0)
    assert torch.all(fc.sigma_x() <= 2.5 + 1e-6)
    assert torch.all(fc.sigma_y() >= 1.0 - 1e-6)
    rho = fc.rho()
    assert torch.all(rho <= 0.8 + 1e-6)
    assert torch.all(rho >= -0.8 - 1e-6)


# --------------------------------------------------------------------------- #
# Per-zone divergence
# --------------------------------------------------------------------------- #


def test_rt_per_zone_divergence_only_affects_target_zone() -> None:
    """Tweak σ_r at zone 7 (TopKey3) only — log-kernel must differ
    only for support shots in zone 7, all others unchanged."""
    rt = RadialTangentZoneKernel()
    # Two support shots, one in TopKey3 (zone 7), one in RA (zone 0).
    support = torch.tensor([[[0.0, 25.0], [0.0, 2.0]]], dtype=torch.float32)  # (1, 2, 2)
    shot = torch.tensor([[1.0, 12.0]], dtype=torch.float32)
    before = rt(support_xy=support, shot_xy=shot)
    with torch.no_grad():
        rt.raw_r[7] = 5.0  # push σ_r at TopKey3 to ~σ_max
    after = rt(support_xy=support, shot_xy=shot)
    # Zone-7 row changed; zone-0 row unchanged.
    assert not torch.equal(before[0, 0], after[0, 0])
    torch.testing.assert_close(before[0, 1], after[0, 1])


def test_fc_per_zone_divergence_only_affects_target_zone() -> None:
    fc = FullCovarianceZoneKernel()
    support = torch.tensor([[[0.0, 25.0], [0.0, 2.0]]], dtype=torch.float32)
    shot = torch.tensor([[1.0, 12.0]], dtype=torch.float32)
    before = fc(support_xy=support, shot_xy=shot)
    with torch.no_grad():
        fc.raw_rho[7] = 5.0  # tilt ρ at zone 7
    after = fc(support_xy=support, shot_xy=shot)
    assert not torch.equal(before[0, 0], after[0, 0])
    torch.testing.assert_close(before[0, 1], after[0, 1])


# --------------------------------------------------------------------------- #
# Origin guard (RT only)
# --------------------------------------------------------------------------- #


def test_rt_origin_guard_uses_isotropic_fallback() -> None:
    """A support shot at the basket has no defined radial frame. The
    kernel must fall back to isotropic ``σ_r²I`` and return finite,
    correct values."""
    rt = RadialTangentZoneKernel(sigma_min=1.0, sigma_max=2.5, sigma_init=1.5, origin_eps=1e-4)
    # Single support shot exactly at origin (basket).
    support = torch.zeros(1, 1, 2)
    shot = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    out = rt(support_xy=support, shot_xy=shot)
    assert torch.isfinite(out).all()
    # σ_r = 1.5 at init; fallback = isotropic σ_r²I.
    expected = _iso_log_kernel(support, shot, sigma=1.5)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=0)


def test_rt_origin_guard_does_not_corrupt_off_origin_rows() -> None:
    """A mixed batch with one origin-support shot and several
    off-origin support shots: only the origin one uses the fallback;
    all others use the radial-tangential kernel."""
    rt = RadialTangentZoneKernel()
    # Push σ_r at zone 0 (RA) up so the anisotropic kernel diverges
    # from the isotropic one.
    with torch.no_grad():
        rt.raw_r[0] = 3.0
    # Two support shots: one at origin (zone-0 fallback), one at
    # (0, 4) (zone-0 RA, normal radial frame).
    support = torch.tensor([[[0.0, 0.0], [0.0, 4.0]]], dtype=torch.float32)
    shot = torch.tensor([[1.0, 5.0]], dtype=torch.float32)
    out = rt(support_xy=support, shot_xy=shot)
    assert torch.isfinite(out).all()
    # The fallback row (origin) uses σ_r²I (≈ σ_max²I after the raw=3
    # push). The non-fallback row uses the full radial-tangential
    # formula; with σ_r ≫ σ_t = 1.5, the two log-kernel values must
    # differ.
    assert not torch.isclose(out[0, 0], out[0, 1], atol=1e-4)


# --------------------------------------------------------------------------- #
# Radial-tangential geometric correctness
# --------------------------------------------------------------------------- #


def test_rt_pure_radial_displacement_uses_only_sigma_r() -> None:
    """A displacement aligned with r̂ (radially outward from basket)
    should evaluate the kernel using only ``σ_r``; ``σ_t`` should have
    no effect for that row."""
    rt = RadialTangentZoneKernel(sigma_min=1.0, sigma_max=2.5, sigma_init=1.5)
    # Support shot in the top-of-key (zone 7), at (0, 25).
    # r̂ = (0, 1), t̂ = (-1, 0).
    # Shot 2 ft further out at (0, 27): δ = (0, 2) is purely radial.
    support = torch.tensor([[[0.0, 25.0]]], dtype=torch.float32)
    shot = torch.tensor([[0.0, 27.0]], dtype=torch.float32)
    out_baseline = rt(support_xy=support, shot_xy=shot)
    with torch.no_grad():
        rt.raw_t[7] = 5.0  # change σ_t at zone 7
    out_after_st_tweak = rt(support_xy=support, shot_xy=shot)
    # σ_t affects only the normalizer (½ log|Σ| has a log σ_t term);
    # the quadratic form δ_t²/σ_t² = 0 because δ_t = 0 here. So the
    # log-kernel still shifts by Δ log σ_t. Verify this targeted shift:
    sigma_t_before = rt.sigma_t()[7].item() / (1.0 + (math.exp(5.0) - 1.0) / (1.0 + math.exp(5.0)))
    # Easier check: the quadratic component (which depends on σ_t)
    # should remain zero; the change is entirely in the −log σ_t term.
    # Validate by computing the expected residual.
    sigma_r = rt.sigma_r()[7].item()
    sigma_t_after = rt.sigma_t()[7].item()
    expected_after = (
        -math.log(2.0 * math.pi)
        - math.log(sigma_r)
        - math.log(sigma_t_after)
        - 0.5 * (4.0 / sigma_r**2)
    )
    torch.testing.assert_close(
        out_after_st_tweak.squeeze(), torch.tensor(expected_after), atol=5e-5, rtol=0
    )
    # Sanity that the baseline matched the radial-only formula too.
    expected_before = (
        -math.log(2.0 * math.pi) - math.log(sigma_r) - math.log(1.5) - 0.5 * (4.0 / sigma_r**2)
    )
    torch.testing.assert_close(
        out_baseline.squeeze(), torch.tensor(expected_before), atol=5e-5, rtol=0
    )
    # And explicitly that the σ_t at zone 7 actually moved.
    assert sigma_t_after > 1.5
    _ = sigma_t_before  # silence unused-var lint


def test_rt_pure_tangential_displacement_uses_only_sigma_t() -> None:
    """A displacement aligned with t̂ should respond only to σ_t."""
    rt = RadialTangentZoneKernel(sigma_min=1.0, sigma_max=2.5, sigma_init=1.5)
    # Support shot at (0, 25). r̂ = (0, 1), t̂ = (-1, 0).
    # Shot at (-2, 25): δ = (-2, 0) is purely tangential (δ_r=0, δ_t=2).
    support = torch.tensor([[[0.0, 25.0]]], dtype=torch.float32)
    shot = torch.tensor([[-2.0, 25.0]], dtype=torch.float32)
    out_before = rt(support_xy=support, shot_xy=shot)
    with torch.no_grad():
        rt.raw_r[7] = 5.0  # change σ_r — should affect ONLY the normalizer
    out_after = rt(support_xy=support, shot_xy=shot)
    # The quadratic component is δ_t²/σ_t² (σ_r doesn't appear since
    # δ_r = 0). The shift in log-kernel between before and after is
    # purely −Δ log σ_r.
    sigma_r_before = 1.5
    sigma_r_after = rt.sigma_r()[7].item()
    expected_delta = -(math.log(sigma_r_after) - math.log(sigma_r_before))
    torch.testing.assert_close(
        out_after - out_before, torch.tensor([[expected_delta]]), atol=5e-5, rtol=0
    )


# --------------------------------------------------------------------------- #
# Gradient flow
# --------------------------------------------------------------------------- #


def test_rt_gradients_flow_to_both_parameter_groups() -> None:
    rt = RadialTangentZoneKernel()
    support, shot = _random_batch(B=2, M=6, seed=7)
    out = rt(support_xy=support, shot_xy=shot)
    out.sum().backward()
    assert rt.raw_r.grad is not None
    assert rt.raw_t.grad is not None
    assert rt.raw_r.grad.abs().sum() > 0
    assert rt.raw_t.grad.abs().sum() > 0


def test_fc_gradients_flow_to_all_parameter_groups() -> None:
    fc = FullCovarianceZoneKernel()
    support, shot = _random_batch(B=2, M=6, seed=7)
    out = fc(support_xy=support, shot_xy=shot)
    out.sum().backward()
    assert fc.raw_sx.grad is not None
    assert fc.raw_sy.grad is not None
    assert fc.raw_rho.grad is not None
    assert fc.raw_sx.grad.abs().sum() > 0
    assert fc.raw_sy.grad.abs().sum() > 0
    assert fc.raw_rho.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# State-dict round-trip
# --------------------------------------------------------------------------- #


def test_rt_state_dict_round_trip() -> None:
    src = RadialTangentZoneKernel()
    with torch.no_grad():
        src.raw_r.copy_(torch.randn_like(src.raw_r))
        src.raw_t.copy_(torch.randn_like(src.raw_t))
    sd = src.state_dict()
    dst = RadialTangentZoneKernel()
    dst.load_state_dict(sd)
    torch.testing.assert_close(dst.sigma_r(), src.sigma_r())
    torch.testing.assert_close(dst.sigma_t(), src.sigma_t())


def test_fc_state_dict_round_trip() -> None:
    src = FullCovarianceZoneKernel()
    with torch.no_grad():
        src.raw_sx.copy_(torch.randn_like(src.raw_sx))
        src.raw_sy.copy_(torch.randn_like(src.raw_sy))
        src.raw_rho.copy_(torch.randn_like(src.raw_rho))
    sd = src.state_dict()
    dst = FullCovarianceZoneKernel()
    dst.load_state_dict(sd)
    torch.testing.assert_close(dst.sigma_x(), src.sigma_x())
    torch.testing.assert_close(dst.sigma_y(), src.sigma_y())
    torch.testing.assert_close(dst.rho(), src.rho())


# --------------------------------------------------------------------------- #
# Input shape validation
# --------------------------------------------------------------------------- #


def test_rt_input_shape_validation() -> None:
    rt = RadialTangentZoneKernel()
    with pytest.raises(ValueError, match=r"\(B, M, 2\)"):
        rt(support_xy=torch.zeros(3, 4), shot_xy=torch.zeros(3, 2))
    with pytest.raises(ValueError, match=r"\(B, 2\)"):
        rt(support_xy=torch.zeros(3, 4, 2), shot_xy=torch.zeros(3, 5, 2))
    with pytest.raises(ValueError, match="batch dim mismatch"):
        rt(support_xy=torch.zeros(3, 4, 2), shot_xy=torch.zeros(2, 2))


def test_fc_input_shape_validation() -> None:
    fc = FullCovarianceZoneKernel()
    with pytest.raises(ValueError, match=r"\(B, M, 2\)"):
        fc(support_xy=torch.zeros(3, 4), shot_xy=torch.zeros(3, 2))
    with pytest.raises(ValueError, match=r"\(B, 2\)"):
        fc(support_xy=torch.zeros(3, 4, 2), shot_xy=torch.zeros(3, 5, 2))
    with pytest.raises(ValueError, match="batch dim mismatch"):
        fc(support_xy=torch.zeros(3, 4, 2), shot_xy=torch.zeros(2, 2))


def test_rt_bad_bounds_rejected() -> None:
    with pytest.raises(ValueError, match="must be <"):
        RadialTangentZoneKernel(sigma_min=2.0, sigma_max=1.0)
    with pytest.raises(ValueError, match="required"):
        RadialTangentZoneKernel(sigma_min=1.0, sigma_max=2.0, sigma_init=3.0)
    with pytest.raises(ValueError, match="origin_eps"):
        RadialTangentZoneKernel(origin_eps=0.0)


def test_fc_bad_bounds_rejected() -> None:
    with pytest.raises(ValueError, match="must be <"):
        FullCovarianceZoneKernel(sigma_min=2.0, sigma_max=1.0)
    with pytest.raises(ValueError, match="rho_max"):
        FullCovarianceZoneKernel(rho_max=1.5)
    with pytest.raises(ValueError, match="rho_init"):
        FullCovarianceZoneKernel(rho_max=0.5, rho_init=0.9)


# --------------------------------------------------------------------------- #
# Numerical sanity: anisotropic kernel matches the closed-form
# bivariate-Gaussian density at random parameters.
# --------------------------------------------------------------------------- #


def test_fc_matches_closed_form_bivariate_gaussian_density() -> None:
    """Sanity: at random (σ_x, σ_y, ρ), the log-kernel must equal the
    standard bivariate-Gaussian log-density formula."""
    fc = FullCovarianceZoneKernel(sigma_min=1.0, sigma_max=2.5, rho_max=0.8)
    with torch.no_grad():
        fc.raw_sx.copy_(torch.tensor([0.5, -0.8, 1.2, 0.0, 0.3, -1.5, 0.7, 0.2]))
        fc.raw_sy.copy_(torch.tensor([-0.3, 1.1, 0.0, 0.6, -0.5, 0.8, 0.4, -0.7]))
        fc.raw_rho.copy_(torch.tensor([0.4, -0.8, 0.0, 1.2, -0.3, 0.5, 0.6, -1.0]))
    # Single support shot in zone 1 (Paint), single observation.
    support = torch.tensor([[[0.0, 8.0]]], dtype=torch.float32)
    shot = torch.tensor([[1.5, 7.0]], dtype=torch.float32)
    out = fc(support_xy=support, shot_xy=shot)
    # Reference: closed-form bivariate-Gaussian log-density at (δ_x, δ_y).
    sx = fc.sigma_x()[1].item()
    sy = fc.sigma_y()[1].item()
    rho = fc.rho()[1].item()
    dx, dy = 1.5, -1.0
    one_minus_r2 = 1.0 - rho**2
    log_det = 2 * math.log(sx) + 2 * math.log(sy) + math.log(one_minus_r2)
    q = (dx**2 / sx**2 - 2 * rho * dx * dy / (sx * sy) + dy**2 / sy**2) / one_minus_r2
    expected = -math.log(2.0 * math.pi) - 0.5 * log_det - 0.5 * q
    torch.testing.assert_close(
        out.squeeze(), torch.tensor(expected, dtype=torch.float32), atol=5e-5, rtol=0
    )


def _zone_sigma_at_init_sanity() -> None:
    """Helper used by the float64 check; suppress unused-import lint."""
    _ = np.zeros(1)

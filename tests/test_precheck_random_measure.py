"""Synthetic-data tests for the random-measure pre-check.

Three tests per spec §6.2 of
[docs/precheck_random_measure_spec.md](../docs/precheck_random_measure_spec.md):

1. ``test_precheck_iid_synthetic`` — shots iid given a known per-player
   multinomial; R should be near 0 and decision should be STOP.
2. ``test_precheck_latent_synthetic`` — shots drawn from a per-game random
   multinomial with known logistic-normal variance; R should recover the
   known excess covariance and decision should be PROCEED.
3. ``test_precheck_calibration_failure_aborts`` — baseline with known
   constant bias on one zone; calibration check should fail and the pipeline
   should report a calibration-failed abort.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from precheck_random_measure import (  # type: ignore[import-not-found]
    N_ZONES_4,
    PROCEED_THRESHOLD,
    STOP_THRESHOLD,
    analyze_stratum,
    apply_decision_rule,
    calibration_check,
    compute_sigma_emp,
    compute_sigma_null,
)


def _draw_iid_multinomial(p: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Return 4-zone proportions from K iid multinomial draws on p."""
    counts = rng.multinomial(K, p)
    return counts / K


def test_precheck_iid_synthetic() -> None:
    """Generate N games where shots are iid given a known per-player p.
    Use p as the baseline (perfectly calibrated). Then R should be near 0
    and the decision rule should fire STOP."""
    rng = np.random.default_rng(0)
    N = 10_000
    # Each "player" has a fixed multinomial. Draw N players.
    p_bar = rng.dirichlet(np.array([2.0, 1.5, 1.0, 2.5]), size=N)  # (N, 4)
    K = rng.integers(15, 25, size=N)  # high stratum
    p_hat = np.stack(
        [_draw_iid_multinomial(p_bar[i], int(K[i]), rng) for i in range(N)],
        axis=0,
    )

    # Hand the core math the perfectly-calibrated baseline.
    sigma_emp = compute_sigma_emp(p_hat, p_bar)
    sigma_null = compute_sigma_null(p_bar, K)
    delta = sigma_emp - sigma_null
    R = float(np.trace(delta) / np.trace(sigma_null))

    # Under H0 with calibrated baseline, R should be near 0. Allow ±0.05.
    assert abs(R) < 0.05, f"R should be near 0 under iid null; got {R}"

    # Decision should be STOP.
    stratum_result = analyze_stratum(p_hat, p_bar, K, n_boot=1000, seed=0)
    decision = apply_decision_rule(stratum_result)
    assert decision == "STOP", (
        f"iid synthetic should give STOP, got {decision}. "
        f"R={stratum_result['R']:.3f}, "
        f"R_ci_95_hi={stratum_result['R_ci_95_hi']:.3f}, "
        f"STOP threshold={STOP_THRESHOLD}"
    )


def test_precheck_latent_synthetic() -> None:
    """Generate N games where each game's p_n = softmax(mu + L*z_n),
    z_n ~ N(0, I), and shots are iid from p_n. The marginal mean E[p_n]
    serves as the baseline. The decision rule should fire PROCEED.

    The injected latent variance is large enough (L scale = 0.6) to push
    R well above the 20% PROCEED threshold."""
    rng = np.random.default_rng(0)
    N = 10_000
    K_const = 20  # high stratum

    # Population-mean logits. Two latent dimensions with non-trivial
    # influence on all 4 zones — pushed to make excess variance >> 20%.
    mu = np.array([0.6, 0.3, -0.5, -0.4])
    L = 0.6 * np.array(
        [
            [1.0, -1.0],
            [-0.5, 0.8],
            [-1.0, -0.5],
            [1.5, 0.7],
        ]
    )  # (4, 2)

    # Per-game latent and resulting per-game p.
    z = rng.standard_normal(size=(N, 2))
    logits = mu[None, :] + z @ L.T  # (N, 4)
    p_n = np.exp(logits)
    p_n = p_n / p_n.sum(axis=-1, keepdims=True)

    # Baseline = the per-game marginal mean. Approximate via Monte Carlo
    # over the latent prior (same for all games here).
    z_mc = rng.standard_normal(size=(20_000, 2))
    logits_mc = mu[None, :] + z_mc @ L.T
    p_mc = np.exp(logits_mc) / np.exp(logits_mc).sum(axis=-1, keepdims=True)
    p_bar_population = p_mc.mean(axis=0)
    p_bar = np.broadcast_to(p_bar_population, (N, N_ZONES_4)).copy()

    # Draw shots iid from each game's p_n.
    K = np.full(N, K_const, dtype=np.int64)
    p_hat = np.stack(
        [_draw_iid_multinomial(p_n[i], K_const, rng) for i in range(N)],
        axis=0,
    )

    stratum_result = analyze_stratum(p_hat, p_bar, K, n_boot=1000, seed=0)
    decision = apply_decision_rule(stratum_result)
    R = stratum_result["R"]
    R_lo99 = stratum_result["R_one_sided_99_lower"]
    assert decision == "PROCEED", (
        f"latent-variance synthetic should give PROCEED, got {decision}. "
        f"R={R:.3f}, R_one_sided_99_lower={R_lo99:.3f}, "
        f"PROCEED threshold={PROCEED_THRESHOLD}, "
        f"3pt_sig={stratum_result['three_pt_significant_positive']}"
    )
    # And R should be substantially above the PROCEED threshold.
    assert R > 0.3, f"injected latent variance should give R >> 0.20; got {R}"


def test_precheck_calibration_failure_aborts() -> None:
    """Inject a known +0.05 systematic bias on one zone of the baseline.
    The calibration check should fail (not pass) so that the downstream
    pipeline can detect it and abort. Updated 2026-06-10: criterion is
    practical effect-size (|gap| < 0.03 on ≥8/10 deciles, no systematic
    monotone drift on the upper tail)."""
    rng = np.random.default_rng(0)
    N = 5_000
    K = rng.integers(15, 25, size=N)
    p_true = rng.dirichlet(np.array([2.0, 1.5, 1.0, 2.5]), size=N)
    p_hat = np.stack(
        [_draw_iid_multinomial(p_true[i], int(K[i]), rng) for i in range(N)],
        axis=0,
    )
    # Bias the baseline on zone 2 by -0.05 (i.e., baseline systematically
    # under-predicts that zone). Re-normalize to stay on the simplex.
    p_bar = p_true.copy()
    p_bar[:, 2] -= 0.05
    p_bar = np.clip(p_bar, 1e-6, None)
    p_bar = p_bar / p_bar.sum(axis=-1, keepdims=True)

    cal = calibration_check(p_hat, p_bar)
    # The biased zone (2 = corner-3) should fail because every decile has
    # |gap| ≈ 0.05 > 0.03 threshold.
    assert not cal["pass"], (
        "calibration check should fail when baseline has known bias on a zone; "
        f"got pass=True. zones: "
        f"{[(z['zone'], z['pass'], z['max_abs_gap']) for z in cal['per_zone']]}"
    )
    # Zone 2 (corner-3) specifically should be flagged.
    corner3 = next(z for z in cal["per_zone"] if z["zone"] == "corner-3")
    assert not corner3["pass"], "corner-3 with +0.05 bias should fail calibration"

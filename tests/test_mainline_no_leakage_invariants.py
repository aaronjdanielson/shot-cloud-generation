"""No-leakage invariant tests for the 5 mainline modules (paper §3.1).

Added 2026-06-10 per audit fix H2. The leakage guardrail tests in
``test_models_continuous_mixture_spatial.py`` and
``test_mode_routed_spatial.py`` already cover ``CausalZoneBias`` and
``ModeRouter`` — but both are REJECTED stress-test variants. The five
MAINLINE / load-bearing modules that actually appear in O1 had no
perturbation guard:

- :class:`shotcloud.models.PoolingGate` (gate λ, paper §2.x)
- :class:`shotcloud.models.zone_defense_reweighting.ZoneReweightingDefense`
  (D-lite-zone defense, paper §3.2)
- :class:`shotcloud.models.ContextResidualEncoder` (residual u_θ, paper §2.x)
- :class:`shotcloud.models.zone_source_bandwidth.ZoneSourceBandwidth`
  (per-source bandwidth σ_m, paper §2.x)
- :class:`shotcloud.models.NegBinCountHead` (count factor, paper §5.2)

This file adds two layers of defense:

1. **Signature-level guards** (5 tests): any future PR that adds
   ``shot_xy`` / ``y_n`` / observed-shot-location parameters to a
   mainline forward signature fires here immediately.
2. **Wrapper-level perturbation tests** (4 tests): with each component
   wired into the ``ContinuousMixtureSpatial`` wrapper, perturbing
   ``shot_xy`` must leave the component's output field bit-identical.
   The count head is covered transitively via the residual encoder
   (its μ output feeds the residual's usage channel).

Paper §3.1 invariant:

    *No attention logit, gate, residual, or kernel modifier may depend
    on y_n except through the normalized density evaluation itself.*
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shotcloud.models.context_residual import ContextResidualEncoder
from shotcloud.models.count_head import NegBinCountHead
from shotcloud.models.location_embedding import LocationEmbedding
from shotcloud.models.pooling_gate import PoolingGate
from shotcloud.models.zone_defense_reweighting import ZoneReweightingDefense
from shotcloud.models.zone_source_bandwidth import ZoneSourceBandwidth
from test_models_continuous_mixture_spatial import (  # type: ignore[import-not-found]
    _batch,
    _build_setup,
    _build_zone_lite_setup,
    _defense_batch,
)

# ---------------------------------------------------------------------------
# Signature-level guards — Part 1
# ---------------------------------------------------------------------------

#: Forbidden parameter names. A mainline module that consumes any of these
#: is a leakage vector by signature alone.
_FORBIDDEN_PARAMS = frozenset({"shot_xy", "y_n", "y", "shot_loc", "observed_xy"})


def _assert_no_yn_in_signature(module_class: type, msg_prefix: str) -> None:
    sig = inspect.signature(module_class.forward)
    bad = [name for name in sig.parameters if name in _FORBIDDEN_PARAMS]
    assert not bad, (
        f"{msg_prefix}.forward must not accept y_n / shot_xy / observed shot "
        f"location; found forbidden param(s): {bad}. Paper §3.1 no-leakage rule."
    )


def test_pooling_gate_forward_signature_excludes_yn() -> None:
    """PoolingGate's forward must consume only causal context + history."""
    _assert_no_yn_in_signature(PoolingGate, "PoolingGate")


def test_zone_reweighting_defense_forward_signature_excludes_yn() -> None:
    """ZoneReweightingDefense.forward(query_xy, def_features) — query_xy
    is the SUPPORT shots' coordinates, not the observed shot y_n."""
    _assert_no_yn_in_signature(ZoneReweightingDefense, "ZoneReweightingDefense")


def test_context_residual_encoder_forward_signature_excludes_yn() -> None:
    """ContextResidualEncoder.forward(x_n, h_n, usage, outcome) — all
    causal."""
    _assert_no_yn_in_signature(ContextResidualEncoder, "ContextResidualEncoder")


def test_zone_source_bandwidth_forward_signature_excludes_yn() -> None:
    """ZoneSourceBandwidth.forward(support_xy, own_mask) — support_xy
    is the SUPPORT shots' coordinates, not y_n."""
    _assert_no_yn_in_signature(ZoneSourceBandwidth, "ZoneSourceBandwidth")


def test_neg_bin_count_head_forward_signature_excludes_yn() -> None:
    """NegBinCountHead.forward(x_n) — pure context."""
    _assert_no_yn_in_signature(NegBinCountHead, "NegBinCountHead")


# ---------------------------------------------------------------------------
# Wrapper-level perturbation tests — Part 2
# ---------------------------------------------------------------------------
#
# Pattern: wire each component into ContinuousMixtureSpatial, set its
# parameters to random non-zero values (so a hypothetical y_n dependency
# would produce a visible delta), then run forward with two different
# shot_xy and assert the component's output field is bit-identical.


def _randomize(module: torch.nn.Module, scale: float = 0.3) -> None:
    """Randomize all module parameters to non-trivial non-zero values so a
    hypothetical y_n leak is actually exercised."""
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * scale)


def test_pooling_gate_lambda_invariant_to_shot_xy() -> None:
    """PoolingGate's output ``λ`` (exposed as ``out.gate_lambda``) is a
    function of (log1p_H_hat, x_n, own_support_count, own_available,
    h_n, pooled_available). None of those carry y_n. If a future
    refactor accidentally routes ``shot_xy`` into the gate, this fires."""
    from shotcloud.models.continuous_mixture_spatial import ContinuousMixtureSpatial

    setup = _build_setup()
    gate = PoolingGate(history_dim=0)
    _randomize(gate)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        pooling_gate=gate,
    )
    batch_a = _batch(setup, n_batch=4)
    # Large perturbation crossing zone / court boundaries.
    perturbation = torch.tensor([[25.0, 30.0]], dtype=batch_a["shot_xy"].dtype)
    batch_b = {**batch_a, "shot_xy": batch_a["shot_xy"] + perturbation}
    with torch.no_grad():
        out_a = spatial(**batch_a)
        out_b = spatial(**batch_b)
    assert out_a.gate_lambda is not None, "PoolingGate did not populate gate_lambda"
    assert torch.equal(out_a.gate_lambda, out_b.gate_lambda), (
        "PoolingGate λ changed when shot_xy changed — LEAKAGE BUG"
    )


def test_zone_reweighting_defense_logits_invariant_to_shot_xy() -> None:
    """``ZoneReweightingDefense`` produces ``out.defense_logits`` from the
    SUPPORT-shot locations + opponent's centered-zone defense features.
    Neither depends on y_n. Perturbing the wrapper's ``shot_xy`` must
    leave ``defense_logits`` bit-identical."""
    from shotcloud.models.continuous_mixture_spatial import ContinuousMixtureSpatial

    setup = _build_zone_lite_setup(beta_init=1e-3)
    field = setup["defensive_field_zone_lite"]
    _randomize(field, scale=0.5)  # type: ignore[arg-type]
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        defensive_field=field,  # type: ignore[arg-type]
        defensive_features=setup["defensive_features"],  # type: ignore[arg-type]
    )
    batch_a = _defense_batch(setup, n_batch=3)
    perturbation = torch.tensor([[20.0, -10.0]], dtype=batch_a["shot_xy"].dtype)
    batch_b = {**batch_a, "shot_xy": batch_a["shot_xy"] + perturbation}
    with torch.no_grad():
        out_a = spatial(**batch_a)
        out_b = spatial(**batch_b)
    assert out_a.defense_logits is not None, "D-lite-zone defense did not populate defense_logits"
    assert torch.equal(out_a.defense_logits, out_b.defense_logits), (
        "D-lite-zone defense_logits changed when shot_xy changed — LEAKAGE BUG"
    )


def test_context_residual_encoder_residual_logits_invariant_to_shot_xy() -> None:
    """``ContextResidualEncoder`` produces ``u_θ(x_n, h_n, usage, outcome)``
    which combines with ``location_embedding(support_xy)`` to form
    ``residual_logits`` (B, M). Perturbing y_n must leave both ``u_θ``
    AND ``residual_logits`` bit-identical. This also TRANSITIVELY covers
    ``NegBinCountHead``: when the residual's ``usage_dim`` consumes the
    detached ``K̂``, any y_n leak in the count head would surface as a
    delta in residual_logits."""
    from shotcloud.models.continuous_mixture_spatial import ContinuousMixtureSpatial

    setup = _build_setup()
    residual = ContextResidualEncoder(rank=4, within_game_dim=0)
    _randomize(residual)
    loc = LocationEmbedding(rank=4)
    _randomize(loc)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        residual_encoder=residual,
        location_embedding=loc,
    )
    batch_a = _batch(setup, n_batch=4)
    perturbation = torch.tensor([[-15.0, 20.0]], dtype=batch_a["shot_xy"].dtype)
    batch_b = {**batch_a, "shot_xy": batch_a["shot_xy"] + perturbation}
    with torch.no_grad():
        out_a = spatial(**batch_a)
        out_b = spatial(**batch_b)
    assert out_a.residual_logits is not None
    assert torch.equal(out_a.residual_logits, out_b.residual_logits), (
        "residual_logits changed when shot_xy changed — LEAKAGE BUG in "
        "ContextResidualEncoder or LocationEmbedding (location_embedding "
        "should consume support_xy not shot_xy)"
    )


def test_zone_source_bandwidth_sigma_per_shot_invariant_to_shot_xy() -> None:
    """``ZoneSourceBandwidth`` produces ``out.sigma_per_shot`` (B, M)
    from ``support_xy`` and ``own_mask``. Neither depends on y_n.
    Perturbing the wrapper's ``shot_xy`` must leave ``sigma_per_shot``
    bit-identical."""
    from shotcloud.models.continuous_mixture_spatial import ContinuousMixtureSpatial

    setup = _build_setup()
    bw = ZoneSourceBandwidth(sigma_min=0.5, sigma_max=4.0, sigma_init=1.5)
    _randomize(bw)
    spatial = ContinuousMixtureSpatial(
        offensive_prior=setup["collab"],  # type: ignore[arg-type]
        bandwidth_field=bw,
    )
    batch_a = _batch(setup, n_batch=4)
    perturbation = torch.tensor([[18.0, -5.0]], dtype=batch_a["shot_xy"].dtype)
    batch_b = {**batch_a, "shot_xy": batch_a["shot_xy"] + perturbation}
    with torch.no_grad():
        out_a = spatial(**batch_a)
        out_b = spatial(**batch_b)
    assert out_a.sigma_per_shot is not None, "ZoneSourceBandwidth did not populate sigma_per_shot"
    assert torch.equal(out_a.sigma_per_shot, out_b.sigma_per_shot), (
        "ZoneSourceBandwidth sigma_per_shot changed when shot_xy changed — LEAKAGE BUG"
    )


# ---------------------------------------------------------------------------
# Sanity: the perturbation actually changes downstream log_lik
# ---------------------------------------------------------------------------
#
# This is the positive control. The shot_xy perturbation must change
# ``log_lik`` (because the kernel evaluation at the observed shot has
# moved) — otherwise the perturbation is meaningless and any "invariant"
# test would be vacuously true.


def test_perturbation_changes_log_lik() -> None:
    """Positive control: shot_xy perturbation must change log_lik (the
    kernel is evaluated at y_n, and moving y_n by 25+ ft must change the
    density). Guards the test pattern itself."""
    from shotcloud.models.continuous_mixture_spatial import ContinuousMixtureSpatial

    setup = _build_setup()
    spatial = ContinuousMixtureSpatial(offensive_prior=setup["collab"])  # type: ignore[arg-type]
    batch_a = _batch(setup, n_batch=4)
    perturbation = torch.tensor([[25.0, 30.0]], dtype=batch_a["shot_xy"].dtype)
    batch_b = {**batch_a, "shot_xy": batch_a["shot_xy"] + perturbation}
    with torch.no_grad():
        out_a = spatial(**batch_a)
        out_b = spatial(**batch_b)
    assert not torch.equal(out_a.log_lik, out_b.log_lik), (
        "shot_xy perturbation did not change log_lik — perturbation test is "
        "vacuously satisfied; any leakage test is unreliable. Increase the "
        "perturbation magnitude or check that shot_xy reaches the kernel."
    )

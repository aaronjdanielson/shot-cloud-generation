"""No-leakage tests for the mainline support-logit, gate, and count modules.

The paper's no-leakage guardrail requires that no attention logit, gate,
residual, or kernel modifier depend on the observed location ``y_n`` except
through the normalized density evaluation itself. This file checks that
contract for the modules of the mainline configuration:

- :class:`shotcloud.models.pooling_gate.PoolingGate` (pooling gate λ)
- :class:`shotcloud.models.zone_defense_reweighting.ZoneReweightingDefense`
  (zone-level opponent reweighting)
- :class:`shotcloud.models.ContextResidualEncoder` (residual input u_θ)
- :class:`shotcloud.models.zone_source_bandwidth.ZoneSourceBandwidth`
  (per-source bandwidth σ_m)
- :class:`shotcloud.models.NegBinCountHead` (count factor)

Two kinds of test are used:

1. **Signature guards** (all five modules): ``forward`` accepts no parameter
   named like an observed shot location (``shot_xy``, ``y_n``, ...).
2. **Perturbation tests** (all but the count head): with the module wired into
   :class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`
   and its parameters randomized, perturbing ``shot_xy`` leaves the module's
   output field bit-identical. A positive control confirms the same
   perturbation does change ``log_lik``.

Guards for the alternative ``CausalZoneBias`` and ``ModeRouter`` modules live
in ``test_models_continuous_mixture_spatial.py`` and
``test_mode_routed_spatial.py``.
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
# Signature-level guards
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
    """``ZoneReweightingDefense.forward`` takes support-shot ``query_xy``, never ``y_n``."""
    _assert_no_yn_in_signature(ZoneReweightingDefense, "ZoneReweightingDefense")


def test_context_residual_encoder_forward_signature_excludes_yn() -> None:
    """``ContextResidualEncoder.forward`` takes only causal ``(x_n, h_n, usage, outcome)``."""
    _assert_no_yn_in_signature(ContextResidualEncoder, "ContextResidualEncoder")


def test_zone_source_bandwidth_forward_signature_excludes_yn() -> None:
    """``ZoneSourceBandwidth.forward`` takes support-shot ``support_xy``, never ``y_n``."""
    _assert_no_yn_in_signature(ZoneSourceBandwidth, "ZoneSourceBandwidth")


def test_neg_bin_count_head_forward_signature_excludes_yn() -> None:
    """``NegBinCountHead.forward`` takes context ``x_n`` only."""
    _assert_no_yn_in_signature(NegBinCountHead, "NegBinCountHead")


# ---------------------------------------------------------------------------
# Wrapper-level perturbation tests
# ---------------------------------------------------------------------------
#
# Pattern: wire each component into ContinuousMixtureSpatial, set its
# parameters to random non-zero values (so a hypothetical y_n dependency
# would produce a visible delta), then run forward with two different
# shot_xy and assert the component's output field is bit-identical.


def _randomize(module: torch.nn.Module, scale: float = 0.3) -> None:
    """Set every parameter to random non-zero values so any ``y_n`` dependence would show."""
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * scale)


def test_pooling_gate_lambda_invariant_to_shot_xy() -> None:
    """The pooling gate ``out.gate_lambda`` is unchanged when ``shot_xy`` is perturbed.

    λ is a function of ``(log1p_h_hat, x_n, own_support_count, own_available,
    h_n, pooled_available)``, none of which carries ``y_n``.
    """
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
    """``out.defense_logits`` is unchanged when ``shot_xy`` is perturbed.

    The logits depend only on support-shot locations and the opponent's
    centered-zone defense features.
    """
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
    """``out.residual_logits`` is unchanged when ``shot_xy`` is perturbed.

    The residual logits ``(B, M)`` combine ``u_θ`` from the context encoder
    with ``location_embedding(support_xy)``, so they depend on support-shot
    locations only.
    """
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
    """``out.sigma_per_shot`` ``(B, M)`` is unchanged when ``shot_xy`` is perturbed.

    The bandwidths depend only on ``support_xy`` and ``own_mask``.
    """
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
# Positive control: the shot_xy perturbation must change ``log_lik``
# (the kernel is evaluated at the moved observation); otherwise every
# invariance test above would hold vacuously.


def test_perturbation_changes_log_lik() -> None:
    """Moving ``shot_xy`` by roughly 40 ft changes ``log_lik``, so the perturbation is effective."""
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

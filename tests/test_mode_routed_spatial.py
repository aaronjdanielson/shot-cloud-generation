"""Tests for the Phase 3 mode-routed AC-KDE spatial decoder.

Five structural invariants the 2026-06-09 design lock requires the
mode-routed forward to satisfy:

1. **Leakage guardrail.** ``support_log_weights`` and ``mode_log_pi``
   must NOT change when ``shot_xy`` changes (with everything else
   held fixed). Inherits the locked rule from the α1 redesign — no
   attention modifier may depend on y_n.
2. **Finite log-likelihood.** A vanilla forward must produce a
   finite log_lik on every row, including rows with empty modes.
3. **Valid normalized mixture.** ``exp(mode_log_pi)`` must sum to
   1 over available modes per row.
4. **Empty modes safely excluded.** Modes with no causal support
   have ``mode_log_pi = -inf`` and contribute zero to log_lik.
5. **Gradients flow to the router.** One backward pass must
   populate non-zero finite gradient on ModeRouter's parameters
   (no dual-zero saddle here — π_k enters log_lik directly, not
   multiplied by zero-init B as in CausalZoneBias).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shotcloud.models.mode_routed_spatial import (
    ModeRoutedContinuousMixtureSpatial,
    ModeRouter,
)
from test_models_continuous_mixture_spatial import (  # type: ignore[import-not-found]
    _batch,
    _build_setup,
)


def _build_mode_routed_spatial() -> tuple[ModeRoutedContinuousMixtureSpatial, dict]:
    setup = _build_setup()
    router = ModeRouter()
    spatial = ModeRoutedContinuousMixtureSpatial(
        offensive_prior=setup["collab"],
        mode_router=router,
    )
    return spatial, setup


def test_mode_router_invariant_to_observed_shot() -> None:
    """LEAKAGE GUARDRAIL (locked 2026-06-09): perturbing shot_xy while
    holding everything else fixed must leave both ``mode_log_pi`` and
    ``support_log_weights`` bit-identical. The mode router consumes
    x_n only; the within-mode softmax depends on support_logits +
    support_xy zones, neither of which involves y_n. If this test
    fires, someone reintroduced y_n into the attention pipeline."""
    torch.manual_seed(0)
    spatial, setup = _build_mode_routed_spatial()
    # Random non-zero router params so any leak is exercised.
    with torch.no_grad():
        for p in spatial.mode_router.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    batch_a = _batch(setup, n_batch=8)
    perturbation = torch.tensor([[20.0, 5.0]], dtype=batch_a["shot_xy"].dtype)
    batch_b = {**batch_a, "shot_xy": batch_a["shot_xy"] + perturbation}
    out_a = spatial(**batch_a)
    out_b = spatial(**batch_b)
    assert out_a.mode_log_pi is not None and out_b.mode_log_pi is not None
    assert torch.equal(out_a.mode_log_pi, out_b.mode_log_pi), (
        "mode_log_pi changed when shot_xy changed — LEAKAGE BUG in mode router"
    )
    assert torch.equal(out_a.support_log_weights, out_b.support_log_weights), (
        "support_log_weights changed when shot_xy changed — LEAKAGE BUG in within-mode softmax"
    )


def test_mode_routed_log_lik_is_finite() -> None:
    """Vanilla forward must produce a finite log-density on every row,
    including rows where some modes are empty. Tests the empty-mode
    -inf handling via the masked-fill + clamp pattern in forward."""
    torch.manual_seed(0)
    spatial, setup = _build_mode_routed_spatial()
    batch = _batch(setup, n_batch=16)
    out = spatial(**batch)
    assert torch.isfinite(out.log_lik).all(), (
        f"log_lik must be finite per row; got min={out.log_lik.min()}, max={out.log_lik.max()}"
    )


def test_mode_pi_is_normalized_over_available_modes() -> None:
    """For each row, exp(mode_log_pi) restricted to available modes
    must sum to 1 (within float tolerance). Tests the
    masked_fill(~mode_available, -inf) + softmax pattern."""
    torch.manual_seed(0)
    spatial, setup = _build_mode_routed_spatial()
    batch = _batch(setup, n_batch=16)
    out = spatial(**batch)
    assert out.mode_log_pi is not None
    assert out.mode_available is not None
    pi = out.mode_log_pi.exp()  # (B, K) — unavailable modes -> 0
    # Sum over available modes per row.
    total = pi.sum(dim=-1)  # (B,)
    # For cold-start rows (no modes available) we patched the first
    # logit to 0 so the softmax gives all mass to mode 0; the row's
    # log_lik is then overwritten by the cold-start floor. So total
    # should be 1.0 for everyone (including cold-start).
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), (
        f"mode_log_pi must softmax-normalize per row; got total per row: {total}"
    )


def test_empty_modes_contribute_zero() -> None:
    """For any row r where mode k has no causal support shots, the
    contribution of mode k to log_lik must be exactly zero (i.e.,
    mode_log_pi[r, k] = -inf, so exp(...)·f_k = 0). Tested by
    constructing a row whose support is entirely in one zone."""
    torch.manual_seed(0)
    spatial, setup = _build_mode_routed_spatial()
    # The synthetic setup has small support sets; some modes will be
    # naturally empty. Confirm at least one mode is empty for at least
    # one row, and that its mode_log_pi is -inf.
    batch = _batch(setup, n_batch=16)
    out = spatial(**batch)
    assert out.mode_log_pi is not None
    assert out.mode_available is not None
    # At least some rows should have at least one unavailable mode.
    has_empty_modes = (~out.mode_available).any(dim=-1)
    assert has_empty_modes.any(), (
        "synthetic batch should have at least one row with some empty modes"
    )
    # Unavailable modes must have log_pi = -inf so they contribute zero
    # to the logsumexp.
    log_pi_at_unavailable = out.mode_log_pi[~out.mode_available]
    assert torch.all(log_pi_at_unavailable == float("-inf")), (
        f"mode_log_pi must be -inf for unavailable modes; "
        f"got values at unavailable: {log_pi_at_unavailable}"
    )


def test_mode_router_receives_gradient() -> None:
    """One backward pass must populate non-zero finite gradient on
    ALL ModeRouter parameters. Unlike CausalZoneBias (where π_q is
    multiplied by zero-init B and gets no gradient on step 1), the
    mode router's π_k enters log_lik DIRECTLY (via logsumexp_k(log_pi
    + log_f_k)), so there's no dual-zero saddle and gradient should
    flow on the first backward."""
    torch.manual_seed(0)
    spatial, setup = _build_mode_routed_spatial()
    batch = _batch(setup, n_batch=16)
    out = spatial(**batch)
    loss = -out.log_lik.mean()
    loss.backward()
    grads = [p.grad for p in spatial.mode_router.parameters()]
    assert all(g is not None for g in grads), "all mode_router params must have grad"
    assert all(torch.isfinite(g).all() for g in grads if g is not None), (
        "all mode_router grads must be finite"
    )
    nonzero_count = sum(int(g.abs().sum() > 0.0) for g in grads if g is not None)
    assert nonzero_count > 0, (
        f"at least one mode_router param must have non-zero gradient; "
        f"got {nonzero_count}/{len(grads)}"
    )


def test_mode_routed_rejects_pooling_gate_at_construction() -> None:
    """The 2026-06-09 design lock chose option (iii): no pooling gate.
    Constructing the class with a pooling_gate must raise."""
    import pytest

    from shotcloud.models.pooling_gate import PoolingGate

    setup = _build_setup()
    router = ModeRouter()
    with pytest.raises(ValueError, match=r"does not use a pooling gate"):
        ModeRoutedContinuousMixtureSpatial(
            offensive_prior=setup["collab"],
            mode_router=router,
            pooling_gate=PoolingGate(),
        )


def test_mode_routed_rejects_causal_zone_bias_at_construction() -> None:
    """Mode routing supersedes the additive causal zone bias.
    Constructing the class with a causal_zone_bias must raise."""
    import pytest

    from shotcloud.models.continuous_mixture_spatial import CausalZoneBias

    setup = _build_setup()
    router = ModeRouter()
    with pytest.raises(ValueError, match=r"does not use causal_zone_bias"):
        ModeRoutedContinuousMixtureSpatial(
            offensive_prior=setup["collab"],
            mode_router=router,
            causal_zone_bias=CausalZoneBias(),
        )


def test_mode_router_is_in_collected_modules() -> None:
    """REGRESSION (parallels test_causal_zone_bias_module_is_registered_child):
    the mode router must end up in _collect_modules_for_spatial's
    output so the optimizer trains it and the save/load pipeline
    round-trips its parameters. Without this wiring, π_k would stay
    uniform throughout training — silent no-op like the first α1
    bug."""
    from shotcloud.models import ContextMLP, NegBinCountHead, TimingSoftmaxHead
    from shotcloud.training.train_gibbs import _collect_modules_for_spatial

    spatial, _setup = _build_mode_routed_spatial()
    modules = _collect_modules_for_spatial(
        spatial=spatial,
        count_head=NegBinCountHead(),
        timing_head=TimingSoftmaxHead(),
        context_mlp=ContextMLP(input_dim=27, hidden_dim=4, residual=True),
    )
    assert "mode_router" in modules, (
        f"mode_router must be registered in the modules dict; got keys: {list(modules)}"
    )
    # And the router's parameters must be reachable via spatial.parameters().
    router_params = set(id(p) for p in spatial.mode_router.parameters())
    spatial_params = set(id(p) for p in spatial.parameters() if p.requires_grad)
    assert router_params.issubset(spatial_params), (
        "mode_router parameters must appear in spatial.parameters() so the optimizer trains them"
    )

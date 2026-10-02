"""Spatial training losses for the Gibbs / continuous-mixture decoders.

The model predicts a discrete distribution over court cells
``log p_Θ(c | context) = log_probs[b, c]``. The observed data,
however, are exact continuous coordinates ``y_b = (x_b, y_b)`` in
court feet, not categorical cell labels. Training the spatial head
against exact-cell cross-entropy throws away that geometric signal —
a prediction one foot from the observed shot pays the same penalty
as a prediction thirty feet away.

This module exposes a distance-aware alternative.

Continuous-coordinate marginal likelihood
-----------------------------------------

Treat each observed coordinate as drawn from a Gaussian observation
kernel around the latent predicted cell center:

.. math::

    K_\\tau(y - x_c) \\propto \\exp(-\\|y - x_c\\|^2 / 2\\tau^2),
    \\qquad
    f_\\Theta(y \\mid \\text{context})
    = \\sum_c p_\\Theta(c \\mid \\text{context})\\, K_\\tau(y - x_c).

The per-shot NLL is then ``-log f_Θ(y_b | context_b)``. Computed in
log-space via ``logsumexp(log_probs + log_K_τ)`` so the existing
log-probability tensor never has to leave log-space.

Recovers the exact-cell loss as :math:`\\tau \\to 0` when the
observed coordinate equals a cell center.

Distance diagnostic
-------------------

``expected_distance_ft`` reports ``E_c[||x_c - y||]`` under the
predicted cell distribution — a geometric sanity metric that responds
to mass moving in the right *direction* even when exact-cell NLL is
slow to improve.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

#: Default rectangular court bounding box ``(x_min, x_max, y_min, y_max)``
#: in feet, used by :func:`half_court_log_normalizer`. Matches the
#: out-of-court filter applied in :func:`shotcloud.data.loaders.load_shots`:
#: shots with ``|x|>25`` ft, ``y<-5`` ft, or ``y>47`` ft are dropped as
#: out-of-court. With the basket at the origin and ``y`` pointing away
#: from the basket, this bounding box covers the entire in-court region
#: that the model ever evaluates.
DEFAULT_COURT_BOUNDS: tuple[float, float, float, float] = (-25.0, 25.0, -5.0, 47.0)


def half_court_log_normalizer(
    support_xy: Tensor,
    sigma: Tensor,
    bounds: tuple[float, float, float, float] = DEFAULT_COURT_BOUNDS,
    *,
    eps: float = 1e-12,
) -> Tensor:
    r"""Analytic ``log Z_m(\mathcal C)`` for an isotropic 2-D Gaussian
    on the rectangular court ``\mathcal C``.

    For an isotropic Gaussian kernel ``K(y; s_m, \sigma_m^2 I)`` centered
    at ``s_m`` with bandwidth ``\sigma_m`` evaluated on the rectangular
    court ``\mathcal C = [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]``,
    the boundary-normalizer

    .. math::

        Z_m(\mathcal C) = \int_\mathcal C K(u; s_m, \sigma_m^2 I)\, du

    factorizes (the Gaussian is separable and the rectangle is
    axis-aligned) into a product of one-dimensional erf differences:

    .. math::

        Z_m
        = \tfrac{1}{4}
          \bigl[
            \operatorname{erf}\bigl((x_{\max}-s_{m,x})/(\sigma_m\sqrt 2)\bigr)
          - \operatorname{erf}\bigl((x_{\min}-s_{m,x})/(\sigma_m\sqrt 2)\bigr)
          \bigr]
          \bigl[
            \operatorname{erf}\bigl((y_{\max}-s_{m,y})/(\sigma_m\sqrt 2)\bigr)
          - \operatorname{erf}\bigl((y_{\min}-s_{m,y})/(\sigma_m\sqrt 2)\bigr)
          \bigr].

    Limits:

    * ``\sigma_m \to 0``: every erf argument saturates to ``\pm 1``, so
      ``Z_m \to 1`` (the kernel is a Dirac mass at ``s_m \in \mathcal C``
      and the on-court integral collects all of it).
    * ``\sigma_m \to \infty``: the differences vanish, ``Z_m \to 0``
      (the kernel spreads over all of ``\mathbb R^2`` and the on-court
      fraction goes to zero).
    * Deep interior of ``\mathcal C`` and modest ``\sigma_m``: erf
      arguments saturate, ``Z_m \approx 1``.

    Parameters
    ----------
    support_xy : Tensor, shape ``(B, M, 2)``
        Per-(row, support) coordinate ``s_m`` in court feet.
    sigma : Tensor, shape ``(B,)`` or ``(B, M)``
        Per-row or per-support-shot Gaussian bandwidth in feet. Must
        be positive (no internal clamping --- callers that allow ``0``
        bandwidth should clamp first).
    bounds : 4-tuple of float
        ``(x_min, x_max, y_min, y_max)`` rectangular court bounding box
        in feet. Defaults to :data:`DEFAULT_COURT_BOUNDS`.
    eps : float
        Lower clamp on ``Z_m`` before the log to guard against numeric
        underflow on extreme-bandwidth configurations. ``log Z_m`` is
        therefore bounded below by ``log(eps)``.

    Returns
    -------
    Tensor of shape ``(B, M)``
        Per-support-shot ``log Z_m``. Same dtype and device as
        ``support_xy``.

    Notes
    -----
    The normalizer is for the **isotropic** kernel only. Anisotropic
    kernels (the radial-tangential and full-covariance stress-test
    variants) need a separate, kernel-specific normalizer that the
    anisotropic-kernel module is responsible for producing.
    """
    if support_xy.dim() != 3 or support_xy.shape[-1] != 2:
        raise ValueError(f"support_xy must be (B, M, 2); got {tuple(support_xy.shape)}")
    if sigma.dim() == 1:
        if sigma.shape[0] != support_xy.shape[0]:
            raise ValueError(
                f"sigma (B,) must have B={support_xy.shape[0]}; got {tuple(sigma.shape)}"
            )
        sig = sigma.unsqueeze(-1)  # (B, 1) → broadcasts to (B, M)
    elif sigma.dim() == 2:
        if sigma.shape != support_xy.shape[:2]:
            raise ValueError(
                f"sigma (B, M) must match support_xy first two dims; got "
                f"{tuple(sigma.shape)} vs {tuple(support_xy.shape[:2])}"
            )
        sig = sigma
    else:
        raise ValueError(f"sigma must be (B,) or (B, M); got {tuple(sigma.shape)}")
    x_min, x_max, y_min, y_max = bounds
    s_x = support_xy[..., 0]  # (B, M)
    s_y = support_xy[..., 1]  # (B, M)
    inv = 1.0 / (sig * math.sqrt(2.0))  # (B, M) or (B, 1)
    erf_xmax = torch.erf((x_max - s_x) * inv)
    erf_xmin = torch.erf((x_min - s_x) * inv)
    erf_ymax = torch.erf((y_max - s_y) * inv)
    erf_ymin = torch.erf((y_min - s_y) * inv)
    z = 0.25 * (erf_xmax - erf_xmin) * (erf_ymax - erf_ymin)
    return torch.log(z.clamp_min(eps))


def _squared_distances(shot_xy: Tensor, cell_centers: Tensor) -> Tensor:
    """``(B, C)`` squared Euclidean distances ``||y_b - x_c||^2`` in ft²."""
    if shot_xy.dim() != 2 or shot_xy.shape[-1] != 2:
        raise ValueError(f"shot_xy must be (B, 2); got {tuple(shot_xy.shape)}")
    if cell_centers.dim() != 2 or cell_centers.shape[-1] != 2:
        raise ValueError(f"cell_centers must be (C, 2); got {tuple(cell_centers.shape)}")
    diff = shot_xy.unsqueeze(1) - cell_centers.unsqueeze(0)  # (B, C, 2)
    return diff.pow(2).sum(dim=-1)  # (B, C)


def continuous_coordinate_nll(
    log_probs: Tensor,
    shot_xy: Tensor,
    cell_centers: Tensor,
    *,
    tau: float = 1.0,
    normalize_kernel: bool = True,
) -> Tensor:
    """Per-shot continuous-coordinate NLL ``-log f_Θ(y_b | context_b)``.

    Parameters
    ----------
    log_probs : Tensor of shape ``(B, C)``
        Log of the predicted cell distribution ``log p_Θ(c | ·)``.
        Rows must be valid log-probabilities (i.e. ``logsumexp`` over
        ``C`` is 0 within float tolerance). Not checked here — the
        upstream Gibbs decoder is responsible.
    shot_xy : Tensor of shape ``(B, 2)``
        Exact observed shot coordinates in court feet. Use the raw
        continuous coordinate (not the snapped cell center).
    cell_centers : Tensor of shape ``(C, 2)``
        Per-cell ``(x, y)`` centers in court feet, in the same flat
        image-layout order ``c = iy*nx + ix`` as ``log_probs``.
    tau : float
        Observation-kernel bandwidth in feet. Smaller → sharper
        (loss approaches exact-cell NLL when ``tau → 0`` and the shot
        lands on a cell center). Default 1.0 ft ≈ one grid cell width.
    normalize_kernel : bool
        When True (default) normalize ``K_τ(y - x_c)`` to sum to 1
        over cells per shot, via log-softmax. Keeps the loss scale
        stable at the court boundary where the un-normalized kernel
        loses mass to absent cells.

    Returns
    -------
    Tensor of shape ``(B,)``
        Per-shot NLL. The caller decides whether to mean over the
        batch — mirrors the existing per-shot pattern in
        ``_epoch()``.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0; got {tau}")
    dist2 = _squared_distances(shot_xy, cell_centers)  # (B, C)
    log_obs_kernel = -0.5 * dist2 / (tau * tau)
    if normalize_kernel:
        log_obs_kernel = log_obs_kernel - torch.logsumexp(log_obs_kernel, dim=-1, keepdim=True)
    log_lik = torch.logsumexp(log_probs + log_obs_kernel, dim=-1)  # (B,)
    return -log_lik


def exact_cell_nll(log_probs: Tensor, cell_idx: Tensor) -> Tensor:
    """Per-shot categorical NLL ``-log p_Θ(c_obs | context)``.

    The legacy spatial training objective. Retained as a comparability
    metric when :func:`continuous_coordinate_nll` is the training
    objective, so cross-run comparison stays meaningful.
    """
    return -log_probs.gather(1, cell_idx.unsqueeze(1)).squeeze(1)


def expected_distance_ft(log_probs: Tensor, shot_xy: Tensor, cell_centers: Tensor) -> Tensor:
    """Per-shot ``E_c[||x_c - y||]`` under the predicted cell distribution.

    Geometric diagnostic for whether the model is concentrating mass
    near the observed shot. Useful when exact-cell NLL barely moves
    but the distribution is in fact rotating toward the right region
    of the court.

    Returns ``(B,)`` in feet.
    """
    dist2 = _squared_distances(shot_xy, cell_centers)
    dist = dist2.clamp_min(0.0).sqrt()  # (B, C)
    probs = log_probs.exp()  # (B, C)
    return (probs * dist).sum(dim=-1)


def continuous_mixture_loglik(
    log_weights: Tensor,
    support_xy: Tensor,
    shot_xy: Tensor,
    sigma: Tensor | None = None,
    *,
    log_kernel: Tensor | None = None,
    log_kernel_extra: Tensor | None = None,
    weights_are_log_probs: bool = False,
    support_mask: Tensor | None = None,
    court_bounds: tuple[float, float, float, float] | None = None,
    log_court_normalizer: Tensor | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Per-row continuous-mixture log-likelihood of the cell-free
    collaborative kernel mixture (paper §"continuous collab KDE").

    Evaluates

    .. math::

        \\log f_\\Theta(y_b)
        = \\operatorname{logsumexp}_m\\!\\left[
            \\log w_{b,m}
            - \\log(2\\pi\\sigma_b^2)
            - \\frac{\\|y_b - s_{b,m}\\|^2}{2\\sigma_b^2}
          \\right].

    The unconstrained 2-D isotropic Gaussian density is on
    :math:`\\mathbb{R}^2`. When the optional ``court_bounds`` argument
    is provided (or an explicit per-(B, M) ``log_court_normalizer``
    is supplied for the anisotropic path), the per-support-shot
    boundary correction

    .. math::

        Z_m(\\mathcal C)
        = \\int_\\mathcal C \\varphi_2(u; s_m, \\sigma_m^2 I)\\, du

    is computed analytically (isotropic case) via
    :func:`half_court_log_normalizer` and subtracted from each
    support shot's log-kernel so the predictive density is a proper
    density on the rectangular court ``\\mathcal C`` rather than on
    :math:`\\mathbb{R}^2`. When both ``court_bounds`` and
    ``log_court_normalizer`` are ``None`` (the default) the function
    behaves bit-identically to the v1 unconstrained-:math:`\\mathbb R^2`
    formulation, preserving every existing trained checkpoint and
    test.

    Parameters
    ----------
    log_weights : Tensor of shape ``(B, M)``
        Unnormalized log-scores per support shot when
        ``weights_are_log_probs=False`` (the default — softmax-normalized
        internally with a numerically stable
        ``log_weights - logsumexp(log_weights)``). If you've already
        normalized externally, pass ``weights_are_log_probs=True``.
    support_xy : Tensor of shape ``(B, M, 2)``
        Per-row support coordinates in court feet.
    shot_xy : Tensor of shape ``(B, 2)``
        Observed shot coordinate per row in court feet.
    sigma : Tensor of shape ``(B,)`` or ``(B, M)``
        Isotropic bandwidth in feet. ``(B,)`` = per-row (one σ shared
        across the row's M support shots — the current fixed/per-row
        path). ``(B, M)`` = per-support-shot (Tier-1a source/zone
        bandwidth: each support shot carries its own σ derived from
        its source ∈ {own, pooled} and zone ∈ [0, N_ZONES)).
    support_mask : Tensor of shape ``(B, M)``, optional
        Bool mask; ``True`` for valid support, ``False`` for padded /
        non-causal / empty-history slots. Invalid slots get a
        ``-inf`` log-weight before normalization so they contribute
        zero mass to the mixture. When ``None`` all slots are valid.
    court_bounds : 4-tuple of float, optional
        ``(x_min, x_max, y_min, y_max)`` rectangular court bounding box
        in feet. When provided (e.g.
        :data:`DEFAULT_COURT_BOUNDS`) the per-support-shot half-court
        normalizer is computed analytically via
        :func:`half_court_log_normalizer` and subtracted from each
        kernel value, so the predictive density is a proper density on
        ``\\mathcal C``. **Only valid with the isotropic ``sigma`` path**
        --- combining ``court_bounds`` with the precomputed
        ``log_kernel`` (anisotropic) path raises. For the anisotropic
        path, the kernel module must supply its own
        ``log_court_normalizer``.
    log_court_normalizer : Tensor of shape ``(B, M)``, optional
        Precomputed per-support-shot ``log Z_m(\\mathcal C)`` for the
        anisotropic-kernel path. When provided, it is subtracted from
        ``log_kernel`` (or from the isotropic ``log_kernel_eff``) and
        ``court_bounds`` must be ``None``. The anisotropic-kernel
        module is responsible for computing this from its own per-zone
        covariance shape.
    eps : float
        Floor on ``sigma**2`` for the divisor.

    Returns
    -------
    Tensor of shape ``(B,)``
        Per-row log-likelihood. Rows whose mask is all-``False``
        (no valid support — e.g. cold-start with no causal analogue
        history) get ``log_lik = -inf`` here; the caller is
        responsible for any floor / fallback policy.
    """
    if log_weights.dim() != 2:
        raise ValueError(f"log_weights must be (B, M); got {tuple(log_weights.shape)}")
    if support_xy.dim() != 3 or support_xy.shape[-1] != 2:
        raise ValueError(f"support_xy must be (B, M, 2); got {tuple(support_xy.shape)}")
    if support_xy.shape[:2] != log_weights.shape:
        raise ValueError(
            f"support_xy first two dims must match log_weights; got "
            f"{tuple(support_xy.shape[:2])} vs {tuple(log_weights.shape)}"
        )
    if shot_xy.dim() != 2 or shot_xy.shape != (log_weights.shape[0], 2):
        raise ValueError(
            f"shot_xy must be (B={log_weights.shape[0]}, 2); got {tuple(shot_xy.shape)}"
        )
    # Mutually exclusive: either pass ``sigma`` (the current isotropic
    # path) and let this function compute the log-kernel, or pass a
    # precomputed ``log_kernel`` (the Tier-2 anisotropic path: the
    # caller's kernel module has already produced the per-shot
    # ``log K(δ)`` from its own per-zone covariance).
    if (sigma is None) == (log_kernel is None):
        raise ValueError(
            "exactly one of `sigma` or `log_kernel` must be provided "
            f"(got sigma={sigma is not None}, log_kernel={log_kernel is not None})"
        )
    if sigma is not None:
        if sigma.dim() == 1:
            if sigma.shape[0] != log_weights.shape[0]:
                raise ValueError(
                    f"sigma (B,) must have B={log_weights.shape[0]}; got {tuple(sigma.shape)}"
                )
        elif sigma.dim() == 2:
            if sigma.shape != log_weights.shape:
                raise ValueError(
                    f"sigma (B, M) must match log_weights shape; got "
                    f"{tuple(sigma.shape)} vs {tuple(log_weights.shape)}"
                )
        else:
            raise ValueError(f"sigma must be (B,) or (B, M); got {tuple(sigma.shape)}")
    else:
        assert log_kernel is not None
        if log_kernel.shape != log_weights.shape:
            raise ValueError(
                f"log_kernel must match log_weights shape (B, M); got "
                f"{tuple(log_kernel.shape)} vs {tuple(log_weights.shape)}"
            )
    if support_mask is not None:
        if support_mask.shape != log_weights.shape:
            raise ValueError(
                f"support_mask must match log_weights shape; got "
                f"{tuple(support_mask.shape)} vs {tuple(log_weights.shape)}"
            )
        log_weights = log_weights.masked_fill(~support_mask, float("-inf"))
        # Cold-start guard. ``logsumexp(all-inf)`` returns NaN in
        # PyTorch (the max-subtraction trick computes -inf - -inf =
        # NaN). Even though we overwrite the row's ``log_lik`` with
        # -inf below via ``torch.where``, the backward pass through
        # ``logsumexp`` already produces ``exp(... - NaN) = NaN`` for
        # those rows, and ``0 * NaN = NaN`` poisons the shared σ /
        # weight gradients. Force one dummy finite entry per
        # cold-start row so the logsumexp is well-defined; the
        # downstream torch.where discards the dummy value.
        no_valid_pre = ~support_mask.any(dim=-1)
        if no_valid_pre.any():
            log_weights = log_weights.clone()
            log_weights[no_valid_pre, 0] = 0.0

    if weights_are_log_probs:
        log_w = log_weights
    else:
        log_w = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)

    if sigma is not None:
        diff = shot_xy.unsqueeze(1) - support_xy  # (B, M, 2)
        dist2 = (diff * diff).sum(dim=-1)  # (B, M)
        # sigma2 broadcasts: (B, 1) when sigma is (B,) (per-row), or (B, M)
        # when sigma is (B, M) (per-support-shot, Tier-1a source/zone σ).
        if sigma.dim() == 1:
            sigma2 = sigma.pow(2).clamp_min(eps).unsqueeze(-1)  # (B, 1)
        else:
            sigma2 = sigma.pow(2).clamp_min(eps)  # (B, M)
        log_kernel_eff = -math.log(2.0 * math.pi) - torch.log(sigma2) - 0.5 * dist2 / sigma2
    else:
        # Anisotropic path: the caller's kernel module has already
        # computed the per-shot ``log K(δ)`` with its own per-zone
        # covariance shape. We just use it as-is.
        assert log_kernel is not None
        log_kernel_eff = log_kernel

    if log_kernel_extra is not None:
        if log_kernel_extra.shape != log_weights.shape:
            raise ValueError(
                f"log_kernel_extra must match log_weights shape (B, M); got "
                f"{tuple(log_kernel_extra.shape)} vs {tuple(log_weights.shape)}"
            )
        # Additive per-(query, support) log multiplier on the kernel.
        # Stratified-court kernel (Phase 1 C1, 2026-06-09): each entry
        # is 0 when the observed shot and the support shot live in the
        # same court stratum (zone), or ``log(epsilon)`` when they
        # don't. The mixture is therefore stratum-respecting: cross-
        # stratum support shots contribute reduced mass even when
        # their Euclidean distance is small.
        log_kernel_eff = log_kernel_eff + log_kernel_extra

    # Half-court boundary correction (AOAS audit item A1, 2026-06-13).
    # Without this step the per-support-shot kernel ``K(y;s_m,σ_m^2 I)``
    # is an unconstrained 2-D Gaussian on ℝ², which leaks mass off-court
    # when s_m is near the rim, baseline, or corner; the resulting
    # ``log f_Θ`` is unnormalized on the court ``C`` and is not
    # directly comparable across architectures that allocate σ
    # differently. ``court_bounds`` opts in to the analytic per-shot
    # log Z_m subtraction (isotropic path), or callers on the
    # anisotropic path can supply ``log_court_normalizer`` precomputed
    # from their kernel module. Both ``None`` preserves the v1
    # behavior bit-exactly.
    if court_bounds is not None and log_court_normalizer is not None:
        raise ValueError(
            "court_bounds and log_court_normalizer are mutually exclusive; "
            "pass one or the other (or neither)"
        )
    if court_bounds is not None:
        if sigma is None:
            raise ValueError(
                "court_bounds requires the isotropic `sigma` path "
                "(precomputed `log_kernel` should pair with the precomputed "
                "`log_court_normalizer` instead)"
            )
        log_z = half_court_log_normalizer(support_xy, sigma, court_bounds)
        log_kernel_eff = log_kernel_eff - log_z
    elif log_court_normalizer is not None:
        if log_court_normalizer.shape != log_weights.shape:
            raise ValueError(
                f"log_court_normalizer must match log_weights shape (B, M); got "
                f"{tuple(log_court_normalizer.shape)} vs {tuple(log_weights.shape)}"
            )
        log_kernel_eff = log_kernel_eff - log_court_normalizer

    log_lik: Tensor = torch.logsumexp(log_w + log_kernel_eff, dim=-1)  # (B,)
    # ``logsumexp`` of an all-``-inf`` row is NaN in PyTorch (the
    # max-of-all-inf in the trick is -inf, so we get
    # ``-inf + log(exp(0) + ...) = -inf + log(N*nan)``). We want
    # a clean ``-inf`` sentinel for rows whose support is entirely
    # masked out (cold-start with no causal analogue history) so the
    # trainer can apply a floor / fallback without NaN-poisoning the
    # backward pass.
    if support_mask is not None:
        no_valid = ~support_mask.any(dim=-1)
        log_lik = torch.where(no_valid, torch.full_like(log_lik, float("-inf")), log_lik)
    return log_lik


def continuous_mixture_nll(
    log_weights: Tensor,
    support_xy: Tensor,
    shot_xy: Tensor,
    sigma: Tensor,
    *,
    weights_are_log_probs: bool = False,
    support_mask: Tensor | None = None,
    court_bounds: tuple[float, float, float, float] | None = None,
    log_court_normalizer: Tensor | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Per-row continuous-mixture NLL (the negation of
    :func:`continuous_mixture_loglik`). Mirrors
    :func:`continuous_coordinate_nll`'s per-shot return signature so
    the trainer can plug it in interchangeably. ``court_bounds`` and
    ``log_court_normalizer`` are forwarded to the underlying loglik;
    see :func:`continuous_mixture_loglik` for the semantics."""
    return -continuous_mixture_loglik(
        log_weights=log_weights,
        support_xy=support_xy,
        shot_xy=shot_xy,
        sigma=sigma,
        weights_are_log_probs=weights_are_log_probs,
        support_mask=support_mask,
        court_bounds=court_bounds,
        log_court_normalizer=log_court_normalizer,
        eps=eps,
    )


def mode_mixture_loglik(
    mode_logits: Tensor,
    mode_mu: Tensor,
    shot_xy: Tensor,
    sigma: Tensor | float,
    *,
    court_bounds: tuple[float, float, float, float] | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Per-row log-likelihood of a small K-mode Gaussian mixture
    (paper §"collaborative mode mixture").

    Used by the cell-free **mode-extraction** spatial path
    (:class:`~shotcloud.models.collaborative_mode_mixture.CollaborativeModeMixtureSpatial`),
    where the mode centers ``μ_k`` are **per-row** convex
    combinations of the attended support coordinates rather than
    fixed/learnable anchors.

    Evaluates

    .. math::

        \\log f_\\Theta(y_b)
        = \\operatorname{logsumexp}_k\\!\\left[
            \\log \\pi_{b,k}
            - \\log(2\\pi \\sigma_{b,k}^2)
            - \\frac{\\|y_b - \\mu_{b,k}\\|^2}{2\\sigma_{b,k}^2}
          \\right].

    Parameters
    ----------
    mode_logits : Tensor of shape ``(B, K)``
        Unnormalized log mode weights; softmax-normalized internally.
    mode_mu : Tensor of shape ``(B, K, 2)``
        Per-row mode centers in court feet.
    shot_xy : Tensor of shape ``(B, 2)``
        Observed shot coordinate per row in court feet.
    sigma : Tensor of shape ``(K,)`` or ``(B,)`` or ``(B, K)`` or scalar
        Per-mode density bandwidth. Broadcast to ``(B, K)`` internally.
    eps : float
        Floor on ``sigma**2`` for the divisor.

    Returns
    -------
    Tensor of shape ``(B,)``
        Per-row log-likelihood.
    """
    if mode_logits.dim() != 2:
        raise ValueError(f"mode_logits must be (B, K); got {tuple(mode_logits.shape)}")
    if mode_mu.dim() != 3 or mode_mu.shape[-1] != 2:
        raise ValueError(f"mode_mu must be (B, K, 2); got {tuple(mode_mu.shape)}")
    if mode_mu.shape[:2] != mode_logits.shape:
        raise ValueError(
            f"mode_mu (B, K) must match mode_logits; got {tuple(mode_mu.shape[:2])} "
            f"vs {tuple(mode_logits.shape)}"
        )
    if shot_xy.shape != (mode_logits.shape[0], 2):
        raise ValueError(
            f"shot_xy must be (B={mode_logits.shape[0]}, 2); got {tuple(shot_xy.shape)}"
        )

    b, k = mode_logits.shape
    if isinstance(sigma, float | int):
        sigma_bk = torch.full(
            (b, k), float(sigma), dtype=mode_logits.dtype, device=mode_logits.device
        )
    elif sigma.dim() == 1 and sigma.shape[0] == k:
        sigma_bk = sigma.unsqueeze(0).expand(b, k)
    elif sigma.dim() == 1 and sigma.shape[0] == b:
        sigma_bk = sigma.unsqueeze(1).expand(b, k)
    elif sigma.dim() == 2 and sigma.shape == (b, k):
        sigma_bk = sigma
    else:
        raise ValueError(f"sigma must broadcast to (B={b}, K={k}); got shape {tuple(sigma.shape)}")

    log_pi = mode_logits - torch.logsumexp(mode_logits, dim=-1, keepdim=True)  # (B, K)
    diff = shot_xy.unsqueeze(1) - mode_mu  # (B, K, 2)
    dist2 = (diff * diff).sum(dim=-1)  # (B, K)
    sigma2 = sigma_bk.pow(2).clamp_min(eps)
    log_kernel = -math.log(2.0 * math.pi) - torch.log(sigma2) - 0.5 * dist2 / sigma2

    # Half-court boundary correction for the mode-mixture path. The
    # mode centers ``μ_k`` are per-row, learned mixture components
    # (not historical shot locations), so they can land anywhere in
    # the modelling region; using ``court_bounds`` here renormalizes
    # each mode's Gaussian on the on-court rectangle, matching the
    # ``continuous_mixture_loglik`` boundary correction.
    if court_bounds is not None:
        log_z = half_court_log_normalizer(mode_mu, sigma_bk, court_bounds)
        log_kernel = log_kernel - log_z

    log_lik: Tensor = torch.logsumexp(log_pi + log_kernel, dim=-1)
    return log_lik


def mode_mixture_nll(
    mode_logits: Tensor,
    mode_mu: Tensor,
    shot_xy: Tensor,
    sigma: Tensor | float,
    *,
    court_bounds: tuple[float, float, float, float] | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Per-row mode-mixture NLL. Negation of :func:`mode_mixture_loglik`."""
    return -mode_mixture_loglik(
        mode_logits, mode_mu, shot_xy, sigma, court_bounds=court_bounds, eps=eps
    )


__all__ = [
    "DEFAULT_COURT_BOUNDS",
    "continuous_coordinate_nll",
    "continuous_mixture_loglik",
    "continuous_mixture_nll",
    "exact_cell_nll",
    "expected_distance_ft",
    "half_court_log_normalizer",
    "mode_mixture_loglik",
    "mode_mixture_nll",
]

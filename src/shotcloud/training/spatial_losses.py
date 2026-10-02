"""Spatial log-likelihoods and losses for the spatial decoders.

Two families of losses are provided.

*Cell-free mixture likelihoods.* :func:`continuous_mixture_loglik`
evaluates the AC-KDE density, a Gaussian kernel mixture over causal
support shots, at the observed coordinate, optionally renormalized to
the rectangular court with :func:`half_court_log_normalizer`.
:func:`mode_mixture_loglik` does the same for a small per-row Gaussian
mode mixture.

*Grid-cell losses.* For decoders that output a distribution over court
cells, ``log p_Θ(c | context) = log_probs[b, c]``, the observed data are
still exact coordinates in feet. Exact-cell cross-entropy
(:func:`exact_cell_nll`) ignores that geometry: a prediction one foot
from the observed shot pays the same penalty as one thirty feet away.
:func:`continuous_coordinate_nll` is a distance-aware alternative, and
:func:`expected_distance_ft` is a geometric diagnostic.

Continuous-coordinate marginal likelihood
-----------------------------------------

Treat each observed coordinate as drawn from a Gaussian observation
kernel around the latent predicted cell center:

.. math::

    K_\\tau(y - x_c) \\propto \\exp(-\\|y - x_c\\|^2 / 2\\tau^2),
    \\qquad
    f_\\Theta(y \\mid \\text{context})
    = \\sum_c p_\\Theta(c \\mid \\text{context})\\, K_\\tau(y - x_c).

The per-shot NLL is ``-log f_Θ(y_b | context_b)``, computed as
``logsumexp(log_probs + log_K_τ)`` so the computation stays in log space.
It recovers the exact-cell loss as :math:`\\tau \\to 0` when the observed
coordinate equals a cell center.

Distance diagnostic
-------------------

:func:`expected_distance_ft` reports ``E_c[||x_c - y||]`` under the
predicted cell distribution, a geometric metric that responds to mass
moving toward the observed shot even when the exact-cell NLL changes
little.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

#: Default rectangular court bounding box ``(x_min, x_max, y_min, y_max)``
#: in feet, used by :func:`half_court_log_normalizer`. Matches the
#: default out-of-court filter of :func:`shotcloud.data.loaders.load_shots`, which
#: drops shots with ``|x| > 25``, ``y < -5`` or ``y > 47`` (basket at the
#: origin, ``y`` pointing away from the basket), so the box covers every
#: location the model evaluates.
DEFAULT_COURT_BOUNDS: tuple[float, float, float, float] = (-25.0, 25.0, -5.0, 47.0)


def half_court_log_normalizer(
    support_xy: Tensor,
    sigma: Tensor,
    bounds: tuple[float, float, float, float] = DEFAULT_COURT_BOUNDS,
    *,
    eps: float = 1e-12,
) -> Tensor:
    r"""Log on-court mass ``log Z_m(\mathcal C)`` of isotropic Gaussian kernels.

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
    The normalizer is for the isotropic kernel only. Anisotropic kernels
    (:mod:`shotcloud.models.anisotropic_kernel`) need a kernel-specific
    normalizer, passed to :func:`continuous_mixture_loglik` as
    ``log_court_normalizer``.
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
        Rows must be normalized (``logsumexp`` over ``C`` equal to 0 up
        to float tolerance); this is not checked.
    shot_xy : Tensor of shape ``(B, 2)``
        Exact observed shot coordinates in court feet. Use the raw
        continuous coordinate (not the snapped cell center).
    cell_centers : Tensor of shape ``(C, 2)``
        Per-cell ``(x, y)`` centers in court feet, in the same flat
        image-layout order ``c = iy*nx + ix`` as ``log_probs``.
    tau : float, default 1.0
        Observation-kernel bandwidth in feet. Smaller values are sharper;
        the loss approaches the exact-cell NLL as ``tau → 0`` when the
        shot lies on a cell center.
    normalize_kernel : bool, default True
        Normalize ``K_τ(y - x_c)`` to sum to one over cells for each shot
        (a log-softmax). Keeps the loss scale stable near the court
        boundary, where the unnormalized kernel loses mass to cells
        outside the grid.

    Returns
    -------
    Tensor of shape ``(B,)``
        Per-shot NLL; reduction over the batch is left to the caller.
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

    The grid-cell training objective for ``spatial_likelihood="cell"``;
    also reported as a comparison metric when
    :func:`continuous_coordinate_nll` is optimized.

    Parameters
    ----------
    log_probs : Tensor of shape ``(B, C)``
        Log cell probabilities.
    cell_idx : Tensor of shape ``(B,)``, int64
        Observed cell per shot.

    Returns
    -------
    Tensor of shape ``(B,)``
        Per-shot NLL.
    """
    return -log_probs.gather(1, cell_idx.unsqueeze(1)).squeeze(1)


def expected_distance_ft(log_probs: Tensor, shot_xy: Tensor, cell_centers: Tensor) -> Tensor:
    """Per-shot ``E_c[||x_c - y||]`` under the predicted cell distribution.

    Geometric diagnostic of whether the predicted mass concentrates near
    the observed shot; it can improve while the exact-cell NLL barely
    changes. ``log_probs`` is ``(B, C)``, ``shot_xy`` is ``(B, 2)`` and
    ``cell_centers`` is ``(C, 2)`` as in :func:`continuous_coordinate_nll`.
    Returns a ``(B,)`` tensor in feet.
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
    """Per-row log-likelihood of the cell-free AC-KDE kernel mixture.

    Evaluates

    .. math::

        \\log f_\\Theta(y_b)
        = \\operatorname{logsumexp}_m\\!\\left[
            \\log w_{b,m}
            - \\log(2\\pi\\sigma_b^2)
            - \\frac{\\|y_b - s_{b,m}\\|^2}{2\\sigma_b^2}
          \\right].

    Each Gaussian kernel is a density on :math:`\\mathbb{R}^2`. With
    ``court_bounds`` (isotropic kernels) or ``log_court_normalizer`` (any
    kernel), each support shot's log-kernel is reduced by the log of its
    on-court mass

    .. math::

        Z_m(\\mathcal C)
        = \\int_\\mathcal C \\varphi_2(u; s_m, \\sigma_m^2 I)\\, du,

    computed analytically by :func:`half_court_log_normalizer` in the
    isotropic case, so the predictive density integrates to one over the
    rectangular court :math:`\\mathcal C`. With neither argument the
    kernels are normalized on :math:`\\mathbb{R}^2`.

    Parameters
    ----------
    log_weights : Tensor of shape ``(B, M)``
        Support log-weights. Unnormalized logits by default, normalized
        internally as ``log_weights - logsumexp(log_weights)``; pass
        ``weights_are_log_probs=True`` if they are already normalized.
    support_xy : Tensor of shape ``(B, M, 2)``
        Per-row support coordinates in court feet.
    shot_xy : Tensor of shape ``(B, 2)``
        Observed shot coordinate per row in court feet.
    sigma : Tensor of shape ``(B,)`` or ``(B, M)``, optional
        Isotropic bandwidth in feet: one per row, or one per support shot
        (e.g. a per-(source, zone) bandwidth). Exactly one of ``sigma``
        and ``log_kernel`` must be given.
    log_kernel : Tensor of shape ``(B, M)``, optional
        Precomputed log-kernel ``log K_m(y_b - s_{b,m})``, e.g. from an
        anisotropic kernel module. Replaces the isotropic kernel.
    log_kernel_extra : Tensor of shape ``(B, M)``, optional
        Additive log-multiplier on each kernel value, e.g. a cross-zone
        attenuation that is 0 when the observed shot and the support shot
        share a court zone and ``log(epsilon)`` otherwise.
    weights_are_log_probs : bool, default False
        Whether ``log_weights`` is already normalized.
    support_mask : Tensor of shape ``(B, M)``, optional
        Boolean mask, ``True`` for valid support and ``False`` for padded
        or empty slots. Invalid slots receive a ``-inf`` log-weight before
        normalization and so carry no mass. ``None`` means all valid.
    court_bounds : 4-tuple of float, optional
        ``(x_min, x_max, y_min, y_max)`` court rectangle in feet, e.g.
        :data:`DEFAULT_COURT_BOUNDS`. Requires the isotropic ``sigma``
        path; mutually exclusive with ``log_court_normalizer``.
    log_court_normalizer : Tensor of shape ``(B, M)``, optional
        Precomputed per-support-shot ``log Z_m(\\mathcal C)``, supplied by
        a kernel module that defines its own kernel shape. Mutually
        exclusive with ``court_bounds``.
    eps : float, default 1e-12
        Floor on ``sigma**2``.

    Returns
    -------
    Tensor of shape ``(B,)``
        Per-row log-likelihood. Rows whose mask is all ``False`` (no valid
        support) get ``-inf``; any floor or fallback is left to the
        caller.

    Raises
    ------
    ValueError
        On shape mismatches, when both or neither of ``sigma`` and
        ``log_kernel`` are given, or when ``court_bounds`` is combined
        with ``log_court_normalizer`` or with ``log_kernel``.
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
    # Either ``sigma`` (isotropic kernel computed here) or a precomputed
    # ``log_kernel`` from a kernel module with its own covariance shape.
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
        # Cold-start guard. ``logsumexp`` over an all-``-inf`` row is NaN
        # (the max-subtraction computes -inf - -inf). Although such rows
        # are overwritten with -inf below via ``torch.where``, the
        # backward pass through ``logsumexp`` would still produce NaN, and
        # ``0 * NaN = NaN`` would poison the shared σ and weight
        # gradients. One finite placeholder entry per cold-start row keeps
        # the logsumexp well defined; ``torch.where`` discards its value.
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
        # sigma2 is (B, 1) for a per-row σ and (B, M) for a per-support σ.
        if sigma.dim() == 1:
            sigma2 = sigma.pow(2).clamp_min(eps).unsqueeze(-1)  # (B, 1)
        else:
            sigma2 = sigma.pow(2).clamp_min(eps)  # (B, M)
        log_kernel_eff = -math.log(2.0 * math.pi) - torch.log(sigma2) - 0.5 * dist2 / sigma2
    else:
        # Precomputed log-kernel from the caller's kernel module.
        assert log_kernel is not None
        log_kernel_eff = log_kernel

    if log_kernel_extra is not None:
        if log_kernel_extra.shape != log_weights.shape:
            raise ValueError(
                f"log_kernel_extra must match log_weights shape (B, M); got "
                f"{tuple(log_kernel_extra.shape)} vs {tuple(log_weights.shape)}"
            )
        # Additive per-(query, support) log-multiplier on the kernel. For
        # the zone-stratified kernel each entry is 0 when the observed
        # shot and the support shot share a court zone and log(epsilon)
        # otherwise, so cross-zone support contributes reduced mass even
        # at small Euclidean distance.
        log_kernel_eff = log_kernel_eff + log_kernel_extra

    # Court boundary correction. On ℝ² each kernel K(y; s_m, σ_m² I)
    # leaks mass off the court when s_m is near an edge of the court
    # rectangle, so log f_Θ is not normalized on the court and is not
    # comparable across models that allocate σ differently. Subtracting
    # log Z_m makes every kernel a density on the court rectangle.
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
    # Rows with no valid support (cold start) get a clean -inf sentinel
    # in place of the placeholder value, so the caller can apply a floor
    # or fallback.
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
    """Per-row continuous-mixture NLL, the negation of :func:`continuous_mixture_loglik`.

    Supports the isotropic ``sigma`` path only; all arguments are
    forwarded unchanged. Returns a ``(B,)`` tensor, like
    :func:`continuous_coordinate_nll`.
    """
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
    """Per-row log-likelihood of a small K-mode Gaussian mixture.

    Used by the mode-mixture spatial decoder
    (:class:`~shotcloud.models.collaborative_mode_mixture.CollaborativeModeMixtureSpatial`),
    an alternative to the support-shot mixture in which the mode centers
    ``μ_k`` are per-row convex combinations of the attended support
    coordinates.

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
    sigma : Tensor of shape ``(K,)`` or ``(B,)`` or ``(B, K)``, or float
        Mode bandwidth in feet, broadcast to ``(B, K)``.
    court_bounds : 4-tuple of float, optional
        ``(x_min, x_max, y_min, y_max)`` court rectangle in feet. When
        given, each mode's Gaussian is renormalized to the rectangle via
        :func:`half_court_log_normalizer`.
    eps : float, default 1e-12
        Floor on ``sigma**2``.

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

    # Court boundary correction, as in ``continuous_mixture_loglik``. Mode
    # centers are learned per row rather than observed shot locations,
    # so they can lie anywhere in the modeling region.
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

"""Conditional mixture density network (MDN) baseline.

A generic neural conditional-density baseline for shot location. It
sits between the classical reference KDE and the structured AC-KDE in
the paper's baselines. The MDN consumes the same causal context
vector (the raw :math:`\\tilde x_n \\in \\mathbb R^{27}`) that the
AC-KDE consumes and produces

.. math::

    p_{\\mathrm{MDN}}(y \\mid x_n)
    = \\sum_{k=1}^{K} \\pi_k(x_n)\\,
      \\mathcal N_2\\!\\bigl(y;\\,\\mu_k(x_n),\\,\\operatorname{diag}(\\sigma_k(x_n))^2\\bigr).

Parameterization (each head is one linear layer on a shared MLP
trunk):

* :math:`\\pi_k(x_n) = \\operatorname{softmax}_k[\\mathbf W^\\pi h(x_n)]`
  — :math:`K` mixture weights.
* :math:`\\mu_k(x_n) = \\mathbf W^\\mu h(x_n)` — :math:`2K` means.
* :math:`\\sigma_k(x_n) = \\varepsilon + \\exp(\\mathbf W^\\sigma h(x_n))`
  — :math:`2K` per-component diagonal scales with a soft floor
  :math:`\\varepsilon` to prevent collapse to a delta. Default
  :math:`\\varepsilon = 0.1` ft.

This is a deliberately plain MDN: shared trunk, diagonal covariance,
no court-aware reparameterisation. It measures how far *generic
context conditioning* gets relative to *structured causal support
borrowing*; it is not a court-specialised neural density estimator.

Sampling uses ancestral mixture-component draws followed by a
Gaussian draw with per-component diagonal scales, with rejection
against the court rectangle (defaults to the same bounds AC-KDE
sampling uses) and a clip-to-bounds fallback after a fixed number
of rejected attempts.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from shotcloud.simulation.sampler import (
    DEFAULT_COURT_XLIM,
    DEFAULT_COURT_YLIM,
)


class ConditionalMDN(nn.Module):
    """Conditional MDN with diagonal-:math:`\\Sigma` components.

    Parameters
    ----------
    input_dim : int, default 27
        Context dimension (matches :data:`shotcloud.data.context.CONTEXT_DIM`).
    hidden_dim : int, default 128
        Width of each shared-trunk MLP layer.
    n_components : int, default 16
        Number of Gaussian mixture components :math:`K`.
    eps_sigma : float, default 0.1
        Soft floor on the per-component standard deviation in feet.
        Added linearly after the exponential of the raw head so
        :math:`\\sigma_k \\ge \\varepsilon` everywhere; prevents
        single components collapsing to a delta during NLL training.
    n_hidden_layers : int, default 2
        Number of GELU-activated linear layers in the trunk.
    init_sigma_ft : float, default 8.0
        Initial per-component standard deviation in feet; must exceed
        ``eps_sigma``.
    court_bounds : tuple of float, default (-25.0, 25.0, -5.0, 47.0)
        ``(x_lo, x_hi, y_lo, y_hi)`` rectangle over which the initial
        component means are scattered.

    Inputs are the raw 27-dim context :math:`\\tilde x_n` (no
    :class:`~shotcloud.models.ContextMLP` in front); the MDN's own trunk learns the
    representation it needs. This deliberately separates the
    baseline's representation learning from AC-KDE's
    :math:`f_{\\mathrm{ctx}}` — the comparison is generic neural
    density on the same input envelope, not on the same learned
    embedding.
    """

    def __init__(
        self,
        input_dim: int = 27,
        hidden_dim: int = 128,
        n_components: int = 16,
        eps_sigma: float = 0.1,
        n_hidden_layers: int = 2,
        init_sigma_ft: float = 8.0,
        court_bounds: tuple[float, float, float, float] = (-25.0, 25.0, -5.0, 47.0),
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive; got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}")
        if n_components <= 0:
            raise ValueError(f"n_components must be positive; got {n_components}")
        if eps_sigma <= 0:
            raise ValueError(f"eps_sigma must be positive; got {eps_sigma}")
        if n_hidden_layers <= 0:
            raise ValueError(f"n_hidden_layers must be positive; got {n_hidden_layers}")
        if init_sigma_ft <= eps_sigma:
            raise ValueError(
                f"init_sigma_ft must exceed eps_sigma; got {init_sigma_ft} <= {eps_sigma}"
            )
        x_lo, x_hi, y_lo, y_hi = court_bounds
        if not (x_lo < x_hi and y_lo < y_hi):
            raise ValueError(f"invalid court_bounds {court_bounds}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_components = int(n_components)
        self.eps_sigma = float(eps_sigma)

        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.GELU())
            in_dim = self.hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.head_logits = nn.Linear(self.hidden_dim, self.n_components)
        self.head_mu = nn.Linear(self.hidden_dim, 2 * self.n_components)
        self.head_log_sigma = nn.Linear(self.hidden_dim, 2 * self.n_components)

        # Court-scale init: zero-init head weights so step-0 predictions
        # depend only on the heads' biases. Bias mu to a coarse scatter
        # across the half-court so each component starts at a different
        # plausible mode; bias log_sigma so initial sigma is broad
        # (init_sigma_ft) and the step-0 log-density at any shot is
        # finite and not catastrophically negative.
        with torch.no_grad():
            self.head_logits.weight.zero_()
            self.head_logits.bias.zero_()
            self.head_mu.weight.zero_()
            self.head_log_sigma.weight.zero_()

            # Seeded uniform random init for mu: scatter the K components
            # across the half-court rectangle.
            gen = torch.Generator().manual_seed(0)
            mu_init_x = torch.rand(self.n_components, generator=gen) * (x_hi - x_lo) + x_lo
            mu_init_y = torch.rand(self.n_components, generator=gen) * (y_hi - y_lo) + y_lo
            mu_bias = torch.stack([mu_init_x, mu_init_y], dim=-1).flatten()  # (2K,)
            self.head_mu.bias.copy_(mu_bias)

            # log_sigma bias chosen so sigma = init_sigma_ft after the
            # exp + eps_sigma floor: solve exp(b) + eps = init_sigma_ft.
            log_sigma_bias = float(torch.log(torch.tensor(init_sigma_ft - eps_sigma)).item())
            self.head_log_sigma.bias.fill_(log_sigma_bias)

    def forward(self, x_n_raw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(log_pi, mu, sigma)`` per row.

        Shapes: ``log_pi`` is ``(B, K)``, ``mu`` and ``sigma`` are
        ``(B, K, 2)``. ``sigma`` is the standard deviation (not
        variance) with the soft floor already added.
        """
        if x_n_raw.dim() != 2 or x_n_raw.shape[-1] != self.input_dim:
            raise ValueError(f"x_n_raw must be (B, {self.input_dim}); got {tuple(x_n_raw.shape)}")
        h = self.trunk(x_n_raw)
        log_pi = torch.log_softmax(self.head_logits(h), dim=-1)
        # Clamp the raw mu and log_sigma head outputs to a safe range
        # before they propagate. The half-court spans 50 ft × 52 ft; mu
        # clamped to [-50, 50] still covers any plausible component
        # placement plus a margin, and log_sigma clamped to [-3, 4]
        # gives σ in [exp(-3), exp(4)] ≈ [0.05, 55] ft before the
        # eps_sigma floor — broad enough to be a uniform-over-court
        # baseline and tight enough to be a near-delta at the rim.
        # Together these bounds rule out NaN/inf in the diff/sigma and
        # log_norm computations regardless of how the head weights drift.
        mu = self.head_mu(h).view(-1, self.n_components, 2).clamp(min=-50.0, max=50.0)
        log_sigma = self.head_log_sigma(h).view(-1, self.n_components, 2).clamp(min=-3.0, max=4.0)
        sigma = torch.exp(log_sigma) + self.eps_sigma
        return log_pi, mu, sigma

    def log_prob(self, y: Tensor, x_n_raw: Tensor) -> Tensor:
        """Per-row log density :math:`\\log p_{\\mathrm{MDN}}(y \\mid x_n)`.

        Shapes: ``y`` is ``(B, 2)``, ``x_n_raw`` is ``(B, input_dim)``.
        Returns ``(B,)``.

        Implementation uses a stable log-sum-exp over component log-
        Gaussian densities; the diagonal covariance reduces to a
        per-dim sum of squared standardised residuals plus a
        :math:`\\sum \\log \\sigma` normaliser.
        """
        if y.dim() != 2 or y.shape[-1] != 2:
            raise ValueError(f"y must be (B, 2); got {tuple(y.shape)}")
        if y.shape[0] != x_n_raw.shape[0]:
            raise ValueError(
                f"y and x_n_raw must agree on batch dim; got {y.shape[0]} vs {x_n_raw.shape[0]}"
            )
        log_pi, mu, sigma = self.forward(x_n_raw)
        diff = y.unsqueeze(1) - mu
        log_norm = (
            -0.5 * ((diff / sigma) ** 2).sum(dim=-1)
            - torch.log(sigma).sum(dim=-1)
            - math.log(2.0 * math.pi)
        )
        return torch.logsumexp(log_pi + log_norm, dim=-1)

    def density_at_xy(self, xy: Tensor, x_n_raw: Tensor) -> Tensor:
        """Plain density (not log) at arbitrary points.

        Shapes: ``xy`` is ``(B, M, 2)`` for ``B`` rows each
        evaluating ``M`` query coordinates, and ``x_n_raw`` is
        ``(B, input_dim)``. Returns ``(B, M)``.

        Used for density-surface metrics: HDR coverage, zone Brier,
        and the smoothed-log score's Monte-Carlo integration.
        """
        if xy.dim() != 3 or xy.shape[-1] != 2:
            raise ValueError(f"xy must be (B, M, 2); got {tuple(xy.shape)}")
        if x_n_raw.shape[0] != xy.shape[0]:
            raise ValueError(
                f"xy and x_n_raw must agree on batch dim; got {xy.shape[0]} vs {x_n_raw.shape[0]}"
            )
        log_pi, mu, sigma = self.forward(x_n_raw)
        # diff: (B, M, K, 2)
        diff = xy.unsqueeze(2) - mu.unsqueeze(1)
        sigma_e = sigma.unsqueeze(1)
        log_norm = (
            -0.5 * ((diff / sigma_e) ** 2).sum(dim=-1)
            - torch.log(sigma_e).sum(dim=-1)
            - math.log(2.0 * math.pi)
        )
        # log_pi: (B, K) → (B, 1, K)
        log_density = torch.logsumexp(log_pi.unsqueeze(1) + log_norm, dim=-1)
        return torch.exp(log_density)

    def sample(
        self,
        x_n_raw: Tensor,
        n_samples: int,
        *,
        court_xlim: tuple[float, float] = DEFAULT_COURT_XLIM,
        court_ylim: tuple[float, float] = DEFAULT_COURT_YLIM,
        max_attempts: int = 8,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw ``n_samples`` locations per row, rejection-sampled to court.

        Shapes: ``x_n_raw`` is ``(B, input_dim)``, returns
        ``(B, n_samples, 2)``. Court bounds default to the
        :data:`shotcloud.simulation.sampler.DEFAULT_COURT_XLIM` /
        ``YLIM`` rectangle (matching AC-KDE's evaluator). After
        ``max_attempts`` rounds of rejection, any remaining
        out-of-court draws are redrawn once more and clipped to the
        rectangle, so every returned point lies inside the court.

        ``generator`` seeds the Gaussian draws only; the mixture
        component indices are drawn from the global RNG.
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive; got {n_samples}")
        log_pi, mu, sigma = self.forward(x_n_raw)
        b = log_pi.shape[0]
        device = x_n_raw.device

        probs = log_pi.exp()
        cat = torch.distributions.Categorical(probs=probs)
        # Categorical.sample((n,)) returns (n, B); transpose to (B, n).
        k_idx = cat.sample((n_samples,)).transpose(0, 1)  # type: ignore[no-untyped-call]  # (B, n)

        # Gather component-specific mu and sigma for every (row, draw).
        gather_idx = k_idx.unsqueeze(-1).expand(b, n_samples, 2)
        mu_chosen = mu.gather(1, gather_idx)
        sigma_chosen = sigma.gather(1, gather_idx)

        x_lo, x_hi = court_xlim
        y_lo, y_hi = court_ylim

        out = torch.empty(b, n_samples, 2, device=device, dtype=mu.dtype)
        out_filled = torch.zeros(b, n_samples, dtype=torch.bool, device=device)
        for _ in range(max_attempts):
            eps = torch.randn(b, n_samples, 2, device=device, dtype=mu.dtype, generator=generator)
            cand = mu_chosen + sigma_chosen * eps
            valid = (
                (cand[..., 0] >= x_lo)
                & (cand[..., 0] <= x_hi)
                & (cand[..., 1] >= y_lo)
                & (cand[..., 1] <= y_hi)
            )
            accept = valid & ~out_filled
            out[accept] = cand[accept]
            out_filled = out_filled | accept
            if out_filled.all():
                break
        if not out_filled.all():
            # Clip residual rejected draws into the legal rectangle.
            eps = torch.randn(b, n_samples, 2, device=device, dtype=mu.dtype, generator=generator)
            cand = mu_chosen + sigma_chosen * eps
            cand[..., 0] = cand[..., 0].clamp(x_lo, x_hi)
            cand[..., 1] = cand[..., 1].clamp(y_lo, y_hi)
            out[~out_filled] = cand[~out_filled]
        return out

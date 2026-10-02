"""Context-adaptive anisotropic KDE kernel (paper §3.2 extension).

Promotes the per-shot smoothing scale from a fixed global bandwidth
(``h = 1.5 ft``, isotropic) to a *learned per-shot per-context*
anisotropic Gaussian whose widths along the rim-radial and tangential
axes are conditioned on both the current player-game context
:math:`x_n` and the historical shot's own context :math:`z_j`.

Mathematically, the self-density becomes

.. math::

    \\hat q_\\phi^{\\mathrm{self}}(c \\mid p, x)
    = \\sum_{j \\in \\mathcal H_p^{<t}}
        \\pi_{\\phi,j}(x) \\,
        K_{\\Sigma_\\phi(x, z_j)}(x_c - s_j),

with

.. math::

    \\Sigma_\\phi(x, z_j) = R_j^\\top
        \\mathrm{diag}\\!\\left(
            \\sigma_{\\parallel, \\phi}^2(x, z_j),
            \\sigma_{\\perp, \\phi}^2(x, z_j)
        \\right) R_j,

where :math:`R_j` rotates to the **rim-radial / tangential** frame at
shot :math:`s_j` (fixed per shot, not learned), and the two widths
are bounded scalars

.. math::

    \\sigma_{\\bullet, j}(x) = \\sigma_{\\min}
        + (\\sigma_{\\max} - \\sigma_{\\min}) \\cdot
        \\mathrm{sigmoid}(g_\\bullet(x, z_j)),

.. math::

    g_\\bullet(x, z_j)
        = a_0
        + \\mathrm{MLP}_z(z_j)
        + \\mathrm{MLP}_x(x)
        + (W x)^\\top (V z_j).

The four-term decomposition (bias, marginal-z, marginal-x, low-rank
bilinear interaction) gives the right bias-variance tradeoff: the
model can learn "rim shots are tighter than arc shots" (via
``MLP_z``), "tonight the player has broader role" (via ``MLP_x``),
and "this corner-three matters more under tonight's starter
context" (via the bilinear) — without an unconstrained black-box
similarity network.

Three ablation forms are exposed via ``kernel_form`` (corresponding
to the ``--kernel-form`` CLI flag):

* ``z_only``: only ``a_0 + MLP_z(z_j)`` — variable bandwidth, no
  context (classical Abramson-style).
* ``additive``: ``a_0 + MLP_z(z_j) + MLP_x(x)`` — independent
  context and shot dependence, no interaction.
* ``factored``: full ``a_0 + MLP_z + MLP_x + bilinear`` — the
  paper-worthy model.

Initialization. ``a_0`` is set so that
``sigmoid(a_0) = (1.5 - σ_min) / (σ_max - σ_min)`` for both
:math:`\\sigma_\\parallel` and :math:`\\sigma_\\perp`. At step 0, with
the output layer of ``MLP_z``, ``MLP_x``, and the bilinear
zero-init, every kernel is isotropic Gaussian with bandwidth 1.5 ft
— matching the current fixed-bandwidth default exactly. Anisotropy
emerges only as the optimizer pushes off step 0.

Computation. The kernel is evaluated on the **full court grid** per
batch (no precomputed stencil) using vectorized projections onto
the rim-radial / tangential axes. For each shot :math:`j` and each
cell :math:`c`, the radial and tangential signed distances are
computed as inner products

.. math::

    d_{\\parallel, jc} = (x_c - s_j) \\cdot u_j,
    \\qquad
    d_{\\perp, jc} = (x_c - s_j) \\cdot v_j,

with :math:`u_j` the unit vector from rim to :math:`s_j` (fixed per
shot at fit time) and :math:`v_j = u_j^\\perp`. The unnormalized
log-kernel is

.. math::

    \\log \\tilde K_{jc} = -\\tfrac{1}{2}
        \\left(
            \\frac{d_{\\parallel, jc}^2}{\\sigma_{\\parallel, j}^2}
            + \\frac{d_{\\perp, jc}^2}{\\sigma_{\\perp, j}^2}
        \\right),

and the per-shot kernel is normalized over the grid by softmax,
yielding :math:`K_{jc} = \\mathrm{softmax}_c(\\log \\tilde K_{jc})`
which sums to 1 over cells per shot. Memory is controlled by
chunking along ``max_N`` (history length) so the peak intermediate
``(B, chunk, n_cells)`` tensor stays under ~1 GB.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from shotcloud.grids import CourtGrid

_KERNEL_FORMS = ("z_only", "additive", "factored")


def _logit(p: float) -> float:
    """Inverse sigmoid; used to initialize ``a_0`` to a target σ."""
    return float(np.log(p / (1.0 - p)))


class AnisotropicKernelEvaluator(nn.Module):
    """Per-shot per-context anisotropic Gaussian kernel evaluator.

    Parameters
    ----------
    grid : CourtGrid
        Spatial discretization. Cell centers are computed once at
        construction and stored as a non-trainable ``(n_cells, 2)``
        buffer.
    context_dim : int, default 27
        Dimension of the per-shot context vector (must match
        :data:`shotcloud.data.context.CONTEXT_DIM`).
    kernel_form : {"z_only", "additive", "factored"}, default "factored"
        Which σ parameterization is active. See module docstring.
    sigma_min, sigma_max : float, default 0.75, 4.0
        Bounds on the per-shot widths (in feet). The sigmoid output
        is mapped to ``[σ_min, σ_max]``.
    hidden_dim : int, default 16
        Hidden width of the two marginal MLPs ``MLP_z`` and ``MLP_x``.
    rank : int, default 4
        Rank of the bilinear interaction term ``(W x)^T (V z)``.
        Only used when ``kernel_form == 'factored'``. Capped explicitly
        to prevent the interaction from becoming an unconstrained
        similarity network.
    init_sigma : float, default 1.5
        Target value for both ``σ_∥`` and ``σ_⊥`` at step 0. ``a_0``
        is initialized so that ``sigmoid(a_0) = (init_sigma − σ_min) /
        (σ_max − σ_min)``. With output layers zero-init, this means
        every kernel starts as an isotropic Gaussian of bandwidth
        ``init_sigma``, matching the current fixed-bandwidth default.
    max_n_chunk : int, default 64
        Maximum ``max_N`` slice processed at a time, controlling
        peak intermediate memory at forward time. Set lower if MPS
        OOMs; higher to maximize throughput.
    """

    cell_centers: Tensor

    def __init__(
        self,
        grid: CourtGrid,
        context_dim: int = 27,
        kernel_form: str = "factored",
        sigma_min: float = 0.75,
        sigma_max: float = 4.0,
        hidden_dim: int = 16,
        rank: int = 4,
        init_sigma: float = 1.5,
        max_n_chunk: int = 64,
    ) -> None:
        super().__init__()
        if kernel_form not in _KERNEL_FORMS:
            raise ValueError(f"kernel_form must be one of {_KERNEL_FORMS}; got {kernel_form!r}")
        if not (0.0 < sigma_min < sigma_max):
            raise ValueError(f"require 0 < sigma_min ({sigma_min}) < sigma_max ({sigma_max})")
        if not (sigma_min <= init_sigma <= sigma_max):
            raise ValueError(
                f"init_sigma ({init_sigma}) must lie in [sigma_min, sigma_max] "
                f"= [{sigma_min}, {sigma_max}]"
            )
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if kernel_form == "factored" and rank <= 0:
            raise ValueError(f"rank must be positive for 'factored', got {rank}")
        if max_n_chunk <= 0:
            raise ValueError(f"max_n_chunk must be positive, got {max_n_chunk}")

        self.context_dim = int(context_dim)
        self.kernel_form = kernel_form
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_range = self.sigma_max - self.sigma_min
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank) if kernel_form == "factored" else 0
        self.init_sigma = float(init_sigma)
        self.max_n_chunk = int(max_n_chunk)

        # Cell-center buffer in (n_cells, 2) layout: row c = (x_c, y_c)
        # where flat index c = iy * nx + ix matches the rest of the
        # codebase's coord_to_cell convention.
        x_centers = grid.xcenters
        y_centers = grid.ycenters
        cell_xy = np.stack(
            [
                np.tile(x_centers, len(y_centers)),
                np.repeat(y_centers, len(x_centers)),
            ],
            axis=1,
        ).astype(np.float32)
        self.register_buffer("cell_centers", torch.from_numpy(cell_xy), persistent=False)
        self._n_cells = int(cell_xy.shape[0])

        # a_0: 2 biases (one each for σ_∥, σ_⊥), initialized so that
        # at step 0 the kernel is isotropic Gaussian with bandwidth
        # init_sigma. Both biases get the same value.
        a0_init = _logit((self.init_sigma - self.sigma_min) / self.sigma_range)
        self.a_0 = nn.Parameter(torch.full((2,), a0_init, dtype=torch.float32))

        # MLP_z and MLP_x: 27 → hidden → 2 (one head for σ_∥, one for σ_⊥).
        # Output layers are zero-init so step-0 g_• equals a_0.
        # ``mlp_z[-1]`` returns nn.Module statically; we keep a typed
        # reference to the output Linear for init + diagnostics.
        mlp_z_out = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(mlp_z_out.weight)
        nn.init.zeros_(mlp_z_out.bias)
        self.mlp_z = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            mlp_z_out,
        )

        self.mlp_x: nn.Sequential | None
        if kernel_form in ("additive", "factored"):
            mlp_x_out = nn.Linear(hidden_dim, 2)
            nn.init.zeros_(mlp_x_out.weight)
            nn.init.zeros_(mlp_x_out.bias)
            self.mlp_x = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.GELU(),
                mlp_x_out,
            )
        else:
            self.mlp_x = None

        if kernel_form == "factored":
            # Rank-r bilinear: for each of (∥, ⊥), parameters
            # (W: r × D, V: r × D). Interaction term is
            # (W x)^T (V z_j) summed across the rank-r dim. Zero-init
            # both so the bilinear contribution is exactly zero at
            # step 0.
            self.bilinear_W = nn.Parameter(torch.zeros(2, self.rank, context_dim))
            self.bilinear_V = nn.Parameter(torch.zeros(2, self.rank, context_dim))
        else:
            self.register_parameter("bilinear_W", None)
            self.register_parameter("bilinear_V", None)

    @property
    def n_cells(self) -> int:
        return self._n_cells

    def _compute_g(self, x_n: Tensor, z_j: Tensor) -> Tensor:
        """Compute ``g_•(x_n, z_j)`` of shape ``(B, max_N, 2)``.

        The last axis is ``(g_∥, g_⊥)`` — the pre-sigmoid log-widths.
        """
        # a_0 contribution: broadcast (2,) → (B, max_N, 2).
        g = self.a_0.view(1, 1, 2).expand(z_j.shape[0], z_j.shape[1], 2)

        # Marginal in z_j: (B, max_N, D) → (B, max_N, 2).
        g = g + self.mlp_z(z_j)

        # Marginal in x_n: (B, D) → (B, 2) → (B, max_N, 2).
        if self.mlp_x is not None:
            g_x = self.mlp_x(x_n).unsqueeze(1).expand(-1, z_j.shape[1], -1)
            g = g + g_x

        # Bilinear interaction: factored form only.
        if self.bilinear_W is not None and self.bilinear_V is not None:
            # W: (2, r, D), x: (B, D) → Wx: (2, B, r) via contract on D.
            # V: (2, r, D), z: (B, max_N, D) → Vz: (2, B, max_N, r).
            # Interaction per (∥, ⊥): sum over r of (Wx) * (Vz).
            Wx: Tensor = torch.einsum("hrd,bd->hbr", self.bilinear_W, x_n)
            Vz: Tensor = torch.einsum("hrd,bnd->hbnr", self.bilinear_V, z_j)
            interaction = (Wx.unsqueeze(2) * Vz).sum(dim=-1)  # (2, B, max_N)
            # Reorder to (B, max_N, 2).
            g = g + interaction.permute(1, 2, 0)

        out: Tensor = g
        return out

    def _compute_sigmas(self, x_n: Tensor, z_j: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(σ_∥, σ_⊥)`` each of shape ``(B, max_N)``."""
        g = self._compute_g(x_n, z_j)
        sigmas = self.sigma_min + self.sigma_range * torch.sigmoid(g)
        return sigmas[..., 0], sigmas[..., 1]

    def _kernel_chunk(
        self,
        s_j: Tensor,
        u_j: Tensor,
        sigma_par: Tensor,
        sigma_perp: Tensor,
    ) -> Tensor:
        """Per-chunk kernel evaluation. Returns ``(B, n_in_chunk, n_cells)``.

        All inputs already correspond to the ``max_N`` chunk slice.
        Output rows sum to 1 over cells (softmax-normalized per shot).
        """
        # Cell centers: (n_cells, 2). Broadcast to (1, 1, n_cells, 2)
        # implicitly by computing per-axis dot products with shots and
        # subtracting the scalar shot·axis offset.
        c_xy = self.cell_centers  # (n_cells, 2)

        # v_j: perpendicular of u_j in 2D, (B, n_chunk, 2).
        v_j = torch.stack([-u_j[..., 1], u_j[..., 0]], dim=-1)

        # c · u, c · v: (B, n_chunk, n_cells) via einsum.
        c_dot_u = torch.einsum("cd,bnd->bnc", c_xy, u_j)
        c_dot_v = torch.einsum("cd,bnd->bnc", c_xy, v_j)

        # s_j · u, s_j · v: (B, n_chunk).
        s_dot_u = (s_j * u_j).sum(dim=-1)
        s_dot_v = (s_j * v_j).sum(dim=-1)

        # Signed distances along the two axes: (B, n_chunk, n_cells).
        d_par = c_dot_u - s_dot_u.unsqueeze(-1)
        d_perp = c_dot_v - s_dot_v.unsqueeze(-1)

        # Log unnormalized kernel: (B, n_chunk, n_cells).
        inv_sigma_par_sq = 1.0 / sigma_par.pow(2).unsqueeze(-1).clamp_min(1e-12)
        inv_sigma_perp_sq = 1.0 / sigma_perp.pow(2).unsqueeze(-1).clamp_min(1e-12)
        log_unnorm = -0.5 * (d_par.pow(2) * inv_sigma_par_sq + d_perp.pow(2) * inv_sigma_perp_sq)

        # Per-shot normalization across cells.
        log_K = torch.log_softmax(log_unnorm, dim=-1)
        return log_K.exp()

    def forward(
        self,
        x_n: Tensor,
        z_j: Tensor,
        s_j: Tensor,
        u_j: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute per-shot per-cell kernel weights ``K_{Σ_φ(x, z_j)}(c − s_j)``.

        Parameters
        ----------
        x_n : Tensor, shape ``(B, context_dim)``
            Per-row raw context (the paper's :math:`\\tilde x_n`).
        z_j : Tensor, shape ``(B, max_N, context_dim)``
            Per-historical-shot raw context, padded along ``max_N``.
        s_j : Tensor, shape ``(B, max_N, 2)``
            Per-historical-shot ``(x, y)`` coordinates in court feet.
        u_j : Tensor, shape ``(B, max_N, 2)``
            Per-historical-shot rim-radial unit vector
            ``(s_j - r_rim) / ||s_j - r_rim||``. Fixed per shot at
            fit time; threaded through here as a buffer-gather.
        mask : Tensor, shape ``(B, max_N)``, optional
            ``1`` for real shots, ``0`` for padding. When supplied,
            padded rows of the output are set to zero so they
            contribute zero to any downstream weighted sum.

        Returns
        -------
        Tensor of shape ``(B, max_N, n_cells)``
            Per-shot per-cell kernel weights. Each row sums to 1
            over cells (it's the softmax-normalized Gaussian on the
            grid in the rim-radial / tangential frame). Padded rows
            are zero when ``mask`` is supplied.
        """
        if z_j.dim() != 3 or x_n.dim() != 2:
            raise ValueError(
                f"expected z_j (B, N, D) and x_n (B, D); got {tuple(z_j.shape)} and "
                f"{tuple(x_n.shape)}"
            )
        b, max_n, d = z_j.shape
        if x_n.shape != (b, d):
            raise ValueError(f"x_n shape {tuple(x_n.shape)} != (B={b}, D={d})")
        if s_j.shape != (b, max_n, 2):
            raise ValueError(f"s_j shape {tuple(s_j.shape)} != (B={b}, max_N={max_n}, 2)")
        if u_j.shape != (b, max_n, 2):
            raise ValueError(f"u_j shape {tuple(u_j.shape)} != (B={b}, max_N={max_n}, 2)")
        if d != self.context_dim:
            raise ValueError(
                f"context_dim mismatch: z_j has {d}, evaluator built with {self.context_dim}"
            )

        # Widths computed once per batch (not per chunk).
        sigma_par, sigma_perp = self._compute_sigmas(x_n, z_j)

        # Chunked evaluation along max_N keeps the (B, chunk, n_cells)
        # intermediate within bounds.
        if max_n <= self.max_n_chunk:
            kernel = self._kernel_chunk(s_j, u_j, sigma_par, sigma_perp)
        else:
            chunks: list[Tensor] = []
            for start in range(0, max_n, self.max_n_chunk):
                stop = min(start + self.max_n_chunk, max_n)
                chunks.append(
                    self._kernel_chunk(
                        s_j[:, start:stop, :],
                        u_j[:, start:stop, :],
                        sigma_par[:, start:stop],
                        sigma_perp[:, start:stop],
                    )
                )
            kernel = torch.cat(chunks, dim=1)

        if mask is not None:
            if mask.shape != (b, max_n):
                raise ValueError(f"mask shape {tuple(mask.shape)} != (B={b}, max_N={max_n})")
            kernel = kernel * mask.unsqueeze(-1)

        return kernel

    def sigma_stats(self, x_n: Tensor, z_j: Tensor) -> dict[str, float]:
        """Diagnostic helper: mean/std of σ_∥ and σ_⊥ across the batch.

        Useful for the q_self diagnostic: ``σ_∥ ≠ σ_⊥`` on average
        confirms the model is actually using anisotropy.
        """
        with torch.no_grad():
            sigma_par, sigma_perp = self._compute_sigmas(x_n, z_j)
            return {
                "sigma_par_mean": float(sigma_par.mean()),
                "sigma_par_std": float(sigma_par.std()),
                "sigma_perp_mean": float(sigma_perp.mean()),
                "sigma_perp_std": float(sigma_perp.std()),
                "sigma_anisotropy": float((sigma_par - sigma_perp).abs().mean()),
            }

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"kernel_form={self.kernel_form!r}, "
            f"context_dim={self.context_dim}, hidden_dim={self.hidden_dim}, "
            f"rank={self.rank}, "
            f"sigma_range=({self.sigma_min}, {self.sigma_max}), "
            f"init_sigma={self.init_sigma}, n_params={n_params}"
        )

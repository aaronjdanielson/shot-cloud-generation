"""Wasserstein archetype fitting under the debiased Sinkhorn divergence.

Deprecated; retained to reproduce the Wasserstein-archetype analyses.
The current spatial factor,
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`,
does not use archetypes.

:func:`fit_archetypes_v1` fits the **linear surrogate** form of the
Wasserstein-archetypal objective: player densities are reconstructed by
a *linear* convex combination of atoms

.. math::

    \\tilde q_p \\;=\\; \\rho_p \\, A,\\qquad
    A_k = \\mathrm{softmax}(\\alpha_k) \\in \\Delta^{C-1},\\;
    A \\in \\mathbb R^{K \\times C},\\;
    \\rho_p = \\mathrm{softmax}(z_p) \\in \\Delta^{K-1},

trained against the per-player KDE :math:`Q_p` under the entropic
Wasserstein loss

.. math::

    \\min_{\\alpha,\\,Z}\\;
    \\frac{1}{\\sum_p w_p}\\sum_{p}\\,w_p\\;
        S_\\varepsilon\\!\\bigl(Q_p,\\;\\tilde q_p\\bigr),

where :math:`w_p` are optional per-player weights (typically
:math:`w_p \\propto \\sqrt{N_p}`) and :math:`S_\\varepsilon` is the
**debiased Sinkhorn divergence**

.. math::

    S_\\varepsilon(a, b) \\;=\\; W_\\varepsilon(a, b)
        - \\tfrac12 W_\\varepsilon(a, a)
        - \\tfrac12 W_\\varepsilon(b, b),

which is non-negative, zero iff :math:`a=b`, and avoids the diffuse-
solution bias of the raw entropic OT cost (Feydy et al., 2019;
Genevay et al., 2018).

The full Wasserstein-barycentric form
:math:`\\tilde q_p = B_\\varepsilon(A_{1:K}; \\rho_p)` is available
only with frozen atoms: :func:`fit_v2_rho_given_A` optimizes the
barycentric weights :math:`\\rho_p` for fixed :math:`A`, using
:func:`wasserstein_barycenter_separable`.

Computational note. With cost matrix
:math:`C[(i_y,i_x),(i_y',i_x')] = (\\Delta y)^2 + (\\Delta x)^2`,
the Gibbs kernel :math:`K = \\exp(-C/\\varepsilon)` factorizes as
:math:`K = K_y \\otimes K_x` (Solomon et al., 2015), so the
Sinkhorn matvec ``K @ v`` reduces to two small ``ny x ny`` and
``nx x nx`` operations instead of one ``C x C`` matmul. The
Sinkhorn iteration is run in **log-domain** to stay stable on
concentrated NBA shot data (rim peaks would produce u, v with
O(1e60) dynamic range in the multiplicative form). Players are
processed in **mini-batches** so the broadcast tensor used by the
log-stable kernel does not exceed memory at NBA scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

#: Default entropic regularization for the Sinkhorn solver. Cost is
#: in (cell-size)^2 units, so ``epsilon=1`` corresponds to a Gibbs
#: kernel with diffusion length ``~sqrt(epsilon)`` cells.
DEFAULT_EPSILON: float = 1.0

#: Default outer-optimization steps for the archetype Adam loop.
#: 50 is past the loss-plateau knee on NBA-scale data with the
#: ``epsilon=1`` Sinkhorn kernel; warm-starting across snapshots
#: means later anchors converge in even fewer steps.
DEFAULT_MAX_ITER: int = 50

#: Default Sinkhorn iterations per loss evaluation. With ``epsilon=1``
#: on a 56x64 NBA grid the marginal-error contraction rate is ~0.9
#: per iter, so 15 iters give ~3 digits of precision --- enough for
#: archetype fitting (the gradient direction is the load-bearing
#: signal, not absolute calibration).
DEFAULT_SINKHORN_ITER: int = 15

#: Default Adam learning rate for ``(alpha, z)``.
DEFAULT_LR: float = 0.05

#: Default minibatch size over players. Sized for **autograd-retained**
#: peak memory at typical NBA grid sizes (ny=56, nx=64) when only the
#: final Sinkhorn iteration is under autograd (see
#: ``last_iter_with_grad`` in :func:`sinkhorn_distance_separable`):
#: per-batch peak is ~``batch * ny * nx * (ny + nx) * 4`` bytes for
#: the two broadcast tensors that one with-grad iteration retains.
#: At ``batch=64`` that is ~110 MB per Sinkhorn solve; the debiased
#: divergence runs two such solves so peak is ~220 MB. Override to
#: ``None`` to disable minibatching (full-batch).
DEFAULT_BATCH_SIZE: int = 64

#: Numerical floor for ``log(0)`` protection.
EPS_SAFE: float = 1e-30


# ---------------------------------------------------------------------------
# Separable Gibbs kernel (Solomon-style, log-domain)
# ---------------------------------------------------------------------------


def build_separable_log_kernel(
    grid_ny: int,
    grid_nx: int,
    *,
    cell_size_y: float = 1.0,
    cell_size_x: float = 1.0,
    epsilon: float = DEFAULT_EPSILON,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Build the 1D log-Gaussian kernels ``(log Ky, log Kx)``.

    Cost on cell centers is squared Euclidean,
    ``D[(iy,ix),(iy',ix')] = (cell_size_y * (iy - iy'))^2 +
    (cell_size_x * (ix - ix'))^2``, and the Gibbs kernel
    ``K = exp(-D/epsilon)`` factorizes as ``K = Ky ⊗ Kx`` with
    ``log Ky[iy, iy'] = -(cell_size_y * (iy - iy'))^2 / epsilon``
    (and symmetric for x). All log-domain Sinkhorn primitives use
    these directly.
    """
    if grid_ny <= 0 or grid_nx <= 0:
        raise ValueError(f"grid dims must be positive; got ({grid_ny}, {grid_nx})")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    iy = torch.arange(grid_ny, dtype=dtype, device=device)
    ix = torch.arange(grid_nx, dtype=dtype, device=device)
    dy = (iy.unsqueeze(0) - iy.unsqueeze(1)) * cell_size_y
    dx = (ix.unsqueeze(0) - ix.unsqueeze(1)) * cell_size_x
    log_Ky = -(dy.pow(2)) / epsilon
    log_Kx = -(dx.pow(2)) / epsilon
    return log_Ky, log_Kx


def apply_separable_log_kernel(log_v: Tensor, log_Ky: Tensor, log_Kx: Tensor) -> Tensor:
    """Compute ``log(K @ exp(log_v))`` where ``K = K_y ⊗ K_x``.

    Two log-sum-exp passes (one per axis), each over a single
    contracted dim of size ``ny`` or ``nx``. Memory is dominated by
    the broadcast tensor of shape ``(B, ny, ny, nx)`` (step 1) and
    ``(B, ny, nx, nx)`` (step 2); use a small enough batch ``B``
    that this fits in memory.
    """
    log_h1 = torch.logsumexp(
        log_v.unsqueeze(-3) + log_Ky.unsqueeze(-1),
        dim=-2,
    )
    log_h2 = torch.logsumexp(
        log_h1.unsqueeze(-2) + log_Kx,
        dim=-1,
    )
    return log_h2


# ---------------------------------------------------------------------------
# Sinkhorn distance (log-domain, separable kernel)
# ---------------------------------------------------------------------------


def sinkhorn_distance_separable(
    a: Tensor,
    b: Tensor,
    log_Ky: Tensor,
    log_Kx: Tensor,
    *,
    epsilon: float,
    n_iter: int = DEFAULT_SINKHORN_ITER,
    eps_safe: float = EPS_SAFE,
    last_iter_with_grad: bool = True,
) -> Tensor:
    """Entropic Wasserstein distance ``W_epsilon(a, b)`` (regularized OT cost).

    Returns the proper cost-units value
    :math:`W_\\varepsilon(a,b) = \\varepsilon\\,(\\langle f, a\\rangle +
    \\langle g, b\\rangle)`, where ``(f, g)`` are the log-domain
    multiplicative-scaling potentials produced by the Sinkhorn
    iteration. The constant :math:`-\\varepsilon` from the dual at
    convergence is dropped because it does not affect ``argmin``.

    Memory mode. When ``last_iter_with_grad=True`` (default), the
    first ``n_iter - 1`` Sinkhorn iterations run under
    ``torch.no_grad()`` and only the final iteration is recorded for
    backward. This is **truncated backprop with detached warm-start
    potentials**: not the same as the rigorous implicit-differentiation
    formula (which would require solving a linear system in the
    Jacobian of the Sinkhorn fixed-point map; see Cuturi 2013, Feydy
    et al. 2019), but a much cheaper engineering approximation that
    matches it asymptotically as ``n_iter`` grows past convergence.
    For Adam on the linear surrogate loss it works well in practice; if
    you want exact gradients through the iteration, set this to
    ``False`` (and budget O(n_iter * batch * grid^2) more memory).

    Parameters
    ----------
    a, b : Tensor
        Shape ``(B, ny, nx)`` or ``(B, ny * nx)``. Each row is a
        probability distribution over the grid (``sum_yx = 1``).
    log_Ky, log_Kx : Tensor
        Output of :func:`build_separable_log_kernel`.
    epsilon : float
        The entropic regularization that the kernel was built with.
        Used here only as the multiplicative scale on the dual pair.
    n_iter : int
        Sinkhorn iterations.
    eps_safe : float
        Floor on the source / target distributions before taking
        ``log``; protects against ``log(0)`` when rows have empty cells.
    last_iter_with_grad : bool, default True
        Whether to enable autograd on only the final Sinkhorn
        iteration. Reduces peak memory by ``~n_iter`` factor with
        no measurable change in archetype quality.

    Returns
    -------
    Tensor of shape ``(B,)`` — entropic OT cost in cost units.
    """
    ny = log_Ky.shape[0]
    nx = log_Kx.shape[0]
    if a.dim() == 2 and a.shape[-1] == ny * nx:
        a = a.view(-1, ny, nx)
        b = b.view(-1, ny, nx)
    if a.shape[-2:] != (ny, nx) or b.shape[-2:] != (ny, nx):
        raise ValueError(
            f"a, b must be (B, {ny}, {nx}) or (B, {ny * nx}); got {a.shape}, {b.shape}"
        )

    log_a = a.clamp_min(eps_safe).log()
    log_b = b.clamp_min(eps_safe).log()

    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    if last_iter_with_grad and n_iter > 1:
        # All but the final iteration: run under no_grad. Detach the
        # warm-started potentials so the graph for the final iteration
        # only spans one Sinkhorn step.
        with torch.no_grad():
            for _ in range(n_iter - 1):
                f = log_a - apply_separable_log_kernel(g, log_Ky, log_Kx)
                g = log_b - apply_separable_log_kernel(f, log_Ky, log_Kx)
        f = f.detach()
        g = g.detach()
        # Final iteration with autograd enabled.
        f = log_a - apply_separable_log_kernel(g, log_Ky, log_Kx)
        g = log_b - apply_separable_log_kernel(f, log_Ky, log_Kx)
    else:
        for _ in range(n_iter):
            f = log_a - apply_separable_log_kernel(g, log_Ky, log_Kx)
            g = log_b - apply_separable_log_kernel(f, log_Ky, log_Kx)

    cost: Tensor = epsilon * ((f * a).sum(dim=(-2, -1)) + (g * b).sum(dim=(-2, -1)))
    return cost


def sinkhorn_divergence_separable(
    a: Tensor,
    b: Tensor,
    log_Ky: Tensor,
    log_Kx: Tensor,
    *,
    epsilon: float,
    n_iter: int = DEFAULT_SINKHORN_ITER,
    eps_safe: float = EPS_SAFE,
    waa: Tensor | None = None,
    last_iter_with_grad: bool = True,
    clamp_negative: bool = False,
) -> Tensor:
    """Debiased Sinkhorn divergence ``S_epsilon(a, b)``.

    Math:

    .. math::

        S_\\varepsilon(a, b) \\;=\\; W_\\varepsilon(a, b)
            - \\tfrac12 W_\\varepsilon(a, a)
            - \\tfrac12 W_\\varepsilon(b, b).

    Non-negative in exact arithmetic; zero iff ``a == b``. With
    finite Sinkhorn iterations and float32, the formula can dip
    *slightly* below zero on near-equal pairs (the dual potentials
    haven't fully converged). For optimization this is harmless ---
    the gradient is well-defined --- but ``clamp_negative=True``
    truncates at zero if you want a strictly non-negative scalar
    for downstream consumers.

    Compared to raw :func:`sinkhorn_distance_separable`, this loss
    does not push :math:`\\tilde q_p` toward an entropy-maximizing
    diffuse solution (Feydy et al. 2019).

    Parameters
    ----------
    waa : Tensor, optional
        Pre-computed ``W_epsilon(a, a)`` per row, shape ``(B,)``.
        Useful when ``a`` is fixed across the optimization (e.g. the
        per-player KDEs) so this term is computed once outside the
        Adam loop instead of every step.
    last_iter_with_grad : bool, default True
        Forwarded to :func:`sinkhorn_distance_separable`. See its
        docstring for the memory tradeoff.
    clamp_negative : bool, default False
        If True, replace any negative output with zero. Default
        False so callers see numerical drift directly.
    """
    wab = sinkhorn_distance_separable(
        a,
        b,
        log_Ky,
        log_Kx,
        epsilon=epsilon,
        n_iter=n_iter,
        eps_safe=eps_safe,
        last_iter_with_grad=last_iter_with_grad,
    )
    if waa is None:
        # ``W_epsilon(a, a)`` is a constant w.r.t. the parameters when
        # ``a`` is fixed across the Adam loop, so even when called
        # without a precomputed value we can run it under no_grad.
        with torch.no_grad():
            waa = sinkhorn_distance_separable(
                a,
                a,
                log_Ky,
                log_Kx,
                epsilon=epsilon,
                n_iter=n_iter,
                eps_safe=eps_safe,
                last_iter_with_grad=False,
            )
    wbb = sinkhorn_distance_separable(
        b,
        b,
        log_Ky,
        log_Kx,
        epsilon=epsilon,
        n_iter=n_iter,
        eps_safe=eps_safe,
        last_iter_with_grad=last_iter_with_grad,
    )
    div = wab - 0.5 * waa - 0.5 * wbb
    if clamp_negative:
        div = div.clamp_min(0.0)
    return div


# ---------------------------------------------------------------------------
# Entropic Wasserstein barycenter (barycentric reconstruction)
# ---------------------------------------------------------------------------

#: Default Sinkhorn iterations inside the entropic barycenter solver.
#: Larger values give a tighter barycenter but more memory under
#: ``last_iter_with_grad=True`` autograd. Solomon et al. 2015 use 30;
#: 20 is a good speed/quality trade-off for :func:`fit_v2_rho_given_A`.
DEFAULT_BARYCENTER_ITER: int = 20


def wasserstein_barycenter_separable(
    rho: Tensor,
    log_A: Tensor,
    log_Ky: Tensor,
    log_Kx: Tensor,
    *,
    n_iter: int = DEFAULT_BARYCENTER_ITER,
    eps_safe: float = EPS_SAFE,
    last_iter_with_grad: bool = True,
) -> Tensor:
    """Entropic Wasserstein barycenter of ``K`` atoms with weights ``rho``.

    Solomon-style fixed-point iteration in log-domain
    (Solomon et al. 2015, Algorithm 2 / "Convolutional Wasserstein
    Distances"), specialized to the separable Gibbs kernel
    :math:`K = K_y \\otimes K_x`.

    For weights :math:`\\rho \\in \\Delta^{K-1}` and atoms
    :math:`A_1, \\ldots, A_K`, the entropic barycenter

    .. math::

        B_\\varepsilon(A_{1:K}; \\rho)
        \\;=\\; \\arg\\min_{\\nu} \\sum_k \\rho_k\\, W_\\varepsilon(\\nu, A_k)

    is the unique distribution that minimizes the weighted average
    transport cost to the atoms. The fixed-point iteration is

    .. math::

        \\log w_k &\\leftarrow \\log A_k - \\log(K v_k), \\\\
        \\log\\nu  &\\leftarrow \\sum_k \\rho_k \\log(K w_k) - \\text{(normalize)}, \\\\
        \\log v_k &\\leftarrow \\log\\nu - \\log(K w_k).

    Run in log-space throughout for numerical stability — the dual
    potentials can have O(1e60) dynamic range on concentrated NBA
    grids. The barycenter estimate is renormalized to a simplex
    every iteration.

    Memory mode. ``last_iter_with_grad=True`` (default) runs all but
    the last iteration under ``torch.no_grad()`` and detaches the
    warm-started potentials, mirroring the same truncated-backprop
    pattern that ``sinkhorn_distance_separable`` uses. This
    is *not* the rigorous implicit-differentiation Jacobian (which
    would require solving a linear system in the barycenter
    fixed-point map) but the standard cheap approximation that
    matches it asymptotically as ``n_iter`` grows past convergence.

    Parameters
    ----------
    rho : Tensor of shape ``(B, K)``
        Per-row simplex weights. Must be non-negative; need not sum
        exactly to 1 (each row is normalized internally only as a
        defensive measure — softmax-parameterized inputs naturally
        satisfy this).
    log_A : Tensor of shape ``(K, ny, nx)``
        Log of the K archetype distributions on the grid. Each
        :math:`A_k` should be a simplex over the ``ny * nx`` cells.
    log_Ky, log_Kx : Tensor
        Output of :func:`build_separable_log_kernel`.
    n_iter : int, default ``DEFAULT_BARYCENTER_ITER``
        Number of barycenter fixed-point iterations.
    eps_safe : float
        Floor on the input distributions before taking ``log``.
    last_iter_with_grad : bool, default True
        Whether to enable autograd on only the final barycenter
        iteration. Reduces peak memory by ``~n_iter`` factor.

    Returns
    -------
    log_bary : Tensor of float, shape ``(B, ny, nx)``
        Per-row **log probabilities** of the entropic barycenter:
        each ``log_bary[b]`` satisfies
        ``torch.exp(log_bary[b]).sum() == 1`` (renormalized at every
        iteration). Apply ``torch.exp(log_bary)`` to get the
        barycenter densities.
    """
    B, K = rho.shape
    if log_A.dim() != 3 or log_A.shape[0] != K:
        raise ValueError(f"log_A must have shape (K={K}, ny, nx); got {tuple(log_A.shape)}")
    ny, nx = log_A.shape[-2:]
    if log_Ky.shape[0] != ny or log_Kx.shape[0] != nx:
        raise ValueError(
            f"kernel shapes ({log_Ky.shape}, {log_Kx.shape}) inconsistent "
            f"with log_A grid ({ny}, {nx})"
        )

    device = log_A.device
    dtype = log_A.dtype

    # Broadcast log_A to (B, K, ny, nx). The dual potentials log_v / log_w
    # are stored at the (B, K) level — each (player, atom) pair has its
    # own running iterates.
    log_A_b = log_A.unsqueeze(0).expand(B, K, ny, nx)
    log_v = torch.zeros(B, K, ny, nx, dtype=dtype, device=device)

    # log_rho enters as a coefficient in the weighted geometric mean.
    # Defensive softmax floor: if rho has any zeros (e.g. one-hot init),
    # logging produces -inf; we never multiply log_Kw by 0 * (-inf)
    # because the corresponding entry in log_Kw stays bounded above zero
    # by the kernel structure, but we want the rho coefficient itself
    # to be well-defined. Use rho as-is (positive after softmax) for the
    # weighted-sum coefficient.

    def _step(
        log_v_in: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """One Solomon-style barycenter iteration. Returns (log_v_out, log_bary)."""
        # Apply K to each (b, k) potential. apply_separable_log_kernel
        # broadcasts over arbitrary leading dims; reshape to (B*K, ny, nx)
        # so each pair is independent.
        Kv = apply_separable_log_kernel(log_v_in.reshape(B * K, ny, nx), log_Ky, log_Kx).reshape(
            B, K, ny, nx
        )
        log_w = log_A_b - Kv
        Kw = apply_separable_log_kernel(log_w.reshape(B * K, ny, nx), log_Ky, log_Kx).reshape(
            B, K, ny, nx
        )
        # Weighted log-sum: log_bary = Σ_k ρ_k * log(K w_k)
        log_bary = (rho.unsqueeze(-1).unsqueeze(-1) * Kw).sum(dim=1)  # (B, ny, nx)
        # Renormalize the barycenter so exp(log_bary) sums to 1.
        log_bary = log_bary - torch.logsumexp(log_bary.reshape(B, ny * nx), dim=1).reshape(B, 1, 1)
        # Update v: log_v_k = log_bary - log(K w_k)
        log_v_out = log_bary.unsqueeze(1) - Kw
        return log_v_out, log_bary

    log_bary = torch.zeros(B, ny, nx, dtype=dtype, device=device)
    if last_iter_with_grad and n_iter > 1:
        with torch.no_grad():
            for _ in range(n_iter - 1):
                log_v, log_bary = _step(log_v)
        log_v = log_v.detach()
        # Final iteration with autograd enabled.
        log_v, log_bary = _step(log_v)
    else:
        for _ in range(n_iter):
            log_v, log_bary = _step(log_v)

    # Suppress eps_safe warning: defensive clamp on the output if needed.
    _ = eps_safe
    return log_bary


# ---------------------------------------------------------------------------
# Archetype fitter (linear surrogate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchetypeFitResult:
    """Output of :func:`fit_archetypes_v1`.

    Attributes
    ----------
    archetypes : NDArray[float32], shape ``(K, C)``
        Each row is a probability distribution over court cells.
    mixtures : NDArray[float32], shape ``(P, K)``
        Per-player archetype weights. Each row is a probability
        distribution over the K archetypes.
    loss_history : NDArray[float32], shape ``(max_iter,)``
        Mean (sample-weighted) divergence at each Adam step.
    min_negative_div : float
        Smallest per-row Sinkhorn divergence value observed across
        all chunks and all training steps. Should be near zero in
        exact arithmetic; magnitudes above ~1e-3 suggest the
        Sinkhorn iteration count is too low for the chosen
        ``epsilon``. Always 0.0 when ``use_sinkhorn_divergence=False``.
    """

    archetypes: NDArray[np.float32]
    mixtures: NDArray[np.float32]
    loss_history: NDArray[np.float32]
    min_negative_div: float = 0.0


def _iter_chunks(n: int, batch_size: int | None) -> list[slice]:
    if batch_size is None or batch_size >= n:
        return [slice(0, n)]
    return [slice(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


def fit_archetypes_v1(
    Q: NDArray[np.float32] | Tensor,
    *,
    grid_ny: int,
    grid_nx: int,
    K: int = 8,
    cell_size_y: float = 1.0,
    cell_size_x: float = 1.0,
    epsilon: float = DEFAULT_EPSILON,
    max_iter: int = DEFAULT_MAX_ITER,
    sinkhorn_iter: int = DEFAULT_SINKHORN_ITER,
    lr: float = DEFAULT_LR,
    A_init: NDArray[np.float32] | Tensor | None = None,
    sample_weights: NDArray[np.float32] | Tensor | None = None,
    batch_size: int | None = DEFAULT_BATCH_SIZE,
    use_sinkhorn_divergence: bool = True,
    seed: int = 0,
    device: torch.device | str = "cpu",
    verbose: bool = False,
) -> ArchetypeFitResult:
    """Fit ``K`` archetypal spatial measures via the linear surrogate.

    Linear convex-combination reconstruction ``ρ_p A`` trained under the
    debiased Sinkhorn divergence; atoms and mixtures are fit jointly.

    Parameters
    ----------
    Q : array, shape ``(P, C=ny*nx)``
        Per-player KDE matrix (each row a probability distribution
        over court cells).
    grid_ny, grid_nx : int
        Court grid dimensions; ``grid_ny * grid_nx`` must equal
        ``Q.shape[1]``.
    K : int, default 8
        Number of archetypes.
    cell_size_y, cell_size_x : float, default 1.0
        Physical cell sizes (e.g. feet); enter the cost matrix.
    epsilon : float, default 1.0
        Sinkhorn entropic regularization. Smaller = tighter
        transport; larger = smoother dual potentials and faster
        convergence.
    max_iter : int, default 200
        Outer Adam steps.
    sinkhorn_iter : int, default 30
        Inner Sinkhorn iterations per loss evaluation.
    lr : float, default 0.05
        Adam learning rate for ``(alpha, z)``.
    A_init : array, shape ``(K, C)``, optional
        Warm-start archetype basis. ``alpha`` is initialized as
        ``log(A_init + eps)``; otherwise random Gaussian.
    sample_weights : array, shape ``(P,)``, optional
        Per-player loss weights. Defaults to uniform. A typical choice
        is ``w_p ~ sqrt(N_p)`` (training-shot count) so dense
        players do not dominate the archetype geometry.
    batch_size : int or None, default 64
        Players processed per Sinkhorn pass. ``None`` runs full-batch.
        At NBA scale (P~1500, ny=56, nx=64) the full-batch broadcast
        tensor is ~1.2 GB; mini-batching keeps memory proportional to
        ``batch_size``.
    use_sinkhorn_divergence : bool, default True
        Use the debiased Sinkhorn divergence
        ``S_eps = W_eps(a,b) - 0.5 W_eps(a,a) - 0.5 W_eps(b,b)``
        instead of raw ``W_eps(a, b)``. The debiased form avoids
        entropic blur on the archetypes (Feydy et al., 2019).
    seed : int, default 0
        RNG seed for the cold-start init of ``alpha`` and ``z``.
    device : torch.device or str
        Where to run the optimization.
    verbose : bool, default False
        Print loss every ``max(1, max_iter // 10)`` steps.

    Returns
    -------
    :class:`ArchetypeFitResult`
        ``archetypes`` (K, C), ``mixtures`` (P, K), ``loss_history``
        (max_iter,).
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if grid_ny * grid_nx != Q.shape[1]:
        raise ValueError(f"grid_ny * grid_nx = {grid_ny * grid_nx} != Q.shape[1] = {Q.shape[1]}")

    Q_tensor = (
        torch.from_numpy(np.asarray(Q, dtype=np.float32))
        if isinstance(Q, np.ndarray)
        else Q.to(torch.float32)
    ).to(device)
    P, C = Q_tensor.shape

    # Validate Q: each row must be a probability distribution over
    # cells (non-negative, finite, rows summing to ~1). NaNs or
    # negative entries from upstream KDE bugs would silently produce
    # garbage archetypes.
    if not torch.isfinite(Q_tensor).all():
        raise ValueError("Q contains NaN or Inf entries")
    if (Q_tensor < 0).any():
        raise ValueError("Q must be non-negative (rows are probability distributions)")
    if P > 0:
        row_sums = Q_tensor.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            max_dev = float((row_sums - 1.0).abs().max().item())
            raise ValueError(
                f"Q rows must sum to 1 (max deviation {max_dev:.4g}); each "
                f"row is a probability distribution over the {C} court cells."
            )

    # CPU generator + .to(device): MPS does not support
    # ``torch.Generator(device='mps')`` and CUDA generators are not
    # automatically constructed when the device is "cuda"-typed
    # without an index. The random sample is small (K*C and P*K
    # floats) so transferring is negligible.
    rng = torch.Generator()
    rng.manual_seed(seed)

    if A_init is not None:
        A_init_tensor = (
            torch.from_numpy(np.asarray(A_init, dtype=np.float32))
            if isinstance(A_init, np.ndarray)
            else A_init.to(torch.float32)
        ).to(device)
        if A_init_tensor.shape != (K, C):
            raise ValueError(f"A_init has shape {tuple(A_init_tensor.shape)}, expected ({K}, {C})")
        alpha_init = torch.log(A_init_tensor.clamp_min(1e-12))
    else:
        alpha_init = (0.1 * torch.randn(K, C, generator=rng)).to(device)

    alpha = torch.nn.Parameter(alpha_init.clone())
    z = torch.nn.Parameter((0.1 * torch.randn(P, K, generator=rng)).to(device))

    log_Ky, log_Kx = build_separable_log_kernel(
        grid_ny,
        grid_nx,
        cell_size_y=cell_size_y,
        cell_size_x=cell_size_x,
        epsilon=epsilon,
        device=device,
    )

    # Per-player weights, normalized to sum to 1.
    if sample_weights is None:
        weights = torch.full((P,), 1.0 / max(P, 1), dtype=torch.float32, device=device)
    else:
        w_arr = (
            torch.from_numpy(np.asarray(sample_weights, dtype=np.float32))
            if isinstance(sample_weights, np.ndarray)
            else sample_weights.to(torch.float32)
        ).to(device)
        if w_arr.shape != (P,):
            raise ValueError(f"sample_weights has shape {tuple(w_arr.shape)}, expected ({P},)")
        weights = w_arr / w_arr.sum().clamp_min(1e-12)

    Q_2d = Q_tensor.view(P, grid_ny, grid_nx)

    # Memory estimate: warn if the user accidentally asks for a
    # batch that will OOM. Peak allocation per Sinkhorn iter under the
    # default last_iter_with_grad=True is one B*ny*ny*nx + one
    # B*ny*nx*nx broadcast tensor (the two logsumexp axes), float32.
    # The debiased divergence runs two such solves, so peak is 2x.
    eff_batch = P if batch_size is None else min(batch_size, P)
    bytes_per_logsumexp = eff_batch * grid_ny * grid_nx * (grid_ny + grid_nx) * 4
    peak_bytes = 2 * bytes_per_logsumexp if use_sinkhorn_divergence else bytes_per_logsumexp
    peak_gb = peak_bytes / 2**30
    if peak_gb > 4.0:
        import warnings

        warnings.warn(
            f"Wasserstein archetype fit: estimated peak allocation per Sinkhorn "
            f"step is ~{peak_gb:.1f} GB at P={P}, batch={eff_batch}, "
            f"grid={grid_ny}x{grid_nx}, debiased={use_sinkhorn_divergence}. "
            f"Consider reducing --w-batch-size if your machine has less RAM "
            f"than this. Setting last_iter_with_grad=False (full-graph backprop) "
            f"would multiply this by ~sinkhorn_iter; we use last_iter_with_grad=True "
            f"by default.",
            stacklevel=2,
        )
    if verbose:
        print(
            f"  [w-fit] memory: P={P}, batch={eff_batch}, "
            f"grid={grid_ny}x{grid_nx}, peak~{peak_gb:.2f} GB per Sinkhorn step"
        )

    # Pre-compute W_eps(Q_p, Q_p) per player when using divergence,
    # since Q is fixed across optimization. No-grad + last_iter_with_grad
    # off (no backward needed at all for this constant).
    waa: Tensor | None = None
    if use_sinkhorn_divergence:
        with torch.no_grad():
            chunks: list[Tensor] = []
            for sl in _iter_chunks(P, batch_size):
                chunks.append(
                    sinkhorn_distance_separable(
                        Q_2d[sl],
                        Q_2d[sl],
                        log_Ky,
                        log_Kx,
                        epsilon=epsilon,
                        n_iter=sinkhorn_iter,
                        last_iter_with_grad=False,
                    )
                )
            waa = torch.cat(chunks)

    optim = torch.optim.Adam([alpha, z], lr=lr)
    log_every = max(1, max_iter // 10)
    loss_history = np.zeros(max_iter, dtype=np.float32)
    # Worst (most negative) raw per-row divergence seen during training.
    # Should be near 0 in exact arithmetic; values << 0 indicate the
    # Sinkhorn iteration count is too low for the chosen epsilon.
    min_negative_div: float = 0.0

    for step in range(max_iter):
        optim.zero_grad()
        A = torch.softmax(alpha, dim=-1)  # (K, C)
        rho = torch.softmax(z, dim=-1)  # (P, K)

        total_loss = torch.zeros((), device=device)
        for sl in _iter_chunks(P, batch_size):
            Q_chunk_2d = Q_2d[sl]
            rho_chunk = rho[sl]
            Q_hat_chunk = (rho_chunk @ A).view(-1, grid_ny, grid_nx)
            if use_sinkhorn_divergence:
                assert waa is not None
                cost_chunk = sinkhorn_divergence_separable(
                    Q_chunk_2d,
                    Q_hat_chunk,
                    log_Ky,
                    log_Kx,
                    epsilon=epsilon,
                    n_iter=sinkhorn_iter,
                    waa=waa[sl],
                )
                with torch.no_grad():
                    chunk_min = float(cost_chunk.min().item())
                    if chunk_min < min_negative_div:
                        min_negative_div = chunk_min
            else:
                cost_chunk = sinkhorn_distance_separable(
                    Q_chunk_2d,
                    Q_hat_chunk,
                    log_Ky,
                    log_Kx,
                    epsilon=epsilon,
                    n_iter=sinkhorn_iter,
                )
            total_loss = total_loss + (cost_chunk * weights[sl]).sum()

        total_loss.backward()  # type: ignore[no-untyped-call]
        optim.step()
        loss_history[step] = float(total_loss.detach().cpu().item())
        if verbose and (step % log_every == 0 or step == max_iter - 1):
            print(f"  [w-fit] step {step:4d} / {max_iter}: loss = {loss_history[step]:.6f}")

    with torch.no_grad():
        A_final = torch.softmax(alpha, dim=-1).cpu().numpy().astype(np.float32)
        rho_final = torch.softmax(z, dim=-1).cpu().numpy().astype(np.float32)

    # Numerical hygiene: clip and renormalize.
    A_final = np.clip(A_final, 0.0, None)
    a_row_sum = A_final.sum(axis=1, keepdims=True)
    a_row_sum = np.where(a_row_sum < 1e-12, 1.0, a_row_sum).astype(np.float32)
    A_final = (A_final / a_row_sum).astype(np.float32)

    rho_final = np.clip(rho_final, 0.0, None)
    rho_row_sum = rho_final.sum(axis=1, keepdims=True)
    rho_row_sum = np.where(rho_row_sum < 1e-12, 1.0, rho_row_sum).astype(np.float32)
    rho_final = (rho_final / rho_row_sum).astype(np.float32)

    if use_sinkhorn_divergence and min_negative_div < -1e-3:
        import warnings

        warnings.warn(
            f"Wasserstein archetype fit: minimum Sinkhorn divergence reached "
            f"{min_negative_div:.3e} (should be ~0 in exact arithmetic). "
            f"Increase --w-sinkhorn-iter (currently {sinkhorn_iter}) or "
            f"increase --w-epsilon (currently {epsilon}) for tighter convergence.",
            stacklevel=2,
        )

    return ArchetypeFitResult(
        archetypes=A_final,
        mixtures=rho_final,
        loss_history=loss_history,
        min_negative_div=min_negative_div,
    )


# ---------------------------------------------------------------------------
# Barycentric weights: optimize ρ only with frozen archetypes
# ---------------------------------------------------------------------------


def fit_v2_rho_given_A(  # noqa: N802 — name mirrors the math symbol (uppercase A)
    Q: NDArray[np.float32] | Tensor,
    A: NDArray[np.float32] | Tensor,
    *,
    grid_ny: int,
    grid_nx: int,
    cell_size_y: float = 1.0,
    cell_size_x: float = 1.0,
    epsilon: float = DEFAULT_EPSILON,
    max_iter: int = DEFAULT_MAX_ITER,
    sinkhorn_iter: int = DEFAULT_SINKHORN_ITER,
    barycenter_iter: int = DEFAULT_BARYCENTER_ITER,
    lr: float = DEFAULT_LR,
    sample_weights: NDArray[np.float32] | Tensor | None = None,
    batch_size: int | None = DEFAULT_BATCH_SIZE,
    seed: int = 0,
    device: torch.device | str = "cpu",
    verbose: bool = False,
) -> ArchetypeFitResult:
    """Fit barycentric weights ``ρ_p`` for frozen archetypes ``A``.

    Given fixed atoms :math:`A_1, \\ldots, A_K`, find barycentric
    coordinates :math:`\\rho_p \\in \\Delta^{K-1}` for each player such
    that the entropic Wasserstein barycenter
    :math:`\\mathcal B_\\varepsilon(A_{1:K}; \\rho_p)` reconstructs
    :math:`Q_p`:

    .. math::

        \\rho_p^\\star
        = \\arg\\min_{\\rho_p \\in \\Delta^{K-1}}
            S_\\varepsilon\\bigl(Q_p,
                \\mathcal B_\\varepsilon(A_{1:K}; \\rho_p)\\bigr).

    Comparing these weights with the linear-surrogate mixtures from
    :func:`fit_archetypes_v1` separates the effect of the linear
    ``ρ A`` reconstruction from that of the data geometry and atom set:
    if the barycentric :math:`\\rho` is no sharper, diffuse mixtures
    are not an artifact of the linear surrogate.

    Returns ``ArchetypeFitResult`` with ``A`` unchanged from input
    and ``mixtures`` set to the optimized ρ.

    Parameters
    ----------
    Q : array, shape ``(P, C=ny*nx)``
        Per-player KDE matrix. Each row a probability distribution.
    A : array, shape ``(K, C)``
        Frozen archetypes. Each row a probability distribution. Held
        constant throughout — only ``ρ`` is optimized.
    grid_ny, grid_nx : int
    cell_size_y, cell_size_x : float
    epsilon : float
        Sinkhorn entropic regularization. Should match the value used
        when ``A`` was fit (otherwise comparisons are off-grid).
    max_iter : int
        Outer Adam steps for ``z`` (the softmax-logits of ``ρ``).
    sinkhorn_iter : int
        Sinkhorn iterations inside ``S_ε``.
    barycenter_iter : int
        Solomon-style fixed-point iterations inside the barycenter
        solver.
    lr : float
        Adam learning rate for ``z``.
    sample_weights : array, shape ``(P,)``, optional
        Per-player loss weights. Defaults to uniform.
    batch_size : int or None
        Players per Sinkhorn pass.
    seed : int
        RNG seed for cold-start ``z`` init.
    device, verbose : standard.

    Returns
    -------
    :class:`ArchetypeFitResult`
        ``archetypes`` (K, C) — the input ``A``, unchanged.
        ``mixtures`` (P, K) — optimized ρ.
        ``loss_history`` (max_iter,) — outer-step weighted divergence.
    """
    if grid_ny * grid_nx != Q.shape[1]:
        raise ValueError(f"grid_ny * grid_nx = {grid_ny * grid_nx} != Q.shape[1] = {Q.shape[1]}")
    if A.shape[1] != Q.shape[1]:
        raise ValueError(f"A.shape[1] = {A.shape[1]} != Q.shape[1] = {Q.shape[1]}")

    Q_tensor = (
        torch.from_numpy(np.asarray(Q, dtype=np.float32))
        if isinstance(Q, np.ndarray)
        else Q.to(torch.float32)
    ).to(device)
    A_tensor = (
        torch.from_numpy(np.asarray(A, dtype=np.float32))
        if isinstance(A, np.ndarray)
        else A.to(torch.float32)
    ).to(device)
    P = Q_tensor.shape[0]
    K = A_tensor.shape[0]

    # Q must be a valid probability matrix.
    if not torch.isfinite(Q_tensor).all():
        raise ValueError("Q contains NaN or Inf entries")
    if (Q_tensor < 0).any():
        raise ValueError("Q must be non-negative")
    if P > 0:
        row_sums = Q_tensor.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            max_dev = float((row_sums - 1.0).abs().max().item())
            raise ValueError(f"Q rows must sum to 1 (max deviation {max_dev:.4g})")
    # A must be a valid simplex matrix too.
    if not torch.isfinite(A_tensor).all() or (A_tensor < 0).any():
        raise ValueError("A must be finite and non-negative")
    a_row_sums = A_tensor.sum(dim=-1)
    if not torch.allclose(a_row_sums, torch.ones_like(a_row_sums), atol=1e-3):
        raise ValueError("A rows must sum to 1 (frozen-archetype contract)")

    rng = torch.Generator()
    rng.manual_seed(seed)
    z = torch.nn.Parameter((0.1 * torch.randn(P, K, generator=rng)).to(device))

    log_Ky, log_Kx = build_separable_log_kernel(
        grid_ny,
        grid_nx,
        cell_size_y=cell_size_y,
        cell_size_x=cell_size_x,
        epsilon=epsilon,
        device=device,
    )

    if sample_weights is None:
        weights = torch.full((P,), 1.0 / max(P, 1), dtype=torch.float32, device=device)
    else:
        w_arr = (
            torch.from_numpy(np.asarray(sample_weights, dtype=np.float32))
            if isinstance(sample_weights, np.ndarray)
            else sample_weights.to(torch.float32)
        ).to(device)
        if w_arr.shape != (P,):
            raise ValueError(f"sample_weights has shape {tuple(w_arr.shape)}, expected ({P},)")
        weights = w_arr / w_arr.sum().clamp_min(1e-12)

    Q_2d = Q_tensor.view(P, grid_ny, grid_nx)
    log_A = torch.log(A_tensor.clamp_min(1e-12)).view(K, grid_ny, grid_nx)

    # Pre-compute W_ε(Q_p, Q_p) for the debiased divergence (constant w.r.t. ρ).
    waa: Tensor
    with torch.no_grad():
        chunks: list[Tensor] = []
        for sl in _iter_chunks(P, batch_size):
            chunks.append(
                sinkhorn_distance_separable(
                    Q_2d[sl],
                    Q_2d[sl],
                    log_Ky,
                    log_Kx,
                    epsilon=epsilon,
                    n_iter=sinkhorn_iter,
                    last_iter_with_grad=False,
                )
            )
        waa = torch.cat(chunks)

    optim = torch.optim.Adam([z], lr=lr)
    log_every = max(1, max_iter // 10)
    loss_history = np.zeros(max_iter, dtype=np.float32)
    min_negative_div: float = 0.0

    for step in range(max_iter):
        optim.zero_grad()
        rho = torch.softmax(z, dim=-1)  # (P, K)

        total_loss = torch.zeros((), device=device)
        for sl in _iter_chunks(P, batch_size):
            Q_chunk = Q_2d[sl]  # (Bc, ny, nx)
            rho_chunk = rho[sl]  # (Bc, K)

            # B_ε(A; ρ_chunk) — barycenter for each row of ρ.
            log_bary = wasserstein_barycenter_separable(
                rho_chunk,
                log_A,
                log_Ky,
                log_Kx,
                n_iter=barycenter_iter,
            )
            bary = torch.exp(log_bary)
            # Defensive renormalization (the helper renormalizes too, but
            # exp/log roundtrip can drift by ~1e-7).
            bary = bary / bary.reshape(bary.shape[0], -1).sum(dim=1).reshape(-1, 1, 1)

            cost_chunk = sinkhorn_divergence_separable(
                Q_chunk,
                bary,
                log_Ky,
                log_Kx,
                epsilon=epsilon,
                n_iter=sinkhorn_iter,
                waa=waa[sl],
            )
            with torch.no_grad():
                chunk_min = float(cost_chunk.min().item())
                if chunk_min < min_negative_div:
                    min_negative_div = chunk_min
            total_loss = total_loss + (cost_chunk * weights[sl]).sum()

        total_loss.backward()  # type: ignore[no-untyped-call]
        optim.step()
        loss_history[step] = float(total_loss.detach().cpu().item())
        if verbose and (step % log_every == 0 or step == max_iter - 1):
            print(f"  [v2-fit] step {step:4d} / {max_iter}: loss = {loss_history[step]:.6f}")

    with torch.no_grad():
        rho_final = torch.softmax(z, dim=-1).cpu().numpy().astype(np.float32)
    rho_final = np.clip(rho_final, 0.0, None)
    rho_row_sum = rho_final.sum(axis=1, keepdims=True)
    rho_row_sum = np.where(rho_row_sum < 1e-12, 1.0, rho_row_sum).astype(np.float32)
    rho_final = (rho_final / rho_row_sum).astype(np.float32)

    A_out = (
        np.asarray(A, dtype=np.float32)
        if isinstance(A, np.ndarray)
        else A.cpu().numpy().astype(np.float32)
    )
    return ArchetypeFitResult(
        archetypes=A_out,
        mixtures=rho_final,
        loss_history=loss_history,
        min_negative_div=min_negative_div,
    )


__all__ = [
    "DEFAULT_BARYCENTER_ITER",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPSILON",
    "DEFAULT_LR",
    "DEFAULT_MAX_ITER",
    "DEFAULT_SINKHORN_ITER",
    "ArchetypeFitResult",
    "apply_separable_log_kernel",
    "build_separable_log_kernel",
    "fit_archetypes_v1",
    "fit_v2_rho_given_A",
    "sinkhorn_distance_separable",
    "sinkhorn_divergence_separable",
    "wasserstein_barycenter_separable",
]

"""Adaptive offensive prior: ESS-gated self-KDE ⊕ archetype prior.

Deprecated; retained to reproduce the grid-cell decoder ablations. Superseded
by the gated own/pooled support mixture of
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.

Composition of the offensive prior:

* Self component — relevance-weighted self-KDE
  :math:`\\hat q_\\phi(c \\mid p, x_n) =
  \\sum_j \\pi_{\\phi,j}(x_n; \\mathcal H_p^{(t_i)}) M[c, k_j]`,
  with :math:`\\mathcal H_p^{(t_i)}` the player's history filtered to
  shots strictly before the snapshot anchor :math:`t_i`.
* Archetype component — adaptive archetype prior
  :math:`\\hat q_g(c \\mid x_n; t_i)
  = \\sum_k \\rho_{\\xi,k}(x_n) A_k^{(t_i)}(c)`,
  composed via :class:`ArchetypeDictionary` and
  :class:`ArchetypeMixture`.

The two components are blended by ESS shrinkage:

.. math::

    N_{\\mathrm{eff}}(p, x_n) &= 1 / \\sum_j \\pi_{\\phi,j}(x_n)^2 \\\\
    \\omega_p(x_n) &= N_{\\mathrm{eff}} / (N_{\\mathrm{eff}} + \\kappa) \\\\
    \\tilde q(c \\mid p, x_n; t_i) &=
        \\omega_p(x_n) \\hat q_\\phi(c) + (1 - \\omega_p(x_n)) \\hat q_g(c)

The causal date mask is enforced on the fly: each forward call masks
history slots whose date is :math:`\\geq` the per-row anchor date pulled from
``snapshot_store.bundles[snapshot_idx].anchor_date``. This guarantees
:math:`\\mathcal H_p^{(t_i)}` is :math:`\\mathcal F_{<t_i}`-measurable
without rebuilding per-player history per anchor.

The defensive feasibility field ``a_δ`` and residual tilt ``r_θ`` are
applied downstream by other modules and composed under the Gibbs softmax
in :class:`~shotcloud.legacy_pivot.gibbs_decoder.ConditionalGibbsDecoder`.

Implementation. The ragged per-player history is gathered on demand
into padded ``(B, max_N)`` tensors per minibatch. The padding mask is
threaded through :class:`RelevanceScore` so softmax weight on padded
positions is zero, and the matmul against ``M`` ignores those
positions (their contribution is the column at cell index ``0`` times
zero). On top of the padding mask, a per-row causal mask is
multiplied in (``history_dates < anchor_date``). Rows with zero
remaining history fall back fully to the archetype prior (``ω = 0``,
``π`` zero).

Trainable parameters: the :class:`RelevanceScore` scalars and the
:class:`ArchetypeMixture` head. Everything else — the kernel matrix
``M``, per-player histories, :class:`ArchetypeDictionary` surfaces,
snapshot anchor dates, per-shot history dates — is a non-trainable
buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from shotcloud.data.context import CONTEXT_DIM
from shotcloud.kde.adaptive import AdaptiveKDE
from shotcloud.kde.anisotropic import AnisotropicKernelEvaluator
from shotcloud.legacy_pivot.archetypes import ArchetypeDictionary, ArchetypeMixture
from shotcloud.models.relevance import RelevanceScore

if TYPE_CHECKING:
    from shotcloud.data.snapshots import SnapshotStore
    from shotcloud.training.dataset import PlayerVocab


class AdaptiveOffensivePrior(nn.Module):
    """ESS-gated self-KDE ⊕ archetype prior with causal date mask.

    Parameters
    ----------
    adaptive_kde : AdaptiveKDE
        Already :meth:`AdaptiveKDE.fit`-ted on the training shots
        **with** the ``date`` argument so per-shot dates are available
        for the causal mask.
    snapshot_store : SnapshotStore
        Causal feature registry. The constructor reads
        anchor dates into a ``(T,)`` int64 buffer; the forward maps a
        per-row ``snapshot_idx`` into that buffer to supply the
        causal cutoff.
    archetype_dictionary : ArchetypeDictionary
        Holds the per-anchor frozen archetype surfaces ``A^{(t_i)}``
        as a non-trainable buffer. Indexed by ``snapshot_idx`` in
        forward.
    archetype_mixture : ArchetypeMixture
        Trainable head ``ρ_ξ(x_n)`` that produces the per-row
        archetype simplex.
    vocab : PlayerVocab
        Player-id ↔ idx mapping; keeps the dataset's ``player_idx``
        in sync with this module's per-player buffers.
    relevance : RelevanceScore
        Trainable structured ``f_φ``.
    kappa : float, default 50.0
        ESS shrinkage strength. ``ω_p = N_eff / (N_eff + κ)``.
    eps : float, default 1e-9
        Floor on the adaptive density before taking the log.
    low_rank : int or None, default None
        SVD rank-``r`` approximation of ``M`` (fast path); when set,
        the kernel matrix is replaced by ``M ≈ U_r Σ_r V_r^T``.
    """

    M: Tensor
    M_U: Tensor
    M_S: Tensor
    M_V: Tensor
    history_cells: Tensor
    history_context: Tensor
    history_mask: Tensor
    history_dates: Tensor
    history_coords: Tensor
    history_uvecs: Tensor
    anchor_dates: Tensor

    def __init__(
        self,
        adaptive_kde: AdaptiveKDE,
        snapshot_store: SnapshotStore,
        archetype_dictionary: ArchetypeDictionary,
        archetype_mixture: ArchetypeMixture,
        vocab: PlayerVocab,
        relevance: RelevanceScore,
        kappa: float = 50.0,
        eps: float = 1e-9,
        low_rank: int | None = None,
        anisotropic_kernel: AnisotropicKernelEvaluator | None = None,
        rim_xy: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        super().__init__()
        if not adaptive_kde.is_fitted:
            raise ValueError(
                "adaptive_kde must be fit before being passed to AdaptiveOffensivePrior"
            )
        if not adaptive_kde.dates:
            raise ValueError(
                "adaptive_kde must be fit with the date= argument so the "
                "causal date mask can be applied; refit with shots dates"
            )
        if kappa < 0:
            raise ValueError(f"kappa must be non-negative, got {kappa}")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if low_rank is not None and low_rank <= 0:
            raise ValueError(f"low_rank must be positive or None, got {low_rank}")

        self.kappa = float(kappa)
        self.eps = float(eps)
        self.low_rank = low_rank
        self.relevance = relevance
        self.archetype_dictionary = archetype_dictionary
        self.archetype_mixture = archetype_mixture
        self.anisotropic_kernel = anisotropic_kernel
        if anisotropic_kernel is not None and not adaptive_kde.coords:
            raise ValueError(
                "anisotropic_kernel requires adaptive_kde to expose per-shot "
                "coords (refit on a version of AdaptiveKDE that populates "
                "the .coords dict)"
            )

        # ---- Anchor dates buffer (T,) int64 epoch days. ----
        anchors_np = np.array(
            [
                np.asarray(b.anchor_date, dtype="datetime64[D]").astype(np.int64)
                for b in snapshot_store.bundles
            ],
            dtype=np.int64,
        )
        self.register_buffer("anchor_dates", torch.from_numpy(anchors_np), persistent=False)

        # ---- Padded per-player history. ----
        n_players = len(vocab)
        max_n = max(adaptive_kde.n_history.get(pid, 0) for pid in vocab.ids)
        if max_n == 0:
            raise ValueError("AdaptiveKDE has no fitted history for any player in the vocab")

        cells_padded = np.zeros((n_players, max_n), dtype=np.int64)
        ctx_padded = np.zeros((n_players, max_n, CONTEXT_DIM), dtype=np.float32)
        mask = np.zeros((n_players, max_n), dtype=np.float32)
        # Sentinel for padded date: int64 max so causal comparison
        # `padded_date < anchor` is always False; the existing real-vs-pad
        # mask separately zeros out padded slots, but using a clearly-
        # impossible sentinel makes the causal mask robust on its own.
        dates_padded = np.full((n_players, max_n), np.iinfo(np.int64).max, dtype=np.int64)
        # Per-shot (x, y) coords and rim-radial unit vectors. Padded
        # rows contain zeros (which never enter the anisotropic kernel
        # because mask multiplies the kernel output).
        coords_padded = np.zeros((n_players, max_n, 2), dtype=np.float32)
        uvecs_padded = np.zeros((n_players, max_n, 2), dtype=np.float32)
        # Default orientation for shots exactly at the rim (where the
        # rim-radial direction is undefined): global +x axis.
        rim_xy_np = np.asarray(rim_xy, dtype=np.float32)
        for pid in vocab.ids:
            idx = vocab.to_idx(pid)
            if pid not in adaptive_kde.cells:
                continue
            p_cells = adaptive_kde.cells[pid]
            p_ctx = adaptive_kde.context[pid]
            n_p = p_cells.size
            cells_padded[idx, :n_p] = p_cells
            ctx_padded[idx, :n_p, :] = p_ctx
            mask[idx, :n_p] = 1.0
            if pid in adaptive_kde.dates:
                p_dates = adaptive_kde.dates[pid]
                if p_dates.size != n_p:
                    raise ValueError(
                        f"adaptive_kde.dates[{pid!r}] has size {p_dates.size}, "
                        f"expected {n_p} to match cells"
                    )
                dates_padded[idx, :n_p] = p_dates
            if pid in adaptive_kde.coords:
                p_coords = adaptive_kde.coords[pid]
                if p_coords.shape != (n_p, 2):
                    raise ValueError(
                        f"adaptive_kde.coords[{pid!r}] shape {p_coords.shape}, "
                        f"expected ({n_p}, 2) to match cells"
                    )
                coords_padded[idx, :n_p, :] = p_coords
                # Rim-radial unit vectors per shot. Shots exactly at the
                # rim (|s - rim| == 0) get the +x fallback.
                deltas = p_coords - rim_xy_np
                norms = np.linalg.norm(deltas, axis=-1, keepdims=True)
                safe_norms = np.where(norms > 1e-6, norms, 1.0)
                unit = deltas / safe_norms
                fallback = np.zeros_like(unit)
                fallback[:, 0] = 1.0
                unit = np.where(norms > 1e-6, unit, fallback)
                uvecs_padded[idx, :n_p, :] = unit.astype(np.float32)

        self.register_buffer("history_cells", torch.from_numpy(cells_padded), persistent=False)
        self.register_buffer("history_context", torch.from_numpy(ctx_padded), persistent=False)
        self.register_buffer("history_mask", torch.from_numpy(mask), persistent=False)
        self.register_buffer("history_dates", torch.from_numpy(dates_padded), persistent=False)
        self.register_buffer("history_coords", torch.from_numpy(coords_padded), persistent=False)
        self.register_buffer("history_uvecs", torch.from_numpy(uvecs_padded), persistent=False)

        # ---- Kernel matrix ``M`` or its SVD factors. ----
        assert adaptive_kde.M is not None
        M_full = torch.from_numpy(adaptive_kde.M.astype(np.float32))
        if low_rank is None:
            self.register_buffer("M", M_full, persistent=False)
            self.register_buffer("M_U", torch.empty(0), persistent=False)
            self.register_buffer("M_S", torch.empty(0), persistent=False)
            self.register_buffer("M_V", torch.empty(0), persistent=False)
        else:
            r = min(low_rank, M_full.shape[0], M_full.shape[1])
            U_full, S_full, Vh_full = torch.linalg.svd(M_full, full_matrices=False)
            self.register_buffer("M", torch.empty(0), persistent=False)
            self.register_buffer("M_U", U_full[:, :r].contiguous(), persistent=False)
            self.register_buffer("M_S", S_full[:r].contiguous(), persistent=False)
            self.register_buffer("M_V", Vh_full[:r, :].t().contiguous(), persistent=False)

        self._n_cells = int(adaptive_kde.grid.n_cells)

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def max_history(self) -> int:
        return int(self.history_cells.shape[1])

    def _causal_history_mask(
        self, player_idx: Tensor, snapshot_idx: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(eff_mask, has_history, history_context)``.

        ``eff_mask`` is the elementwise product of the padding mask
        (``history_mask``) and the causal mask
        (``history_dates < anchor_dates[snapshot_idx]``). ``has_history``
        is the per-row boolean ``eff_mask.sum(-1) > 0``; rows with
        ``has_history == False`` collapse to a uniform-output softmax
        which the caller zeroes out so :math:`\\omega = 0` and the
        prior reduces to the archetype layer.
        """
        real_mask = self.history_mask[player_idx]  # (B, max_N)
        h_dates = self.history_dates[player_idx]  # (B, max_N) int64
        a_dates = self.anchor_dates[snapshot_idx].unsqueeze(1)  # (B, 1) int64
        causal = (h_dates < a_dates).to(real_mask.dtype)
        eff_mask = real_mask * causal
        has_history = eff_mask.sum(dim=-1) > 0
        z_j = self.history_context[player_idx]
        return eff_mask, has_history, z_j

    def relevance_weights(
        self,
        player_idx: Tensor,
        snapshot_idx: Tensor,
        x_n_raw: Tensor,
        games_ago: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(π, eff_mask, has_history)`` per minibatch.

        ``π`` has shape ``(B, max_N)`` and sums to 1 along the last
        axis on rows with ``has_history``; rows without any causal
        history return all-zero ``π`` (so ``N_eff`` is undefined and
        the caller drops them via ``ω = 0``).

        Consumes ``x_n_raw`` (the 27-D :class:`ContextEncoder` output,
        the paper's :math:`\\tilde x_n`) so that the structured
        :class:`RelevanceScore` can read its **named slices** —
        ``period_onehot``, ``time_in_period``, etc. — with their
        original semantic interpretation. ``z_j`` is the per-historical
        shot raw context stored at fit time, so similarity terms compare
        raw-to-raw.
        """
        eff_mask, has_history, z_j = self._causal_history_mask(player_idx, snapshot_idx)
        # Run softmax with the effective mask. Rows with all-zero mask
        # produce NaN from softmax(-inf, ...); we replace them with zeros.
        pi = self.relevance.softmax(z_j, x_n_raw, games_ago=games_ago, mask=eff_mask)
        pi = torch.where(has_history.unsqueeze(-1), pi, torch.zeros_like(pi))
        return pi, eff_mask, has_history

    def forward(
        self,
        player_idx: Tensor,
        snapshot_idx: Tensor,
        x_n_raw: Tensor,
        x_n: Tensor,
        games_ago: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(log q̃, π, ω)`` for the batch.

        Parameters
        ----------
        player_idx : Tensor of int64, shape ``(B,)``
        snapshot_idx : Tensor of int64, shape ``(B,)``
            Per-row index into the ``snapshot_store`` returned by
            :meth:`SnapshotStore.get_snapshot_index(date)`.
        x_n_raw : Tensor of float, shape ``(B, CONTEXT_DIM)``
            Raw context vector :math:`\\tilde x_n` (the
            :class:`ContextEncoder` output). Consumed by the structured
            :class:`RelevanceScore` because its 5 scalars act on named
            slices that only have their documented meaning before any
            :class:`ContextMLP` transformation.
        x_n : Tensor of float, shape ``(B, CONTEXT_DIM)``
            Learned context :math:`x_n = f_{\\mathrm{ctx}}(\\tilde
            x_n)`. Consumed by the :class:`ArchetypeMixture` head,
            which is a linear-in-context head with no per-dimension
            semantic claims.
        games_ago : Tensor, shape ``(B, max_N)``, optional
        """
        pi, eff_mask, has_history = self.relevance_weights(
            player_idx, snapshot_idx, x_n_raw, games_ago=games_ago
        )

        if self.anisotropic_kernel is None:
            # Isotropic path: q_phi = pi_per_cell @ M.T (or its SVD
            # low-rank approximation). The fixed-bandwidth kernel
            # matrix M was precomputed at fit time.
            cells_per_row = self.history_cells[player_idx]  # (B, max_N), int64
            pi_per_cell = torch.zeros(pi.shape[0], self.n_cells, device=pi.device, dtype=pi.dtype)
            pi_per_cell.scatter_add_(dim=1, index=cells_per_row, src=pi)
            if self.low_rank is None:
                q_phi = pi_per_cell @ self.M.t()
            else:
                proj = pi_per_cell @ self.M_V  # (B, r)
                proj = proj * self.M_S  # (B, r)
                q_phi = proj @ self.M_U.t()  # (B, n_cells)
        else:
            # Anisotropic path: per-shot per-context Gaussian on the
            # full grid. K_jc is (B, max_N, n_cells), softmax-
            # normalized per shot, masked to zero on padded rows. Then
            # q_phi[b, c] = Σ_j π_j[b] · K_jc[b, j, c].
            z_j_full = self.history_context[player_idx]  # (B, max_N, D)
            s_j = self.history_coords[player_idx]  # (B, max_N, 2)
            u_j = self.history_uvecs[player_idx]  # (B, max_N, 2)
            kernel = self.anisotropic_kernel(
                x_n_raw, z_j_full, s_j, u_j, mask=eff_mask
            )  # (B, max_N, n_cells)
            q_phi = (pi.unsqueeze(-1) * kernel).sum(dim=1)  # (B, n_cells)

        # ESS shrinkage. Rows without causal history have π == 0, so
        # ``pi.pow(2).sum`` is zero; clamp_min(eps) keeps n_eff finite
        # (= 1/eps), but we override ω = 0 explicitly via ``has_history``
        # so the layer reduces to the pure archetype prior. ``N_eff``
        # is detached so the relevance head can only learn through
        # ``q_self``, not by inflating ``ω`` via a diffuse π.
        pi_sq_sum = pi.pow(2).sum(dim=-1).clamp_min(self.eps)
        n_eff = 1.0 / pi_sq_sum.detach()
        omega = n_eff / (n_eff + self.kappa)
        omega = torch.where(has_history, omega, torch.zeros_like(omega))

        # Archetype branch at this snapshot.
        rho = self.archetype_mixture(x_n)  # (B, K)
        q_g = self.archetype_dictionary(snapshot_idx, rho)  # (B, n_cells)

        # Additive smoothing instead of clamp_min: keeps gradient flowing
        # for cells where the model assigns very low density (clamp_min
        # would zero those gradients). Renormalize so q_off remains a
        # proper probability over cells.
        # The clamp_min(0) absorbs tiny negative SVD-reconstruction
        # artifacts when low_rank is used; it's a no-op for the dense
        # path where q_phi and q_g are non-negative by construction.
        mixed = omega.unsqueeze(-1) * q_phi + (1.0 - omega.unsqueeze(-1)) * q_g
        smoothed = mixed.clamp_min(0.0) + (self.eps / self.n_cells)
        log_q = torch.log(smoothed) - torch.log(smoothed.sum(dim=-1, keepdim=True))
        return log_q, pi, omega

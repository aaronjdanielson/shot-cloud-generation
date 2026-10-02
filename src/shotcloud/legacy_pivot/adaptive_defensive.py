"""Adaptive defensive field — opponent-conditioned spatial reweighting (paper §3.4).

This module is the per-opponent analog of
:class:`shotcloud.legacy_pivot.adaptive_prior.AdaptiveOffensivePrior`. For
opponent :math:`d` and player-game context :math:`x_n`, it computes
the opponent-conditioned reweighting field

.. math::

    a_\\delta(c \\mid d, x_n)
    = \\sum_{j \\in \\mathcal D_d^{<t}}
        \\omega_{\\delta,j}(x_n) \\, K_h(c - s_j^{\\mathrm{opp}}),

where :math:`\\mathcal D_d^{<t}` is the causal pool of shots taken
*against* opponent :math:`d` strictly before snapshot anchor :math:`t`,
and :math:`\\omega_{\\delta,j}(x_n)` are relevance weights produced by a
structured :class:`~shotcloud.models.RelevanceScore`. The field is a
simplex over cells (the weights sum to 1 by construction); the Gibbs
decoder consumes :math:`\\log a_\\delta(c)` as a log-additive term
in its energy

.. math::

    \\log p(c) = \\log q^{\\mathrm{off}}(c) + \\log a_\\delta(c)
        + r_\\theta(c) - \\log Z.

Three structural differences from :class:`AdaptiveOffensivePrior`:

1. **No archetype shrinkage / no ESS gate.** The defensive field is
   a single-component KDE; there is no defensive analog of the
   archetype dictionary. Sparse-history opponents fall through to a
   uniform feasibility surface (mathematically equivalent to "no
   opponent effect" — a constant offset that the Gibbs partition
   function absorbs).
2. **Indexed by opponent, not player.** The padded history buffers
   are keyed by ``opp_idx``; the relevance score uses the *shooter's*
   per-shot context :math:`z_j^{\\mathrm{opp}} = x_n^{\\text{(j)}}`
   captured at the time of each historical opponent shot.
3. **Same** :class:`AdaptiveKDE` **machinery, opponent-grouped at fit
   time.** The underlying kernel matrix and per-key (cells, context,
   dates) histories are produced by calling
   :meth:`AdaptiveKDE.fit(player_id=opponent_codes, ...)` — the
   ``player_id`` argument is the grouping key, semantically
   "opponent" in this use.

Causal date masking follows the same Proposition-1 contract as the
offensive prior: each forward call masks history slots whose date
is :math:`\\geq` the per-row anchor date pulled from the
SnapshotStore. Rows with no causal history fall back to uniform
feasibility (``log a_δ(c) = -log C``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from shotcloud.data.context import CONTEXT_DIM
from shotcloud.kde.adaptive import AdaptiveKDE
from shotcloud.models.relevance import RelevanceScore

if TYPE_CHECKING:
    from shotcloud.data.snapshots import SnapshotStore
    from shotcloud.training.dataset import OpponentVocab


class AdaptiveDefensiveField(nn.Module):
    """Per-opponent context-adaptive spatial reweighting field (paper §3.4).

    Parameters
    ----------
    defensive_kde : AdaptiveKDE
        Already :meth:`AdaptiveKDE.fit`-ted on the training shots
        with ``player_id=opponent_codes`` and ``date=...`` so the
        per-opponent histories carry per-shot dates for the causal
        mask. The kernel matrix ``M`` from this fit is reused as a
        non-trainable buffer.
    snapshot_store : SnapshotStore
        Causal feature registry (paper §2.5). The constructor reads
        anchor dates into a ``(T,)`` int64 buffer; the forward maps
        a per-row ``snapshot_idx`` into that buffer to supply the
        causal cutoff.
    opp_vocab : OpponentVocab
        ``opponent_id ↔ idx`` mapping; aligns the dataset's
        ``opponent_idx`` with this module's per-opponent buffers.
    relevance : RelevanceScore
        Trainable structured similarity head. Independent from the
        offensive prior's relevance module — defensive feasibility
        has its own learnable similarity logic.
    eps : float, default 1e-9
        Floor on the field before taking the log.
    low_rank : int or None, default None
        Optional SVD rank-``r`` approximation of ``M`` (mirrors the
        offensive prior's fast path). ``None`` keeps the dense matrix.
    """

    M: Tensor
    M_U: Tensor
    M_S: Tensor
    M_V: Tensor
    history_cells: Tensor
    history_context: Tensor
    history_mask: Tensor
    history_dates: Tensor
    anchor_dates: Tensor

    def __init__(
        self,
        defensive_kde: AdaptiveKDE,
        snapshot_store: SnapshotStore,
        opp_vocab: OpponentVocab,
        relevance: RelevanceScore,
        eps: float = 1e-9,
        low_rank: int | None = None,
    ) -> None:
        super().__init__()
        if not defensive_kde.is_fitted:
            raise ValueError(
                "defensive_kde must be fit before being passed to AdaptiveDefensiveField"
            )
        if not defensive_kde.dates:
            raise ValueError(
                "defensive_kde must be fit with the date= argument so the "
                "causal date mask can be applied; refit with shot dates"
            )
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if low_rank is not None and low_rank <= 0:
            raise ValueError(f"low_rank must be positive or None, got {low_rank}")

        self.eps = float(eps)
        self.low_rank = low_rank
        self.relevance = relevance

        # ---- Anchor dates buffer (T,) int64 epoch days. ----
        anchors_np = np.array(
            [
                np.asarray(b.anchor_date, dtype="datetime64[D]").astype(np.int64)
                for b in snapshot_store.bundles
            ],
            dtype=np.int64,
        )
        self.register_buffer("anchor_dates", torch.from_numpy(anchors_np), persistent=False)

        # ---- Padded per-opponent history. ----
        n_opps = len(opp_vocab)
        max_n = max(defensive_kde.n_history.get(oid, 0) for oid in opp_vocab.ids)
        if max_n == 0:
            raise ValueError("defensive_kde has no fitted history for any opponent in the vocab")

        cells_padded = np.zeros((n_opps, max_n), dtype=np.int64)
        ctx_padded = np.zeros((n_opps, max_n, CONTEXT_DIM), dtype=np.float32)
        mask = np.zeros((n_opps, max_n), dtype=np.float32)
        # Sentinel for padded date: int64 max so causal comparison
        # `padded_date < anchor` is always False; the explicit padding
        # mask zeroes out padded slots, but using an impossible sentinel
        # keeps the causal mask robust on its own.
        dates_padded = np.full((n_opps, max_n), np.iinfo(np.int64).max, dtype=np.int64)
        for oid in opp_vocab.ids:
            idx = opp_vocab.to_idx(oid)
            if oid not in defensive_kde.cells:
                continue
            o_cells = defensive_kde.cells[oid]
            o_ctx = defensive_kde.context[oid]
            n_o = o_cells.size
            cells_padded[idx, :n_o] = o_cells
            ctx_padded[idx, :n_o, :] = o_ctx
            mask[idx, :n_o] = 1.0
            if oid in defensive_kde.dates:
                o_dates = defensive_kde.dates[oid]
                if o_dates.size != n_o:
                    raise ValueError(
                        f"defensive_kde.dates[{oid!r}] has size {o_dates.size}, "
                        f"expected {n_o} to match cells"
                    )
                dates_padded[idx, :n_o] = o_dates

        self.register_buffer("history_cells", torch.from_numpy(cells_padded), persistent=False)
        self.register_buffer("history_context", torch.from_numpy(ctx_padded), persistent=False)
        self.register_buffer("history_mask", torch.from_numpy(mask), persistent=False)
        self.register_buffer("history_dates", torch.from_numpy(dates_padded), persistent=False)

        # ---- Kernel matrix ``M`` or its SVD factors. ----
        assert defensive_kde.M is not None
        M_full = torch.from_numpy(defensive_kde.M.astype(np.float32))
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

        self._n_cells = int(defensive_kde.grid.n_cells)

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def max_history(self) -> int:
        return int(self.history_cells.shape[1])

    def _causal_history_mask(
        self, opp_idx: Tensor, snapshot_idx: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(eff_mask, has_history, history_context)``.

        ``eff_mask`` is the elementwise product of the padding mask
        (``history_mask``) and the causal mask
        (``history_dates < anchor_dates[snapshot_idx]``). ``has_history``
        is the per-row boolean ``eff_mask.sum(-1) > 0``; rows with
        ``has_history == False`` collapse to a uniform feasibility
        field downstream.
        """
        real_mask = self.history_mask[opp_idx]  # (B, max_N)
        h_dates = self.history_dates[opp_idx]  # (B, max_N) int64
        a_dates = self.anchor_dates[snapshot_idx].unsqueeze(1)  # (B, 1) int64
        causal = (h_dates < a_dates).to(real_mask.dtype)
        eff_mask = real_mask * causal
        has_history = eff_mask.sum(dim=-1) > 0
        z_j = self.history_context[opp_idx]
        return eff_mask, has_history, z_j

    def relevance_weights(
        self,
        opp_idx: Tensor,
        snapshot_idx: Tensor,
        x_n_raw: Tensor,
        games_ago: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(π, eff_mask, has_history)`` per minibatch.

        ``π`` has shape ``(B, max_N)`` and sums to 1 along the last
        axis on rows with ``has_history``; rows without any causal
        defensive history return all-zero ``π`` (which the caller
        replaces with the uniform-feasibility fallback). Consumes
        ``x_n_raw`` (the :class:`ContextEncoder` output) so that the
        structured :class:`RelevanceScore` reads its named slices with
        their documented meaning.
        """
        eff_mask, has_history, z_j = self._causal_history_mask(opp_idx, snapshot_idx)
        pi = self.relevance.softmax(z_j, x_n_raw, games_ago=games_ago, mask=eff_mask)
        pi = torch.where(has_history.unsqueeze(-1), pi, torch.zeros_like(pi))
        return pi, eff_mask, has_history

    def forward(
        self,
        opp_idx: Tensor,
        snapshot_idx: Tensor,
        x_n_raw: Tensor,
        games_ago: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(log a_δ, π, has_history)`` for the batch.

        Parameters
        ----------
        opp_idx : Tensor of int64, shape ``(B,)``
            Per-row opponent index from
            :meth:`OpponentVocab.to_idx`.
        snapshot_idx : Tensor of int64, shape ``(B,)``
            Per-row snapshot index from
            :meth:`SnapshotStore.get_snapshot_index(date)`.
        x_n_raw : Tensor of float, shape ``(B, CONTEXT_DIM)``
            Raw context vector :math:`\\tilde x_n` consumed by the
            structured :class:`RelevanceScore`. The defensive field
            has no archetype mixture or other linear-in-context head,
            so it only needs raw context — no learned ``x_n``.
        games_ago : Tensor, shape ``(B, max_N)``, optional

        Returns
        -------
        log_a_delta : Tensor of float, shape ``(B, n_cells)``
            Log of the per-row reweighting field. Rows with no causal
            history contribute ``-log(n_cells)`` everywhere — a
            constant the Gibbs partition function absorbs.
        pi : Tensor of float, shape ``(B, max_N)``
            The relevance softmax weights (zeros on
            non-``has_history`` rows).
        has_history : Tensor of bool, shape ``(B,)``
            Whether the row had any causal opponent shot in support.
        """
        pi, _eff_mask, has_history = self.relevance_weights(
            opp_idx, snapshot_idx, x_n_raw, games_ago=games_ago
        )

        cells_per_row = self.history_cells[opp_idx]  # (B, max_N), int64
        pi_per_cell = torch.zeros(pi.shape[0], self.n_cells, device=pi.device, dtype=pi.dtype)
        pi_per_cell.scatter_add_(dim=1, index=cells_per_row, src=pi)
        if self.low_rank is None:
            field = pi_per_cell @ self.M.t()
        else:
            proj = pi_per_cell @ self.M_V  # (B, r)
            proj = proj * self.M_S
            field = proj @ self.M_U.t()  # (B, n_cells)

        # Uniform-feasibility fallback for rows with no causal history.
        # log(1 / n_cells) is a constant under the Gibbs partition
        # function, so the row contributes zero to the energy.
        uniform = torch.full_like(field, 1.0 / self.n_cells)
        a_delta = torch.where(has_history.unsqueeze(-1), field, uniform)
        # Additive smoothing instead of clamp_min, parallel to the
        # offensive prior's Decision-1 fix (2026-05-16). Keeps gradient
        # flowing for under-supported cells. Renormalize so the field
        # remains a proper probability — the constant offset slides
        # through the Gibbs partition function untouched. The
        # ``clamp_min(0)`` removes tiny negative SVD-reconstruction
        # artifacts that would otherwise produce log(negative) NaNs;
        # gradient still flows for all non-negative values.
        smoothed = a_delta.clamp_min(0.0) + (self.eps / self.n_cells)
        log_a_delta = torch.log(smoothed) - torch.log(smoothed.sum(dim=-1, keepdim=True))
        return log_a_delta, pi, has_history

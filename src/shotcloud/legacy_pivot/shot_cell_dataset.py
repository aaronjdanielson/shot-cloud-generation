"""Pre-pivot per-shot dataset.

Moved from ``shotcloud.training.dataset`` to
``shotcloud.legacy_pivot`` on 2026-05-15. Replaced in production
by :class:`~shotcloud.training.GibbsShotDataset` (which routes
``snapshot_idx`` and per-game count/timing targets in addition to
the per-shot tuple).

:class:`ShotCellDataset` yields per-shot
``(player_idx, cell_idx, log_q0_p, opponent_idx, x_n)`` 5-tuples
ready for cross-entropy loss against the pre-Gibbs decoder.
``log_q0_p`` is a precomputed log base-measure vector for the
player; the decoder needs it as input. ``opponent_idx`` is
meaningful only when a :class:`DefensiveKDE` is attached (Phase
2 — pre-pivot); defaults to ``0`` otherwise so the tuple shape
is invariant.

Two base-measure variants are supported: :class:`KDEProduct`
(geometric mixture with hand-picked or learnable weights) and
:class:`HierarchicalKDEBase` (additive Bayesian shrinkage, no
mixture weights).
"""

from __future__ import annotations

from dataclasses import dataclass  # noqa: F401 — re-exported via TypeAlias chain below

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from shotcloud.data.context import CONTEXT_DIM, ContextEncoder
from shotcloud.grids import CourtGrid
from shotcloud.kde import HierarchicalKDEBase
from shotcloud.legacy import KDEProduct
from shotcloud.legacy_pivot.defensive_kde import DefensiveKDE
from shotcloud.training.dataset import OpponentVocab, PlayerVocab

BaseMeasure = KDEProduct | HierarchicalKDEBase


class ShotCellDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    """Per-shot training examples: ``(player_idx, cell_idx, log_q0_p, opponent_idx, x_n)``.

    The 5-tuple contract is invariant across modes — sentinel zeros
    are returned for unused channels (``opponent_idx`` when no defensive
    KDE; ``x_n`` shape ``(0,)`` when no context encoder).

    Parameters
    ----------
    shots_df : DataFrame
        Must contain ``x``, ``y``, ``player_id`` columns. When
        ``defensive_kde`` is provided, must additionally contain an
        ``opponent`` column.
    base_measure : KDEProduct or HierarchicalKDEBase
    grid : CourtGrid
    vocab : PlayerVocab, optional
    dtype : torch.dtype, default ``torch.float32``
    cache_components : bool, default False
    defensive_kde : DefensiveKDE, optional
    opponent_vocab : OpponentVocab, optional
    context_encoder : ContextEncoder, optional
    """

    def __init__(
        self,
        shots_df: pd.DataFrame,
        base_measure: BaseMeasure,
        grid: CourtGrid,
        vocab: PlayerVocab | None = None,
        dtype: torch.dtype = torch.float32,
        cache_components: bool = False,
        defensive_kde: DefensiveKDE | None = None,
        opponent_vocab: OpponentVocab | None = None,
        context_encoder: ContextEncoder | None = None,
    ) -> None:
        for col in ("x", "y", "player_id"):
            if col not in shots_df.columns:
                raise KeyError(f"shots_df must contain column {col!r}")

        if cache_components and not isinstance(base_measure, KDEProduct):
            raise ValueError(
                "cache_components=True requires a KDEProduct base measure; "
                f"got {type(base_measure).__name__}. The components (a_p, a_g, a_0) "
                "are only meaningful for the geometric KDE-product mixture."
            )

        if defensive_kde is not None:
            if not defensive_kde.is_fitted:
                raise ValueError("defensive_kde must be fit before being passed to ShotCellDataset")
            if "opponent" not in shots_df.columns:
                raise KeyError("defensive_kde requires shots_df to have an 'opponent' column")
            if opponent_vocab is None:
                opponent_vocab = OpponentVocab.from_ids(defensive_kde.opponents)
        elif opponent_vocab is not None:
            raise ValueError(
                "opponent_vocab provided without defensive_kde — pass them together or omit both."
            )

        if vocab is None:
            vocab = PlayerVocab.from_ids(list(base_measure.hierarchical_kde.player_density_grid))
        self.vocab = vocab
        self.opponent_vocab = opponent_vocab

        df = shots_df[
            shots_df["player_id"].astype(str).isin({str(pid) for pid in vocab.ids})
        ].copy()
        if defensive_kde is not None:
            assert opponent_vocab is not None
            df = df[
                df["opponent"].astype(str).isin({str(oid) for oid in opponent_vocab.ids})
            ].copy()

        x = df["x"].to_numpy(dtype=np.float64)
        y = df["y"].to_numpy(dtype=np.float64)
        cells = grid.coord_to_cell(x, y)
        valid = cells >= 0
        df = df.loc[valid].reset_index(drop=True)
        cells = cells[valid].astype(np.int64)

        if len(df) == 0:
            raise ValueError("no in-court shots remained after filtering")

        log_q0_cache: dict[int, NDArray[np.float64]] = {}
        for pid in vocab.ids:
            log_q0_cache[vocab.to_idx(pid)] = (
                base_measure.log_density(pid).ravel().astype(np.float64)
            )

        n_cells = grid.n_cells
        n_players = len(vocab)
        log_q0_table = np.empty((n_players, n_cells), dtype=np.float64)
        for idx, vec in log_q0_cache.items():
            log_q0_table[idx] = vec
        self.log_q0_table: Tensor = torch.from_numpy(log_q0_table).to(dtype=dtype)

        self.log_qp_table: Tensor | None = None
        self.log_qg_table: Tensor | None = None
        self.log_ql_vector: Tensor | None = None
        if cache_components:
            assert isinstance(base_measure, KDEProduct)
            kde = base_measure.hierarchical_kde
            log_qp_table = np.empty((n_players, n_cells), dtype=np.float64)
            log_qg_table = np.empty((n_players, n_cells), dtype=np.float64)
            for pid in vocab.ids:
                idx = vocab.to_idx(pid)
                log_qp_table[idx] = np.log(kde.player_density(pid, hierarchical=False)).ravel()
                position = kde.player_position[str(pid)]
                log_qg_table[idx] = np.log(kde.position_density(position)).ravel()
            log_ql = np.log(kde.league_density()).ravel()
            self.log_qp_table = torch.from_numpy(log_qp_table).to(dtype=dtype)
            self.log_qg_table = torch.from_numpy(log_qg_table).to(dtype=dtype)
            self.log_ql_vector = torch.from_numpy(log_ql).to(dtype=dtype)

        self.log_qd_table: Tensor | None = None
        if defensive_kde is not None:
            assert opponent_vocab is not None
            n_opps = len(opponent_vocab)
            log_qd_table = np.empty((n_opps, n_cells), dtype=np.float64)
            for oid in opponent_vocab.ids:
                log_qd_table[opponent_vocab.to_idx(oid)] = defensive_kde.log_density(oid).ravel()
            self.log_qd_table = torch.from_numpy(log_qd_table).to(dtype=dtype)

        player_idx = np.array([vocab.to_idx(pid) for pid in df["player_id"]], dtype=np.int64)
        self.player_idx: Tensor = torch.from_numpy(player_idx)
        self.cell_idx: Tensor = torch.from_numpy(cells)
        if defensive_kde is not None:
            assert opponent_vocab is not None
            opp_idx = np.array([opponent_vocab.to_idx(o) for o in df["opponent"]], dtype=np.int64)
            self.opponent_idx: Tensor = torch.from_numpy(opp_idx)
        else:
            self.opponent_idx = torch.zeros(len(df), dtype=torch.int64)

        self.context_features: Tensor
        if context_encoder is not None:
            ctx = context_encoder.transform(df)
            if ctx.shape != (len(df), CONTEXT_DIM):
                raise ValueError(
                    f"context_encoder.transform produced shape {ctx.shape}; "
                    f"expected ({len(df)}, {CONTEXT_DIM})"
                )
            self.context_features = torch.from_numpy(ctx).to(dtype=dtype)
        else:
            self.context_features = torch.empty((len(df), 0), dtype=dtype)

    @property
    def has_components(self) -> bool:
        return self.log_qp_table is not None

    @property
    def has_defensive(self) -> bool:
        return self.log_qd_table is not None

    @property
    def has_context(self) -> bool:
        return self.context_features.shape[-1] > 0

    @property
    def context_dim(self) -> int:
        return int(self.context_features.shape[-1])

    def __len__(self) -> int:
        return int(self.cell_idx.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        p = self.player_idx[idx]
        c = self.cell_idx[idx]
        opp = self.opponent_idx[idx]
        x_n = self.context_features[idx]
        return p, c, self.log_q0_table[p], opp, x_n

"""Canonical per-shot context vector ``x_n`` for AA-KDE.

The :class:`ContextEncoder` is the single source of truth for ``x_n``,
the per-shot context vector consumed by every learnable component of
the spatial decoder (paper §3): the offensive self-relevance score,
the archetype mixture, the defensive feasibility relevance score,
the residual perturbation encoder, and the count and timing heads
(paper §4).

**Causal Snapshot Principle (paper §2.5).** Every coordinate of
``x_n`` either:

1. comes from the per-shot row itself (period, time-in-period —
   intrinsically pregame-known by virtue of being part of the
   game's clock state at the moment the shot is taken),
2. comes from the joined per-game row (starter, minutes — pregame
   known via the box-score join), or
3. is looked up from the bundle ``S(t)`` via
   :meth:`shotcloud.data.SnapshotStore.get_snapshot` for the
   shot's date (role profile, position mixture, opp-efficiency
   bucket — all fit causally on shots before the bundle's anchor),
4. is a fixed training-window normalization (date range for
   season-recency, mean/std for minutes-z-score) — pregame-known
   constants computed once.

There is no fourth category. Anyone reading this code can verify
:meth:`ContextEncoder.transform` does not reach into a per-game row's
future shots, into a snapshot's future, or into anything outside the
bundle and the row.

Layout (see :data:`FEATURE_LAYOUT`):

==========================  =========  ===================================
range                         dim        meaning
==========================  =========  ===================================
``0:4``                        4        period one-hot ``[Q1, Q2, Q3, Q4]``
``4``                          1        time elapsed in current period (0..1)
``5``                          1        season recency (0..1 over training range)
``6``                          1        starter indicator (0/1)
``7``                          1        minutes z-scored
``8``                          1        home/away indicator (1=home, 0=visitor)
``9``                          1        recent 3PA fraction (recency-weighted, ``[0, 1]``)
``10``                         1        recent usage rate (recency-weighted, z-scored)
``11``                         1        recent FGA per game (recency-weighted, z-scored)
``12:20``                      8        role profile (paper §3.3a)
``20:23``                      3        position mixture over (G, W, B)
``23:27``                      4        opponent-efficiency one-hot
==========================  =========  ===================================

Total: ``CONTEXT_DIM = 27``.

The three ``recent_*`` features are causal: a row's recent values
reflect the player's prior games (date strictly before the row),
exponentially decayed (default half-life 30 days). They are computed
once at game-log load time by
:func:`shotcloud.data.game_logs.load_game_logs` and joined per-shot
by :func:`shotcloud.data.game_logs.join_game_logs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shotcloud.data.snapshots import (
    N_OPP_EFFICIENCY_BINS,
    POSITION_MIXTURE_DIM,
    ROLE_PROFILE_DIM,
    SnapshotStore,
)

# Layout constants — keep in sync with the docstring.
N_PERIOD_ONEHOT: int = 4
PERIOD_SECONDS: float = 12.0 * 60.0

# CONTEXT_DIM = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 8 + 3 + 4 = 27
CONTEXT_DIM: int = (
    N_PERIOD_ONEHOT
    + 1  # time_in_period
    + 1  # season_recency
    + 1  # starter
    + 1  # minutes_norm
    + 1  # home_away
    + 1  # recent_3pa_frac
    + 1  # recent_usage
    + 1  # recent_fga
    + ROLE_PROFILE_DIM
    + POSITION_MIXTURE_DIM
    + N_OPP_EFFICIENCY_BINS
)

#: Slice ranges into the (N, CONTEXT_DIM) output, keyed by feature name.
#: Downstream models pull slices by name (never by hardcoded indices),
#: so adding a new feature here is the only place to update.
FEATURE_LAYOUT: dict[str, slice] = {
    "period_onehot": slice(0, N_PERIOD_ONEHOT),
    "time_in_period": slice(N_PERIOD_ONEHOT, N_PERIOD_ONEHOT + 1),
    "season_recency": slice(N_PERIOD_ONEHOT + 1, N_PERIOD_ONEHOT + 2),
    "starter": slice(N_PERIOD_ONEHOT + 2, N_PERIOD_ONEHOT + 3),
    "minutes_norm": slice(N_PERIOD_ONEHOT + 3, N_PERIOD_ONEHOT + 4),
    "home_away": slice(N_PERIOD_ONEHOT + 4, N_PERIOD_ONEHOT + 5),
    "recent_3pa_frac": slice(N_PERIOD_ONEHOT + 5, N_PERIOD_ONEHOT + 6),
    "recent_usage": slice(N_PERIOD_ONEHOT + 6, N_PERIOD_ONEHOT + 7),
    "recent_fga": slice(N_PERIOD_ONEHOT + 7, N_PERIOD_ONEHOT + 8),
    "role_profile": slice(
        N_PERIOD_ONEHOT + 8,
        N_PERIOD_ONEHOT + 8 + ROLE_PROFILE_DIM,
    ),
    "position_mixture": slice(
        N_PERIOD_ONEHOT + 8 + ROLE_PROFILE_DIM,
        N_PERIOD_ONEHOT + 8 + ROLE_PROFILE_DIM + POSITION_MIXTURE_DIM,
    ),
    "opp_efficiency_onehot": slice(
        N_PERIOD_ONEHOT + 8 + ROLE_PROFILE_DIM + POSITION_MIXTURE_DIM,
        CONTEXT_DIM,
    ),
}

#: Default minutes mean (NBA-wide rough average; used as a fallback
#: when no training stats are provided).
DEFAULT_MINUTES_MEAN: float = 24.0

#: Default minutes std (z-score scale).
DEFAULT_MINUTES_STD: float = 10.0

#: Default mean and std for ``recent_usage`` z-scoring (NBA-wide
#: rough averages: usage ≈ 0.50 with std ≈ 0.25 across role players).
DEFAULT_RECENT_USAGE_MEAN: float = 0.50
DEFAULT_RECENT_USAGE_STD: float = 0.25

#: Default mean and std for ``recent_fga`` z-scoring (NBA-wide
#: rough averages across all minute-bins: ~6 FGA with std ~5).
DEFAULT_RECENT_FGA_MEAN: float = 6.0
DEFAULT_RECENT_FGA_STD: float = 5.0


@dataclass(frozen=True)
class ContextEncoder:
    """Build per-shot context vectors ``x_n`` of shape ``(N, CONTEXT_DIM)``.

    The encoder is a small immutable record holding a
    :class:`shotcloud.data.SnapshotStore` (when run in full AA-KDE
    mode) plus four pregame-known normalization constants. Every
    coordinate of every produced row is derivable from the input row,
    the snapshot bundle at the row's date, or these constants. There
    is no in-encoder learning.

    Parameters
    ----------
    snapshot_store : SnapshotStore | None
        The causal feature registry. When non-None, the encoder fills
        the snapshot-derived slices (``role_profile``,
        ``position_mixture``, ``opp_efficiency_onehot``) by looking up
        a bundle for each row's date. When None, those slices stay
        zero; this is a transitional path for legacy callers that
        have not yet migrated to AA-KDE. New code should always pass
        a ``SnapshotStore``.
    date_min, date_max : np.datetime64
        Inclusive training-window bounds, used to normalize
        ``season_recency`` to ``[0, 1]``. Pregame-known constants.
    minutes_mean, minutes_std : float
        Z-score normalization for the ``minutes`` field.
    n_train : int
        Number of training rows used at fit time (diagnostic only).
    """

    snapshot_store: SnapshotStore | None
    date_min: np.datetime64
    date_max: np.datetime64
    minutes_mean: float = DEFAULT_MINUTES_MEAN
    minutes_std: float = DEFAULT_MINUTES_STD
    recent_usage_mean: float = DEFAULT_RECENT_USAGE_MEAN
    recent_usage_std: float = DEFAULT_RECENT_USAGE_STD
    recent_fga_mean: float = DEFAULT_RECENT_FGA_MEAN
    recent_fga_std: float = DEFAULT_RECENT_FGA_STD
    n_train: int = 0

    @classmethod
    def fit(
        cls,
        snapshot_store_or_shots: SnapshotStore | pd.DataFrame,
        shots: pd.DataFrame | None = None,
        *,
        date_min: np.datetime64 | pd.Timestamp | str | None = None,
        date_max: np.datetime64 | pd.Timestamp | str | None = None,
        minutes_mean: float | None = None,
        minutes_std: float | None = None,
        recent_usage_mean: float | None = None,
        recent_usage_std: float | None = None,
        recent_fga_mean: float | None = None,
        recent_fga_std: float | None = None,
    ) -> ContextEncoder:
        """Build an encoder from a snapshot store + training stats.

        New AA-KDE callers should pass a ``SnapshotStore`` as the
        first argument, optionally followed by training shots from
        which to derive normalization stats. Legacy callers may pass
        a DataFrame as the single positional argument; the encoder
        will run in *no-snapshot* mode (snapshot-derived slices stay
        zero) and derive stats from the DataFrame. Both forms produce
        a ``CONTEXT_DIM``-wide output.

        Parameters
        ----------
        snapshot_store_or_shots : SnapshotStore | DataFrame
            Either the causal feature registry or, for legacy
            callers, the training shots frame (no snapshot lookups
            will be performed).
        shots : DataFrame, optional
            Training shots (the same window used to build
            ``snapshot_store``); used to derive normalization stats.
            Ignored in legacy single-arg mode.
        date_min, date_max : datetime-like, optional
            Override the training-window bounds.
        minutes_mean, minutes_std : float, optional
            Override the minutes z-score parameters.
        """
        # Polymorphic dispatch: DataFrame as first arg → legacy mode.
        store: SnapshotStore | None
        if isinstance(snapshot_store_or_shots, pd.DataFrame):
            store = None
            stat_shots: pd.DataFrame | None = snapshot_store_or_shots
        else:
            store = snapshot_store_or_shots
            stat_shots = shots

        # Default date span: anchor span if we have a store, else from
        # the stat_shots frame.
        if store is not None:
            anchors = store.anchor_dates
            d_min = cast("np.datetime64", anchors[0])
            d_max = cast("np.datetime64", anchors[-1])
        else:
            d_min = np.datetime64("2014-10-01", "D")  # placeholder; overridden below
            d_max = np.datetime64("2025-06-30", "D")

        if date_min is not None:
            d_min = np.datetime64(date_min).astype("datetime64[D]")
        if date_max is not None:
            d_max = np.datetime64(date_max).astype("datetime64[D]")

        if stat_shots is not None and "date" in stat_shots.columns:
            shot_dates = pd.to_datetime(stat_shots["date"]).dropna()
            if len(shot_dates) > 0:
                if date_min is None:
                    d_min = shot_dates.min().to_datetime64().astype("datetime64[D]")
                if date_max is None:
                    d_max = shot_dates.max().to_datetime64().astype("datetime64[D]")

        m_mean = minutes_mean if minutes_mean is not None else DEFAULT_MINUTES_MEAN
        m_std = minutes_std if minutes_std is not None else DEFAULT_MINUTES_STD
        if stat_shots is not None and "minutes" in stat_shots.columns and minutes_mean is None:
            mins = pd.to_numeric(stat_shots["minutes"], errors="coerce").dropna()
            if len(mins) > 0:
                m_mean = float(mins.mean())
                if minutes_std is None:
                    m_std = float(max(mins.std(), 1.0))

        # Recent-feature stats: derive from training shots when available
        # and not explicitly overridden. Z-score on the *non-zero* subset:
        # zero-imputed first-game rows would otherwise pull the mean toward
        # zero and break the z-score interpretation.
        ru_mean = recent_usage_mean if recent_usage_mean is not None else DEFAULT_RECENT_USAGE_MEAN
        ru_std = recent_usage_std if recent_usage_std is not None else DEFAULT_RECENT_USAGE_STD
        rf_mean = recent_fga_mean if recent_fga_mean is not None else DEFAULT_RECENT_FGA_MEAN
        rf_std = recent_fga_std if recent_fga_std is not None else DEFAULT_RECENT_FGA_STD
        if stat_shots is not None:
            if "recent_usage" in stat_shots.columns and recent_usage_mean is None:
                ru = pd.to_numeric(stat_shots["recent_usage"], errors="coerce")
                ru = ru[ru > 0].dropna()
                if len(ru) > 0:
                    ru_mean = float(ru.mean())
                    if recent_usage_std is None:
                        ru_std = float(max(ru.std(), 1e-3))
            if "recent_fga" in stat_shots.columns and recent_fga_mean is None:
                rf = pd.to_numeric(stat_shots["recent_fga"], errors="coerce")
                rf = rf[rf > 0].dropna()
                if len(rf) > 0:
                    rf_mean = float(rf.mean())
                    if recent_fga_std is None:
                        rf_std = float(max(rf.std(), 1e-3))

        n = len(stat_shots) if stat_shots is not None else 0
        return cls(
            snapshot_store=store,
            date_min=d_min.astype("datetime64[D]"),
            date_max=d_max.astype("datetime64[D]"),
            minutes_mean=m_mean,
            minutes_std=m_std,
            recent_usage_mean=ru_mean,
            recent_usage_std=ru_std,
            recent_fga_mean=rf_mean,
            recent_fga_std=rf_std,
            n_train=n,
        )

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> NDArray[np.float32]:
        """Compute the per-row context array, shape ``(N, CONTEXT_DIM)``.

        Each row of `df` is one shot. The function looks up the
        snapshot bundle at the row's date and pulls the role profile,
        position mixture, and opp-efficiency bucket from it; the rest
        comes from row-local fields with sensible fallbacks.

        Required columns (any subset; missing columns leave their
        feature slice as zeros): ``date``, ``period``,
        ``time_remaining_sec``, ``player_id``, ``opponent``,
        ``starter``, ``minutes``.

        Raises
        ------
        ValueError
            If any row's date precedes the snapshot store's earliest
            anchor (no causal snapshot is available).
        """
        n = len(df)
        out = np.zeros((n, CONTEXT_DIM), dtype=np.float32)
        if n == 0:
            return out

        # ------------------------------------------------------------------
        # Per-shot timing features (vectorized).
        # ------------------------------------------------------------------
        if "period" in df.columns:
            period_raw = pd.to_numeric(df["period"], errors="coerce").to_numpy()
            period = np.where(np.isnan(period_raw), 1, period_raw).astype(np.int64)
            period = np.clip(period, 1, N_PERIOD_ONEHOT)
            for p in range(1, N_PERIOD_ONEHOT + 1):
                out[period == p, p - 1] = 1.0

        if "time_remaining_sec" in df.columns and "period" in df.columns:
            total_elapsed = pd.to_numeric(df["time_remaining_sec"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            period_idx = (
                pd.to_numeric(df["period"], errors="coerce").fillna(1).to_numpy(dtype=np.float64)
            )
            period_start = (period_idx - 1) * PERIOD_SECONDS
            in_period = (total_elapsed - period_start) / PERIOD_SECONDS
            in_period = np.clip(in_period, 0.0, 1.0)
            in_period = np.nan_to_num(in_period, nan=0.5, posinf=1.0, neginf=0.0)
            out[:, FEATURE_LAYOUT["time_in_period"].start] = in_period.astype(np.float32)

        # ------------------------------------------------------------------
        # Season recency (vectorized; pregame-known fixed normalization).
        # ------------------------------------------------------------------
        if "date" in df.columns:
            dates_arr = (
                pd.to_datetime(df["date"], errors="coerce")
                .fillna(pd.Timestamp(self.date_min))
                .to_numpy(dtype="datetime64[ns]")
                .astype("datetime64[D]")
            )
            day = np.timedelta64(1, "D")
            delta = (dates_arr - self.date_min) / day
            total = (self.date_max - self.date_min) / day
            recency = (
                np.clip(delta.astype(np.float32) / float(total), 0.0, 1.0)
                if total > 0
                else np.zeros(n, dtype=np.float32)
            )
            out[:, FEATURE_LAYOUT["season_recency"].start] = recency

        # ------------------------------------------------------------------
        # Per-game features from the joined game-log row (vectorized).
        # ------------------------------------------------------------------
        if "starter" in df.columns:
            starter = pd.to_numeric(df["starter"], errors="coerce").fillna(0)
            out[:, FEATURE_LAYOUT["starter"].start] = starter.astype(np.float32).to_numpy()
        if "minutes" in df.columns:
            mins = pd.to_numeric(df["minutes"], errors="coerce").fillna(self.minutes_mean)
            out[:, FEATURE_LAYOUT["minutes_norm"].start] = (
                (mins.astype(np.float32) - self.minutes_mean) / max(self.minutes_std, 1e-3)
            ).to_numpy()
        if "home_away" in df.columns:
            ha = pd.to_numeric(df["home_away"], errors="coerce").fillna(0)
            out[:, FEATURE_LAYOUT["home_away"].start] = ha.astype(np.float32).to_numpy()

        if "recent_3pa_frac" in df.columns:
            r3 = pd.to_numeric(df["recent_3pa_frac"], errors="coerce").fillna(0.0)
            # Already a [0,1] fraction; pass through (clip for robustness).
            out[:, FEATURE_LAYOUT["recent_3pa_frac"].start] = np.clip(
                r3.astype(np.float32).to_numpy(), 0.0, 1.0
            )
        if "recent_usage" in df.columns:
            ru = pd.to_numeric(df["recent_usage"], errors="coerce").fillna(self.recent_usage_mean)
            out[:, FEATURE_LAYOUT["recent_usage"].start] = (
                (ru.astype(np.float32) - self.recent_usage_mean) / max(self.recent_usage_std, 1e-3)
            ).to_numpy()
        if "recent_fga" in df.columns:
            rf = pd.to_numeric(df["recent_fga"], errors="coerce").fillna(self.recent_fga_mean)
            out[:, FEATURE_LAYOUT["recent_fga"].start] = (
                (rf.astype(np.float32) - self.recent_fga_mean) / max(self.recent_fga_std, 1e-3)
            ).to_numpy()

        # ------------------------------------------------------------------
        # Snapshot lookups: bundle the rows by snapshot index, then
        # apply per-bundle player/opp lookups. Each bundle handles its
        # own subset of rows. Skipped entirely in legacy
        # no-snapshot mode (snapshot_store is None).
        # ------------------------------------------------------------------
        if self.snapshot_store is None or "date" not in df.columns:
            return out

        dates_arr = (
            pd.to_datetime(df["date"], errors="coerce")
            .fillna(pd.Timestamp(self.date_min))
            .to_numpy(dtype="datetime64[ns]")
            .astype("datetime64[D]")
        )

        first_anchor = self.snapshot_store.bundles[0].anchor_date.astype("datetime64[D]")
        if (dates_arr < first_anchor).any():
            n_bad = int((dates_arr < first_anchor).sum())
            raise ValueError(
                f"{n_bad} of {n} rows have date before the earliest snapshot anchor "
                f"{first_anchor}; either filter shots to date >= {first_anchor}, "
                f"or extend the snapshot anchor grid further back in time."
            )

        # Group rows by snapshot index for efficient per-bundle processing.
        anchor_dates = self.snapshot_store.anchor_dates
        snapshot_idx = np.searchsorted(anchor_dates, dates_arr, side="right") - 1
        # snapshot_idx[i] = bundle index for shot i; >= 0 by construction
        # (we already filtered out pre-first-anchor dates above)

        player_ids = (
            df["player_id"].to_numpy() if "player_id" in df.columns else np.zeros(n, dtype=np.int64)
        )
        opps = df["opponent"].to_numpy() if "opponent" in df.columns else np.array([""] * n)

        rp_slice = FEATURE_LAYOUT["role_profile"]
        pm_slice = FEATURE_LAYOUT["position_mixture"]
        opp_slice = FEATURE_LAYOUT["opp_efficiency_onehot"]

        # Default position mixture: uniform fallback for unknown players.
        uniform_pos = np.full(POSITION_MIXTURE_DIM, 1.0 / POSITION_MIXTURE_DIM, dtype=np.float32)

        for i_snap in np.unique(snapshot_idx):
            bundle = self.snapshot_store.bundles[int(i_snap)]
            mask = snapshot_idx == i_snap
            row_indices = np.where(mask)[0]

            for i_row in row_indices:
                pid = player_ids[i_row]
                if pid is None:
                    out[i_row, pm_slice] = uniform_pos
                    continue
                try:
                    pid_int = int(pid)
                except (TypeError, ValueError):
                    out[i_row, pm_slice] = uniform_pos
                    continue
                p_idx = bundle.player_idx(pid_int)
                if p_idx is None:
                    # Unknown player at this anchor: zeros for role profile,
                    # uniform for position mixture (signals neutral).
                    out[i_row, pm_slice] = uniform_pos
                else:
                    out[i_row, rp_slice] = bundle.role_profiles[p_idx]
                    out[i_row, pm_slice] = bundle.position_mixtures[p_idx]

                opp = opps[i_row]
                if opp is None or (isinstance(opp, float) and np.isnan(opp)):
                    continue
                bin_idx = bundle.opp_strength_bin(str(opp))
                if bin_idx is not None and 0 <= bin_idx < N_OPP_EFFICIENCY_BINS:
                    out[i_row, opp_slice.start + bin_idx] = 1.0
                # else: unknown opp; opp slice stays zero (neutral)

        return cast("NDArray[np.float32]", out)


@dataclass(frozen=True)
class _LegacyOppBucket:
    """Backward-compat shim: a fitted opp-efficiency bucket map.

    Retained so legacy callers that don't yet have a SnapshotStore can
    still import the module without breaking. Any new code should use
    :class:`ContextEncoder` directly.
    """

    opp_strength_bucket: dict[str, int] = field(default_factory=dict)


__all__ = [
    "CONTEXT_DIM",
    "DEFAULT_MINUTES_MEAN",
    "DEFAULT_MINUTES_STD",
    "FEATURE_LAYOUT",
    "N_PERIOD_ONEHOT",
    "PERIOD_SECONDS",
    "ContextEncoder",
]

"""Causal player trait vectors for the collaborative KDE.

Builds a 26-dimensional trait vector per (player, snapshot). Trait
vectors drive analogue retrieval (:mod:`shotcloud.models.analogue_retrieval`)
and the shooter-similarity term of the collaborative KDE. Every
slot at snapshot ``m`` uses only games and shots strictly before that
snapshot's anchor date, including the population statistics that
standardize the biographical slots.

Trait vector layout — exactly 26 dims (see :data:`SLOT_NAMES`):

```text
Block A (biographical, populated even at cold-start):
    0  height_z              (per-snapshot z-score of inches)
    1  weight_z              (per-snapshot z-score of pounds)
    2  age_z                 (per-snapshot z-score of years at anchor)
    3  bio_pos_PG            (one-hot)
    4  bio_pos_SG
    5  bio_pos_SF
    6  bio_pos_PF
    7  bio_pos_C
Block B (play-derived, MULTIPLIED by missingness):
    8  shot_pos_G            (from SnapshotBundle.position_mixtures)
    9  shot_pos_W
   10  shot_pos_B
   11  fga_3pa_frac          (derived from role_profile)
   12  fga_2pa_frac          (1 − 3PA fraction)
   13  usage                 (recency-weighted Hollinger-style usage)
   14  log1p_minutes_M       (log(1 + M_p^<t_m))
   15  log1p_fga_S           (log(1 + S_p^<t_m))
   16  log_shot_density      (log((S+1)/(M+1)))
   17  role_rim              (zone-0 rate from role_profile)
   18  role_paint
   19  role_midrange
   20  role_corner3
   21  role_atb3
   22  role_mean_dist
   23  role_std_dist
   24  role_entropy
Missingness indicator:
   25  m_play                (1 if player had any FGA before anchor)
```

Block A's continuous slots are z-scored at each snapshot against the
*reference population*: the players with at least one game in the
``reference_window_days`` before the snapshot's anchor date. The mean and
standard deviation therefore describe the league as it stood at the
anchor and never depend on players who debut later. Every player in the
vocabulary, including one who has not yet played, is standardized with
those statistics, with age evaluated at the anchor date. Missing
biographical values are imputed to the post-z-score mean (zero), placing
such players at the population-average coordinate; when the reference
population is empty or has zero variance, the slot is zero for everyone.

Block B is multiplied by ``m_play`` so players with no prior field-goal
attempts get exact zeros across the play-derived block. The explicit
``m_play`` slot at index 25 lets a consumer distinguish "no history"
from "a genuine zero in the player's play pattern".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shotcloud.data.player_bio import (
    POSITION_GROUPS,
    compute_age_years,
    position_group_onehot_matrix,
)
from shotcloud.data.snapshots import ROLE_PROFILE_DIM, SnapshotStore

#: Total dimension of the per-(player, snapshot) trait vector.
TRAIT_DIM: Final[int] = 26

#: Symbolic slot names, one per trait slot; checked against
#: :data:`TRAIT_DIM` at import.
SLOT_NAMES: Final[tuple[str, ...]] = (
    "height_z",
    "weight_z",
    "age_z",
    *(f"bio_pos_{g}" for g in POSITION_GROUPS),
    "shot_pos_G",
    "shot_pos_W",
    "shot_pos_B",
    "fga_3pa_frac",
    "fga_2pa_frac",
    "usage",
    "log1p_minutes_M",
    "log1p_fga_S",
    "log_shot_density",
    "role_rim",
    "role_paint",
    "role_midrange",
    "role_corner3",
    "role_atb3",
    "role_mean_dist",
    "role_std_dist",
    "role_entropy",
    "m_play",
)
assert len(SLOT_NAMES) == TRAIT_DIM, (
    f"SLOT_NAMES has {len(SLOT_NAMES)} entries; expected {TRAIT_DIM}"
)

#: End (exclusive) of Block A, the biographical slots 0-7.
BLOCK_A_END: Final[int] = 8
#: End (exclusive) of Block B, the play-derived slots 8-24.
BLOCK_B_END: Final[int] = 25
#: Index of the ``m_play`` missingness indicator.
M_PLAY_SLOT: Final[int] = 25

#: Hollinger-style usage coefficient on free-throw attempts.
_USAGE_FT_WEIGHT: Final[float] = 0.44


@dataclass(frozen=True)
class PlayerTraitsTable:
    """Causal trait tensor of shape ``(n_players, n_snapshots, TRAIT_DIM)``.

    Attributes
    ----------
    traits : NDArray[float32], shape ``(n_players, n_snapshots, 26)``
        Causal trait values. Row ``p`` is keyed by ``player_ids[p]``;
        column ``m`` is keyed by ``snapshot_anchors[m]``.
    player_ids : NDArray[int64], shape ``(n_players,)``
        Player IDs, in the order of the ``vocab_ids`` passed to
        :func:`build_player_traits_table`.
    snapshot_anchors : NDArray[datetime64[D]], shape ``(n_snapshots,)``
        Snapshot anchor dates in the order of ``SnapshotStore.bundles``.
    """

    traits: NDArray[np.float32]
    player_ids: NDArray[np.int64]
    snapshot_anchors: NDArray[np.datetime64]

    @property
    def n_players(self) -> int:
        """Number of players (rows)."""
        return int(self.traits.shape[0])

    @property
    def n_snapshots(self) -> int:
        """Number of snapshots (columns)."""
        return int(self.traits.shape[1])

    @property
    def trait_dim(self) -> int:
        """Width of each trait vector."""
        return int(self.traits.shape[2])

    @property
    def slot_names(self) -> tuple[str, ...]:
        """Names of the trait slots (:data:`SLOT_NAMES`)."""
        return SLOT_NAMES


def _recency_weight(delta_days: NDArray[np.float64], half_life_days: float) -> NDArray[np.float64]:
    """Exponential weights ``2^(-delta / half_life)`` on each game."""
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")
    out: NDArray[np.float64] = np.exp(-math.log(2.0) * delta_days / half_life_days)
    return out


def _recency_aggregate_per_snapshot(
    game_logs: pd.DataFrame,
    ref_date: np.datetime64,
    half_life_days: float,
) -> pd.DataFrame:
    """Per-player recency-weighted (M, S, usage_rate) at ``ref_date``.

    Returns a dataframe with columns ``[player_id, M, S, usage_rate]``,
    one row per player with at least one game before ``ref_date``.
    Players with no games before the anchor are absent; the caller
    handles them via the missingness indicator. With recency weights
    ``w_g = 2^(-(t - date(g)) / half_life_days)`` over games strictly
    before ``t = ref_date``:

    * M = Σ_g w_g · minutes_g
    * S = Σ_g w_g · FGA_g
    * usage_rate = Σ_g w_g · (FGA + 0.44·FTA + TOV)_g / Σ_g w_g · minutes_g
    """
    # Work in pd.Timestamp because game_date may be datetime64 of any
    # precision; day deltas go through total_seconds / 86400 because
    # pandas will not cast a DatetimeArray to datetime64[D].
    ref_ts = pd.Timestamp(ref_date)
    game_dates = pd.to_datetime(game_logs["game_date"], errors="coerce")
    mask = (game_dates < ref_ts) & game_dates.notna()
    gl = game_logs.loc[mask, ["player_id", "minutes", "fga", "fta", "tov"]].copy()
    if gl.empty:
        return pd.DataFrame(
            {
                "player_id": pd.Series([], dtype=np.int64),
                "M": pd.Series([], dtype=np.float64),
                "S": pd.Series([], dtype=np.float64),
                "usage_rate": pd.Series([], dtype=np.float64),
            }
        )

    delta_days = (ref_ts - game_dates.loc[mask]).dt.total_seconds().to_numpy(
        dtype=np.float64
    ) / 86400.0
    w = _recency_weight(delta_days, half_life_days)

    minutes = gl["minutes"].fillna(0).to_numpy(dtype=np.float64)
    fga = gl["fga"].fillna(0).to_numpy(dtype=np.float64)
    fta = gl["fta"].fillna(0).to_numpy(dtype=np.float64)
    tov = gl["tov"].fillna(0).to_numpy(dtype=np.float64)

    gl_w = pd.DataFrame(
        {
            "player_id": gl["player_id"].to_numpy(),
            "w_min": w * minutes,
            "w_fga": w * fga,
            "w_usage_num": w * (fga + _USAGE_FT_WEIGHT * fta + tov),
        }
    )
    agg = gl_w.groupby("player_id", as_index=False).sum()
    agg = agg.rename(columns={"w_min": "M", "w_fga": "S"})
    agg["usage_rate"] = agg["w_usage_num"] / agg["M"].clip(lower=1e-6)
    return agg[["player_id", "M", "S", "usage_rate"]].reset_index(drop=True)


def _z_score_with_nan_imputation(
    values: NDArray[np.float64],
    reference: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Z-score a 1-D array against its ``reference`` rows, imputing NaN to 0.

    The mean and standard deviation are taken over the finite entries
    selected by ``reference``; every entry is standardized with them.
    """
    finite_mask = np.isfinite(values)
    reference_mask = finite_mask & reference
    if not reference_mask.any():
        # No finite reference values: the z-score is undefined, so return zeros.
        return np.zeros_like(values, dtype=np.float32)
    reference_vals = values[reference_mask]
    mean = float(reference_vals.mean())
    std = float(reference_vals.std())
    if std < 1e-9:
        # Zero variance: the z-score is undefined, so return zeros.
        return np.zeros_like(values, dtype=np.float32)
    z = (values - mean) / std
    z[~finite_mask] = 0.0  # NaN → post-z-score mean
    return z.astype(np.float32)


def build_player_traits_table(
    snapshot_store: SnapshotStore,
    vocab_ids: list[int] | NDArray[np.int64],
    bio_df: pd.DataFrame,
    game_logs_df: pd.DataFrame,
    *,
    recency_half_life_days: float = 30.0,
    reference_window_days: float = 365.0,
) -> PlayerTraitsTable:
    """Build the causal trait table for the collaborative KDE.

    For each snapshot, Block A is filled from ``bio_df`` and standardized
    against the players active in the ``reference_window_days`` before the
    bundle's anchor date. Block B is filled from the snapshot bundle
    (position mixture, role profile) and from recency-weighted aggregates
    of ``game_logs_df`` over games strictly before the anchor.

    Parameters
    ----------
    snapshot_store : SnapshotStore
        Provides the anchor dates and the per-snapshot role profiles and
        position mixtures, each fit on shots before its anchor.
    vocab_ids : sequence of int
        The player IDs the model addresses. Output rows are aligned
        to this order. Players not in ``bio_df`` get NaN bio fields
        (handled by NaN-imputation in z-scoring + the missingness
        indicator).
    bio_df : DataFrame
        Output of :func:`shotcloud.data.player_bio.load_player_bio`.
        Must contain at minimum the columns required by that loader.
    game_logs_df : DataFrame
        Per-game NBA logs (player_id, game_date, minutes, fga, fta,
        tov). Other columns ignored.
    recency_half_life_days : float, default 30.0
        Exponential half-life for recency-weighted aggregates of
        minutes, FGA, and usage. Matches
        :data:`shotcloud.data.game_logs.DEFAULT_RECENCY_HALFLIFE_DAYS`.
    reference_window_days : float, default 365.0
        Length of the window before each anchor that defines the
        reference population for the height, weight, and age z-scores:
        vocabulary players with at least one game in
        ``[anchor - reference_window_days, anchor)``.

    Returns
    -------
    PlayerTraitsTable
        Traits of shape ``(len(vocab_ids), n_snapshots, TRAIT_DIM)``.
    """
    if reference_window_days <= 0:
        raise ValueError(f"reference_window_days must be positive, got {reference_window_days}")
    player_ids = np.asarray(vocab_ids, dtype=np.int64)
    n_players = len(player_ids)
    n_snapshots = len(snapshot_store.bundles)

    # Vocabulary row and date of every game, for the per-snapshot reference
    # population of the biographical z-scores.
    row_of_player = pd.Series(np.arange(n_players), index=player_ids)
    game_dates = pd.to_datetime(game_logs_df["game_date"], errors="coerce")
    game_rows = game_logs_df["player_id"].map(row_of_player)
    known_game = (game_dates.notna() & game_rows.notna()).to_numpy()
    game_dates_np = game_dates.to_numpy()[known_game]
    game_rows_np = game_rows.to_numpy()[known_game].astype(np.int64)
    reference_window = pd.Timedelta(days=reference_window_days)

    # ---- Bio block aligned to vocab order (height/weight/birthdate/position) -----
    bio_indexed = bio_df.set_index("player_id")
    height_raw = np.full(n_players, np.nan, dtype=np.float64)
    weight_raw = np.full(n_players, np.nan, dtype=np.float64)
    birthdates = np.full(n_players, np.datetime64("NaT"), dtype="datetime64[ns]")
    bio_positions = pd.Series([pd.NA] * n_players, dtype="string")

    for i, pid in enumerate(player_ids):
        if int(pid) not in bio_indexed.index:
            continue
        row = bio_indexed.loc[int(pid)]
        if not pd.isna(row.get("height_inches")):
            height_raw[i] = float(row["height_inches"])
        if not pd.isna(row.get("weight_lbs")):
            weight_raw[i] = float(row["weight_lbs"])
        if not pd.isna(row.get("birthdate")):
            birthdates[i] = np.datetime64(row["birthdate"], "ns")
        if not pd.isna(row.get("position_group")):
            bio_positions.iloc[i] = row["position_group"]

    # Bio position one-hot is static across snapshots (player's bio
    # position doesn't change with time).
    pos_onehot = position_group_onehot_matrix(bio_positions).astype(np.float32)
    assert pos_onehot.shape == (n_players, len(POSITION_GROUPS))

    # ---- Output buffer + per-snapshot fill --------------------------------------
    traits = np.zeros((n_players, n_snapshots, TRAIT_DIM), dtype=np.float32)
    snapshot_anchors = np.array(
        [b.anchor_date.astype("datetime64[D]") for b in snapshot_store.bundles],
        dtype="datetime64[D]",
    )

    bio_birthdate_series = pd.Series(birthdates)

    for m_idx, bundle in enumerate(snapshot_store.bundles):
        anchor = bundle.anchor_date

        # Block A: per-snapshot z-scored height, weight, age + static one-hot.
        # The z-score statistics come from players active in the window
        # before the anchor, so they never depend on later debuts.
        anchor_ts = pd.Timestamp(anchor)
        in_window = (game_dates_np >= (anchor_ts - reference_window).to_datetime64()) & (
            game_dates_np < anchor_ts.to_datetime64()
        )
        reference = np.zeros(n_players, dtype=np.bool_)
        reference[game_rows_np[in_window]] = True

        ages = compute_age_years(bio_birthdate_series, anchor)
        height_z = _z_score_with_nan_imputation(height_raw, reference)
        weight_z = _z_score_with_nan_imputation(weight_raw, reference)
        age_z = _z_score_with_nan_imputation(ages, reference)

        traits[:, m_idx, 0] = height_z
        traits[:, m_idx, 1] = weight_z
        traits[:, m_idx, 2] = age_z
        traits[:, m_idx, 3:8] = pos_onehot

        # Block B inputs from the snapshot bundle. Players with no shots
        # before the anchor are absent from the bundle and default to
        # zero; the missingness mask zeros them out in any case.
        bundle_ids = np.asarray(bundle.player_ids, dtype=np.int64)
        # Vectorized map vocab_id → bundle_idx (or -1 if absent).
        bundle_idx_lookup: dict[int, int] = {int(pid): bi for bi, pid in enumerate(bundle_ids)}
        vocab_to_bundle = np.array(
            [bundle_idx_lookup.get(int(pid), -1) for pid in player_ids],
            dtype=np.int64,
        )
        in_bundle = vocab_to_bundle >= 0
        bundle_idx_safe = np.where(in_bundle, vocab_to_bundle, 0)

        # Shot-based position mixture (3-d) and role profile (8-d) per
        # vocab row.
        pos_mix_full = np.zeros((n_players, 3), dtype=np.float32)
        role_full = np.zeros((n_players, ROLE_PROFILE_DIM), dtype=np.float32)
        pos_mix_full[in_bundle] = bundle.position_mixtures[bundle_idx_safe[in_bundle]]
        role_full[in_bundle] = bundle.role_profiles[bundle_idx_safe[in_bundle]]

        # 3PA fraction = corner3 + atb3 (slots 3 and 4 of role profile).
        fga_3pa_frac = role_full[:, 3] + role_full[:, 4]
        fga_2pa_frac = role_full[:, 0] + role_full[:, 1] + role_full[:, 2]

        # Recency-weighted game-log aggregates at this anchor.
        gl_agg = _recency_aggregate_per_snapshot(game_logs_df, anchor, recency_half_life_days)
        gl_indexed = gl_agg.set_index("player_id")
        m_arr = np.zeros(n_players, dtype=np.float64)
        s_arr = np.zeros(n_players, dtype=np.float64)
        usage_arr = np.zeros(n_players, dtype=np.float64)
        for i, pid in enumerate(player_ids):
            if int(pid) in gl_indexed.index:
                row = gl_indexed.loc[int(pid)]
                m_arr[i] = float(row["M"])
                s_arr[i] = float(row["S"])
                usage_arr[i] = float(row["usage_rate"])

        # Missingness indicator: 1 if player has any FGA evidence before anchor.
        m_play = (s_arr > 0).astype(np.float32)

        log1p_m = np.log1p(m_arr).astype(np.float32)
        log1p_s = np.log1p(s_arr).astype(np.float32)
        log_density = np.log((s_arr + 1.0) / (m_arr + 1.0)).astype(np.float32)

        # Stack Block B raw values (still in vocab order), pre-missingness.
        block_b_raw = np.zeros((n_players, BLOCK_B_END - BLOCK_A_END), dtype=np.float32)
        block_b_raw[:, 0:3] = pos_mix_full
        block_b_raw[:, 3] = fga_3pa_frac
        block_b_raw[:, 4] = fga_2pa_frac
        block_b_raw[:, 5] = usage_arr.astype(np.float32)
        block_b_raw[:, 6] = log1p_m
        block_b_raw[:, 7] = log1p_s
        block_b_raw[:, 8] = log_density
        block_b_raw[:, 9:] = role_full

        # Zero the play-derived block for players without prior FGA.
        traits[:, m_idx, BLOCK_A_END:BLOCK_B_END] = block_b_raw * m_play[:, None]
        traits[:, m_idx, M_PLAY_SLOT] = m_play

    return PlayerTraitsTable(
        traits=traits,
        player_ids=player_ids,
        snapshot_anchors=snapshot_anchors,
    )

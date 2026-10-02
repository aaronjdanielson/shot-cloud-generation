"""Data loading and causal feature construction.

This package turns raw NBA shot and game-log tables into the inputs of
the shot-cloud model: canonical shot tables (:func:`load_shots`), the
game-log join (:func:`join_game_logs`), the eight-zone taxonomy, the
per-shot context vector ``x_n`` (:class:`ContextEncoder`), and the
:class:`SnapshotStore` registry through which features derived from
training data are read causally. Within-game history, prior-outcome,
timing, on-court, and player-trait featurizers live in their own
submodules.
"""

from shotcloud.data.context import CONTEXT_DIM, FEATURE_LAYOUT, ContextEncoder
from shotcloud.data.game_logs import (
    CANONICAL_COLUMNS as GAME_LOG_COLUMNS,
)
from shotcloud.data.game_logs import (
    STARTER_MINUTES_THRESHOLD,
    join_game_logs,
    load_game_logs,
    load_starters,
)
from shotcloud.data.loaders import load_shots
from shotcloud.data.positions import (
    POSITION_GROUPS,
    assign_position_group,
    derive_positions_from_ra_rate,
    ra_rate_per_player,
)
from shotcloud.data.role_profile import (
    ROLE_FEATURE_NAMES,
    build_role_profiles,
    build_role_profiles_dataframe,
)
from shotcloud.data.schemas import NBA_STATS_RENAME, OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from shotcloud.data.snapshots import (
    N_OPP_EFFICIENCY_BINS,
    POSITION_MIXTURE_DIM,
    ROLE_PROFILE_DIM,
    SnapshotBundle,
    SnapshotStore,
    build_snapshot_store_from_shots,
)
from shotcloud.data.splits import split_by_season, split_fractional
from shotcloud.data.zones import (
    N_ZONES,
    ZONE_NAMES,
    zone_cell_mask,
    zone_cell_masks_per_zone,
    zone_from_strings,
    zone_from_xy,
    zone_from_xy_vectorized,
)

__all__ = [
    "CONTEXT_DIM",
    "FEATURE_LAYOUT",
    "GAME_LOG_COLUMNS",
    "NBA_STATS_RENAME",
    "N_OPP_EFFICIENCY_BINS",
    "N_ZONES",
    "OPTIONAL_COLUMNS",
    "POSITION_GROUPS",
    "POSITION_MIXTURE_DIM",
    "REQUIRED_COLUMNS",
    "ROLE_FEATURE_NAMES",
    "ROLE_PROFILE_DIM",
    "STARTER_MINUTES_THRESHOLD",
    "ZONE_NAMES",
    "ContextEncoder",
    "SnapshotBundle",
    "SnapshotStore",
    "assign_position_group",
    "build_role_profiles",
    "build_role_profiles_dataframe",
    "build_snapshot_store_from_shots",
    "derive_positions_from_ra_rate",
    "join_game_logs",
    "load_game_logs",
    "load_shots",
    "load_starters",
    "ra_rate_per_player",
    "split_by_season",
    "split_fractional",
    "zone_cell_mask",
    "zone_cell_masks_per_zone",
    "zone_from_strings",
    "zone_from_xy",
    "zone_from_xy_vectorized",
]

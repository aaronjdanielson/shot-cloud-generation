"""Own-history buckets for stratifying evaluation results.

Validation shots and player-games are bucketed by the player's causal
own-shot count ``Ĥ_p(t_n)`` at the game's snapshot (a player trait).
Pooled support is expected to matter most for cold-start and
sparse-history rows, so stratifying by ``Ĥ`` shows where in the history
distribution the collaborative support helps. The pooling diagnostics
(``scripts/pooling_diagnostics.py``) and the sparse-player evaluation
(``scripts/sparse_player_eval.py``) share these edges so their tables
align row for row.

The first bucket is ``0`` (no own history -- cold start). The remaining
buckets are half-open ``(lo, hi]`` so boundary values are unambiguous.
"""

from __future__ import annotations

from typing import Final

#: ``(lo, hi, label)`` triples. ``lo`` is *exclusive*, ``hi`` is
#: *inclusive*, except for the first bucket which is exactly 0 (lo=−0.5,
#: hi=0.5 so any float close to zero falls there). The labels are stable
#: identifiers — change them and downstream JSON / paper tables break.
HISTORY_BUCKETS: Final[list[tuple[float, float, str]]] = [
    (-0.5, 0.5, "0"),
    (0.5, 25.0, "1-25"),
    (25.0, 100.0, "26-100"),
    (100.0, 300.0, "101-300"),
    (300.0, 1000.0, "301-1000"),
    (1000.0, float("inf"), "1001+"),
]

#: Bucket labels in canonical sort order. Use this when iterating in
#: the order you want shown in a table.
BUCKET_ORDER: Final[tuple[str, ...]] = tuple(label for _, _, label in HISTORY_BUCKETS)


def history_bucket(h_hat: float) -> str:
    """Return the bucket label for an own-causal-shot count ``h_hat``.

    ``Ĥ`` is non-negative by construction; negative inputs fall in the
    ``"0"`` bucket and values past the last edge in ``"1001+"``.
    """
    for lo, hi, label in HISTORY_BUCKETS:
        if lo < h_hat <= hi:
            return label
    # Reached only for h_hat <= -0.5, outside every bucket.
    return BUCKET_ORDER[0] if h_hat <= 0.5 else BUCKET_ORDER[-1]


__all__ = ["BUCKET_ORDER", "HISTORY_BUCKETS", "history_bucket"]

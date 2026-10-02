"""Shared history-bucket definitions for the H̄-stratified diagnostics.

Two callers both want to bucket samples (val shots, val player-games)
by the trait-derived own-causal-shot count ``Ĥ_p(t_n)``:

* :mod:`scripts.pooling_diagnostics` — the pooled-mass-vs-history
  monotonicity check (introduced 2026-05-20).
* :mod:`scripts.sparse_player_eval` — the H̄-stratified cloud-metric
  readout that's the headline diagnostic for the PR3 retrieval design
  (2026-05-24): pooled support is supposed to matter mostly for
  cold-start / sparse-history rows; the sparse-player evaluation is
  what tells us *where* in the H̄ distribution the retrieval design
  actually helps.

The bucket edges match the pooling diagnostic's original cuts so
the two readouts can be aligned row-for-row in a paper figure.

The first bucket is ``0`` (exact zero own history — cold-start). The
remaining buckets are half-open ``(lo, hi]`` to keep the boundary
behavior unambiguous.
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

    Negative inputs (shouldn't happen — Ĥ ≥ 0 by construction) clamp
    to the ``"0"`` bucket; values past the last edge fall in
    ``"1001+"``.
    """
    for lo, hi, label in HISTORY_BUCKETS:
        if lo < h_hat <= hi:
            return label
    # Fallthrough only for h_hat strictly outside the union, which is
    # only possible for h_hat <= -0.5 (impossible Ĥ).
    return BUCKET_ORDER[0] if h_hat <= 0.5 else BUCKET_ORDER[-1]


__all__ = ["BUCKET_ORDER", "HISTORY_BUCKETS", "history_bucket"]

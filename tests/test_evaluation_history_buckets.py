"""Tests for :mod:`shotcloud.evaluation.history_buckets`.

The bucket edges are stable identifiers consumed by downstream JSON
output (pooling.json, sparse_eval.json) and paper tables. The tests
fence against accidental edge-shift or label rename.
"""

from __future__ import annotations

import pytest

from shotcloud.evaluation.history_buckets import (
    BUCKET_ORDER,
    HISTORY_BUCKETS,
    history_bucket,
)


def test_bucket_labels_and_order_are_stable() -> None:
    """The bucket labels and their order are load-bearing identifiers.
    Tests downstream of this module (pooling, sparse-player eval) match
    on these strings.
    """
    assert BUCKET_ORDER == ("0", "1-25", "26-100", "101-300", "301-1000", "1001+")
    assert tuple(label for _, _, label in HISTORY_BUCKETS) == BUCKET_ORDER


@pytest.mark.parametrize(
    "h_hat, expected",
    [
        # Exact zero → "0".
        (0.0, "0"),
        (0.4, "0"),  # < 0.5 → "0"
        (0.5, "0"),  # boundary inclusive on first bucket
        # 0.5 < h ≤ 25 → "1-25".
        (1.0, "1-25"),
        (10.0, "1-25"),
        (25.0, "1-25"),
        # 25 < h ≤ 100 → "26-100".
        (25.5, "26-100"),
        (50.0, "26-100"),
        (100.0, "26-100"),
        # 100 < h ≤ 300 → "101-300".
        (100.5, "101-300"),
        (200.0, "101-300"),
        (300.0, "101-300"),
        # 300 < h ≤ 1000 → "301-1000".
        (300.5, "301-1000"),
        (750.0, "301-1000"),
        (1000.0, "301-1000"),
        # 1000 < h → "1001+".
        (1000.5, "1001+"),
        (5000.0, "1001+"),
    ],
)
def test_history_bucket_classification(h_hat: float, expected: str) -> None:
    assert history_bucket(h_hat) == expected


def test_bucket_edges_are_disjoint_and_total() -> None:
    """Every nonneg float should land in exactly one bucket. Spot-check
    with a representative sweep — the cuts are deterministic enough
    that a sweep is the right test (not random sampling)."""
    sweep = [
        0.0,
        0.1,
        0.5,
        0.6,
        1.0,
        25.0,
        25.5,
        100.0,
        100.5,
        300.0,
        300.5,
        1000.0,
        1000.5,
        5000.0,
    ]
    labels = [history_bucket(h) for h in sweep]
    # Every result is in the canonical bucket set.
    for lab in labels:
        assert lab in BUCKET_ORDER
    # Monotone non-decreasing position in BUCKET_ORDER as h_hat grows.
    positions = [BUCKET_ORDER.index(lab) for lab in labels]
    for i in range(1, len(positions)):
        assert positions[i] >= positions[i - 1], (
            f"non-monotone at sweep[{i - 1}, {i}] = ({sweep[i - 1]}, {sweep[i]}) -> "
            f"({labels[i - 1]}, {labels[i]})"
        )


def test_history_bucket_negative_clamps_to_first_bucket() -> None:
    """Ĥ < 0 shouldn't happen (Ĥ is a count) but the function must
    return a valid bucket label rather than crash on the fallthrough."""
    assert history_bucket(-1.0) == "0"

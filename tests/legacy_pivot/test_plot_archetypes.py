"""Tests for ``scripts/legacy_pivot/plot_archetypes.py``: checkpoint loaders and figures.

The figure-rendering helpers are exercised end-to-end on a tiny
synthetic checkpoint dir to confirm they don't raise; we don't
assert pixel-level output beyond "the file exists and is non-empty".
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import numpy as np
import pytest

# Put scripts/legacy_pivot/ on sys.path so the script can be imported as a module.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "legacy_pivot"))
sys.path.insert(0, str(_REPO / "scripts"))

import plot_archetypes  # noqa: E402

import pretrain_snapshots  # noqa: E402
from shotcloud.grids import CourtGrid  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_checkpoint(
    tmp_dir: Path, anchor: str, *, K: int = 4, C: int | None = None, seed: int = 0
) -> Path:
    """Write a synthetic _AnchorFitOutput checkpoint at tmp_dir/<anchor>.npz.

    Defaults to C = canonical CourtGrid n_cells so the figure renderer
    accepts the result. Override C only for loader-validation tests.
    """
    if C is None:
        C = CourtGrid().n_cells
    rng = np.random.default_rng(seed)
    A = rng.dirichlet(np.ones(C), size=K).astype(np.float32)
    P = 30
    rho_raw = rng.dirichlet(np.ones(K), size=P).astype(np.float32)
    pair_l1 = np.zeros((K, K), dtype=np.float32)
    pair_sw = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(i + 1, K):
            pair_l1[i, j] = float(np.abs(A[i] - A[j]).sum())
            pair_sw[i, j] = float(rng.uniform(1.0, 10.0))
    out = pretrain_snapshots._AnchorFitOutput(
        archetypes=A,
        mixtures=rho_raw,
        archetype_player_ids=np.arange(P, dtype=np.int64),
        n_players_fit=P,
        final_loss=2.0,
        wallclock_s=1.0,
        archetype_usage=rho_raw.sum(axis=0).astype(np.float32),
        archetype_entropy=np.full(K, 5.0, dtype=np.float32),
        mixture_entropy_summary={
            "min": 1.0,
            "p10": 1.5,
            "median": 2.0,
            "p90": 2.5,
            "max": 3.0,
            "mean": 2.0,
            "log2_K_ceiling": float(np.log2(K)),
        },
        dead_archetypes=0,
        min_negative_div=0.0,
        archetype_pair_l1=pair_l1,
        archetype_pair_sw=pair_sw,
        archetype_pair_l1_min=float(pair_l1[np.triu_indices(K, k=1)].min()),
        archetype_pair_l1_median=float(np.median(pair_l1[np.triu_indices(K, k=1)])),
        archetype_pair_sw_min=float(pair_sw[np.triu_indices(K, k=1)].min()),
        archetype_pair_sw_median=float(np.median(pair_sw[np.triu_indices(K, k=1)])),
    )
    path = tmp_dir / f"{anchor}.npz"
    pretrain_snapshots._save_anchor_checkpoint(path, np.datetime64(anchor), out, K=K, C=C)
    return path


# ---------------------------------------------------------------------------
# _views_from_checkpoint_dir
# ---------------------------------------------------------------------------


def test_views_from_checkpoint_dir_sorts_by_anchor_date(tmp_path: Path) -> None:
    """Views are returned in chronological order regardless of glob order."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    # Write out of order to make the sort observable.
    _write_minimal_checkpoint(ckpt_dir, "2023-12-01", K=4, seed=2)
    _write_minimal_checkpoint(ckpt_dir, "2023-10-01", K=4, seed=0)
    _write_minimal_checkpoint(ckpt_dir, "2023-11-01", K=4, seed=1)

    views = plot_archetypes._views_from_checkpoint_dir(ckpt_dir)
    assert [v.anchor_label for v in views] == ["2023-10-01", "2023-11-01", "2023-12-01"]


def test_views_from_checkpoint_dir_raises_on_empty(tmp_path: Path) -> None:
    empty_dir = tmp_path / "no_checkpoints"
    empty_dir.mkdir()
    with pytest.raises(SystemExit, match=r"no \.npz checkpoints"):
        plot_archetypes._views_from_checkpoint_dir(empty_dir)


def test_views_from_checkpoint_dir_raises_on_K_mismatch(tmp_path: Path) -> None:
    """A run that switched K mid-way produces a useless temporal grid; refuse."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    C = CourtGrid().n_cells
    _write_minimal_checkpoint(ckpt_dir, "2023-10-01", K=4, C=C)
    _write_minimal_checkpoint(ckpt_dir, "2023-11-01", K=8, C=C)

    with pytest.raises(SystemExit, match="checkpoint shape mismatch"):
        plot_archetypes._views_from_checkpoint_dir(ckpt_dir)


def test_views_from_checkpoint_dir_raises_on_C_mismatch(tmp_path: Path) -> None:
    """Different grids across anchors → unaligned cells → meaningless gallery."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _write_minimal_checkpoint(ckpt_dir, "2023-10-01", K=4, C=50)
    _write_minimal_checkpoint(ckpt_dir, "2023-11-01", K=4, C=60)

    with pytest.raises(SystemExit, match="checkpoint shape mismatch"):
        plot_archetypes._views_from_checkpoint_dir(ckpt_dir)


# ---------------------------------------------------------------------------
# Temporal evolution + drift figures
# ---------------------------------------------------------------------------


def test_plot_archetype_temporal_evolution_renders(tmp_path: Path) -> None:
    """Figure produces a non-empty PNG without raising."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for i, anchor in enumerate(["2023-10-01", "2023-11-01", "2023-12-01"]):
        _write_minimal_checkpoint(ckpt_dir, anchor, K=4, seed=i)
    views = plot_archetypes._views_from_checkpoint_dir(ckpt_dir)

    out_path = tmp_path / "temporal.png"
    plot_archetypes.plot_archetype_temporal_evolution(views, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # non-trivial PNG


def test_plot_archetype_temporal_evolution_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="no views"):
        plot_archetypes.plot_archetype_temporal_evolution([], out_path=Path("/tmp/x.png"))


def test_plot_archetype_temporal_drift_renders(tmp_path: Path) -> None:
    """Drift line plot renders for >= 2 anchors."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for i, anchor in enumerate(["2023-10-01", "2023-11-01", "2023-12-01", "2024-01-01"]):
        _write_minimal_checkpoint(ckpt_dir, anchor, K=4, seed=i)
    views = plot_archetypes._views_from_checkpoint_dir(ckpt_dir)

    out_path = tmp_path / "drift.png"
    plot_archetypes.plot_archetype_temporal_drift(views, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000


def test_plot_archetype_temporal_drift_skips_when_single_anchor(tmp_path: Path) -> None:
    """Drift requires >= 2 anchors; with 1, function silently no-ops."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _write_minimal_checkpoint(ckpt_dir, "2023-10-01", K=4)
    views = plot_archetypes._views_from_checkpoint_dir(ckpt_dir)

    out_path = tmp_path / "drift_should_not_exist.png"
    plot_archetypes.plot_archetype_temporal_drift(views, out_path=out_path)
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# Position-usage figure
# ---------------------------------------------------------------------------


def _view_with_positions(K: int = 4, P: int = 30, seed: int = 0) -> plot_archetypes._ArchetypeView:
    """Construct a synthetic _ArchetypeView with both archetype mixtures
    and position mixtures populated, for the position-usage figure."""
    rng = np.random.default_rng(seed)
    C = CourtGrid().n_cells
    A = rng.dirichlet(np.ones(C), size=K).astype(np.float32)
    rho = rng.dirichlet(np.ones(K), size=P).astype(np.float32)
    pids = np.arange(P, dtype=np.int64)
    pos = rng.dirichlet(np.ones(3), size=P).astype(np.float32)
    return plot_archetypes._ArchetypeView(
        anchor_label="2023-10-01",
        A=A,
        mixtures=rho,
        player_ids=pids,
        usage=rho.sum(axis=0).astype(np.float32),
        archetype_entropy=None,
        pair_l1=None,
        pair_sw=None,
        causal_anchor=np.datetime64("2023-10-01"),
        position_mixtures=pos,
        position_player_ids=pids,
    )


def test_plot_archetype_position_usage_renders(tmp_path: Path) -> None:
    """The 2-panel position-usage figure produces a non-empty PNG."""
    view = _view_with_positions()
    out_path = tmp_path / "position_usage.png"
    plot_archetypes.plot_archetype_position_usage(view, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000


def test_plot_archetype_position_usage_skips_when_positions_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without position_mixtures, function prints and returns; no file written."""
    view = _view_with_positions()
    view_no_pos = plot_archetypes._ArchetypeView(
        anchor_label=view.anchor_label,
        A=view.A,
        mixtures=view.mixtures,
        player_ids=view.player_ids,
        usage=view.usage,
        archetype_entropy=view.archetype_entropy,
        pair_l1=view.pair_l1,
        pair_sw=view.pair_sw,
        causal_anchor=view.causal_anchor,
        position_mixtures=None,
        position_player_ids=None,
    )
    out_path = tmp_path / "should_not_exist.png"
    plot_archetypes.plot_archetype_position_usage(view_no_pos, out_path=out_path)
    captured = capsys.readouterr()
    assert "no position_mixtures" in captured.out
    assert not out_path.exists()


def test_plot_archetype_position_usage_handles_unmatched_players(tmp_path: Path) -> None:
    """When archetype.player_ids has pids not in position_player_ids, those
    players get the uniform (1/3, 1/3, 1/3) fallback. Figure still renders."""
    rng = np.random.default_rng(0)
    K, P_arch = 4, 20
    C = CourtGrid().n_cells
    A = rng.dirichlet(np.ones(C), size=K).astype(np.float32)
    rho = rng.dirichlet(np.ones(K), size=P_arch).astype(np.float32)
    arch_pids = np.arange(P_arch, dtype=np.int64)
    # Position info covers only half the archetype players (10 missing).
    P_pos = 10
    pos = rng.dirichlet(np.ones(3), size=P_pos).astype(np.float32)
    pos_pids = np.arange(P_pos, dtype=np.int64)
    view = plot_archetypes._ArchetypeView(
        anchor_label="2023-10-01",
        A=A,
        mixtures=rho,
        player_ids=arch_pids,
        usage=rho.sum(axis=0).astype(np.float32),
        archetype_entropy=None,
        pair_l1=None,
        pair_sw=None,
        causal_anchor=np.datetime64("2023-10-01"),
        position_mixtures=pos,
        position_player_ids=pos_pids,
    )
    out_path = tmp_path / "position_usage_partial.png"
    plot_archetypes.plot_archetype_position_usage(view, out_path=out_path)
    assert out_path.exists()

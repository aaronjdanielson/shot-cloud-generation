"""Tests for :mod:`scripts.pretrain_snapshots` helpers.

The full ``run_pretrain`` pipeline is tested end-to-end on synthetic
data; the soft position-mixture and anchor-grid helpers get unit
tests.

The script's heavyweight NMF fitter is exercised through a small
end-to-end run on a tiny synthetic shot table so we get coverage of
the warm-start path without the real NBA dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo scripts/ to sys.path so test can import the script as a module.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import pretrain_snapshots  # noqa: E402
from shotcloud.data import POSITION_MIXTURE_DIM  # noqa: E402

# ---------------------------------------------------------------------------
# soft_position_mixture
# ---------------------------------------------------------------------------


def test_soft_position_mixture_pure_guard() -> None:
    """RA-rate well below 0.15 → all guard mass."""
    out = pretrain_snapshots.soft_position_mixture(0.05)
    assert out.shape == (POSITION_MIXTURE_DIM,)
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0], atol=1e-6)


def test_soft_position_mixture_pure_big() -> None:
    """RA-rate above 0.40 → all big mass."""
    out = pretrain_snapshots.soft_position_mixture(0.60)
    np.testing.assert_allclose(out, [0.0, 0.0, 1.0], atol=1e-6)


def test_soft_position_mixture_pure_wing() -> None:
    """RA-rate in the wing plateau (0.25–0.30) → all wing mass."""
    out = pretrain_snapshots.soft_position_mixture(0.275)
    np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-6)


def test_soft_position_mixture_in_between_smoothly_interpolates() -> None:
    """RA-rate between breakpoints → mass shared across two adjacent
    categories, sums to 1."""
    out_low = pretrain_snapshots.soft_position_mixture(0.20)  # halfway guard↔wing
    np.testing.assert_allclose(out_low.sum(), 1.0, atol=1e-6)
    assert out_low[0] > 0 and out_low[1] > 0
    assert out_low[2] == 0  # no big mass

    out_high = pretrain_snapshots.soft_position_mixture(0.35)  # halfway wing↔big
    np.testing.assert_allclose(out_high.sum(), 1.0, atol=1e-6)
    assert out_high[1] > 0 and out_high[2] > 0
    assert out_high[0] == 0  # no guard mass


def test_soft_position_mixture_always_sums_to_one() -> None:
    for r in np.linspace(0.0, 1.0, 21):
        out = pretrain_snapshots.soft_position_mixture(float(r))
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-5, err_msg=f"r={r}")
        assert (out >= 0).all()


def test_soft_position_mixture_clamps_to_unit_range() -> None:
    """Out-of-range inputs (negative or > 1) clamp to a valid simplex."""
    np.testing.assert_allclose(
        pretrain_snapshots.soft_position_mixture(-0.5),
        pretrain_snapshots.soft_position_mixture(0.0),
    )
    np.testing.assert_allclose(
        pretrain_snapshots.soft_position_mixture(1.5),
        pretrain_snapshots.soft_position_mixture(1.0),
    )


# ---------------------------------------------------------------------------
# build_position_mixtures
# ---------------------------------------------------------------------------


def test_build_position_mixtures_rim_big_classified_correctly() -> None:
    """Player whose shots are all RA → big-dominated mixture."""
    rng = np.random.default_rng(0)
    shots = pd.DataFrame(
        [
            {
                "player_id": 1,
                "x": float(rng.normal(0, 0.5)),
                "y": float(rng.normal(2.5, 0.5)),  # RA region
            }
            for _ in range(100)
        ]
    )
    out = pretrain_snapshots.build_position_mixtures(shots)
    assert 1 in out
    assert out[1].argmax() == 2  # big


def test_build_position_mixtures_perimeter_player_classified_as_guard() -> None:
    """Player whose shots are all 3-point territory → guard-dominated mixture."""
    rng = np.random.default_rng(1)
    shots = pd.DataFrame(
        [
            {
                "player_id": 2,
                "x": float(rng.normal(20, 0.5)),
                "y": float(rng.normal(20, 0.5)),  # well outside RA
            }
            for _ in range(100)
        ]
    )
    out = pretrain_snapshots.build_position_mixtures(shots)
    assert out[2].argmax() == 0  # guard


def test_build_position_mixtures_min_shots_filter() -> None:
    """min_shots filter drops sparse players."""
    shots = pd.DataFrame(
        [
            {"player_id": 1, "x": 0.0, "y": 2.5},  # 1 shot
            *[{"player_id": 2, "x": 0.0, "y": 2.5} for _ in range(50)],
        ]
    )
    out = pretrain_snapshots.build_position_mixtures(shots, min_shots=10)
    assert 1 not in out
    assert 2 in out


def test_build_position_mixtures_empty_input() -> None:
    out = pretrain_snapshots.build_position_mixtures(pd.DataFrame(columns=["player_id", "x", "y"]))
    assert out == {}


# ---------------------------------------------------------------------------
# monthly_anchor_grid
# ---------------------------------------------------------------------------


def test_monthly_anchor_grid_strictly_increasing() -> None:
    anchors = pretrain_snapshots.monthly_anchor_grid(
        np.datetime64("2018-01-01", "D"),
        np.datetime64("2020-12-31", "D"),
    )
    assert len(anchors) > 0
    for i in range(len(anchors) - 1):
        assert anchors[i + 1] > anchors[i]


def test_monthly_anchor_grid_first_of_month() -> None:
    anchors = pretrain_snapshots.monthly_anchor_grid(
        np.datetime64("2018-01-15", "D"),
        np.datetime64("2018-06-30", "D"),
    )
    # All anchors should be on day 1 of a month
    for a in anchors:
        ts = pd.Timestamp(a)
        assert ts.day == 1


def test_monthly_anchor_grid_count() -> None:
    """Approximately one anchor per month over 3 years = ~36."""
    anchors = pretrain_snapshots.monthly_anchor_grid(
        np.datetime64("2018-01-01", "D"),
        np.datetime64("2020-12-01", "D"),
    )
    assert 35 <= len(anchors) <= 36


# ---------------------------------------------------------------------------
# End-to-end pipeline (small synthetic dataset)
# ---------------------------------------------------------------------------


_PLAYERS = tuple(range(1001, 1013))  # 12 synthetic players → enough for K=4 NMF


def _synthetic_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Synthetic shots + game-logs CSV pair under tmp_path.

    Twelve players with varied shot-cluster centers spanning rim,
    midrange, and three-point territory, so per-anchor NMF has
    enough samples and the archetype basis has room to differentiate.
    """
    rng = np.random.default_rng(0)
    # Per-player shot-cluster centers (in tenths of feet to match NBA Stats).
    centers = {
        pid: (
            float(rng.uniform(-220, 220)),
            float(rng.uniform(0, 280)),
        )
        for pid in _PLAYERS
    }
    rows = []
    for date in pd.date_range("2018-01-01", "2020-12-31", freq="W"):
        # Each week, every player takes ~3 shots near their cluster center.
        for pid in _PLAYERS:
            cx, cy = centers[pid]
            for _ in range(3):
                rows.append(
                    {
                        "GAME_ID": int(rng.integers(20180000, 20200000)),
                        "GAME_DATE": pd.Timestamp(date).strftime("%Y%m%d"),
                        "game_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                        "PERIOD": int(rng.choice([1, 2, 3, 4])),
                        "MINUTES_REMAINING": int(rng.integers(0, 12)),
                        "SECONDS_REMAINING": int(rng.integers(0, 60)),
                        "PLAYER_ID": int(pid),
                        "TEAM_NAME": "TestTeam",
                        "HTM": "TST",
                        "VTM": str(rng.choice(["BOS", "LAL", "GSW"])),
                        "LOC_X": float(rng.normal(cx, 30.0)),
                        "LOC_Y": float(rng.normal(cy, 30.0)),
                        "SHOT_ATTEMPTED_FLAG": 1,
                        "SHOT_MADE_FLAG": int(rng.random() < 0.45),
                    }
                )
    shots_path = tmp_path / "shots.csv"
    pd.DataFrame(rows).to_csv(shots_path, index=False)

    # Game logs (one row per (player, game) pair seen in the shots).
    gl_rows = []
    for game_id in pd.unique(pd.DataFrame(rows)["GAME_ID"]):
        for pid in _PLAYERS:
            gl_rows.append({"player_id": int(pid), "game_id": int(game_id), "minutes": 25})
    gl_path = tmp_path / "game_logs.csv"
    pd.DataFrame(gl_rows).to_csv(gl_path, index=False)

    return shots_path, gl_path


def test_run_pretrain_end_to_end(tmp_path: Path) -> None:
    """Tiny synthetic run produces a valid snapshot store + manifest."""
    shots_path, gl_path = _synthetic_dataset(tmp_path)
    output_path = tmp_path / "snapshots.pt"

    store = pretrain_snapshots.run_pretrain(
        shots_path=shots_path,
        game_logs_path=gl_path,
        starters_path=None,
        output_path=output_path,
        start_date="2018-01-01",
        end_date="2020-12-31",
        K=4,
        min_shots=5,
        bandwidth=1.5,
        kappa=200.0,
        recency_half_life_days=365.0,
        # Tiny iteration counts for test speed; these only need to be
        # non-zero for the fit to produce simplex-valid output.
        w_max_iter=5,
        w_sinkhorn_iter=8,
        seed=0,
        verbose=False,
        with_archetypes=True,
    )

    # Store is non-empty; archetype surfaces are populated.
    assert len(store) > 0
    for bundle in store.bundles:
        assert bundle.archetype_surfaces is not None
        assert bundle.archetype_surfaces.shape == (4, store.bundles[0].archetype_surfaces.shape[1])

    # Causality holds (the script asserts this internally; double-check here).
    shots_df = pd.read_csv(shots_path)
    shots_df["date"] = pd.to_datetime(shots_df["game_date"])
    # Reload via load_shots to match the script's processing path
    from shotcloud.data import load_shots as _load_shots

    shots_loaded = _load_shots(shots_path)
    store.assert_causal(shots_loaded)

    # Output artifact + manifest exist
    assert output_path.exists()
    manifest_path = output_path.with_suffix(".manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_anchors"] == len(store)
    assert manifest["K"] == 4
    assert all(p["has_archetypes"] for p in manifest["per_anchor"])


def test_run_pretrain_warm_starts_archetypes(tmp_path: Path) -> None:
    """Successive snapshot bundles should have archetypes that drift
    (warm-started fit) rather than being completely independent."""
    shots_path, gl_path = _synthetic_dataset(tmp_path)
    output_path = tmp_path / "snapshots.pt"

    store = pretrain_snapshots.run_pretrain(
        shots_path=shots_path,
        game_logs_path=gl_path,
        starters_path=None,
        output_path=output_path,
        start_date="2018-01-01",
        end_date="2020-12-31",
        K=4,
        min_shots=5,
        w_max_iter=10,
        w_sinkhorn_iter=8,
        seed=0,
        verbose=False,
        with_archetypes=True,
    )

    # With warm-starting, consecutive snapshots' archetype surfaces should
    # be similar (small TV distance) — at least more similar than to a
    # randomly-permuted version.
    assert len(store) >= 2
    a0 = store.bundles[0].archetype_surfaces
    a1 = store.bundles[1].archetype_surfaces
    assert a0 is not None and a1 is not None
    # Per-archetype TV distance.
    same_label_tv = 0.5 * np.abs(a0 - a1).sum(axis=1).mean()
    # Permute archetype labels and compare; warm-started fits keep labels
    # aligned, so the same-label TV should be smaller than the
    # permuted-label TV on average.
    perm = np.array([1, 0, 3, 2])  # arbitrary non-identity permutation
    permuted_tv = 0.5 * np.abs(a0 - a1[perm]).sum(axis=1).mean()
    assert same_label_tv <= permuted_tv * 1.1, (
        f"warm-start failed to align labels: same={same_label_tv:.4f}, permuted={permuted_tv:.4f}"
    )


def test_run_pretrain_manifest_contents(tmp_path: Path) -> None:
    shots_path, gl_path = _synthetic_dataset(tmp_path)
    output_path = tmp_path / "snapshots.pt"

    pretrain_snapshots.run_pretrain(
        shots_path=shots_path,
        game_logs_path=gl_path,
        starters_path=None,
        output_path=output_path,
        start_date="2018-01-01",
        end_date="2020-12-31",
        K=3,
        min_shots=5,
        w_max_iter=5,
        w_sinkhorn_iter=8,
        seed=0,
        verbose=False,
        with_archetypes=True,
    )

    manifest = json.loads(output_path.with_suffix(".manifest.json").read_text())
    assert manifest["K"] == 3
    assert manifest["params"]["min_shots"] == 5
    assert "first_anchor" in manifest and "last_anchor" in manifest
    assert manifest["n_anchors"] == len(manifest["per_anchor"])


# ---------------------------------------------------------------------------
# Per-anchor checkpointing + resume
# ---------------------------------------------------------------------------


def test_save_load_anchor_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Helpers persist all _AnchorFitOutput fields losslessly."""
    pair_l1 = np.zeros((4, 4), dtype=np.float32)
    pair_l1[np.triu_indices(4, k=1)] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    pair_sw = np.zeros((4, 4), dtype=np.float32)
    pair_sw[np.triu_indices(4, k=1)] = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    out = pretrain_snapshots._AnchorFitOutput(
        archetypes=np.full((4, 50), 1.0 / 50, dtype=np.float32),
        mixtures=np.full((10, 4), 0.25, dtype=np.float32),
        archetype_player_ids=np.arange(10, dtype=np.int64),
        n_players_fit=10,
        final_loss=1.234,
        wallclock_s=2.5,
        archetype_usage=np.array([2.5, 2.5, 2.5, 2.5], dtype=np.float32),
        archetype_entropy=np.array([5.6, 5.6, 5.6, 5.6], dtype=np.float32),
        mixture_entropy_summary={
            "min": 1.0,
            "p10": 1.5,
            "median": 2.0,
            "p90": 2.5,
            "max": 3.0,
            "mean": 2.0,
            "log2_K_ceiling": 2.0,
        },
        dead_archetypes=0,
        min_negative_div=-1e-5,
        archetype_pair_l1=pair_l1,
        archetype_pair_sw=pair_sw,
        archetype_pair_l1_min=0.1,
        archetype_pair_l1_median=0.35,
        archetype_pair_sw_min=1.1,
        archetype_pair_sw_median=1.35,
    )
    anchor = np.datetime64("2018-06-01")
    path = pretrain_snapshots._checkpoint_path_for(tmp_path, anchor)
    pretrain_snapshots._save_anchor_checkpoint(path, anchor, out, K=4, C=50)
    assert path.exists()
    loaded = pretrain_snapshots._load_anchor_checkpoint(
        path, expected_anchor=anchor, expected_K=4, expected_C=50
    )
    np.testing.assert_array_equal(loaded.archetypes, out.archetypes)
    np.testing.assert_array_equal(loaded.mixtures, out.mixtures)
    np.testing.assert_array_equal(loaded.archetype_player_ids, out.archetype_player_ids)
    np.testing.assert_array_equal(loaded.archetype_usage, out.archetype_usage)
    np.testing.assert_array_equal(loaded.archetype_entropy, out.archetype_entropy)
    np.testing.assert_array_equal(loaded.archetype_pair_l1, out.archetype_pair_l1)
    np.testing.assert_array_equal(loaded.archetype_pair_sw, out.archetype_pair_sw)
    assert loaded.n_players_fit == out.n_players_fit
    assert loaded.final_loss == out.final_loss
    assert loaded.wallclock_s == out.wallclock_s
    assert loaded.dead_archetypes == out.dead_archetypes
    assert loaded.min_negative_div == out.min_negative_div
    assert loaded.mixture_entropy_summary == out.mixture_entropy_summary
    assert loaded.archetype_pair_l1_min == out.archetype_pair_l1_min
    assert loaded.archetype_pair_l1_median == out.archetype_pair_l1_median
    assert loaded.archetype_pair_sw_min == out.archetype_pair_sw_min
    assert loaded.archetype_pair_sw_median == out.archetype_pair_sw_median


def test_save_load_anchor_checkpoint_handles_fallback_no_mixtures(tmp_path: Path) -> None:
    """Checkpoints written from a fallback path (n_players < K) have no
    mixtures / player_ids. Round-trip must preserve None."""
    out = pretrain_snapshots._AnchorFitOutput(
        archetypes=np.full((4, 50), 1.0 / 50, dtype=np.float32),
        mixtures=None,
        archetype_player_ids=None,
        n_players_fit=2,
        final_loss=None,
        wallclock_s=0.1,
        archetype_usage=None,
        archetype_entropy=None,
        mixture_entropy_summary=None,
        dead_archetypes=0,
        min_negative_div=0.0,
    )
    anchor = np.datetime64("2018-01-01")
    path = pretrain_snapshots._checkpoint_path_for(tmp_path, anchor)
    pretrain_snapshots._save_anchor_checkpoint(path, anchor, out, K=4, C=50)
    loaded = pretrain_snapshots._load_anchor_checkpoint(
        path, expected_anchor=anchor, expected_K=4, expected_C=50
    )
    assert loaded.mixtures is None
    assert loaded.archetype_player_ids is None
    assert loaded.archetype_usage is None
    assert loaded.mixture_entropy_summary is None
    assert loaded.final_loss is None
    assert loaded.archetype_pair_l1 is None
    assert loaded.archetype_pair_sw is None
    assert loaded.archetype_pair_l1_min is None
    assert loaded.archetype_pair_sw_min is None


def test_load_anchor_checkpoint_rejects_K_mismatch(tmp_path: Path) -> None:
    """Loading with a different K than the checkpoint was fit with raises."""
    out = pretrain_snapshots._AnchorFitOutput(
        archetypes=np.full((4, 50), 1.0 / 50, dtype=np.float32),
        mixtures=None,
        archetype_player_ids=None,
        n_players_fit=0,
        final_loss=None,
        wallclock_s=0.0,
        archetype_usage=None,
        archetype_entropy=None,
        mixture_entropy_summary=None,
        dead_archetypes=0,
        min_negative_div=0.0,
    )
    anchor = np.datetime64("2018-06-01")
    path = pretrain_snapshots._checkpoint_path_for(tmp_path, anchor)
    pretrain_snapshots._save_anchor_checkpoint(path, anchor, out, K=4, C=50)
    import pytest

    with pytest.raises(ValueError, match="K=4"):
        pretrain_snapshots._load_anchor_checkpoint(
            path, expected_anchor=anchor, expected_K=3, expected_C=50
        )
    with pytest.raises(ValueError, match="C="):
        pretrain_snapshots._load_anchor_checkpoint(
            path, expected_anchor=anchor, expected_K=4, expected_C=60
        )
    with pytest.raises(ValueError, match="anchor"):
        pretrain_snapshots._load_anchor_checkpoint(
            path,
            expected_anchor=np.datetime64("2019-06-01"),
            expected_K=4,
            expected_C=50,
        )


def test_run_pretrain_writes_checkpoints_per_anchor(tmp_path: Path) -> None:
    """When --checkpoint-dir is set, every assembled bundle gets a .npz."""
    shots_path, gl_path = _synthetic_dataset(tmp_path)
    output_path = tmp_path / "snapshots.pt"
    ckpt_dir = tmp_path / "checkpoints"

    store = pretrain_snapshots.run_pretrain(
        shots_path=shots_path,
        game_logs_path=gl_path,
        starters_path=None,
        output_path=output_path,
        start_date="2018-01-01",
        end_date="2020-12-31",
        K=4,
        min_shots=5,
        w_max_iter=5,
        w_sinkhorn_iter=8,
        seed=0,
        verbose=False,
        with_archetypes=True,
        checkpoint_dir=ckpt_dir,
    )

    # One checkpoint per assembled bundle, named by anchor date.
    saved = sorted(p.name for p in ckpt_dir.glob("*.npz"))
    expected = sorted(f"{b.anchor_date}.npz" for b in store.bundles)
    assert saved == expected
    # Each is loadable with the same K and C the run used.
    for bundle in store.bundles:
        path = ckpt_dir / f"{bundle.anchor_date}.npz"
        loaded = pretrain_snapshots._load_anchor_checkpoint(
            path,
            expected_anchor=bundle.anchor_date,
            expected_K=4,
            expected_C=bundle.archetype_surfaces.shape[1],  # type: ignore[union-attr]
        )
        np.testing.assert_allclose(loaded.archetypes, bundle.archetype_surfaces, atol=1e-6)


def test_run_pretrain_resume_skips_existing_anchor_fits(tmp_path: Path) -> None:
    """Second run with --resume reuses checkpoints (no refit) and produces
    archetypes byte-identical to the loaded files. Warm-start chain
    is preserved because each loaded archetype seeds the next anchor."""
    shots_path, gl_path = _synthetic_dataset(tmp_path)
    output_first = tmp_path / "snapshots_first.pt"
    output_second = tmp_path / "snapshots_resume.pt"
    ckpt_dir = tmp_path / "checkpoints"

    # First pass: cold run, writes checkpoints.
    store_first = pretrain_snapshots.run_pretrain(
        shots_path=shots_path,
        game_logs_path=gl_path,
        starters_path=None,
        output_path=output_first,
        start_date="2018-01-01",
        end_date="2020-12-31",
        K=4,
        min_shots=5,
        w_max_iter=5,
        w_sinkhorn_iter=8,
        seed=0,
        verbose=False,
        with_archetypes=True,
        checkpoint_dir=ckpt_dir,
    )
    # Snapshot mtimes; resume must NOT rewrite them.
    mtimes = {p.name: p.stat().st_mtime_ns for p in ckpt_dir.glob("*.npz")}

    # Second pass: --resume should load every checkpoint and skip fitting.
    store_resume = pretrain_snapshots.run_pretrain(
        shots_path=shots_path,
        game_logs_path=gl_path,
        starters_path=None,
        output_path=output_second,
        start_date="2018-01-01",
        end_date="2020-12-31",
        K=4,
        min_shots=5,
        # Use a very different seed and far fewer iters: if the fit
        # were actually rerun, archetypes would differ. Resume must
        # reproduce the first pass byte-for-byte.
        w_max_iter=99,
        w_sinkhorn_iter=99,
        seed=999,
        verbose=False,
        with_archetypes=True,
        checkpoint_dir=ckpt_dir,
        resume=True,
    )

    # Same number of bundles; archetypes match exactly per anchor.
    assert len(store_first) == len(store_resume)
    for b_first, b_resume in zip(store_first.bundles, store_resume.bundles, strict=True):
        assert b_first.archetype_surfaces is not None
        assert b_resume.archetype_surfaces is not None
        np.testing.assert_array_equal(b_first.archetype_surfaces, b_resume.archetype_surfaces)
    # mtimes unchanged → no re-write happened on resume.
    new_mtimes = {p.name: p.stat().st_mtime_ns for p in ckpt_dir.glob("*.npz")}
    assert mtimes == new_mtimes


def test_run_pretrain_overwrite_checkpoints_refits(tmp_path: Path) -> None:
    """--overwrite-checkpoints ignores existing files and rewrites them."""
    shots_path, gl_path = _synthetic_dataset(tmp_path)
    output_first = tmp_path / "snapshots_first.pt"
    output_second = tmp_path / "snapshots_refit.pt"
    ckpt_dir = tmp_path / "checkpoints"

    pretrain_snapshots.run_pretrain(
        shots_path=shots_path,
        game_logs_path=gl_path,
        starters_path=None,
        output_path=output_first,
        start_date="2018-01-01",
        end_date="2020-12-31",
        K=4,
        min_shots=5,
        w_max_iter=5,
        w_sinkhorn_iter=8,
        seed=0,
        verbose=False,
        with_archetypes=True,
        checkpoint_dir=ckpt_dir,
    )
    mtimes_before = {p.name: p.stat().st_mtime_ns for p in ckpt_dir.glob("*.npz")}

    pretrain_snapshots.run_pretrain(
        shots_path=shots_path,
        game_logs_path=gl_path,
        starters_path=None,
        output_path=output_second,
        start_date="2018-01-01",
        end_date="2020-12-31",
        K=4,
        min_shots=5,
        w_max_iter=5,
        w_sinkhorn_iter=8,
        seed=0,
        verbose=False,
        with_archetypes=True,
        checkpoint_dir=ckpt_dir,
        resume=True,
        overwrite_checkpoints=True,
    )
    mtimes_after = {p.name: p.stat().st_mtime_ns for p in ckpt_dir.glob("*.npz")}
    # Every file's mtime should have advanced.
    for name in mtimes_before:
        assert mtimes_after[name] > mtimes_before[name], name


def test_run_pretrain_refuses_to_overwrite_output(tmp_path: Path) -> None:
    """Default behavior: output_path that already exists → FileExistsError."""
    import pytest

    shots_path, gl_path = _synthetic_dataset(tmp_path)
    output_path = tmp_path / "snapshots.pt"
    output_path.write_bytes(b"existing artifact")  # simulate a precious prior file

    with pytest.raises(FileExistsError, match="--overwrite-output"):
        pretrain_snapshots.run_pretrain(
            shots_path=shots_path,
            game_logs_path=gl_path,
            starters_path=None,
            output_path=output_path,
            start_date="2018-01-01",
            end_date="2020-12-31",
            K=4,
            min_shots=5,
            w_max_iter=2,
            w_sinkhorn_iter=4,
            seed=0,
            verbose=False,
            with_archetypes=True,
        )
    # Untouched: the prior file is still there.
    assert output_path.read_bytes() == b"existing artifact"


def test_run_pretrain_first_anchor_decouples_data_window_from_anchors(
    tmp_path: Path,
) -> None:
    """--first-anchor restricts anchors without restricting the data
    window: shots before --first-anchor are still available as causal
    history to the assembled bundles."""
    shots_path, gl_path = _synthetic_dataset(tmp_path)
    full_output = tmp_path / "snapshots_full.pt"
    eval_output = tmp_path / "snapshots_eval.pt"

    # Full pass: data window 2018-01-01 .. 2020-12-31, all anchors.
    full = pretrain_snapshots.run_pretrain(
        shots_path=shots_path,
        game_logs_path=gl_path,
        starters_path=None,
        output_path=full_output,
        start_date="2018-01-01",
        end_date="2020-12-31",
        K=4,
        min_shots=5,
        w_max_iter=2,
        w_sinkhorn_iter=4,
        seed=0,
        verbose=False,
        with_archetypes=True,
    )

    # Eval pass: same data window, anchors restricted to 2020 only.
    eval_run = pretrain_snapshots.run_pretrain(
        shots_path=shots_path,
        game_logs_path=gl_path,
        starters_path=None,
        output_path=eval_output,
        start_date="2018-01-01",
        end_date="2020-12-31",
        first_anchor="2020-01-01",
        K=4,
        min_shots=5,
        w_max_iter=2,
        w_sinkhorn_iter=4,
        seed=0,
        verbose=False,
        with_archetypes=True,
    )

    # Eval pass has strictly fewer bundles, all on or after 2020-01-01.
    assert len(eval_run) < len(full)
    assert len(eval_run) > 0
    eval_dates = {str(b.anchor_date) for b in eval_run.bundles}
    full_dates = {str(b.anchor_date) for b in full.bundles}
    assert eval_dates.issubset(full_dates)
    for d in eval_dates:
        assert d >= "2020-01-01"

    # The 2020 anchors in the eval run see the same causal history as
    # in the full run (since the data window and underlying shots are
    # identical), so the per-anchor n_players matches.
    full_by_date = {str(b.anchor_date): len(b.player_ids) for b in full.bundles}
    for b in eval_run.bundles:
        assert len(b.player_ids) == full_by_date[str(b.anchor_date)]


def test_run_pretrain_first_anchor_outside_window_raises(tmp_path: Path) -> None:
    """--first-anchor before --start-date or after --end-date is rejected."""
    import pytest

    shots_path, gl_path = _synthetic_dataset(tmp_path)
    output_path = tmp_path / "snapshots.pt"

    with pytest.raises(SystemExit, match="--first-anchor"):
        pretrain_snapshots.run_pretrain(
            shots_path=shots_path,
            game_logs_path=gl_path,
            starters_path=None,
            output_path=output_path,
            start_date="2018-01-01",
            end_date="2020-12-31",
            first_anchor="2017-06-01",  # before start-date
            K=4,
            min_shots=5,
            w_max_iter=2,
            w_sinkhorn_iter=4,
            seed=0,
            verbose=False,
            with_archetypes=True,
        )

    with pytest.raises(SystemExit, match="--first-anchor"):
        pretrain_snapshots.run_pretrain(
            shots_path=shots_path,
            game_logs_path=gl_path,
            starters_path=None,
            output_path=output_path,
            start_date="2018-01-01",
            end_date="2020-12-31",
            first_anchor="2021-06-01",  # after end-date
            K=4,
            min_shots=5,
            w_max_iter=2,
            w_sinkhorn_iter=4,
            seed=0,
            verbose=False,
            with_archetypes=True,
        )


def test_run_pretrain_overwrite_output_replaces_existing(tmp_path: Path) -> None:
    """With --overwrite-output, the existing file is replaced cleanly."""
    shots_path, gl_path = _synthetic_dataset(tmp_path)
    output_path = tmp_path / "snapshots.pt"
    output_path.write_bytes(b"existing artifact")

    pretrain_snapshots.run_pretrain(
        shots_path=shots_path,
        game_logs_path=gl_path,
        starters_path=None,
        output_path=output_path,
        start_date="2018-01-01",
        end_date="2020-12-31",
        K=4,
        min_shots=5,
        w_max_iter=2,
        w_sinkhorn_iter=4,
        seed=0,
        verbose=False,
        with_archetypes=True,
        overwrite_output=True,
    )
    # Replaced — no longer the placeholder bytes.
    assert output_path.read_bytes() != b"existing artifact"


# ---------------------------------------------------------------------------
# Farthest-point cold-start initialization (collapse-prevention)
# ---------------------------------------------------------------------------


def test_fps_init_returns_K_valid_simplex_rows() -> None:
    """FPS init produces K rows of shape (C,), each summing to ~1, non-negative."""
    rng = np.random.default_rng(0)
    P, C, K = 30, 50, 8
    Q = rng.dirichlet(np.ones(C), size=P).astype(np.float32)  # (P, C) simplex rows
    A0 = pretrain_snapshots._fps_init_from_player_kdes(Q, K)
    assert A0.shape == (K, C)
    assert A0.dtype == np.float32
    assert (A0 >= 0).all()
    np.testing.assert_allclose(A0.sum(axis=1), 1.0, atol=1e-5)


def test_fps_init_picks_distinct_rows_when_P_geq_K() -> None:
    """The farthest-point algorithm picks K mutually-distinct seed rows.

    Pairwise L1 between the K seeds should all be strictly positive (no
    duplicates). Smoothing means the returned rows aren't byte-identical
    to any single Q row, but the underlying selection is unique.
    """
    rng = np.random.default_rng(0)
    P, C, K = 20, 100, 8
    Q = rng.dirichlet(np.ones(C), size=P).astype(np.float32)
    A0 = pretrain_snapshots._fps_init_from_player_kdes(Q, K)
    pair_l1 = np.array([np.abs(A0[i] - A0[j]).sum() for i in range(K) for j in range(i + 1, K)])
    # Min pairwise L1 is well above zero — atoms are not collapsed.
    assert pair_l1.min() > 0.05, f"FPS produced near-duplicate atoms: {pair_l1}"


def test_fps_init_is_deterministic() -> None:
    """FPS over Q is deterministic — no RNG dependency once Q is fixed."""
    rng = np.random.default_rng(0)
    Q = rng.dirichlet(np.ones(50), size=20).astype(np.float32)
    A0_a = pretrain_snapshots._fps_init_from_player_kdes(Q, K=6)
    A0_b = pretrain_snapshots._fps_init_from_player_kdes(Q, K=6)
    np.testing.assert_array_equal(A0_a, A0_b)


def test_fps_init_first_seed_is_nearest_to_mean() -> None:
    """First selected row should be the one closest to mean(Q) in L2.

    Construct Q so the mean is unambiguous — one row exactly equals
    the mean, K - 1 rows are far away. FPS must pick the central row first.
    """
    C = 50
    central = np.full(C, 1.0 / C, dtype=np.float32)  # exactly uniform = mean of below
    extreme_rows = []
    for k in range(7):
        v = np.zeros(C, dtype=np.float32)
        v[k * 7 % C] = 1.0  # near-deltas, far from uniform
        extreme_rows.append(v)
    # Build Q so that mean = central exactly: include enough copies of `central`
    # to pull the mean toward it. Use 7 extremes + 7 centrals.
    Q = np.stack([*extreme_rows, *([central] * 7)], axis=0)
    A0 = pretrain_snapshots._fps_init_from_player_kdes(Q, K=4, smoothing_eps=0.0)
    # The first seed should be the central row (closest to mean).
    np.testing.assert_allclose(A0[0], central, atol=1e-5)


def test_fps_init_raises_when_n_players_lt_K() -> None:
    """FPS init contract: n_players >= K. Below that, callers should
    use _seed_archetypes_from_player_kdes instead."""
    import pytest

    Q = np.full((3, 50), 1.0 / 50, dtype=np.float32)
    with pytest.raises(ValueError, match="n_players >= K"):
        pretrain_snapshots._fps_init_from_player_kdes(Q, K=8)


# ---------------------------------------------------------------------------
# Pairwise archetype distance diagnostics
# ---------------------------------------------------------------------------


def test_pairwise_archetype_distances_zero_for_identical_atoms() -> None:
    """All-identical archetypes give an upper-tri matrix of zeros."""
    K, C = 5, 50
    A = np.full((K, C), 1.0 / C, dtype=np.float32)
    centers = np.stack([np.linspace(-25, 25, C), np.zeros(C)], axis=1).astype(np.float64)
    L1, SW, stats = pretrain_snapshots._pairwise_archetype_distances(A, centers, sw_n_projections=8)
    assert L1.shape == (K, K)
    assert SW.shape == (K, K)
    np.testing.assert_allclose(L1, 0.0, atol=1e-6)
    np.testing.assert_allclose(SW, 0.0, atol=1e-6)
    assert stats["l1_min"] == 0.0
    assert stats["sw_min"] == 0.0


def test_pairwise_archetype_distances_strict_upper_triangular() -> None:
    """Diagonal and lower-triangular entries are zero; only k < l populated."""
    rng = np.random.default_rng(0)
    K, C = 4, 30
    A = rng.dirichlet(np.ones(C), size=K).astype(np.float32)
    centers = np.stack([np.linspace(-25, 25, C), np.zeros(C)], axis=1).astype(np.float64)
    L1, SW, _ = pretrain_snapshots._pairwise_archetype_distances(A, centers, sw_n_projections=8)
    # Diagonal is zero
    np.testing.assert_array_equal(np.diag(L1), 0.0)
    np.testing.assert_array_equal(np.diag(SW), 0.0)
    # Lower triangle is zero
    for i in range(K):
        for j in range(i):
            assert L1[i, j] == 0.0
            assert SW[i, j] == 0.0
    # Upper triangle is positive (random simplex rows are distinct)
    for i in range(K):
        for j in range(i + 1, K):
            assert L1[i, j] > 0.0
            assert SW[i, j] >= 0.0


def test_pairwise_archetype_distances_summary_matches_upper_tri() -> None:
    """summary stats agree with manual reduction over upper-triangular entries."""
    rng = np.random.default_rng(0)
    K, C = 4, 30
    A = rng.dirichlet(np.ones(C), size=K).astype(np.float32)
    centers = np.stack([np.linspace(-25, 25, C), np.zeros(C)], axis=1).astype(np.float64)
    L1, SW, stats = pretrain_snapshots._pairwise_archetype_distances(A, centers, sw_n_projections=8)
    upper_l1 = L1[np.triu_indices(K, k=1)].astype(np.float64)
    upper_sw = SW[np.triu_indices(K, k=1)].astype(np.float64)
    # Stats are computed from a Python list of float64; assert near-equal
    # to a tolerance that absorbs the float32 round-trip in the matrix.
    np.testing.assert_allclose(stats["l1_min"], float(upper_l1.min()), atol=1e-6)
    np.testing.assert_allclose(stats["l1_median"], float(np.median(upper_l1)), atol=1e-6)
    np.testing.assert_allclose(stats["sw_min"], float(upper_sw.min()), atol=1e-6)
    np.testing.assert_allclose(stats["sw_median"], float(np.median(upper_sw)), atol=1e-6)

"""Tests for :mod:`scripts.archetype_stability`.

Three layers of testing:

1. **Pairwise stability primitives.** Hungarian matching recovers
   a known permutation; pair stats are zero on identity; SW matrix
   is symmetric in the right places.
2. **Per-seed fit determinism.** Fits seeded with the same random
   integer reproduce byte-for-byte; different seeds produce
   different archetypes.
3. **End-to-end smoke test.** A small synthetic dataset runs through
   ``run_stability`` and produces all expected output files +
   resume-safe behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import numpy as np
import pandas as pd

# Add repo scripts/ to sys.path so test can import the script.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "legacy_pivot"))
sys.path.insert(0, str(_REPO / "scripts"))

import archetype_stability  # noqa: E402

from shotcloud.grids import CourtGrid  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic data
# ---------------------------------------------------------------------------


def _synthetic_shots(tmp_path: Path) -> Path:
    """Write a small shots CSV with 6 players, 3 distinct shot regions."""
    rng = np.random.default_rng(0)
    rows = []
    # 6 players × 3 cluster regions × ~50 shots/each
    centers_per_region = [
        (0.0, 25.0),  # rim cluster
        (0.0, 150.0),  # mid cluster
        (200.0, 220.0),  # corner-3 cluster (NBA tenths-of-feet convention)
    ]
    for pid in range(1, 7):
        cx, cy = centers_per_region[(pid - 1) % 3]
        for i in range(80):
            rows.append(
                {
                    "GAME_ID": int(rng.integers(20180000, 20200000)),
                    "GAME_DATE": "20230715",
                    "game_date": "2023-07-15",
                    "PERIOD": (i % 4) + 1,
                    "MINUTES_REMAINING": int(rng.integers(0, 12)),
                    "SECONDS_REMAINING": int(rng.integers(0, 60)),
                    "PLAYER_ID": pid,
                    "TEAM_NAME": "TEST",
                    "HTM": "TST",
                    "VTM": "OPP",
                    "LOC_X": float(rng.normal(cx, 15.0)),
                    "LOC_Y": float(rng.normal(cy, 15.0)),
                    "SHOT_ATTEMPTED_FLAG": 1,
                    "SHOT_MADE_FLAG": int(rng.random() < 0.45),
                }
            )
    out_path = tmp_path / "shots.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# 1. Pairwise stability primitives
# ---------------------------------------------------------------------------


def _make_fit(seed: int, A: np.ndarray) -> archetype_stability.SeededFit:
    """Construct a SeededFit from a given (K, C) archetype matrix."""
    K = A.shape[0]
    P = 5
    rho = np.full((P, K), 1.0 / K, dtype=np.float32)
    pids = np.arange(P, dtype=np.int64)
    n_shots = np.full(P, 100, dtype=np.int64)
    return archetype_stability.SeededFit(
        seed=seed,
        archetypes=A.astype(np.float32),
        mixtures=rho,
        player_ids=pids,
        n_shots=n_shots,
        final_loss=0.0,
    )


def _line_centers(C: int) -> np.ndarray:
    return np.stack([np.linspace(-25, 25, C), np.zeros(C)], axis=1).astype(np.float64)


def test_pairwise_stats_zero_on_identity() -> None:
    """A fit compared to itself has zero matched SW and L1."""
    rng = np.random.default_rng(0)
    A = rng.dirichlet(np.ones(50), size=4).astype(np.float32)
    centers = _line_centers(50)
    fit = _make_fit(seed=0, A=A)
    stats = archetype_stability._compute_pair_stats(fit, fit, centers, n_projections=20, sw_seed=0)
    assert stats.sw_mean < 1e-5
    assert stats.sw_max < 1e-5
    assert stats.l1_mean < 1e-5
    assert stats.l1_max < 1e-5
    # On identity, the optimal permutation is the identity itself.
    assert stats.perm_b_to_a == tuple(range(4))


def test_hungarian_recovers_known_permutation() -> None:
    """Permute the rows of A and verify Hungarian matching recovers
    the inverse permutation."""
    rng = np.random.default_rng(0)
    K = 4
    A = rng.dirichlet(np.ones(50), size=K).astype(np.float32)
    perm = np.array([2, 0, 3, 1])  # arbitrary non-identity permutation
    A_perm = A[perm]

    fit_a = _make_fit(seed=0, A=A)
    fit_b = _make_fit(seed=1, A=A_perm)
    centers = _line_centers(50)
    stats = archetype_stability._compute_pair_stats(
        fit_a, fit_b, centers, n_projections=50, sw_seed=0
    )
    # perm_b_to_a[k] should equal the index in A_perm that matches A[k];
    # since A_perm[i] == A[perm[i]], we want perm_b_to_a[k] == argmax_i [perm[i] == k]
    # — i.e. the inverse permutation.
    inverse = np.argsort(perm).tolist()
    assert list(stats.perm_b_to_a) == inverse
    # Matched distances are essentially zero (same surfaces, just reordered).
    assert stats.sw_max < 1e-5
    assert stats.l1_max < 1e-5


def test_pairwise_sw_matrix_diagonal_is_zero() -> None:
    """SW(A_k, A_k) should be zero for every k."""
    rng = np.random.default_rng(0)
    A = rng.dirichlet(np.ones(50), size=4).astype(np.float32)
    centers = _line_centers(50)
    D = archetype_stability._pairwise_sw_matrix(A, A, centers, n_projections=20, seed=0)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-5)
    # Off-diagonal entries are positive (random simplex rows are distinct).
    off_diag = D[~np.eye(4, dtype=bool)]
    assert (off_diag > 0).all()


def test_align_to_reference_recovers_known_permutation(tmp_path: Path) -> None:
    """The aligned gallery preserves the reference and permutes others
    so each row corresponds to the same archetype motif."""
    rng = np.random.default_rng(0)
    K, C = 4, 50
    A = rng.dirichlet(np.ones(C), size=K).astype(np.float32)
    perm = np.array([1, 3, 0, 2])
    A_perm = A[perm]

    fit_a = _make_fit(seed=0, A=A)
    fit_b = _make_fit(seed=1, A=A_perm)
    centers = _line_centers(C)
    aligned = archetype_stability._align_to_reference(
        [fit_a, fit_b],
        reference_idx=0,
        cell_centers=centers,
        n_projections=50,
        sw_seed=0,
    )
    assert len(aligned) == 2
    # Reference is unchanged.
    np.testing.assert_array_equal(aligned[0], A)
    # Other fit is reordered to match the reference.
    np.testing.assert_allclose(aligned[1], A, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Per-seed fit determinism
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_archetypes() -> None:
    """Two fits with the same seed and same Q produce identical archetypes."""
    rng = np.random.default_rng(0)
    Q = rng.dirichlet(np.ones(56 * 64), size=10).astype(np.float32)
    n_shots = np.full(10, 100, dtype=np.int64)
    pids = np.arange(10, dtype=np.int64)
    grid = CourtGrid()
    fit_a = archetype_stability._fit_one_seed(
        Q,
        n_shots,
        pids,
        grid=grid,
        K=4,
        seed=42,
        epsilon=1.0,
        max_iter=3,
        sinkhorn_iter=4,
        lr=0.05,
        batch_size=8,
        device="cpu",
        verbose=False,
    )
    fit_b = archetype_stability._fit_one_seed(
        Q,
        n_shots,
        pids,
        grid=grid,
        K=4,
        seed=42,
        epsilon=1.0,
        max_iter=3,
        sinkhorn_iter=4,
        lr=0.05,
        batch_size=8,
        device="cpu",
        verbose=False,
    )
    np.testing.assert_array_equal(fit_a.archetypes, fit_b.archetypes)


def test_different_seeds_produce_different_archetypes() -> None:
    """Different seeds → different Dirichlet A_init → different archetypes
    after a few optimization steps."""
    rng = np.random.default_rng(0)
    Q = rng.dirichlet(np.ones(56 * 64), size=10).astype(np.float32)
    n_shots = np.full(10, 100, dtype=np.int64)
    pids = np.arange(10, dtype=np.int64)
    grid = CourtGrid()
    fit_a = archetype_stability._fit_one_seed(
        Q,
        n_shots,
        pids,
        grid=grid,
        K=4,
        seed=0,
        epsilon=1.0,
        max_iter=3,
        sinkhorn_iter=4,
        lr=0.05,
        batch_size=8,
        device="cpu",
        verbose=False,
    )
    fit_b = archetype_stability._fit_one_seed(
        Q,
        n_shots,
        pids,
        grid=grid,
        K=4,
        seed=1,
        epsilon=1.0,
        max_iter=3,
        sinkhorn_iter=4,
        lr=0.05,
        batch_size=8,
        device="cpu",
        verbose=False,
    )
    # At least one cell of at least one archetype differs measurably.
    assert (
        (fit_a.archetypes - fit_b.archetypes).abs().max() > 1e-6
        if hasattr(fit_a.archetypes, "abs")
        else float(np.abs(fit_a.archetypes - fit_b.archetypes).max()) > 1e-6
    )


# ---------------------------------------------------------------------------
# 3. End-to-end smoke test
# ---------------------------------------------------------------------------


def test_run_stability_end_to_end(tmp_path: Path) -> None:
    """Full pipeline on synthetic data: fit 3 seeds, compute pairwise
    stability, write all outputs."""
    shots_path = _synthetic_shots(tmp_path)
    output_dir = tmp_path / "stability"
    summary = archetype_stability.run_stability(
        shots_path=shots_path,
        anchor="2024-04-01",
        start_date="2023-01-01",
        K=3,
        seeds=[0, 1, 2],
        min_shots=20,
        # Tiny iteration counts for test speed.
        max_iter=3,
        sinkhorn_iter=4,
        batch_size=8,
        device="cpu",
        output_dir=output_dir,
        sw_n_projections=8,
        verbose=False,
    )
    # Per-seed checkpoints exist.
    assert (output_dir / "seed_0.npz").exists()
    assert (output_dir / "seed_1.npz").exists()
    assert (output_dir / "seed_2.npz").exists()
    # Summary JSON exists and is consistent.
    summary_path = output_dir / "stability_summary.json"
    assert summary_path.exists()
    on_disk = json.loads(summary_path.read_text())
    assert on_disk["K"] == 3
    assert on_disk["seeds"] == [0, 1, 2]
    assert on_disk["n_pairs"] == 3  # 3 choose 2
    # Figures exist.
    assert (output_dir / "stability_heatmap.png").exists()
    assert (output_dir / "gallery_aligned.png").exists()
    # Returned summary matches the on-disk one.
    assert summary["K"] == 3
    assert summary["n_pairs"] == 3


# ---------------------------------------------------------------------------
# 4. Within-fit atom distinctness diagnostic
# ---------------------------------------------------------------------------


def test_within_fit_stats_distinct_atoms_flag_ok() -> None:
    """Far-apart simplex rows on a 1-D line of cell centers produce
    well-separated atoms; flag should be ``ok``."""
    C = 50
    centers = _line_centers(C)
    # Four atoms each concentrated at a different cell — far apart on
    # the line of centers.
    K = 4
    A = np.zeros((K, C), dtype=np.float32)
    for k in range(K):
        # Place the mass at evenly-spaced cell positions.
        idx = int((k + 1) * C / (K + 1))
        A[k, idx] = 1.0
    fit = _make_fit(seed=0, A=A)
    stats = archetype_stability._compute_within_fit_stats(fit, centers, n_projections=50, sw_seed=0)
    # The closest pair is adjacent atoms; spacing is C/(K+1) cells along the
    # line. With C=50 and span 50 ft, that's ~10 ft. SW between two
    # delta-like masses on the line is roughly E[|cos θ|] * Δx ≈ 0.64 * 10 = 6.4 ft.
    assert stats.pair_sw_min > archetype_stability.WITHIN_FIT_EFFECTIVE_RANK_THRESHOLD
    assert stats.flag == "ok"


def test_within_fit_stats_duplicate_atoms_flag_duplicate_warning() -> None:
    """Identical (or near-identical) atoms produce a duplicate flag."""
    C = 50
    centers = _line_centers(C)
    K = 4
    rng = np.random.default_rng(0)
    base = rng.dirichlet(np.ones(C)).astype(np.float32)
    # All K rows are the same surface — pure duplicates.
    A = np.tile(base[None, :], (K, 1))
    fit = _make_fit(seed=0, A=A)
    stats = archetype_stability._compute_within_fit_stats(fit, centers, n_projections=50, sw_seed=0)
    assert stats.pair_sw_min < archetype_stability.WITHIN_FIT_DUPLICATE_THRESHOLD
    assert stats.flag == "duplicate_warning"


def test_within_fit_stats_in_between_flag_effective_rank_warning() -> None:
    """Atoms near each other but not duplicates → effective-rank warning."""
    C = 50
    centers = _line_centers(C)
    K = 3
    A = np.zeros((K, C), dtype=np.float32)
    # Three atoms at adjacent cells on a 50-cell line. Centers span 50 ft,
    # so adjacent cells are ~1 ft apart. SW between adjacent delta atoms
    # is ~0.64 ft — which sits in the (0.3, 1.5) effective-rank band.
    A[0, 24] = 1.0
    A[1, 25] = 1.0
    A[2, 26] = 1.0
    fit = _make_fit(seed=0, A=A)
    stats = archetype_stability._compute_within_fit_stats(fit, centers, n_projections=50, sw_seed=0)
    assert (
        archetype_stability.WITHIN_FIT_DUPLICATE_THRESHOLD
        <= stats.pair_sw_min
        < archetype_stability.WITHIN_FIT_EFFECTIVE_RANK_THRESHOLD
    )
    assert stats.flag == "effective_rank_warning"


def test_within_fit_stats_in_summary_json(tmp_path: Path) -> None:
    """End-to-end: the new within-fit fields land in stability_summary.json."""
    shots_path = _synthetic_shots(tmp_path)
    output_dir = tmp_path / "stability"
    archetype_stability.run_stability(
        shots_path=shots_path,
        anchor="2024-04-01",
        start_date="2023-01-01",
        K=3,
        seeds=[0, 1, 2],
        min_shots=20,
        max_iter=3,
        sinkhorn_iter=4,
        batch_size=8,
        device="cpu",
        output_dir=output_dir,
        sw_n_projections=8,
        verbose=False,
    )
    summary = json.loads((output_dir / "stability_summary.json").read_text())
    assert "within_fit" in summary
    wf = summary["within_fit"]
    # Per-seed dicts have an entry for each seed (string keys to be JSON-friendly).
    for key in (
        "pair_sw_min_by_seed",
        "pair_sw_median_by_seed",
        "pair_sw_max_by_seed",
        "pair_l1_min_by_seed",
        "pair_l1_median_by_seed",
        "effective_rank_flag_by_seed",
    ):
        assert key in wf
        assert set(wf[key].keys()) == {"0", "1", "2"}
    # Threshold values are recorded.
    assert wf["thresholds"]["duplicate_sw_ft"] == archetype_stability.WITHIN_FIT_DUPLICATE_THRESHOLD
    assert (
        wf["thresholds"]["effective_rank_sw_ft"]
        == archetype_stability.WITHIN_FIT_EFFECTIVE_RANK_THRESHOLD
    )
    # Each flag is one of the three valid values.
    for flag in wf["effective_rank_flag_by_seed"].values():
        assert flag in {"ok", "effective_rank_warning", "duplicate_warning"}
    # Aggregate counts agree with the per-seed flags.
    flags = list(wf["effective_rank_flag_by_seed"].values())
    assert wf["n_seeds_with_duplicate_warning"] == flags.count("duplicate_warning")
    assert wf["n_seeds_with_effective_rank_warning"] == flags.count("effective_rank_warning")


# ---------------------------------------------------------------------------
# 5. Per-atom usage diagnostic (dead / dominant flags)
# ---------------------------------------------------------------------------


def _make_fit_with_rho(seed: int, rho: np.ndarray, K: int) -> archetype_stability.SeededFit:
    """Construct a SeededFit with a custom mixture matrix `rho`."""
    P = rho.shape[0]
    A = np.full((K, 50), 1.0 / 50, dtype=np.float32)  # archetypes irrelevant for this test
    return archetype_stability.SeededFit(
        seed=seed,
        archetypes=A,
        mixtures=rho.astype(np.float32),
        player_ids=np.arange(P, dtype=np.int64),
        n_shots=np.full(P, 100, dtype=np.int64),
        final_loss=0.0,
    )


def test_usage_stats_no_flags_on_uniform_mixture() -> None:
    """Uniform ρ_p across all atoms gives usage_k = 1/K — no dead, no dominant."""
    K = 4
    P = 20
    rho = np.full((P, K), 1.0 / K, dtype=np.float32)
    fit = _make_fit_with_rho(seed=0, rho=rho, K=K)
    stats = archetype_stability._compute_usage_stats(fit)
    np.testing.assert_allclose(stats.usage, [1.0 / K] * K, atol=1e-6)
    assert stats.dead_atoms == ()
    assert stats.dominant_atoms == ()


def test_usage_stats_flags_dead_atom() -> None:
    """An atom that no player uses → flagged as dead."""
    K = 4
    P = 20
    # Atom 0 used heavily; atoms 1-3 each get a small share. Atom 3
    # is set below the dead threshold (0.01).
    rho = np.full((P, K), 0.0, dtype=np.float32)
    rho[:, 0] = 0.50
    rho[:, 1] = 0.30
    rho[:, 2] = 0.195
    rho[:, 3] = 0.005  # dead: < 0.01
    fit = _make_fit_with_rho(seed=0, rho=rho, K=K)
    stats = archetype_stability._compute_usage_stats(fit)
    assert 3 in stats.dead_atoms
    # Atom 0 is heavy but at exactly 0.50 — NOT > 0.50, so not dominant.
    assert 0 not in stats.dominant_atoms


def test_usage_stats_flags_dominant_atom() -> None:
    """An atom whose mean usage exceeds 0.50 → flagged as dominant."""
    K = 4
    P = 20
    # Atom 0 strongly dominates.
    rho = np.full((P, K), 0.05, dtype=np.float32)
    rho[:, 0] = 0.85  # dominant: > 0.50
    fit = _make_fit_with_rho(seed=0, rho=rho, K=K)
    stats = archetype_stability._compute_usage_stats(fit)
    assert 0 in stats.dominant_atoms


# ---------------------------------------------------------------------------
# 6. Paper-grade certification
# ---------------------------------------------------------------------------


def _make_within(seed: int, pair_sw_min: float) -> archetype_stability.WithinFitStats:
    return archetype_stability.WithinFitStats(
        seed=seed,
        pair_sw_min=pair_sw_min,
        pair_sw_median=pair_sw_min + 1.0,
        pair_sw_max=pair_sw_min + 2.0,
        pair_l1_min=0.0,
        pair_l1_median=0.0,
        pair_l1_max=0.0,
        flag="ok" if pair_sw_min >= 1.5 else "effective_rank_warning",
    )


def _make_usage(
    seed: int,
    K: int,
    *,
    dead: list[int] | None = None,
    dominant: list[int] | None = None,
) -> archetype_stability.UsageStats:
    dead = list(dead) if dead else []
    dominant = list(dominant) if dominant else []
    usage_vals = [1.0 / K] * K
    for k in dead:
        usage_vals[k] = 0.001
    for k in dominant:
        usage_vals[k] = 0.85
    return archetype_stability.UsageStats(
        seed=seed,
        usage=tuple(usage_vals),
        dead_atoms=tuple(dead),
        dominant_atoms=tuple(dominant),
    )


def test_certification_passes_when_all_four_rules_met() -> None:
    """Cross-seed mean SW < 1.5, all within-fit pair_sw > 1.5, no dead, no dominant."""
    within = [_make_within(s, pair_sw_min=2.0) for s in range(3)]
    usage = [_make_usage(s, K=4) for s in range(3)]
    cert = archetype_stability._compute_certification(0.5, within, usage)
    assert cert["certified"] is True
    assert cert["cross_seed_stable"] is True
    assert cert["within_fit_distinct"] is True
    assert cert["no_dead_atoms"] is True
    assert cert["no_dominant_atoms"] is True


def test_certification_fails_on_each_individual_rule() -> None:
    base_within = [_make_within(s, pair_sw_min=2.0) for s in range(3)]
    base_usage = [_make_usage(s, K=4) for s in range(3)]

    # Rule 1: cross-seed unstable.
    cert = archetype_stability._compute_certification(2.0, base_within, base_usage)
    assert cert["certified"] is False
    assert cert["cross_seed_stable"] is False

    # Rule 2: at least one seed has within-fit pair_sw < 1.5.
    bad_within = [_make_within(0, 0.5), *base_within[1:]]
    cert = archetype_stability._compute_certification(0.5, bad_within, base_usage)
    assert cert["certified"] is False
    assert cert["within_fit_distinct"] is False

    # Rule 3: at least one seed has a dead atom.
    bad_usage = [_make_usage(0, K=4, dead=[3]), *base_usage[1:]]
    cert = archetype_stability._compute_certification(0.5, base_within, bad_usage)
    assert cert["certified"] is False
    assert cert["no_dead_atoms"] is False

    # Rule 4: at least one seed has a dominant atom.
    bad_usage = [_make_usage(0, K=4, dominant=[0]), *base_usage[1:]]
    cert = archetype_stability._compute_certification(0.5, base_within, bad_usage)
    assert cert["certified"] is False
    assert cert["no_dominant_atoms"] is False


def test_certification_in_summary_json(tmp_path: Path) -> None:
    """End-to-end: the certification block lands in stability_summary.json."""
    shots_path = _synthetic_shots(tmp_path)
    output_dir = tmp_path / "stability"
    archetype_stability.run_stability(
        shots_path=shots_path,
        anchor="2024-04-01",
        start_date="2023-01-01",
        K=3,
        seeds=[0, 1, 2],
        min_shots=20,
        max_iter=3,
        sinkhorn_iter=4,
        batch_size=8,
        device="cpu",
        output_dir=output_dir,
        sw_n_projections=8,
        verbose=False,
    )
    summary = json.loads((output_dir / "stability_summary.json").read_text())

    # Top-level certification block exists with all 5 boolean fields.
    assert "certification" in summary
    cert = summary["certification"]
    for key in (
        "certified",
        "cross_seed_stable",
        "within_fit_distinct",
        "no_dead_atoms",
        "no_dominant_atoms",
    ):
        assert key in cert
        assert isinstance(cert[key], bool)
    # Threshold values are recorded.
    for key in (
        "cross_seed_sw_ft",
        "within_fit_sw_ft",
        "dead_atom_usage",
        "dominant_atom_usage",
    ):
        assert key in cert["thresholds"]

    # Usage block exists with per-seed entries.
    assert "usage" in summary
    u = summary["usage"]
    for key in (
        "usage_by_seed",
        "dead_atoms_by_seed",
        "dominant_atoms_by_seed",
        "n_seeds_with_dead_atoms",
        "n_seeds_with_dominant_atoms",
    ):
        assert key in u
    assert set(u["usage_by_seed"].keys()) == {"0", "1", "2"}


def test_usage_per_seed_figure_renders(tmp_path: Path) -> None:
    usage = [
        archetype_stability.UsageStats(
            seed=s, usage=tuple([1.0 / 4] * 4), dead_atoms=(), dominant_atoms=()
        )
        for s in range(3)
    ]
    out_path = tmp_path / "usage.png"
    archetype_stability.plot_usage_per_seed(
        usage, [0, 1, 2], out_path=out_path, anchor_label="2024-01-01"
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 1000


def test_run_stability_resume_skips_existing_seeds(tmp_path: Path) -> None:
    """With --resume, seeds whose .npz already exists are loaded, not refit."""
    shots_path = _synthetic_shots(tmp_path)
    output_dir = tmp_path / "stability"

    # First pass: fit 2 seeds.
    archetype_stability.run_stability(
        shots_path=shots_path,
        anchor="2024-04-01",
        start_date="2023-01-01",
        K=3,
        seeds=[0, 1],
        min_shots=20,
        max_iter=3,
        sinkhorn_iter=4,
        batch_size=8,
        device="cpu",
        output_dir=output_dir,
        sw_n_projections=8,
        verbose=False,
    )
    seed0_mtime = (output_dir / "seed_0.npz").stat().st_mtime_ns
    seed1_mtime = (output_dir / "seed_1.npz").stat().st_mtime_ns

    # Second pass with --resume and one new seed: existing seeds are not re-fit.
    archetype_stability.run_stability(
        shots_path=shots_path,
        anchor="2024-04-01",
        start_date="2023-01-01",
        K=3,
        seeds=[0, 1, 2],
        min_shots=20,
        # Different fitter args to confirm resumed seeds aren't re-fit:
        max_iter=99,
        sinkhorn_iter=99,
        batch_size=8,
        device="cpu",
        output_dir=output_dir,
        sw_n_projections=8,
        resume=True,
        verbose=False,
    )
    assert (output_dir / "seed_0.npz").stat().st_mtime_ns == seed0_mtime
    assert (output_dir / "seed_1.npz").stat().st_mtime_ns == seed1_mtime
    assert (output_dir / "seed_2.npz").exists()

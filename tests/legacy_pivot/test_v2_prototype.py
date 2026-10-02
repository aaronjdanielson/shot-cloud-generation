"""Smoke and verdict tests for ``scripts/legacy_pivot/v2_prototype.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

# Add repo scripts/ to sys.path.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import v2_prototype  # noqa: E402

from shotcloud.grids import CourtGrid  # noqa: E402
from shotcloud.legacy_pivot.wasserstein_fit import (  # noqa: E402
    build_separable_log_kernel,
    fit_archetypes_v1,
    wasserstein_barycenter_separable,
)


def _gaussian_atom(ny: int, nx: int, cy: float, cx: float, sigma: float = 1.5) -> np.ndarray:
    iy = np.arange(ny, dtype=np.float64)
    ix = np.arange(nx, dtype=np.float64)
    Y, X = np.meshgrid(iy, ix, indexing="ij")
    p = np.exp(-((Y - cy) ** 2 + (X - cx) ** 2) / (2 * sigma**2))
    return (p / p.sum()).flatten().astype(np.float32)


def _synthetic_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Tiny shots CSV + a V1 checkpoint .npz for the prototype to load.

    Builds shots whose KDEs are well-aligned with synthetic
    archetypes, fits V1 to get the checkpoint, then writes a shots
    CSV that overlaps the same anchor's causal window.
    """
    grid = CourtGrid()
    ny, nx = grid.ny, grid.nx

    # Synthetic shots: 12 players × 3 distinct cluster regions.
    rng = np.random.default_rng(0)
    centers_per_region = [
        (0.0, 25.0),  # rim cluster
        (0.0, 150.0),  # mid cluster
        (200.0, 220.0),  # corner-3 cluster
    ]
    rows = []
    for pid in range(1, 13):
        cx, cy = centers_per_region[(pid - 1) % 3]
        for i in range(120):
            rows.append(
                {
                    "GAME_ID": int(rng.integers(20180000, 20200000)),
                    "GAME_DATE": "20230701",
                    "game_date": "2023-07-01",
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
    shots_path = tmp_path / "shots.csv"
    pd.DataFrame(rows).to_csv(shots_path, index=False)

    # Build a synthetic V1 checkpoint. We don't need to actually run V1 — we
    # construct an .npz with the same schema as a stability seed_*.npz.
    K = 3
    A = np.stack(
        [
            _gaussian_atom(ny, nx, 30, 32, sigma=3),
            _gaussian_atom(ny, nx, 40, 32, sigma=3),
            _gaussian_atom(ny, nx, 22, 50, sigma=3),
        ]
    )
    P = 12
    rho = rng.dirichlet(np.ones(K), size=P).astype(np.float32)
    pids = np.arange(1, P + 1, dtype=np.int64)
    n_shots = np.full(P, 120, dtype=np.int64)

    checkpoint_path = tmp_path / "v1_seed_0.npz"
    np.savez_compressed(
        checkpoint_path,
        seed=np.array(0, dtype=np.int64),
        archetypes=A,
        mixtures=rho,
        player_ids=pids,
        n_shots=n_shots,
        final_loss=np.array(1.0, dtype=np.float64),
    )
    return shots_path, checkpoint_path


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def test_verdict_cell_1_strong_case_for_v2() -> None:
    """delta_h substantially negative + V2 loss comparable/lower → cell 1.

    V2 ρ is sharper and reconstruction stays comparable.
    """
    h_v1 = np.full(50, 0.95, dtype=np.float64)
    h_v2_seeds = [np.full(50, 0.30, dtype=np.float64)]
    verdict = v2_prototype._make_verdict(h_v1, h_v2_seeds, v1_loss=2.0, v2_loss=2.0)
    assert verdict["cell"] == "1_strong_case_for_v2"
    assert verdict["loss_comparable"] is True
    assert verdict["delta_h_norm"] < -0.10
    assert "Strong case for full V2" in verdict["conclusion"]


def test_verdict_cell_2_atoms_not_barycentric_optimal() -> None:
    """delta_h substantially negative BUT V2 loss much higher → cell 2.

    V2 sharpens ρ but reconstruction is much worse, indicating that the
    V1 atoms are not barycentric-optimal.
    """
    h_v1 = np.full(50, 0.95, dtype=np.float64)
    h_v2_seeds = [np.full(50, 0.30, dtype=np.float64)]
    verdict = v2_prototype._make_verdict(h_v1, h_v2_seeds, v1_loss=2.0, v2_loss=4.0)
    assert verdict["cell"] == "2_atoms_not_barycentric_optimal"
    assert verdict["loss_comparable"] is False
    assert "JOINT" in verdict["conclusion"]


def test_verdict_cell_3_diffuse_rho_intrinsic() -> None:
    """delta_h ≈ 0 + V2 loss comparable → cell 3.

    Diffuse ρ is intrinsic to the data geometry rather than an artifact
    of the V1 surrogate.
    """
    h_v1 = np.full(50, 0.50, dtype=np.float64)
    h_v2_seeds = [np.full(50, 0.51, dtype=np.float64)]
    verdict = v2_prototype._make_verdict(h_v1, h_v2_seeds, v1_loss=2.0, v2_loss=2.1)
    assert verdict["cell"] == "3_diffuse_rho_intrinsic"
    assert verdict["loss_comparable"] is True
    assert "intrinsic" in verdict["conclusion"]


def test_verdict_cell_4_low_dim_manifold() -> None:
    """delta_h positive → cell 4 (low-dimensional manifold).

    V2 entropy higher than V1's selects this cell regardless of the loss
    axis: the entropy direction alone rules out the V1 surrogate as the
    source of diffuse ρ.
    """
    h_v1 = np.full(50, 0.30, dtype=np.float64)
    h_v2_seeds = [np.full(50, 0.50, dtype=np.float64)]
    verdict = v2_prototype._make_verdict(h_v1, h_v2_seeds, v1_loss=2.0, v2_loss=2.0)
    assert verdict["cell"] == "4_low_dim_manifold"
    assert verdict["delta_h_norm"] > 0.05
    assert "manifold" in verdict["conclusion"]
    # The cell-4 conclusion carries no "Rethink" recommendation.
    assert "Rethink" not in verdict["conclusion"]


def test_verdict_includes_both_axes_in_output() -> None:
    """Verdict dict carries both delta_h_norm and delta_loss + thresholds
    so downstream consumers can re-interpret with different bands.
    """
    h_v1 = np.full(50, 0.40, dtype=np.float64)
    h_v2_seeds = [np.full(50, 0.30, dtype=np.float64)]
    verdict = v2_prototype._make_verdict(h_v1, h_v2_seeds, v1_loss=2.0, v2_loss=2.5)
    for key in (
        "v1_mean_h_norm",
        "v2_mean_h_norm",
        "delta_h_norm",
        "v1_reconstruction_loss",
        "v2_reconstruction_loss",
        "delta_loss",
        "delta_loss_rel",
        "loss_comparable",
        "cell",
        "conclusion",
        "thresholds",
    ):
        assert key in verdict, f"missing {key} in verdict output"
    for key in ("delta_h_sharper", "delta_h_equal", "loss_comparable_rel"):
        assert key in verdict["thresholds"]


# ---------------------------------------------------------------------------
# Atom stats
# ---------------------------------------------------------------------------


def test_atom_stats_dead_and_dominant() -> None:
    rho = np.array(
        [
            [0.85, 0.10, 0.05, 0.000],  # atom 3 below dead threshold (0.01)
            [0.85, 0.10, 0.05, 0.000],
        ],
        dtype=np.float32,
    )
    n_dead, n_dominant = v2_prototype._atom_stats(rho)
    assert n_dead == 1  # atom 3
    assert n_dominant == 1  # atom 0 (mean = 0.85 > 0.50)


def test_atom_stats_no_flags_on_uniform() -> None:
    rho = np.full((10, 4), 0.25, dtype=np.float32)
    n_dead, n_dominant = v2_prototype._atom_stats(rho)
    assert n_dead == 0
    assert n_dominant == 0


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_run_v2_prototype_end_to_end(tmp_path: Path) -> None:
    """The full pipeline: load checkpoint, build Q, fit V2, write outputs."""
    shots_path, ckpt_path = _synthetic_dataset(tmp_path)
    output_dir = tmp_path / "v2_out"
    summary = v2_prototype.run_v2_prototype(
        shots_path=shots_path,
        v1_checkpoint=ckpt_path,
        anchor="2024-04-01",
        start_date="2023-01-01",
        output_dir=output_dir,
        seeds=[0, 1],
        max_players=12,
        min_shots=20,
        max_iter=5,
        sinkhorn_iter=4,
        barycenter_iter=4,
        batch_size=12,
        device="cpu",
        verbose=False,
    )

    # Outputs land on disk.
    assert (output_dir / "v2_summary.json").exists()
    assert (output_dir / "v2_rho.npz").exists()
    assert (output_dir / "v2_vs_v1_entropy.png").exists()

    # Summary has expected structure.
    on_disk = json.loads((output_dir / "v2_summary.json").read_text())
    assert on_disk["K"] == 3
    assert on_disk["seeds"] == [0, 1]
    assert "v1" in on_disk
    assert "v2_by_seed" in on_disk
    assert len(on_disk["v2_by_seed"]) == 2
    assert "verdict" in on_disk
    assert "conclusion" in on_disk["verdict"]
    # V1 reconstruction loss on the same subset is recorded for
    # comparable comparison with V2's loss.
    assert "reconstruction_loss_subset" in on_disk["v1"]
    assert isinstance(on_disk["v1"]["reconstruction_loss_subset"], (int, float))
    # Verdict carries the joint axes (delta_h_norm + delta_loss) and a cell.
    v = on_disk["verdict"]
    assert v["cell"] in {
        "1_strong_case_for_v2",
        "2_atoms_not_barycentric_optimal",
        "3_diffuse_rho_intrinsic",
        "4_low_dim_manifold",
        "mixed",
    }
    assert "v1_reconstruction_loss" in v
    assert "v2_reconstruction_loss" in v
    assert "delta_loss" in v
    assert "loss_comparable" in v

    # In-memory summary matches.
    assert summary["K"] == 3


def test_v2_recovery_better_than_v1_on_real_v1_archetypes(tmp_path: Path) -> None:
    """Run V1 to fit archetypes, then V2 to refit ρ on the SAME data with
    those archetypes frozen. V2 should recover ρ closer to the data-generating
    truth (which we know because we generated Q via barycenters)."""
    grid = CourtGrid()
    ny, nx = grid.ny, grid.nx
    K = 3

    # Build atoms + true ρ + Q via real entropic barycenters on the canonical grid.
    A = np.stack(
        [
            _gaussian_atom(ny, nx, 8, 12, sigma=3),
            _gaussian_atom(ny, nx, 30, 32, sigma=3),
            _gaussian_atom(ny, nx, 22, 50, sigma=3),
        ]
    )
    P = 16
    rng = np.random.default_rng(0)
    true_rhos = rng.dirichlet(0.3 * np.ones(K), size=P).astype(np.float32)

    import torch

    log_Ky, log_Kx = build_separable_log_kernel(ny, nx, epsilon=1.0)
    log_A = torch.log(torch.from_numpy(A).clamp_min(1e-12)).view(K, ny, nx)
    with torch.no_grad():
        log_bary = wasserstein_barycenter_separable(
            torch.from_numpy(true_rhos),
            log_A,
            log_Ky,
            log_Kx,
            n_iter=30,
            last_iter_with_grad=False,
        )
        Q = torch.exp(log_bary).view(P, ny * nx).numpy().astype(np.float32)
    Q = Q / Q.sum(axis=1, keepdims=True)

    # V1: fit ρ holding A fixed (warm-start at truth).
    v1 = fit_archetypes_v1(
        Q,
        grid_ny=ny,
        grid_nx=nx,
        K=K,
        cell_size_y=(grid.ylim[1] - grid.ylim[0]) / ny,
        cell_size_x=(grid.xlim[1] - grid.xlim[0]) / nx,
        epsilon=1.0,
        max_iter=100,
        sinkhorn_iter=15,
        lr=0.1,
        A_init=A,
        batch_size=P,
        seed=0,
        device="cpu",
        verbose=False,
    )
    v1_err = float(np.abs(v1.mixtures - true_rhos).mean(axis=1).mean())

    # V2: fit ρ via fit_v2_rho_given_A.
    from shotcloud.legacy_pivot.wasserstein_fit import fit_v2_rho_given_A

    v2 = fit_v2_rho_given_A(
        Q,
        A,
        grid_ny=ny,
        grid_nx=nx,
        cell_size_y=(grid.ylim[1] - grid.ylim[0]) / ny,
        cell_size_x=(grid.xlim[1] - grid.xlim[0]) / nx,
        epsilon=1.0,
        max_iter=100,
        sinkhorn_iter=15,
        barycenter_iter=20,
        lr=0.1,
        batch_size=P,
        seed=0,
        device="cpu",
        verbose=False,
    )
    v2_err = float(np.abs(v2.mixtures - true_rhos).mean(axis=1).mean())

    assert v2_err < v1_err, (
        f"V2 must recover ρ_true better than V1: V1={v1_err:.4f}, V2={v2_err:.4f}"
    )

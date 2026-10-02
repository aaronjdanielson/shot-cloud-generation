"""Classical-baseline ablation runner — first paper-shaped output.

Computes Tier-1 metrics ([plan.md §4](../../docs/plan.md)) for four classical
models against a held-out test set:

1. **League KDE** — same density for every player.
2. **Player KDE (raw)** — per-player KDE, no shrinkage.
3. **Hierarchical KDE** — per-player KDE shrunk toward the position prior.
4. **KDE product** — geometric product of player × position × league.

These four are the **classical part of the empirical ladder** from
[plan.md §3](../../docs/plan.md). Once training is wired up, we add
"KDE product + low-rank tilt (zero-init)" and "KDE product + low-rank
tilt (trained)" as additional rows.

The runner returns an :class:`AblationResult` with two DataFrames:

- ``summary`` — one row per model, aggregated metrics.
- ``per_player`` — one row per ``(model, player)``, raw metrics.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from shotcloud.evaluation.wasserstein import sliced_wasserstein
from shotcloud.grids import CourtGrid
from shotcloud.kde import HierarchicalKDE
from shotcloud.legacy import KDEProduct
from shotcloud.legacy_pivot.checkpoint import DecoderCheckpoint
from shotcloud.legacy_pivot.eval_metrics import zone_distribution_5, zone_kl_divergence
from shotcloud.legacy_pivot.eval_retrieval import top_k_retrieval

DensityFn = Callable[[object], NDArray[np.float64]]


@dataclass(frozen=True)
class _TestGroup:
    """Per-player test data after preprocessing."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    cells: NDArray[np.int64]
    shots: NDArray[np.float64]


@dataclass
class AblationResult:
    """Output of :func:`run_ablation`."""

    summary: pd.DataFrame
    per_player: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"AblationResult(\n"
            f"  summary: {len(self.summary)} models\n"
            f"  per_player: {len(self.per_player)} (model, player) rows\n"
            f")"
        )


def _build_density_fns(
    kde: HierarchicalKDE,
    weights: Mapping[str, float],
) -> dict[str, DensityFn]:
    """Return ``{model_name: player_id → (ny, nx) density}`` for the 4 baselines."""
    league = kde.league_density()
    product = KDEProduct(hierarchical_kde=kde, weights=dict(weights))

    return {
        "League KDE": lambda _: league,
        "Player KDE (raw)": lambda pid: kde.player_density(pid, hierarchical=False),
        "Hierarchical KDE": lambda pid: kde.player_density(pid, hierarchical=True),
        "KDE product": lambda pid: product.density(pid),
    }


def _sample_cells_from_density(
    density: NDArray[np.float64],
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Vectorized categorical sampling from a flattened density."""
    flat = density.ravel()
    cumsum = np.cumsum(flat)
    u = rng.uniform(0.0, 1.0, size=n)
    cells = np.searchsorted(cumsum, u, side="left")
    return np.clip(cells, 0, flat.size - 1).astype(np.int64)


def _sample_shots_from_density(
    density: NDArray[np.float64],
    n: int,
    grid: CourtGrid,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample n (x, y) shots from a (ny, nx) density grid."""
    cells = _sample_cells_from_density(density, n, rng)
    x, y = grid.dequantize(cells, rng)
    return np.stack([x, y], axis=1)


def run_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    grid: CourtGrid | None = None,
    weights_kde_product: Mapping[str, float] | None = None,
    bandwidth: float = 1.5,
    kappa: float = 500.0,
    recency_half_life_days: float | None = None,
    R_samples: int = 100,
    sw_n_projections: int = 100,
    retrieval_n_projections: int = 100,
    position_map: Mapping[object, str] | None = None,
    date_col: str = "date",
    min_train_shots: int | None = None,
    max_train_shots: int | None = None,
    extra_models: Mapping[str, DensityFn] | None = None,
    seed: int = 42,
    progress: bool = False,
) -> AblationResult:
    """Run the v1 classical-baseline ablation table.

    Parameters
    ----------
    train_df, test_df : DataFrame
        Must contain ``x``, ``y``, ``player_id`` columns. Train data fits
        the KDEs; test data scores them.
    grid : CourtGrid, optional
        Defaults to ``CourtGrid()`` (canonical 64×56 half-court).
    weights_kde_product : Mapping, optional
        Weights for the KDE product. Defaults to
        ``{"a_p": 1.0, "a_g": 0.3, "a_0": 0.2}``.
    bandwidth, kappa : KDE hyperparameters.
    recency_half_life_days : float or None, default None
        If set and ``train_df`` has a ``date_col`` column, fit the KDE
        with exponential recency weighting (shot_flow's winning model
        uses ``365``). ``None`` disables recency.
    R_samples : int, default 100
        Games to sample per test player per model — each game has
        ``mean(test shots / player)`` shots, so the generated cloud has
        ``R_samples × ~20`` shots.
    sw_n_projections, retrieval_n_projections : int, default 100
        Sliced-Wasserstein projection counts for per-player metric and
        retrieval distance respectively.
    position_map : Mapping, optional
        ``player_id → position_label``. If ``None``, every player is
        assigned to a single "ALL" group, making position-level shrinkage
        degenerate. Use :func:`shotcloud.data.derive_positions_from_ra_rate`
        for the data-driven option that mirrors shot_flow's winning setup.
    date_col : str, default "date"
        Name of the date column for recency weighting. Ignored if
        ``recency_half_life_days`` is None.
    min_train_shots, max_train_shots : int, optional
        Filter test players by their **training-set** shot count, so the
        ablation can be sliced into sparse vs. dense regimes. The
        position-group prior is expected to help on sparse players
        (low ``max_train_shots``) and matter less on dense ones
        (high ``min_train_shots``). Both are inclusive. ``None`` means
        no bound on that side.
    extra_models : Mapping[str, callable], optional
        Additional model rows beyond the four classical baselines.
        Each entry maps a display name (e.g.,
        ``"KDE product + tilt (trained)"``) to a callable
        ``player_id → (ny, nx) density grid``. The function is invoked
        once per test player; the same Tier-1 metrics are computed.
        Useful for plugging in trained checkpoints — see
        :func:`shotcloud.training.load_decoder_checkpoint`.
    seed : int, default 42
        RNG seed for sampling and SW projections.
    progress : bool, default False
        Print per-stage timing to stdout — useful for long real-data
        runs where the pairwise SW retrieval (O(n_players²) per model)
        is the dominant cost.

    Returns
    -------
    AblationResult
    """
    if weights_kde_product is None:
        weights_kde_product = {"a_p": 1.0, "a_g": 0.3, "a_0": 0.2}
    if grid is None:
        grid = CourtGrid()

    for required in ("x", "y", "player_id"):
        for name, df in (("train_df", train_df), ("test_df", test_df)):
            if required not in df.columns:
                raise KeyError(f"{name} must contain column {required!r}")

    # Restrict to players present in both train and test.
    train_players = set(train_df["player_id"].unique())
    test_players = set(test_df["player_id"].unique())
    common = sorted(train_players & test_players)
    if not common:
        raise ValueError("no overlap in player_id between train_df and test_df")

    # Optional sparse/dense filter on training-set shot count.
    if min_train_shots is not None or max_train_shots is not None:
        train_counts = train_df[train_df["player_id"].isin(common)]["player_id"].value_counts()
        keep: list[object] = []
        for pid in common:
            n = int(train_counts.get(pid, 0))
            if min_train_shots is not None and n < min_train_shots:
                continue
            if max_train_shots is not None and n > max_train_shots:
                continue
            keep.append(pid)
        if not keep:
            raise ValueError(
                "no players satisfy the train-shot filter "
                f"(min={min_train_shots}, max={max_train_shots})"
            )
        common = keep

    train_df = train_df[train_df["player_id"].isin(common)].copy()
    test_df = test_df[test_df["player_id"].isin(common)].copy()

    # Build (or default) the position map.
    if position_map is None:
        position_map = {pid: "ALL" for pid in common}
    pos_arr = np.array([position_map.get(pid, "ALL") for pid in train_df["player_id"]])

    # Fit the hierarchical KDE on train.
    use_recency = recency_half_life_days is not None and date_col in train_df.columns
    kde = HierarchicalKDE(
        grid=grid,
        bandwidth=bandwidth,
        kappa=kappa,
        recency_half_life_days=recency_half_life_days if use_recency else None,
    )
    if progress:
        print(f"[ablation] fitting KDE on {len(train_df):,} train shots ({len(common)} players)...")
        _t0 = time.perf_counter()
    kde.fit(
        x=train_df["x"].to_numpy(dtype=np.float64),
        y=train_df["y"].to_numpy(dtype=np.float64),
        player_id=train_df["player_id"].to_numpy(),
        position=pos_arr,
        date=train_df[date_col].to_numpy() if use_recency else None,
    )
    if progress:
        print(f"[ablation]   KDE fit in {time.perf_counter() - _t0:.1f}s")

    density_fns = _build_density_fns(kde, weights_kde_product)
    if extra_models:
        for name, fn in extra_models.items():
            if name in density_fns:
                raise ValueError(f"extra_models name {name!r} collides with a built-in baseline")
            density_fns[name] = fn

    # Per-test-player real shots (and cell indices).
    test_groups: dict[object, _TestGroup] = {}
    for pid in common:
        player_test = test_df[test_df["player_id"] == pid]
        if len(player_test) == 0:
            continue
        x = player_test["x"].to_numpy(dtype=np.float64)
        y = player_test["y"].to_numpy(dtype=np.float64)
        cells = grid.coord_to_cell(x, y)
        valid = cells >= 0
        if not valid.any():
            continue
        test_groups[pid] = _TestGroup(
            x=x[valid],
            y=y[valid],
            cells=cells[valid].astype(np.int64),
            shots=np.stack([x[valid], y[valid]], axis=1),
        )
    if not test_groups:
        raise ValueError("no test players have valid in-court shots")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    real_clouds: dict[object, NDArray[np.float64]] = {
        pid: g.shots for pid, g in test_groups.items()
    }
    avg_test_K = int(np.mean([len(g.cells) for g in test_groups.values()]))
    n_per_player = max(R_samples * max(avg_test_K, 1), 100)

    n_models = len(density_fns)
    n_test = len(test_groups)
    if progress:
        print(
            f"[ablation] {n_test} test players x {n_models} models, "
            f"{n_per_player:,} sampled shots/player"
        )

    for model_idx, (model_name, density_fn) in enumerate(density_fns.items(), start=1):
        gen_clouds: dict[object, NDArray[np.float64]] = {}
        if progress:
            _t_model = time.perf_counter()
            print(f"[ablation]   model {model_idx}/{n_models}: {model_name}")
        for pid, group in test_groups.items():
            density = density_fn(pid)
            log_density = np.log(np.maximum(density.ravel(), 1e-300))

            # NLL on the test cells.
            nll = -float(log_density[group.cells].mean())

            # Sample a generated cloud and compute SW + Zone KL.
            gen_shots = _sample_shots_from_density(density, n_per_player, grid, rng)
            sw = sliced_wasserstein(
                group.shots, gen_shots, n_projections=sw_n_projections, seed=seed
            )
            real_zones = zone_distribution_5(group.x, group.y)
            gen_zones = zone_distribution_5(gen_shots[:, 0], gen_shots[:, 1])
            zkl = zone_kl_divergence(gen_zones, real_zones)

            gen_clouds[pid] = gen_shots
            rows.append(
                {
                    "model": model_name,
                    "player_id": pid,
                    "n_test_shots": len(group.cells),
                    "nll_per_shot": nll,
                    "zone_kl_5": zkl,
                    "sliced_wasserstein": sw,
                }
            )

        if progress:
            _per_model_t = time.perf_counter() - _t_model
            print(
                f"[ablation]     per-player metrics done in {_per_model_t:.1f}s "
                f"-- starting retrieval ({n_test}x{n_test} SW grid)..."
            )

        # Retrieval is a model-level metric (uses every player's gen cloud).
        if len(real_clouds) >= 2:
            _t_retrieval = time.perf_counter()
            retrieval = top_k_retrieval(
                real_clouds,
                gen_clouds,
                k=min(5, len(real_clouds)),
                n_projections=retrieval_n_projections,
                seed=seed,
            )
            for r in rows[-len(test_groups) :]:
                r["retrieval_top1"] = retrieval.top1_accuracy
                r["retrieval_topk"] = retrieval.topk_accuracy
                r["retrieval_mean_rank"] = retrieval.mean_rank
                r["retrieval_k"] = retrieval.k
            if progress:
                _retr_t = time.perf_counter() - _t_retrieval
                _model_total = time.perf_counter() - _t_model
                print(
                    f"[ablation]     retrieval done in {_retr_t:.1f}s  "
                    f"(model total {_model_total:.1f}s, "
                    f"top-1={retrieval.top1_accuracy:.3f})"
                )

    per_player = pd.DataFrame(rows)
    summary = (
        per_player.groupby("model")
        .agg(
            n_players=("player_id", "nunique"),
            n_test_shots=("n_test_shots", "sum"),
            nll_mean=("nll_per_shot", "mean"),
            nll_median=("nll_per_shot", "median"),
            zone_kl_mean=("zone_kl_5", "mean"),
            sw_mean=("sliced_wasserstein", "mean"),
            sw_median=("sliced_wasserstein", "median"),
            retrieval_top1=("retrieval_top1", "first"),
            retrieval_topk=("retrieval_topk", "first"),
        )
        .reset_index()
        .sort_values("nll_mean")
        .reset_index(drop=True)
    )

    return AblationResult(summary=summary, per_player=per_player)


def model_order() -> Sequence[str]:
    """Stable display order for the 4 v1 models (worst → best baseline)."""
    return (
        "League KDE",
        "Player KDE (raw)",
        "Hierarchical KDE",
        "KDE product",
    )


def make_density_fn_from_checkpoint(
    checkpoint: DecoderCheckpoint,
    base_measure: KDEProduct,
    grid: CourtGrid,
) -> DensityFn:
    """Build a ``density_fn(player_id) → (ny, nx)`` from a trained checkpoint.

    Use as the value in :func:`run_ablation`'s ``extra_models`` to add a
    "KDE product + low-rank tilt (trained)" row alongside the four
    classical baselines.

    Parameters
    ----------
    checkpoint : DecoderCheckpoint
        Loaded via :func:`shotcloud.training.load_decoder_checkpoint`.
        The encoder vocab must cover every test player run_ablation
        encounters; players in the test set but not in the vocab are
        already filtered out by run_ablation's train/test player
        intersection step (``common = train_players & test_players``),
        so callers using a matching train split are safe.
    base_measure : KDEProduct
        Built on the same training data as the checkpoint, with matching
        bandwidth/kappa/weights/recency. Mismatched configs will produce
        a misleading evaluation.
    grid : CourtGrid
        Must match ``base_measure.grid``.

    Returns
    -------
    callable
        ``density_fn(player_id) → (grid.ny, grid.nx)`` numpy array.
        Strictly positive (post-softmax) and sums to 1.
    """
    if base_measure.grid is not grid and base_measure.grid != grid:
        # The grid identity check is a sanity guard — `KDEProduct.grid` is
        # the hierarchical KDE's grid, so this should always pass when
        # `base_measure` was built on `grid`.
        raise ValueError("base_measure.grid does not match the supplied grid")

    encoder = checkpoint.encoder
    decoder = checkpoint.decoder
    vocab = checkpoint.vocab
    encoder.eval()
    decoder.eval()
    dtype = decoder.V.dtype

    def density_fn(player_id: object) -> NDArray[np.float64]:
        log_q0 = base_measure.log_density(player_id).ravel()
        log_q0_t = torch.as_tensor(log_q0, dtype=dtype).unsqueeze(0)
        idx = torch.tensor([vocab.to_idx(player_id)], dtype=torch.long)
        u = encoder(idx)
        with torch.no_grad():
            probs = decoder.probs(log_q0_t, u).numpy().ravel()
        return probs.reshape(grid.ny, grid.nx).astype(np.float64)

    return density_fn

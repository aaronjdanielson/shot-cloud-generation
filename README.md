# shotcloud

Role-aware KDE-tilted marked point process for basketball shot clouds.

This repository contains the research paper and the production Python package
for modeling NBA shot clouds as **KDE-anchored spatial measures with small
learned contextual deformations**.

The model:

> a role-aware marked point process whose spatial decoder is a low-rank
> exponential tilt of a hierarchical KDE-product base measure.

See [docs/plan.md](docs/plan.md) for the full project plan, including the paper
thesis, formal model, empirical ladder, and software architecture.

## Status

Pre-alpha. Bootstrap complete; model implementation in progress. See
[docs/log.md](docs/log.md) for the running development log.

## Install (development)

```bash
# Requires uv (https://docs.astral.sh/uv/) and Python 3.11+
uv sync --all-extras
```

This creates a `.venv/` and installs `shotcloud` in editable mode plus all
optional extras (`data`, `viz`, `train`, `dev`).

For a minimal install:

```bash
uv sync
```

## Quickstart

The public API is being built out. The intended minimal example is:

```python
from shotcloud import CourtGrid, HierarchicalKDE, KDEProduct, LowRankTiltDecoder
from shotcloud.models import RoleAwareTimingModel, ShotCloudProcess

grid = CourtGrid(xlim=(-25, 25), ylim=(-5, 47), nx=64, ny=56)

kde = HierarchicalKDE(grid=grid, bandwidth=1.5, kappa=500)
kde.fit(shots_train)

q0 = KDEProduct(
    player_kde=kde.player,
    position_kde=kde.position,
    league_kde=kde.league,
    weights=dict(a_p=1.0, a_g=0.3, a_0=0.2),
)
decoder = LowRankTiltDecoder(n_cells=grid.n_cells, rank=8, zero_init=True)
timing = RoleAwareTimingModel(...)

model = ShotCloudProcess(
    timing_model=timing,
    spatial_decoder=decoder,
    base_measure=q0,
)

cloud = model.sample_shot_cloud(context=ctx, R=500)
```

See [docs/plan.md §5.2](docs/plan.md#52-key-api-contracts) for the full API
contract.

## Development

```bash
make test       # run pytest with coverage
make lint       # ruff check
make format     # ruff format
make typecheck  # mypy
```

Pre-commit hooks (`ruff`, `mypy`) run on every commit:

```bash
uv run pre-commit install
```

## Coordinate convention

NBA `LOC_X` / `LOC_Y` are tenths of feet. After dividing by 10:

- Basket at origin `(0, 0)`.
- `x` positive = right side of the court (offensive view).
- `y` positive = away from basket toward midcourt.
- Default grid extent: `x ∈ [-25, 25]`, `y ∈ [-5, 47]` (full half-court).

## License

MIT — see [LICENSE](LICENSE).

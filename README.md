# shotcloud

Adaptive Collaborative KDE (AC-KDE) for forecasting NBA player-game shot clouds.

This repository holds the `shotcloud` Python package and the paper *Adaptive
Collaborative KDE for Forecasting NBA Player-Game Shot Clouds*
([paper/shot_cloud.tex](paper/shot_cloud.tex)).

Each player-game is modeled as a marked point process with three factors:

- **Count** — the number of shots, from a negative-binomial count head.
- **Timing** — when each shot is taken, from a 48-bin softmax over game minutes.
- **Location** — where each shot is taken, from AC-KDE: a continuous kernel
  mixture over *support shots* drawn from the player's own history and from
  pooled shots of analogue players. Support weights combine shooter similarity,
  shot-level attention, a low-rank contextual residual, and opponent
  reweighting; a pooling gate splits mass between own and pooled support.

Every feature is causal: it is computed only from information available before
the game being predicted.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync --all-extras   # package (editable) plus the data, viz, train, and dev extras
uv sync                # runtime dependencies only
```

## Data

The scripts read three raw tables and build two derived artifacts. Locations
come from environment variables, with these defaults:

| Variable | Default | Contents |
| --- | --- | --- |
| `SHOTS` | `$HOME/Dropbox/shot_flow/data/shot_data.csv` | NBA Stats shot events |
| `GAME_LOGS` | `$HOME/Dropbox/shot_flow/data/player_game_logs.csv` | per-player box scores |
| `STARTERS` | `$HOME/Dropbox/shot_flow/data/starters.csv` | starter status per player-game |
| `PLAYER_BIO` | `data/player_bio.csv` | height, weight, position, birth date; fetched by `scripts/fetch_player_bio.py` |
| `SNAPSHOTS` | `data/snapshots_K4.pt` | causal snapshot store; built by `scripts/pretrain_snapshots.py` |

Set `SHOT_FLOW_DATA` to move all three raw tables at once, and `DEVICE`
(`mps` by default) to choose `cpu`, `cuda`, or `mps`.

## Running the code

The command-line scripts live in [scripts/](scripts/). Most have a
ready-to-run launcher of the same name in [scripts/runs/](scripts/runs/):

```bash
scripts/runs/train_gibbs.sh               # train AC-KDE with the paper's configuration
tail -f logs/train_gibbs_*.log            # follow its log
```

A launcher starts its job in the background under `nohup caffeinate`, so the
run survives a closed terminal and the machine stays awake until it finishes.
It prints the PID and the log file, which is written to `logs/`.

```bash
scripts/runs/train_gibbs.sh --n-epochs 5 --output-dir outputs/scratch   # override flags
SHOTCLOUD_FOREGROUND=1 scripts/runs/evaluate_shot_clouds.sh             # run in this shell
SHOTS=/data/shots.csv DEVICE=cpu scripts/runs/train_count_head.sh       # other data, other device
```

Extra arguments are forwarded to the Python script and take precedence over
the launcher's own. Each script documents its inputs, outputs, and options in
its module docstring and under `--help`:

```bash
uv run python scripts/train_gibbs.py --help
```

The pipeline, in order:

| Step | Launcher | Writes |
| --- | --- | --- |
| Fetch player bios | `scripts/runs/fetch_player_bio.sh` | player bio table |
| Build the snapshot store | `scripts/runs/pretrain_snapshots.sh` | snapshot store and manifest |
| Pretrain the count head | `scripts/runs/train_count_head.sh` | count-head run directory |
| Pretrain the timing head | `scripts/runs/train_timing_head.sh` | timing-head run directory |
| Train AC-KDE | `scripts/runs/train_gibbs.sh` | `modules.pt`, `manifest.json`, `history.json` |
| Finite-cloud metrics | `scripts/runs/evaluate_shot_clouds.sh` | cloud metrics JSON |
| Density-surface metrics | `scripts/runs/evaluate_density_surfaces.sh` | density metrics JSON |
| Reference KDE baseline | `scripts/runs/evaluate_baseline_kde.sh` | cloud and density metrics JSON |
| MDN baseline | `scripts/runs/train_mdn.sh`, `scripts/runs/evaluate_mdn.sh` | MDN run directory and metrics |
| Sample shot clouds | `scripts/runs/sample_shot_clouds.sh` | sampled clouds |
| Figures | `scripts/runs/build_attention_figure.sh`, `scripts/runs/plot_hero_shot_clouds.sh` | PNG figures |

These launchers write to fresh output paths, so they do not overwrite the
results the paper reads. The remaining launchers in `scripts/runs/` cover
diagnostics and audits.

## Reproducing the paper

[replication/](replication/) is a numbered pipeline from raw data to the
compiled paper:

```bash
replication/run_all.sh                    # everything, in order
SKIP_ABLATIONS=1 replication/run_all.sh   # without the comparison training runs
DRY_RUN=1 replication/run_all.sh          # print what would run; execute nothing
```

Stages write to the paths the paper reads and skip any step whose outputs
already exist, so a rerun resumes where it stopped. See
[replication/README.md](replication/README.md) for the stages, the table that
maps each figure and table of the paper to its source file, and the items the
pipeline does not cover.

The scripts `scripts/run_experiment_*.sh` and `scripts/eval_*.sh` hold the
exact configuration of each run reported in the paper; the replication stages
call them.

## Repository layout

```
src/shotcloud/
  data/          loaders, context encoder, causal snapshot store, player features
  grids/         court grid and coordinate conversions
  kde/           kernel density engines
  features/      opponent, matchup, and usage features
  models/        AC-KDE spatial factor, count and timing heads, baselines
  training/      datasets, losses, and the joint trainer
  evaluation/    finite-cloud and density-surface metrics, calibration
  simulation/    shot-cloud sampling
  baselines/     causal reference KDE
  viz/           shot-cloud figures
  legacy/, legacy_pivot/   deprecated components, kept for reproducibility
scripts/         command-line entry points and experiment launchers
scripts/runs/    one launcher per script
replication/     the paper's pipeline
tests/           test suite
paper/           paper source
abstract/        conference abstract
docs/            working log and design notes
```

## Development

```bash
uv run pytest
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run mypy src
```

## Coordinate convention

Raw NBA `LOC_X` / `LOC_Y` are tenths of feet; the loader divides by 10.

- Basket at the origin `(0, 0)`; units are feet.
- `x` positive toward the right side of the court.
- `y` positive away from the basket, toward midcourt.
- Default court extent: `x ∈ [-25, 25]`, `y ∈ [-5, 47]`.

## License

MIT — see [LICENSE](LICENSE).

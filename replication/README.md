# Replication

A numbered pipeline that regenerates the reported results, from raw data to
the metrics and figures. The paper is in preparation and its source is not
distributed with the repository; the last two stages compile it when the
source is present and are skipped otherwise.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.11+; run `uv sync --all-extras`
  from the repository root.
- For stages 9 and 10 only: a LaTeX distribution with `latexmk` and
  `pdflatex`, and `pdftotext`.
- Network access to the NBA Stats API for stage 0, which fetches the raw
  tables below; locations are read from environment variables.

| Variable | Default | Contents |
| --- | --- | --- |
| `RAW_DATA` | `data/raw` | directory holding the raw tables |
| `GAME_LOGS` | `$RAW_DATA/player_game_logs.csv` | per-player box scores (stage 0) |
| `SHOTS` | `$RAW_DATA/shot_data.csv` | NBA Stats shot events (stage 0) |
| `STARTERS` | `$RAW_DATA/starters.csv` | starter status per player-game (stage 0) |
| `PLAYER_BIO` | `data/player_bio.csv` | player bio table (stage 0) |
| `SNAPSHOTS` | `data/snapshots_K4.pt` | causal snapshot store (built in stage 1) |
| `DEVICE` | `mps` | device for the stages that take one (`cpu`, `cuda`, `mps`) |

## Running

```bash
replication/run_all.sh                    # the whole pipeline, in order
SKIP_ABLATIONS=1 replication/run_all.sh   # leave out stage 7
replication/04_ac_kde.sh                  # a single stage
DRY_RUN=1 replication/run_all.sh          # print what would run; execute nothing
```

Each script detaches under `nohup caffeinate`, so the run survives a closed
terminal and the machine stays awake. It prints its PID and a log file under
`logs/`; follow progress with `tail -f`. Set `SHOTCLOUD_FOREGROUND=1` to run in
the current shell instead.

A step is skipped when its outputs already exist, so an interrupted run resumes
where it stopped and finished work is never overwritten. `FORCE=1` runs every
step regardless; the game-log fetcher and the count-head, timing-head, and
AC-KDE training scripts still refuse to overwrite an existing output, so move
it aside first.

## Stages

| Stage | Produces |
| --- | --- |
| `00_data.sh` | the four raw tables, fetched from NBA Stats |
| `01_snapshots.sh` | `data/snapshots_K4.pt` and its manifest |
| `02_count_head.sh` | `outputs/count_head_v1/` |
| `03_timing_head.sh` | `outputs/timing_head_v2_masked/` |
| `04_ac_kde.sh` | `outputs/joint_b2_outcome_residual_v1/` (the mainline model) |
| `05_baselines.sh` | `outputs/baseline_kde/`, `outputs/mdn_v1/` |
| `06_evaluate.sh` | mainline metrics: finite-cloud, density-surface, autonomous rollout, support audit, count audit, attention statistics |
| `07_ablations.sh` | five comparison runs and their evaluations against the mainline |
| `08_figures.sh` | `outputs/figures/paper/*.png` |
| `09_paper.sh` | the compiled paper, when `paper/` is present |
| `10_abstract.sh` | the compiled abstract, when `abstract/` is present |

The stages call the launchers in `scripts/` and `scripts/runs/`, which hold the
exact hyperparameters; nothing is duplicated here. Stages 2 to 4 and 7 are
training runs and dominate the wall-clock time. All runs use seed 0 and the
same split: training through 2023-06-30, validation through 2024-04-30, and
evaluation on the 2000 largest validation player-games with at least three
shots.

## Reported tables and figures

| Element | Source | Stage |
| --- | --- | --- |
| Self-bootstrap floor figure | `outputs/figures/paper/self_bootstrap_floor.png` | 6, 8 |
| Reference KDE, MDN, and AC-KDE table | `{cloud,density}_metrics.json` in `outputs/baseline_kde/`, `outputs/mdn_v1/`, `outputs/joint_b2_outcome_residual_v1/` | 5, 6 |
| Pooling-gate dynamics figure | `outputs/figures/paper/dynamics_history.png` | 8 |
| Count validation table | `outputs/count_head_v1/diagnostics.json`, `outputs/ablation_b2/count_head_audit.json` | 2, 6 |
| Timing validation table | `outputs/timing_head_v2_masked/diagnostics.json` | 3 |
| Autonomous rollout table | `outputs/joint_b2_outcome_residual_v1/autonomous_rollout.json` | 6 |
| Outcome-residual table | `cloud_metrics.json` of the mainline and of `outputs/joint_b2_with_frozen_count_and_ctx_v1/` | 6, 7 |
| Attention-over-support figure | `outputs/figures/paper/attention_over_support.png` | 8 |
| Attention statistics table | `outputs/attention_diagnostics/per_game.json` | 6 |
| Stress-test table | `outputs/joint_b2_outcome_{spatial_hawkes,stratified_kernel,causal_zone_bias,mode_routed}_v1/` | 7 |
| Residual ablation table and summary figure | see below | — |
| Example-game figure | see below | — |

## Not covered by the pipeline

- **Residual ablation chain.** The ablation table and
  `outputs/figures/paper/ablation_summary.png` read four runs that have no
  launcher in `scripts/`:
  `outputs/retrieval_gate_zonegamma_def_res_sigma15_20ep_bw1a`,
  `outputs/retrieval_zonegamma_tier1a_usageres_20ep`,
  `outputs/retrieval_zonegamma_tier1a_khatonly_20ep`, and
  `outputs/retrieval_zonegamma_tier1a_usageres_khat_20ep`. Each run's full
  configuration is recorded in its `manifest.json`. With those runs present,
  `uv run python scripts/build_ablation_b2.py` writes
  `outputs/ablation_b2/summary.{json,md}`; without them,
  `scripts/build_paper_figures.py` in stage 8 fails.
- **Example-game figure.** The paper includes
  `outputs/figures/hero_shot_clouds/Nikola_Joki__2024-04-02__SAS_at_DEN/overlay.png`,
  drawn by `scripts/plot_hero_shot_clouds.py`. That script does not load
  checkpoints with an outcome residual, so it cannot draw the figure from the
  mainline run; `scripts/runs/hero_b2.sh` draws the same games from the
  full-residual run of the ablation chain.
- **Presence-model paragraph.** `scripts/run_experiment_presence_then_timing.sh`
  reproduces it from the play-by-play events in `$PBP_EVENTS`, fetched by
  `scripts/runs/fetch_play_by_play.sh` (2024-25 regular season, one feed per
  game).

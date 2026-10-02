# Replication

A numbered pipeline that regenerates the artifacts behind *Adaptive
Collaborative KDE for Forecasting NBA Player-Game Shot Clouds*, from raw data
to the compiled paper.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.11+; run `uv sync --all-extras`
  from the repository root.
- A LaTeX distribution with `latexmk` and `pdflatex` (stages 9 and 10);
  `pdftotext` for stage 10.
- The raw inputs below. Locations are read from environment variables and
  default to a sibling `shot_flow` checkout.

| Variable | Default | Contents |
| --- | --- | --- |
| `SHOT_FLOW_DATA` | `$HOME/Dropbox/shot_flow/data` | directory holding the three raw tables |
| `SHOTS` | `$SHOT_FLOW_DATA/shot_data.csv` | NBA Stats shot events |
| `GAME_LOGS` | `$SHOT_FLOW_DATA/player_game_logs.csv` | per-player box scores |
| `STARTERS` | `$SHOT_FLOW_DATA/starters.csv` | starter status per player-game |
| `PLAYER_BIO` | `data/player_bio.csv` | player bio table (fetched in stage 0) |
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
step regardless; the count-head, timing-head, and AC-KDE training scripts still
refuse to overwrite an existing run directory, so move that directory aside
first.

## Stages

| Stage | Produces |
| --- | --- |
| `00_data.sh` | checks the raw inputs; fetches `data/player_bio.csv` |
| `01_snapshots.sh` | `data/snapshots_K4.pt` and its manifest |
| `02_count_head.sh` | `outputs/count_head_v1/` |
| `03_timing_head.sh` | `outputs/timing_head_v2_masked/` |
| `04_ac_kde.sh` | `outputs/joint_b2_outcome_residual_v1/` (the mainline model) |
| `05_baselines.sh` | `outputs/baseline_kde/`, `outputs/mdn_v1/` |
| `06_evaluate.sh` | mainline metrics: finite-cloud, density-surface, autonomous rollout, support audit, count audit, attention statistics |
| `07_ablations.sh` | five comparison runs and their evaluations against the mainline |
| `08_figures.sh` | `outputs/figures/paper/*.png` |
| `09_paper.sh` | `paper/shot_cloud.pdf` |
| `10_abstract.sh` | `abstract/SSAC27_submission/SSAC27_abstract.pdf` |

The stages call the launchers in `scripts/` and `scripts/runs/`, which hold the
exact hyperparameters; nothing is duplicated here. Stages 2 to 4 and 7 are
training runs and dominate the wall-clock time. All runs use seed 0 and the
split used throughout the paper: training through 2023-06-30, validation
through 2024-04-30, and evaluation on the 2000 largest validation player-games
with at least three shots.

## Paper elements

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
  reproduces it and needs play-by-play events from a separate source
  (`EVENTS_DIR`).

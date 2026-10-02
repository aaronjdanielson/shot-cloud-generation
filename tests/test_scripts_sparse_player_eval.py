"""Tests for :mod:`scripts.sparse_player_eval` (the H̄-stratified
cloud-metric post-processor introduced 2026-05-24).

Two run modes covered:

* **Legacy / two-way**: single ``--cloud-metrics`` + ``--label`` plus
  ``--baseline-cloud-metrics`` + ``--baseline-label``. The reference
  label defaults to the baseline label.
* **Multi-run** (PR3 pooled_max sweep, 2026-05-24): repeated
  ``--cloud-metrics LABEL:PATH`` entries plus ``--reference-label``.
  The output JSON carries ``delta_vs_reference[label][bucket]`` and
  ``ratio_vs_reference[label][bucket]`` for every non-reference label.

The unit tests verify the bucketing logic, the aggregation, the
empty-bucket shape, the missing-h_hat error path, the parser for
``LABEL:PATH``, the reference-label resolution rules, and the
end-to-end main in both modes — without invoking any model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sparse_player_eval as spe  # type: ignore[import-not-found]

_METRICS = (
    "energy_distance",
    "sliced_wasserstein",
    "zone_l1",
    "rim_distance_w1",
    "rim_distance_ks",
    "mean_shot_distance_err_ft",
)


def _fake_game(
    *,
    player_id: str,
    h_hat: float,
    energy: float,
    self_energy: float = 1.0,
) -> dict[str, Any]:
    """Build a minimal per_game record. The non-energy metrics get
    deterministic scaled values so the aggregation tests can pin them."""
    model = {k: float(energy * (1.0 + i * 0.1)) for i, k in enumerate(_METRICS)}
    sb = {k: float(self_energy * (1.0 + i * 0.1)) for i, k in enumerate(_METRICS)}
    return {
        "player_id": player_id,
        "game_idx": hash((player_id, h_hat, energy)) % 10_000,
        "k_obs": 30,
        "h_hat": h_hat,
        "model": model,
        "self_bootstrap": sb,
        "gap": {k: float(model[k] - sb[k]) for k in _METRICS},
        "ratio": {k: float(model[k] / sb[k]) for k in _METRICS},
    }


def _write_cm(path: Path, games: list[dict[str, Any]], *, has_sb: bool = True) -> None:
    payload: dict[str, Any] = {
        "config": {"include_self_bootstrap": has_sb},
        "per_game": games,
    }
    path.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def test_bucket_games_partitions_by_h_hat() -> None:
    games = [
        _fake_game(player_id="p1", h_hat=0.0, energy=2.0),
        _fake_game(player_id="p2", h_hat=10.0, energy=2.0),
        _fake_game(player_id="p3", h_hat=50.0, energy=2.0),
        _fake_game(player_id="p4", h_hat=200.0, energy=2.0),
        _fake_game(player_id="p5", h_hat=500.0, energy=2.0),
        _fake_game(player_id="p6", h_hat=2000.0, energy=2.0),
    ]
    grouped = spe._bucket_games(games)
    assert [len(grouped[b]) for b in spe.BUCKET_ORDER] == [1, 1, 1, 1, 1, 1]
    assert grouped["0"][0]["player_id"] == "p1"
    assert grouped["1-25"][0]["player_id"] == "p2"
    assert grouped["1001+"][0]["player_id"] == "p6"


def test_bucket_games_skips_missing_h_hat_silently() -> None:
    """Records without h_hat are dropped by `_bucket_games` so the
    caller can detect partial coverage via the missing-row count."""
    g1 = _fake_game(player_id="p1", h_hat=10.0, energy=2.0)
    g2 = _fake_game(player_id="p2", h_hat=10.0, energy=2.0)
    del g2["h_hat"]
    grouped = spe._bucket_games([g1, g2])
    total = sum(len(v) for v in grouped.values())
    assert total == 1
    assert grouped["1-25"][0]["player_id"] == "p1"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_bucket_aggregate_empty_bucket_shape() -> None:
    """An empty bucket must still emit the full shape with NaNs so
    callers iterate uniformly."""
    out = spe._bucket_aggregate([], has_self_bootstrap=True)
    assert out["n_games"] == 0
    for k in _METRICS:
        assert out["model"][k] != out["model"][k]  # NaN
        assert out["gap"][k] != out["gap"][k]


def test_bucket_aggregate_means_match_arithmetic_mean() -> None:
    games = [
        _fake_game(player_id="p1", h_hat=10.0, energy=2.0, self_energy=1.0),
        _fake_game(player_id="p2", h_hat=20.0, energy=4.0, self_energy=2.0),
    ]
    out = spe._bucket_aggregate(games, has_self_bootstrap=True)
    assert out["n_games"] == 2
    assert out["mean_h_hat"] == 15.0
    assert out["model"]["energy_distance"] == pytest.approx(3.0)
    assert out["self_bootstrap"]["energy_distance"] == pytest.approx(1.5)
    assert out["gap"]["energy_distance"] == pytest.approx(1.5)
    assert out["ratio"]["energy_distance"] == pytest.approx(2.0)


def test_bucket_aggregate_without_self_bootstrap() -> None:
    games = [
        _fake_game(player_id="p1", h_hat=10.0, energy=2.0),
        _fake_game(player_id="p2", h_hat=20.0, energy=4.0),
    ]
    out = spe._bucket_aggregate(games, has_self_bootstrap=False)
    assert "model" in out
    assert "self_bootstrap" not in out
    assert "gap" not in out


# ---------------------------------------------------------------------------
# Cloud-metrics + pooling I/O
# ---------------------------------------------------------------------------


def test_read_cloud_metrics_rejects_missing_h_hat(tmp_path: Path) -> None:
    g_ok = _fake_game(player_id="p1", h_hat=10.0, energy=2.0)
    g_bad = _fake_game(player_id="p2", h_hat=10.0, energy=2.0)
    del g_bad["h_hat"]
    p = tmp_path / "cm.json"
    _write_cm(p, [g_ok, g_bad])
    with pytest.raises(ValueError, match="missing 'h_hat'"):
        spe._read_cloud_metrics(p)


def test_read_cloud_metrics_succeeds_when_all_h_hat_present(tmp_path: Path) -> None:
    games = [_fake_game(player_id=f"p{i}", h_hat=float(i), energy=2.0) for i in range(3)]
    p = tmp_path / "cm.json"
    _write_cm(p, games)
    pg, has_sb = spe._read_cloud_metrics(p)
    assert len(pg) == 3
    assert has_sb is True


def test_read_cloud_metrics_falls_back_when_config_missing_self_bootstrap(
    tmp_path: Path,
) -> None:
    """Some older cloud_metrics files may not record
    ``include_self_bootstrap`` in their config; infer it from the
    presence of ``self_bootstrap`` on the first per_game record."""
    games = [_fake_game(player_id="p1", h_hat=10.0, energy=2.0)]
    p = tmp_path / "cm.json"
    p.write_text(json.dumps({"config": {}, "per_game": games}))
    _, has_sb = spe._read_cloud_metrics(p)
    assert has_sb is True


def test_read_pooling_finds_aggregate_nested_layout(tmp_path: Path) -> None:
    """``pooled_by_history_bucket`` lives under ``aggregate`` in the
    actual on-disk layout pooling_diagnostics.py writes."""
    pooling_path = tmp_path / "pooling.json"
    pooling_path.write_text(
        json.dumps({"aggregate": {"pooled_by_history_bucket": {"26-100": 0.31}}})
    )
    out = spe._read_pooling(pooling_path)
    assert out == {"26-100": pytest.approx(0.31)}


def test_read_pooling_falls_back_to_top_level(tmp_path: Path) -> None:
    """Top-level placement is also accepted for older / hand-edited files."""
    pooling_path = tmp_path / "pooling.json"
    pooling_path.write_text(json.dumps({"pooled_by_history_bucket": {"26-100": 0.31}}))
    out = spe._read_pooling(pooling_path)
    assert out == {"26-100": pytest.approx(0.31)}


# ---------------------------------------------------------------------------
# Argument parsing: LABEL:PATH
# ---------------------------------------------------------------------------


def test_parse_cm_arg_recognizes_label_colon_path() -> None:
    label, path = spe._parse_cm_arg("pm50:/abs/path/to/cm.json")
    assert label == "pm50"
    assert path == Path("/abs/path/to/cm.json")


def test_parse_cm_arg_falls_back_to_bare_path_when_prefix_has_slash() -> None:
    """A path-typical prefix (contains ``/``) is not a label — the whole
    value is the path."""
    label, path = spe._parse_cm_arg("./outputs/run_a:b/cm.json")
    assert label is None
    assert path == Path("./outputs/run_a:b/cm.json")


def test_parse_cm_arg_falls_back_to_bare_path_when_prefix_has_dot() -> None:
    """A prefix containing ``.`` (file extension or relative path marker)
    is not a label."""
    label, path = spe._parse_cm_arg("./cm.json")
    assert label is None
    assert path == Path("./cm.json")


def test_parse_cm_arg_accepts_alphanumeric_label() -> None:
    label, path = spe._parse_cm_arg("retrieval_v2:/x/y.json")
    assert label == "retrieval_v2"
    assert path == Path("/x/y.json")


def test_parse_cm_arg_no_colon_is_bare_path() -> None:
    label, path = spe._parse_cm_arg("/abs/path/cm.json")
    assert label is None
    assert path == Path("/abs/path/cm.json")


# ---------------------------------------------------------------------------
# Spec resolution and reference-label resolution
# ---------------------------------------------------------------------------


def _ns(**kwargs: Any) -> Any:
    """Minimal argparse.Namespace-like for spec resolution tests."""
    import argparse as _arg

    return _arg.Namespace(**kwargs)


def test_resolve_specs_multi_run_uses_parsed_labels(tmp_path: Path) -> None:
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p3 = tmp_path / "c.json"
    args = _ns(
        cloud_metrics=[("pm50", p1), ("pm100", p2), ("pm500", p3)],
        label=None,
        baseline_cloud_metrics=None,
        baseline_label="baseline",
    )
    specs = spe._resolve_specs(args)
    assert [lab for lab, _ in specs] == ["pm50", "pm100", "pm500"]
    assert [path for _, path in specs] == [p1, p2, p3]


def test_resolve_specs_legacy_label_applies_to_first_bare_path(tmp_path: Path) -> None:
    """Bare-path --cloud-metrics + --label legacy form is preserved."""
    p1 = tmp_path / "a.json"
    args = _ns(
        cloud_metrics=[(None, p1)],
        label="my_run",
        baseline_cloud_metrics=None,
        baseline_label="baseline",
    )
    specs = spe._resolve_specs(args)
    assert specs == [("my_run", p1)]


def test_resolve_specs_legacy_baseline_is_appended(tmp_path: Path) -> None:
    p1 = tmp_path / "a.json"
    p_base = tmp_path / "base.json"
    args = _ns(
        cloud_metrics=[(None, p1)],
        label="retrieval",
        baseline_cloud_metrics=p_base,
        baseline_label="fixed_lr",
    )
    specs = spe._resolve_specs(args)
    assert specs == [("retrieval", p1), ("fixed_lr", p_base)]


def test_resolve_specs_duplicate_labels_raise(tmp_path: Path) -> None:
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    args = _ns(
        cloud_metrics=[("pm50", p1), ("pm50", p2)],
        label=None,
        baseline_cloud_metrics=None,
        baseline_label="baseline",
    )
    with pytest.raises(ValueError, match="duplicate label"):
        spe._resolve_specs(args)


def test_resolve_reference_label_uses_explicit_when_given(tmp_path: Path) -> None:
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    args = _ns(
        cloud_metrics=[("pm50", p1), ("pm500", p2)],
        label=None,
        baseline_cloud_metrics=None,
        baseline_label="baseline",
        reference_label="pm500",
    )
    specs = [("pm50", p1), ("pm500", p2)]
    assert spe._resolve_reference_label(args, specs) == "pm500"


def test_resolve_reference_label_defaults_to_last_in_multi_run(tmp_path: Path) -> None:
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p3 = tmp_path / "c.json"
    args = _ns(
        cloud_metrics=[("pm50", p1), ("pm100", p2), ("pm500", p3)],
        label=None,
        baseline_cloud_metrics=None,
        baseline_label="baseline",
        reference_label=None,
    )
    specs = [("pm50", p1), ("pm100", p2), ("pm500", p3)]
    assert spe._resolve_reference_label(args, specs) == "pm500"


def test_resolve_reference_label_defaults_to_baseline_label_in_legacy_mode(tmp_path: Path) -> None:
    """The legacy ``--baseline-cloud-metrics`` flag makes the baseline
    label the implicit reference (back-compat for callers that don't
    pass ``--reference-label``)."""
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    args = _ns(
        cloud_metrics=[(None, p1)],
        label="retrieval",
        baseline_cloud_metrics=p2,
        baseline_label="fixed_lr",
        reference_label=None,
    )
    specs = [("retrieval", p1), ("fixed_lr", p2)]
    assert spe._resolve_reference_label(args, specs) == "fixed_lr"


def test_resolve_reference_label_returns_none_for_single_run(tmp_path: Path) -> None:
    p1 = tmp_path / "a.json"
    args = _ns(
        cloud_metrics=[("pm50", p1)],
        label=None,
        baseline_cloud_metrics=None,
        baseline_label="baseline",
        reference_label=None,
    )
    specs = [("pm50", p1)]
    assert spe._resolve_reference_label(args, specs) is None


def test_resolve_reference_label_rejects_unknown_label(tmp_path: Path) -> None:
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    args = _ns(
        cloud_metrics=[("pm50", p1), ("pm500", p2)],
        label=None,
        baseline_cloud_metrics=None,
        baseline_label="baseline",
        reference_label="not_a_label",
    )
    specs = [("pm50", p1), ("pm500", p2)]
    with pytest.raises(ValueError, match="not among"):
        spe._resolve_reference_label(args, specs)


# ---------------------------------------------------------------------------
# Δ-vs-reference computation
# ---------------------------------------------------------------------------


def test_compute_delta_vs_reference_uses_gap_when_self_bootstrap_present() -> None:
    """Δ = label_gap − reference_gap; ratio = label_gap / reference_gap."""
    runs: dict[str, dict[str, Any]] = {
        "pm50": {
            "by_bucket": {
                b: spe._bucket_aggregate(
                    [_fake_game(player_id="p", h_hat=10.0, energy=3.0, self_energy=1.0)],
                    has_self_bootstrap=True,
                )
                if b == "1-25"
                else spe._bucket_aggregate([], has_self_bootstrap=True)
                for b in spe.BUCKET_ORDER
            }
        },
        "pm500": {
            "by_bucket": {
                b: spe._bucket_aggregate(
                    [_fake_game(player_id="p", h_hat=10.0, energy=2.5, self_energy=1.0)],
                    has_self_bootstrap=True,
                )
                if b == "1-25"
                else spe._bucket_aggregate([], has_self_bootstrap=True)
                for b in spe.BUCKET_ORDER
            }
        },
    }
    delta, ratio = spe._compute_delta_vs_reference(runs, "pm500", has_self_bootstrap=True)
    # pm50 only — pm500 is the reference, excluded.
    assert set(delta.keys()) == {"pm50"}
    assert set(ratio.keys()) == {"pm50"}
    # pm50 gap_E = 3 − 1 = 2; pm500 gap_E = 2.5 − 1 = 1.5. Δ = 0.5, ratio ≈ 1.333.
    assert delta["pm50"]["1-25"]["energy_distance"] == pytest.approx(0.5)
    assert ratio["pm50"]["1-25"]["energy_distance"] == pytest.approx(2.0 / 1.5)


def test_compute_delta_vs_reference_skips_empty_buckets() -> None:
    empty_buckets = {
        b: spe._bucket_aggregate([], has_self_bootstrap=True) for b in spe.BUCKET_ORDER
    }
    runs: dict[str, dict[str, Any]] = {
        "a": {"by_bucket": dict(empty_buckets)},
        "b": {"by_bucket": dict(empty_buckets)},
    }
    delta, _ = spe._compute_delta_vs_reference(runs, "b", has_self_bootstrap=True)
    assert delta["a"] == {}


def test_compute_delta_vs_reference_falls_back_to_model_without_sb() -> None:
    """No self_bootstrap → Δ over the raw model metric, not the gap."""
    g_a = _fake_game(player_id="p", h_hat=10.0, energy=3.0)
    g_b = _fake_game(player_id="p", h_hat=10.0, energy=2.0)
    runs: dict[str, dict[str, Any]] = {
        "a": {
            "by_bucket": {
                b: spe._bucket_aggregate([g_a] if b == "1-25" else [], has_self_bootstrap=False)
                for b in spe.BUCKET_ORDER
            }
        },
        "b": {
            "by_bucket": {
                b: spe._bucket_aggregate([g_b] if b == "1-25" else [], has_self_bootstrap=False)
                for b in spe.BUCKET_ORDER
            }
        },
    }
    delta, _ = spe._compute_delta_vs_reference(runs, "b", has_self_bootstrap=False)
    # Δ_a = model_a − model_b = 3 − 2 = 1.
    assert delta["a"]["1-25"]["energy_distance"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End-to-end main
# ---------------------------------------------------------------------------


def test_main_single_run_writes_expected_json(tmp_path: Path) -> None:
    games = [
        _fake_game(player_id="p1", h_hat=0.0, energy=5.0),
        _fake_game(player_id="p2", h_hat=10.0, energy=3.0),
        _fake_game(player_id="p3", h_hat=200.0, energy=2.0),
    ]
    cm_path = tmp_path / "cm.json"
    _write_cm(cm_path, games)
    out_path = tmp_path / "sparse_eval.json"
    rc = spe.main(
        [
            "--cloud-metrics",
            str(cm_path),
            "--label",
            "test_run",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    output = json.loads(out_path.read_text())
    assert output["buckets"] == list(spe.BUCKET_ORDER)
    assert "test_run" in output["runs"]
    by = output["runs"]["test_run"]["by_bucket"]
    assert by["0"]["n_games"] == 1
    assert by["1-25"]["n_games"] == 1
    assert by["101-300"]["n_games"] == 1
    # No reference for a single run → Δ/ratio absent, reference_label None.
    assert output["reference_label"] is None
    assert "delta_vs_reference" not in output
    assert "ratio_vs_reference" not in output


def test_main_legacy_baseline_mode_emits_delta_vs_reference(tmp_path: Path) -> None:
    run_games = [_fake_game(player_id="p1", h_hat=10.0, energy=3.0, self_energy=1.0)]
    base_games = [_fake_game(player_id="p1", h_hat=10.0, energy=4.0, self_energy=1.0)]
    cm_run = tmp_path / "cm_run.json"
    cm_base = tmp_path / "cm_base.json"
    _write_cm(cm_run, run_games)
    _write_cm(cm_base, base_games)
    out_path = tmp_path / "sparse_eval.json"
    spe.main(
        [
            "--cloud-metrics",
            str(cm_run),
            "--baseline-cloud-metrics",
            str(cm_base),
            "--label",
            "retrieval",
            "--baseline-label",
            "fixed_lr",
            "--output",
            str(out_path),
        ]
    )
    output = json.loads(out_path.read_text())
    assert "retrieval" in output["runs"]
    assert "fixed_lr" in output["runs"]
    assert output["reference_label"] == "fixed_lr"
    # Δ keyed by non-reference label.
    assert "retrieval" in output["delta_vs_reference"]
    assert output["delta_vs_reference"]["retrieval"]["1-25"]["energy_distance"] < 0


def test_main_multi_run_with_reference_label_emits_delta_for_all_non_reference(
    tmp_path: Path,
) -> None:
    """The PR3 pooled_max sweep call: four runs, pm500 as reference,
    three Δ entries keyed by the non-reference labels."""
    # Construct four runs with increasing performance (smaller energy_distance).
    energies = {"pm50": 5.0, "pm100": 4.5, "pm200": 4.2, "pm500": 4.0}
    files: dict[str, Path] = {}
    for label, e in energies.items():
        games = [_fake_game(player_id="p1", h_hat=50.0, energy=e, self_energy=1.0)]
        p = tmp_path / f"cm_{label}.json"
        _write_cm(p, games)
        files[label] = p
    out_path = tmp_path / "sparse_eval.json"
    argv = []
    for label in ("pm50", "pm100", "pm200", "pm500"):
        argv += ["--cloud-metrics", f"{label}:{files[label]}"]
    argv += [
        "--reference-label",
        "pm500",
        "--output",
        str(out_path),
    ]
    spe.main(argv)
    output = json.loads(out_path.read_text())
    assert set(output["runs"].keys()) == {"pm50", "pm100", "pm200", "pm500"}
    assert output["reference_label"] == "pm500"
    # pm500 absent from delta; the other three present.
    assert set(output["delta_vs_reference"].keys()) == {"pm50", "pm100", "pm200"}
    # All three should have positive Δ at bucket "26-100" (worse than pm500).
    for lab in ("pm50", "pm100", "pm200"):
        d = output["delta_vs_reference"][lab]["26-100"]["energy_distance"]
        assert d > 0, f"expected positive Δ for {lab} vs pm500; got {d}"
    # The Δ should also be monotone: pm50 worst, pm200 closest to pm500.
    assert (
        output["delta_vs_reference"]["pm50"]["26-100"]["energy_distance"]
        > output["delta_vs_reference"]["pm100"]["26-100"]["energy_distance"]
        > output["delta_vs_reference"]["pm200"]["26-100"]["energy_distance"]
    )
    # Ratios > 1 for non-reference labels (worse-than-reference gap).
    for lab in ("pm50", "pm100", "pm200"):
        r = output["ratio_vs_reference"][lab]["26-100"]["energy_distance"]
        assert r > 1.0


def test_main_attaches_pooled_mass_to_first_run(tmp_path: Path) -> None:
    """``--pooling`` overlays pooled_mass onto the FIRST run only."""
    games = [_fake_game(player_id="p1", h_hat=10.0, energy=3.0)]
    cm_a = tmp_path / "a.json"
    cm_b = tmp_path / "b.json"
    _write_cm(cm_a, games)
    _write_cm(cm_b, games)
    pooling_path = tmp_path / "pooling.json"
    pooling_path.write_text(
        json.dumps({"aggregate": {"pooled_by_history_bucket": {"1-25": 0.523}}})
    )
    out_path = tmp_path / "sparse_eval.json"
    spe.main(
        [
            "--cloud-metrics",
            f"runA:{cm_a}",
            "--cloud-metrics",
            f"runB:{cm_b}",
            "--pooling",
            str(pooling_path),
            "--reference-label",
            "runB",
            "--output",
            str(out_path),
        ]
    )
    output = json.loads(out_path.read_text())
    # First run has pooled_mass populated; second run does not.
    assert output["runs"]["runA"]["by_bucket"]["1-25"].get("pooled_mass") == pytest.approx(0.523)
    assert output["runs"]["runB"]["by_bucket"]["1-25"].get("pooled_mass") is None


def test_main_rejects_unknown_reference_label(tmp_path: Path) -> None:
    games = [_fake_game(player_id="p1", h_hat=10.0, energy=3.0)]
    cm_a = tmp_path / "a.json"
    cm_b = tmp_path / "b.json"
    _write_cm(cm_a, games)
    _write_cm(cm_b, games)
    out_path = tmp_path / "sparse_eval.json"
    with pytest.raises(ValueError, match="not among"):
        spe.main(
            [
                "--cloud-metrics",
                f"runA:{cm_a}",
                "--cloud-metrics",
                f"runB:{cm_b}",
                "--reference-label",
                "runC",
                "--output",
                str(out_path),
            ]
        )

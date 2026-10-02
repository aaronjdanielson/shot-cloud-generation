"""Match the MDN evaluation's player-games to the AC-KDE evaluation's player-games.

    .venv/bin/python abstract/SSAC27_submission/match_mdn_games.py     (from the repo root)

The MDN evaluator builds its validation set with a player vocabulary fitted on the training
window, so players with no training shots (rookies) are absent and its 2,000 largest games are
not the 2,000 the other evaluators score. Its records also carry a vocabulary index instead of
the NBA player id, and a different game enumeration.

This script recovers the shared games. It maps the vocabulary index back to the NBA id, then
for each player lists both evaluations' games in game-index order (order of first appearance in
the validation window, the same in both). Restricted to games with at least MIN_K shots, which
both evaluations keep in full, the two shot-count sequences must be identical; a player for
whom they are not is dropped and reported. The result, evidence/mdn_game_match.json, pairs each
AC-KDE record position with an MDN record position.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from shotcloud.data.game_logs import join_game_logs, load_game_logs
from shotcloud.data.loaders import load_shots
from shotcloud.training.dataset import PlayerVocab

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
SHOTS = "/Users/aarondanielson/Dropbox/shot_flow/data/shot_data.csv"
GAME_LOGS = "/Users/aarondanielson/Dropbox/shot_flow/data/player_game_logs.csv"
TRAIN_END = np.datetime64("2023-06-30")
MIN_K = 19  # both evaluations cut inside the 18-shot games, so only 19+ is complete in each


def by_player(records: list[dict], pid_of) -> dict[str, list[tuple[int, int, int]]]:
    out: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for pos, g in enumerate(records):
        if g["k_obs"] >= MIN_K:
            out[pid_of(g)].append((g["game_idx"], g["k_obs"], pos))
    return {p: sorted(v) for p, v in out.items()}


def main() -> None:
    ack = json.load(open(EVIDENCE / "ackde_cloud_metrics.json"))["per_game"]
    mdn = json.load(open(EVIDENCE / "mdn_cloud_metrics.json"))["per_game"]

    shots = join_game_logs(load_shots(SHOTS), load_game_logs(GAME_LOGS))
    train = shots[shots["date"].to_numpy().astype("datetime64[D]") <= TRAIN_END]
    vocab = PlayerVocab.from_ids(train["player_id"].astype(str).tolist())

    a = by_player(ack, lambda g: str(g["player_id"]))
    m = by_player(mdn, lambda g: vocab.to_id(int(g["player_id"])))

    pairs: list[tuple[int, int]] = []
    out_of_vocab, mismatched = [], []
    for pid, games in a.items():
        if pid not in vocab.id_to_idx:
            out_of_vocab.append(pid)
            continue
        other = m.get(pid, [])
        if [k for _, k, _ in games] != [k for _, k, _ in other]:
            mismatched.append(pid)
            continue
        pairs.extend((ga[2], gm[2]) for ga, gm in zip(games, other))

    n_ack = sum(len(v) for v in a.values())
    n_mdn = sum(len(v) for v in m.values())
    print(f"games with {MIN_K}+ shots: AC-KDE evaluation {n_ack}, MDN evaluation {n_mdn}, matched {len(pairs)}")
    print(f"players absent from the training vocabulary: {len(out_of_vocab)} {sorted(out_of_vocab)}")
    print(f"players dropped for unequal shot-count sequences: {len(mismatched)} {sorted(mismatched)}")
    json.dump(
        {
            "min_k": MIN_K,
            "pairs_ackde_pos_mdn_pos": sorted(pairs),
            "out_of_vocab_players": sorted(out_of_vocab),
            "mismatched_players": sorted(mismatched),
        },
        open(EVIDENCE / "mdn_game_match.json", "w"),
    )
    print("wrote evidence/mdn_game_match.json")


if __name__ == "__main__":
    main()

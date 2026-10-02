"""Training: dataset, trainer, history.

Pre-pivot artifacts (``train_decoder``, ``TrainHistory``,
``DecoderCheckpoint``, ``ShotCellDataset``, the regularizer helpers)
moved to :mod:`shotcloud.legacy_pivot` on 2026-05-15.
"""

from shotcloud.training.dataset import OpponentVocab, PlayerVocab
from shotcloud.training.gibbs_dataset import N_TIMING_BINS, GibbsShotDataset, PerGameTable
from shotcloud.training.train_gibbs import (
    CountOnlyTrainHistory,
    GibbsTrainHistory,
    PresenceTrainHistory,
    TimingOnlyTrainHistory,
    train_count_only,
    train_gibbs,
    train_presence_only,
    train_timing_only,
)

__all__ = [
    "N_TIMING_BINS",
    "CountOnlyTrainHistory",
    "GibbsShotDataset",
    "GibbsTrainHistory",
    "OpponentVocab",
    "PerGameTable",
    "PlayerVocab",
    "PresenceTrainHistory",
    "TimingOnlyTrainHistory",
    "train_count_only",
    "train_gibbs",
    "train_presence_only",
    "train_timing_only",
]

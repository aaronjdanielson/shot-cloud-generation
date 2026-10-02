"""Grid-cell Gibbs decoder, Wasserstein archetypes, and their training harness.

Deprecated; retained to reproduce the grid-cell decoder ablations and to keep
their tests runnable. Superseded by the continuous-mixture AC-KDE spatial
factor, :class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.
The dormant grid branch of :mod:`shotcloud.training.train_gibbs` imports the
grid-cell classes from here.

Grid-cell spatial decoder:

* ``gibbs_decoder`` -- :class:`ConditionalGibbsDecoder`, the softmax over
  court cells of offensive prior, defensive field, and low-rank residual.
* ``tilt_decoder`` -- :class:`LowRankTiltDecoder`, the low-rank residual
  ``u^T V`` added to a log base measure.
* ``adaptive_prior`` -- :class:`AdaptiveOffensivePrior`, the gated mixture
  of relevance-weighted self-KDE and archetypal prior.
* ``adaptive_defensive`` -- :class:`AdaptiveDefensiveField`, the
  per-opponent context-adaptive feasibility field on the grid.
* ``archetypes`` -- :class:`ArchetypeDictionary` and
  :class:`ArchetypeMixture`, the archetype basis and its mixture weights.
* ``wasserstein_fit`` -- Sinkhorn-divergence archetype fitting.

Earlier spatial-decoder components:

* ``defensive_kde`` -- non-adaptive per-opponent :class:`DefensiveKDE`.
* ``defensive_scale`` -- :class:`LearnableDefensiveScale`, a scalar or
  context-dependent weight on a defensive log-density.
* ``timing`` -- :class:`TimingModel` protocol and
  :class:`ConstantRateTimingModel`.
* ``marked_process`` -- :class:`ShotCloudProcess`, a sampler composing the
  timing model and a spatial density.
* ``_context_mlp`` -- small MLP shared by ``defensive_scale`` and
  :mod:`shotcloud.legacy.temperature`.

Training and evaluation harness for the low-rank tilt decoder:

* ``trainer`` -- :func:`train_decoder`.
* ``regularizers`` -- entropy and ESS regularizers on relevance weights.
* ``shot_cell_dataset`` -- :class:`ShotCellDataset`, the per-shot dataset.
* ``checkpoint`` -- :class:`DecoderCheckpoint` save and load.
* ``eval_ablation``, ``eval_diagnostics``, ``eval_metrics``,
  ``eval_retrieval`` -- ablation runner and evaluation metrics.

See :mod:`shotcloud.legacy` for the older static KDE-product components.
"""

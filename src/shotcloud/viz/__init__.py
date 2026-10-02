"""Visualization of shot clouds as 3D "energy body" renderings.

:mod:`shotcloud.viz.energy_body` renders a single shot cloud;
:mod:`shotcloud.viz.energy_body_overlay` renders predicted-versus-observed
comparisons.
"""

from shotcloud.viz.energy_body import EnergyBodyConfig, estimate_density, render_energy_body

__all__ = ["EnergyBodyConfig", "estimate_density", "render_energy_body"]

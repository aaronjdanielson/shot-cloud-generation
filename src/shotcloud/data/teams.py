"""NBA team identifiers.

Maps the NBA Stats ``TEAM_ID`` integers to the 3-letter abbreviations
used in the ``HTM`` / ``VTM`` columns of the ``ShotChartDetail`` CSV.
:func:`shotcloud.data.load_shots` uses the mapping to derive the
per-shot ``home_away`` feature.

The 30 IDs are NBA Stats' canonical team identifiers and are constant
over the seasons covered by the data (2014-15 onward): rebrands such
as Brooklyn (2012) and Charlotte (2014) kept their IDs, and the most
recent franchise relocation (Seattle to Oklahoma City, 2008) predates
the data.
"""

from __future__ import annotations

from typing import Final

#: NBA Stats ``TEAM_ID`` → 3-letter abbreviation, matching the codes in
#: the ``HTM`` / ``VTM`` columns of ``ShotChartDetail`` exports.
NBA_TEAM_ID_TO_ABBREV: Final[dict[int, str]] = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612739: "CLE",
    1610612740: "NOP",
    1610612741: "CHI",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612751: "BKN",
    1610612752: "NYK",
    1610612753: "ORL",
    1610612754: "IND",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612760: "OKC",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612763: "MEM",
    1610612764: "WAS",
    1610612765: "DET",
    1610612766: "CHA",
}

__all__ = ["NBA_TEAM_ID_TO_ABBREV"]

#!/usr/bin/env python3
"""Grid debiasing for the N26 HST heliostack (Napier et al. 2026, PSJ 7, 117).

Seed for ``SurveyCharacterizations``: copy this file (or ``hst_n26.toml``)
into that project and keep the survey definition with its characterization
files.

N26 reanalyses the Bernstein et al. 2004 GO-9433 ACS/WFC F606W field as a
single 15-day heliostack (not a multi-epoch AND). Orbits are well enough
determined to use ``aq_grid`` cells in (a, q, sin i_free, H).

Expected layout under ``--root``::

    colour.toml                      # "t-g" for F606W STMAG (see examples/colour.toml)
    characterization/
      N26/
        N26.csv                      # columns named in detection_columns below
        pointings.list               # 6-tile 400" x 600" mosaic, one line
        HST.csv                      # ICRF observer positions
        N26.eff                      # filter= t (F606W STMAG, neither r nor w)

with ``pointings.list``::

    rect 0.1111111 0.1666667 14:07:53.33 -11:21:38 2452672.8585 1.0 HST.csv N26.eff

Run with the installed ``survey_debias`` command::

    survey_debias --survey hst_n26.py:N26_HELIOSTACK --root .

or directly as a script (same options apart from ``--survey``)::

    python hst_n26.py --root .
"""
from pathlib import Path

from survey_debias import GridSurvey

N26_HELIOSTACK = GridSurvey(
    name="N26 heliostack",
    model_band="r",
    colour_file="colour.toml",
    detection_columns={
        "mag": "m_stmag",
        "name": "name",
        "a": "a",
        "e": "e",
        "i": "i",
        "ifree": "ifree",
        "d_bary": "d_bary",
        "comp": "comp",
    },
    detection_defaults={"survey": "N26", "block": "N26"},
    surveys=("N26",),
    paper_reference_jd=2452672.8585,
    rate_cut_min_arcsec_hr=0.05,
    rate_cut_max_arcsec_hr=6.4,
    detections_relpath="characterization/N26/N26.csv",
    detections_full_name="characterization/N26/N26.detections-full",
    results_name="characterization/N26/N26.bias.csv",
    check_detected_title="detected flag≥4 (single 15-day stack)",
    bias_method="aq_grid",
)

if __name__ == "__main__":
    from survey_debias.grid_bias_run import main

    main(survey=N26_HELIOSTACK, default_root=Path.cwd())

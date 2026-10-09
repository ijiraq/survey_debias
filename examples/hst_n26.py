#!/usr/bin/env python3
"""Grid debiasing for the N26 HST heliostack (Napier et al. 2026, PSJ 7, 117).

Seed for ``hst-tno-followup``: copy this file (or ``hst_n26.toml``) into
that project and keep the survey definition with its characterization files.

N26 reanalyses the Bernstein et al. 2004 GO-9433 ACS/WFC F606W field as a
single 15-day heliostack (not a multi-epoch AND). Orbits are well enough
determined to use ``aq_grid`` cells in (a, q, sin i_free, H).

Expected layout under ``--root``::

    data/n26_detections.csv          # columns named in detection_columns below
    colour.toml                      # "F-g" for F606W STMAG
    characterization/
      N26/
        pointings.list               # 6-tile 400" x 600" mosaic, one line
        HST.csv                      # ICRF observer positions
        N26.eff                      # filter= F

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
    detection_columns={
        "mag": "m_stmag",
        "name": "name",
        "a": "a",
        "e": "e",
        "i": "i",
        "d_bary": "d_bary",
        "comp": "comp",
    },
    detection_defaults={"survey": "N26", "block": "N26"},
    paper_reference_jd=2452672.8585,
    rate_cut_min_arcsec_hr=0.05,
    rate_cut_max_arcsec_hr=6.4,
    detections_relpath="data/n26_detections.csv",
    detections_full_name="N26-free-cla_m.detections-full",
    check_detected_title="detected flag≥4 (single 15-day stack)",
    bias_method="aq_grid",
)

if __name__ == "__main__":
    from survey_debias.grid_bias_run import main

    main(survey=N26_HELIOSTACK, default_root=Path.cwd())

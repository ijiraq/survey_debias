#!/usr/bin/env python3
"""Grid debiasing for JWST GO 1568 Sample A (Eduardo et al. 2026, AJ 172, 187).

Seed for ``SurveyCharacterizations``: copy this file (or ``jwst_sample_a.toml``)
into that project and keep the survey definition with its characterization
files.

Short arcs do not constrain (a, e), so this survey uses ``model_ae``: cells in
(r, ecliptic i, H) with (a, e) drawn from the OSSOS model near (r, i).

Expected layout under ``--root``::

    colour.toml                      # "H-g" for F150W2 (see examples/colour.toml)
    characterization/
      jwst_sample_a/
        jwst_sample_a.csv            # columns named in detection_columns below
        survey.conf                  # detections_required = 3
        epoch1/ epoch2/ epoch3/      # pointings.list, jwst_sample_a.eff, JWST.csv in each

Each ``pointings.list`` holds the mosaic for that epoch, for example::

    0.2236068 0.2236068 209.3875 -10.865278 2459969.32118 1.0 JWST.csv jwst_sample_a.eff

Run with the installed ``survey_debias`` command::

    survey_debias --survey jwst_sample_a.py:JWST_SAMPLE_A --root . \\
        --model /path/to/Models/OSSOS

or directly as a script (same options apart from ``--survey``)::

    python jwst_sample_a.py --root . --model /path/to/Models/OSSOS
"""
from pathlib import Path

from survey_debias import GridSurvey

JWST_SAMPLE_A = GridSurvey(
    name="JWST Sample A",
    model_band="r",
    colour_file="colour.toml",
    detection_columns={
        "mag": "m_f150w2",
        "name": "name",
        "a": "a",
        "e": "e",
        "i": "i",
        "d_bary": "d_bary",
        "comp": "comp",
    },
    detection_defaults={"survey": "jwst_sample_a", "block": "jwst_sample_a"},
    surveys=("jwst_sample_a",),
    paper_reference_jd=2459974.5,
    rate_cut_min_arcsec_hr=0.03,
    rate_cut_max_arcsec_hr=8.66,
    detections_relpath="characterization/jwst_sample_a/jwst_sample_a.csv",
    detections_full_name="characterization/jwst_sample_a/jwst_sample_a.detections-full",
    results_name="characterization/jwst_sample_a/jwst_sample_a.bias.csv",
    check_detected_title="detected at all 3 epochs",
    bias_method="model_ae",
)

if __name__ == "__main__":
    from survey_debias.grid_bias_run import main

    main(survey=JWST_SAMPLE_A, default_root=Path.cwd())

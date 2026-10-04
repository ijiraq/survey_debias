#!/usr/bin/env python3
"""Grid debiasing for the N26 HST heliostack (Napier et al. 2026, PSJ 7, 117).

Seed for ``hst-tno-followup``: copy this file into that project and keep the
survey definition with its characterization files.

N26 reanalyses the Bernstein et al. 2004 GO-9433 ACS/WFC F606W field as a
single 15-day heliostack (not a multi-epoch AND). Orbits are well enough
determined to use ``aq_grid`` cells in (a, q, sin i_free, H).

Expected layout under ``--root``::

    data/n26_detections.csv          # name,a,e,i,d_bary,comp,m_stmag[,ifree]
    characterization/
      pointings.template
      HST.csv                        # ICRF observer positions
      N26.eff

Run with survey_debias on PYTHONPATH (or installed)::

    PYTHONPATH=/path/to/survey_debias python hst_n26.py --root .
"""
from pathlib import Path

from grid_bias import GridSurvey
from grid_bias_run import main

# Napier et al.: r_AB = STMAG_F606W − 0.3.
STMAG_F606W_TO_R_AB = -0.3

# 6-tile 400″ × 600″ mosaic (~0.02 deg²) at 14:07:53.33 −11:21:38.
# Single heliostack at the midpoint of the 15-day span.
N26_HELIOSTACK = GridSurvey(
    name="N26 heliostack",
    field_ra_deg=15.0 * (14.0 + 7.0 / 60.0 + 53.33 / 3600.0),
    field_dec_deg=-(11.0 + 21.0 / 60.0 + 38.0 / 3600.0),
    mosaic_width_deg=400.0 / 3600.0,
    mosaic_height_deg=600.0 / 3600.0,
    epoch_jd=(2452672.8585,),
    mag_color_offset=STMAG_F606W_TO_R_AB,
    mag_column="m_stmag",
    observer_csv="HST.csv",
    eff_file="N26.eff",
    paper_reference_jd=2452672.8585,
    rate_cut_min_arcsec_hr=0.05,
    rate_cut_max_arcsec_hr=6.4,
    epoch_layout="flat",
    detections_relpath="data/n26_detections.csv",
    detections_full_name="N26-free-cla_m.detections-full",
    check_detected_title="detected flag≥4 (single 15-day stack)",
    bias_method="aq_grid",
)

if __name__ == "__main__":
    main(survey=N26_HELIOSTACK, default_root=Path.cwd())

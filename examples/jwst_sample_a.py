#!/usr/bin/env python3
"""Grid debiasing for JWST GO 1568 Sample A (Eduardo et al. 2026, AJ 172, 187).

Seed for ``jwst-tno-followup``: copy this file into that project and keep the
survey definition with its characterization files.

Short arcs do not constrain (a, e), so this survey uses ``model_ae``: cells in
(r, ecliptic i, H) with (a, e) drawn from the OSSOS model near (r, i).

Expected layout under ``--root``::

    data/jwst_sampleA.csv            # name,a,e,i,d_bary,comp,m_f150w2[,ifree]
    characterization/
      pointings.template
      epoch1/ epoch2/ epoch3/        # JWST.csv and JWST_sampleA.eff in each

Run with survey_debias on PYTHONPATH (or installed)::

    PYTHONPATH=/path/to/survey_debias python jwst_sample_a.py --root . \\
        --model /path/to/OSSOS/models
"""
import math
from pathlib import Path

from grid_bias import GridSurvey
from grid_bias_run import main

# Eduardo et al. 2026 ICRS mosaic centre (13:57:33, −10:51:55).
# CADC proposal-1568 detector centroids average ~3″ east of this.
# Epochs are the CADC shift-and-stack midpoints; 2459974.5 is the paper's
# orbit-fit reference, not a visit.
JWST_SAMPLE_A = GridSurvey(
    name="JWST Sample A",
    field_ra_deg=209.3875,
    field_dec_deg=-10.865278,
    mosaic_width_deg=math.sqrt(0.05),
    mosaic_height_deg=math.sqrt(0.05),
    epoch_jd=(2459969.32118, 2459973.96785, 2459979.90854),
    mag_color_offset=1.0,  # m_r = m_F150W2 + 1
    mag_column="m_f150w2",
    observer_csv="JWST.csv",
    eff_file="JWST_sampleA.eff",
    paper_reference_jd=2459974.5,
    rate_cut_min_arcsec_hr=0.03,
    rate_cut_max_arcsec_hr=8.66,
    epoch_layout="subdir",
    detections_relpath="data/jwst_sampleA.csv",
    detections_full_name="JWST-free-cla_m.detections-full",
    check_detected_title="detected flag≥4 at all 3 epochs",
    bias_method="model_ae",
)

if __name__ == "__main__":
    main(survey=JWST_SAMPLE_A, default_root=Path.cwd())

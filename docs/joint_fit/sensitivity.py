#!/usr/bin/env python3
"""MLE sensitivity of a joint fit to the colour spread and to JWST's fill factor.

    python docs/joint_fit/sensitivity.py <joint.toml> [out.csv]

Rows: sigma_c = 0.1, 0.2, 0.3 with the configuration as written, then the
JWST Sample A eps scaled by (1.0 / 0.91)^3 (fill factor 1 instead of 0.91 at
each of the three epochs; E26's normalisation).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from survey_debias import hfit, joint, joint_cli  # noqa: E402

JWST_FILL = 0.91


def run(config, out=None):
    cfg, base = joint_cli.load_config(config)
    rows = []
    cases = [("sigma_c", sc, None) for sc in (0.1, 0.2, 0.3)] + [("jwst_fill", 1.0, 1.0)]
    for label, value, fill in cases:
        like, _ = joint_cli.build_terms(cfg, base, sigma_c=value if label == "sigma_c" else None)
        if fill is not None:
            for t in like.surveys:
                if t.name == "jwst_sample_a":
                    scale = (fill / JWST_FILL) ** 3
                    t.eps.eps = t.eps.eps * scale
                    t.eps.boot = t.eps.boot * scale
        f = joint.fit_joint(like)
        a, b, hb, ho = f.theta[:4]
        row = {"case": label, "value": value, "alpha": a, "beta": b, "h_b": hb, "h_o": ho,
               "loglike": f.loglike,
               **{n: v for n, v in zip(like.names[4:], f.theta[4:])},
               **{f"N(<{h})": float(hfit.cumulative(np.array([h]), a, b, hb, ho)[0])
                  for h in (8.0, 10.0, 12.0)}}
        print(row, flush=True)
        rows.append(row)
    if out:
        with Path(out).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    run(*sys.argv[1:3])

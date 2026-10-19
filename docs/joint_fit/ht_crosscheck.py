#!/usr/bin/env python3
"""Horvitz-Thompson cross-check of the joint fit's normalisation.

Each detection's bias is P(detect by any block of its survey | its r, i, H)
for the cold model, from the survey's plant table: plants within +/- DR au
and +/- DI deg of the detection, weighted by w S_bar(H_f + mu) (the
corrected geometric probability and each block's selection), summed over
blocks. sum 1/bias over a survey's detections brighter than H_lim estimates
the cold population with i < 5 deg there (about 0.76 of all cold objects in
the OSSOS 1.0 model), to compare with the joint fit's N(<H_lim).

    python docs/joint_fit/ht_crosscheck.py <joint.toml> <joint_summary.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from survey_debias import hfit, joint_cli  # noqa: E402
from survey_debias.survey_efficiency import PlantTable  # noqa: E402

DR, DI = 1.0, 0.5


def detection_bias(plants, sel, block, r, inc, h_f):
    """Mean over the block's plants in the (r, i) window of w 1(in field) S_bar(h_f + mu)."""
    window = (plants.block == block) & (np.abs(plants.r - r) < DR) & (np.abs(plants.i - inc) < DI)
    if not window.any():
        return np.nan
    m = window & plants.in_field
    s = np.array([sel.measured_selection(np.array([h_f + plants.mu[j]]), float(plants.rate[j]))[0]
                  for j in np.flatnonzero(m)])
    return float(np.sum(plants.w[m] * s) / window.sum())


def main(config, summary_path):
    cfg, base = joint_cli.load_config(config)
    like, info = joint_cli.build_terms(cfg, base)
    summary = json.loads(Path(summary_path).read_text())
    mle = summary["mle"]
    root = Path(info["root"])
    for term, spec in zip(like.surveys, cfg["survey"]):
        name = term.name
        plants = PlantTable.load(root / "characterization" / name / "plants_cold.npz")
        incs = _inclinations(spec, root, name)
        s = term.like.sample
        h_r = s.h - term.colour
        h_lim = float(np.percentile(h_r, 70))
        n_hat, used, missing = 0.0, 0, 0
        for k in range(len(s)):
            if h_r[k] > h_lim:
                continue
            # Union over the survey's blocks: each block alone would estimate the
            # whole population again.
            b = sum(np.nan_to_num(detection_bias(plants, sel, g.split("/", 1)[1], float(s.d[k]),
                                                 incs[str(s.names[k])], float(s.h[k])))
                    for g, sel in term.like.selections.items())
            if not b > 0:
                missing += 1
                continue
            n_hat += 1.0 / b
            used += 1
        model = hfit.cumulative(np.array([h_lim]), mle["alpha"], mle["beta"], mle["h_b"],
                                mle["h_o"])[0]
        print(f"{name}: HT N(H_r < {h_lim:.2f}) = {n_hat:.4g} from {used} detections "
              f"({missing} with no plants in their cell); joint fit N(<{h_lim:.2f}) = {model:.4g}"
              f"; ratio {n_hat / model:.2f}")


def _inclinations(spec, root, name):
    from survey_debias import detections_io

    if spec.get("format") == "release":
        return {r["name"]: float(r["i"]) for r in
                detections_io.convert(root / "characterization", name, detections_io.ColdCut())}
    rows = joint_cli._read_csv(root / spec["detections"])
    key = spec.get("columns", {}).get("name", "name")
    return {r[key]: float(r["i"]) for r in rows}


if __name__ == "__main__":
    main(*sys.argv[1:3])

#!/usr/bin/env python3
"""Synthetic validation of the joint fit through the real JWST and OSSOS characterizations.

Each realisation draws a known tapered cold LF (colours from N(cbar_f, 0.2))
through OSSSSim with the real .eff files (synthetic.inject), then fits it
with the joint likelihood and the eps tables written by survey_debias_eff.
The first realisation is also fitted with sigma_c = 0.1 and 0.3.

Run from the repository root with ../SurveyCharacterizations and the
OSSOS model next to it::

    python docs/joint_fit/validate_synthetic.py [n_realisations]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from survey_debias import joint, joint_cli, synthetic  # noqa: E402
from survey_debias.survey_efficiency import EfficiencyTable, ModelPopulation, SampleCut  # noqa: E402

DEV = HERE.parents[2]
ROOT = DEV / "SurveyCharacterizations"
CHAR = ROOT / "characterization"
MODEL = DEV / "KuiperBeltModels" / "OSSOS" / "Classical-ModelUsed.dat"
OUT = HERE / "synthetic"
TRUTH_SHAPE = (0.25, 0.20, 10.0)
OSSOS_EXPECTED = 270.0
SIGMA_C = 0.2
SURVEYS = {
    "OSSOS_v11": dict(colour=0.0, h_range=(3.0, 9.8), plants=20000,
                      cut=SampleCut(i_max=5.0), exclude=("2015BS",)),
    "jwst_sample_a": dict(colour=-1.0, h_range=(5.0, 14.5), plants=8000,
                          cut=SampleCut(i_max=5.0001, r_range=(38.0, 55.0)), exclude=()),
}

CONFIG = """\
[joint]
name = "synthetic {k}"
root = "{root}"
colour_group = "cold"
eps_tag = "cold"
sigma_delta = 0.05
sigma_c = 0.2
priority = ["OSSOS_v11", "jwst_sample_a"]

[[survey]]
name = "jwst_sample_a"
format = "csv"
detections = "{dir}/jwst_sample_a.csv"
columns = {{ mag = "mag", d_bary = "d_bary", name = "name", block = "block", h = "H" }}

[[survey]]
name = "OSSOS_v11"
format = "csv"
detections = "{dir}/OSSOS_v11.csv"
columns = {{ mag = "mag", d_bary = "d_bary", name = "name", block = "block", h = "H" }}
exclude_blocks = ["2015BS"]
"""


def truth():
    eps = EfficiencyTable.load(CHAR / "OSSOS_v11" / "eps_cold.csv")
    h_o = synthetic.h_o_for_count(eps, TRUTH_SHAPE, OSSOS_EXPECTED, 0.0, SIGMA_C, True)
    theta = (*TRUTH_SHAPE, h_o)
    eps_j = EfficiencyTable.load(CHAR / "jwst_sample_a" / "eps_cold.csv")
    n_j = synthetic.expected_count(eps_j, theta, -1.0, SIGMA_C, False)
    return theta, {"OSSOS_v11": OSSOS_EXPECTED, "jwst_sample_a": n_j}


def realise(k, theta, model):
    d = OUT / f"real{k}"
    for name, spec in SURVEYS.items():
        rows = synthetic.inject(CHAR, name, model, theta, spec["colour"], SIGMA_C,
                                spec["h_range"], spec["plants"], seed=1000 * k + len(name),
                                cut=spec["cut"], colour_file=ROOT / "colour.toml",
                                exclude_blocks=spec["exclude"])
        synthetic.write_catalogue(rows, d / f"{name}.csv",
                                  f"synthetic {name}, realisation {k}, truth {theta}")
    cfg = d / "joint.toml"
    cfg.write_text(CONFIG.format(k=k, root=ROOT, dir=d))
    return cfg


def fit(cfg, sigma_c=None):
    c, base = joint_cli.load_config(cfg)
    like, _ = joint_cli.build_terms(c, base, sigma_c=sigma_c)
    f = joint.fit_joint(like)
    counts = joint_cli.summarize_counts(like, f.theta)
    rel = f.profile_loglike - np.nanmax(f.profile_loglike)
    inside = f.profile_alpha[rel >= -0.5]
    return f, counts, (float(inside.min()), float(inside.max()))


def main():
    n_real = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    model = ModelPopulation.from_file(MODEL)
    theta, expected = truth()
    print("truth", theta, "expected", expected, flush=True)
    rows = []
    for k in range(1, n_real + 1):
        cfg = realise(k, theta, model)
        for sc in ([None, 0.1, 0.3] if k == 1 else [None]):
            f, counts, prof = fit(cfg, sc)
            row = {"realisation": k, "sigma_c_fit": sc if sc is not None else SIGMA_C,
                   **{n: float(v) for n, v in zip(["alpha", "beta", "h_b", "h_o", "delta_H"],
                                                  f.theta)},
                   "alpha_lo": prof[0], "alpha_hi": prof[1],
                   "n_ossos": counts["OSSOS_v11"]["n"], "n_jwst": counts["jwst_sample_a"]["n"]}
            print(row, flush=True)
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "recovery.csv").open("w", newline="") as fh:
        fh.write(f"# truth alpha beta h_b h_o: {' '.join(f'{x:.4f}' for x in theta)}\n")
        fh.write(f"# expected detections: {expected}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, theta)


def plot(rows, theta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from survey_debias import hfit

    base = [r for r in rows if r["sigma_c_fit"] == SIGMA_C]
    hg = np.linspace(5.0, 13.5, 300)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    ax = axes[0]
    ax.plot(hg, hfit.cumulative(hg, *theta), color="k", lw=2.2, label="truth")
    for r in base:
        ax.plot(hg, hfit.cumulative(hg, r["alpha"], r["beta"], r["h_b"], r["h_o"]),
                color="#1f4e79", alpha=0.6, lw=1.2)
    ax.plot([], [], color="#1f4e79", label=f"fits ({len(base)} realisations)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$H_r$")
    ax.set_ylabel(r"$N(<H)$")
    ax.legend(frameon=False)
    ax = axes[1]
    slope = lambda h, a, b, hb: a + b * 10 ** (-b * (h - hb))  # noqa: E731
    ax.plot(hg, slope(hg, *theta[:3]), color="k", lw=2.2, label="truth")
    for r in base:
        ax.plot(hg, slope(hg, r["alpha"], r["beta"], r["h_b"]), color="#1f4e79", alpha=0.6)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel(r"$H_r$")
    ax.set_ylabel(r"local slope $d\log_{10}N/dH$")
    fig.tight_layout()
    fig.savefig(OUT / "recovery.png", dpi=150)


if __name__ == "__main__":
    main()

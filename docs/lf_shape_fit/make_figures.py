#!/usr/bin/env python3
"""Generate figures and results for docs/lf_shape_fit/lf_shape_fit.tex.

The demo fits the dynamically cold objects of JWST Sample A with the
production likelihood (``survey_debias.hfit.ConditionalLikelihood``). The
selection and photometric model are read from the three epoch ``.eff`` files
of the SurveyCharacterizations ``jwst_sample_a`` survey, so S(m) is the
product over epochs (the cube of the single-epoch efficiency).

Run from the repository root, with ``../SurveyCharacterizations`` checked
out next to this repository::

    python docs/lf_shape_fit/make_figures.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy import optimize

# The figures document this checkout's code, not an installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from survey_debias import hfit  # noqa: E402
from survey_debias.efficiency import (  # noqa: E402
    SurveySelection, parse_efficiency_spec, read_efficiency_file,
)
from survey_debias.grid_bias import apparent_to_Hr  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
RES = HERE / "results"
SURVEY_DIR = HERE.parents[2] / "SurveyCharacterizations" / "characterization" / "jwst_sample_a"
GROUP = "jwst_sample_a/jwst_sample_a"

LN10 = math.log(10.0)
# colour.toml: r - F150W2 = 1.0, so (block filter) - model_band = -1.0.
COLOUR = -1.0
# The pre-rebuild Sample A .eff: the same Eq. (8) numbers as a single-epoch
# tanh, applied once (no cube). Kept to show where alpha = 0.05 came from.
OLD_SPEC = "single:0.96,28.92,0.61,30.5"
DIST_REL_ERR = 0.05
# Points in the plotted alpha profile (geometric over the prior, 0.05-2).
PROFILE_POINTS = 61


def style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
    })


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Survey, sample and likelihoods


def load_selection() -> SurveySelection:
    return SurveySelection([read_efficiency_file(SURVEY_DIR / f"epoch{i}" / "jwst_sample_a.eff")
                            for i in (1, 2, 3)])


def old_selection() -> SurveySelection:
    return SurveySelection([parse_efficiency_spec(OLD_SPEC)])


def make_sample(names, m, d, source) -> hfit.HSample:
    m = np.asarray(m, dtype=float)
    d = np.asarray(d, dtype=float)
    h = np.array([apparent_to_Hr(mi, di, colour=COLOUR) for mi, di in zip(m, d)])
    n = len(m)
    return hfit.HSample(
        names=np.asarray(names), h=h, m=m, bias=np.ones(n),
        rate=hfit.opposition_rate(d), h_column="H_r", source=source,
        group=np.array([GROUP] * n, dtype=object), d=d,
    )


def load_cold() -> hfit.HSample:
    path = SURVEY_DIR / "jwst_sample_a.csv"
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["comp"] == "cold"]
    return make_sample([r["name"] for r in rows],
                       [float(r["m_f150w2"]) for r in rows],
                       [float(r["d_bary"]) for r in rows], str(path))


def with_distance_error(sample: hfit.HSample, rel: float) -> hfit.HSample:
    out = sample.subset(np.ones(len(sample), dtype=bool))
    out.d_err = rel * out.d
    return out


def build(sample, selection, measurement):
    h_min = float(sample.h.min()) - 1.0
    like = hfit.ConditionalLikelihood(sample, selection, h_min, measurement=measurement)
    return like, hfit.default_priors(sample.h, h_min)


def alpha_exp_interval(like, alpha_hat, ll_hat):
    """68% interval (Delta lnL = -0.5) of the single-exponential alpha."""
    def gap(a):
        return like.loglike_exponential(a) - (ll_hat - 0.5)

    lo = hi = float("nan")
    if gap(1e-3) < 0:
        lo = optimize.brentq(gap, 1e-3, alpha_hat)
    if gap(3.0) < 0:
        hi = optimize.brentq(gap, alpha_hat, 3.0)
    return lo, hi


def local_slope(h, alpha, beta, h_b):
    """d log10 N(<H) / dH: alpha for H >> H_B, steeper brightward of H_B."""
    return alpha + beta * 10.0 ** (-beta * (np.asarray(h, dtype=float) - h_b))


def at_prior_edge(theta, priors) -> list[str]:
    out = []
    for name, v in zip(hfit.PARAM_NAMES, theta):
        lo, hi = priors.bounds[name]
        if min(v - lo, hi - v) < 1e-3 * (hi - lo):
            out.append(name)
    return out


def fit_case(sample, selection, measurement, label):
    """hfit.fit_mle (grid starts plus the alpha-profile refit), and a finer
    profile for plotting."""
    like, priors = build(sample, selection, measurement)
    mle = hfit.fit_mle(like, priors)
    lo, hi = alpha_exp_interval(like, mle.alpha_exp, mle.loglike_exp)
    return {
        "label": label, "like": like, "priors": priors, "mle": mle,
        "theta": np.asarray(mle.theta), "loglike": mle.loglike,
        "grid_mle": np.asarray(mle.grid_theta), "grid_loglike": mle.grid_loglike,
        "profile": hfit.profile_alpha(like, priors, n_points=PROFILE_POINTS),
        "alpha_exp_lo": lo, "alpha_exp_hi": hi,
        "at_edge": at_prior_edge(mle.theta, priors),
    }


# ---------------------------------------------------------------------------
# Figures


def fig_selection_and_scatter(selection, old, sample):
    m = np.linspace(24.0, 31.0, 1401)
    rate = float(np.median(sample.rate))
    single = selection.epochs[0]
    meas = selection.measurement
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    ax = axes[0]
    ax.plot(m, old(m, rate), color="0.55", lw=1.6, ls=":",
            label="old: single-epoch tanh, not cubed")
    ax.plot(m, single(m, rate), color="#1f4e79", lw=1.8, ls="--",
            label=r"single epoch $p_{\rm det}(m)$ (erfc)")
    ax.plot(m, selection(m, rate), color="#c45c26", lw=2.2,
            label=r"three epochs $S(m)=p_{\rm det}^3$")
    ax.plot(m, selection.measured_selection(m, rate, 0.02), color="#2a9d8f", lw=1.4,
            label=r"$\bar S(m_t)$ with photometric scatter")
    ax.axvline(sample.m.max(), color="0.3", lw=1.0, ls="-.",
               label=fr"faintest cold detection ({sample.m.max():.2f})")
    ax.axvline(single.blocks[0].mag_lim, color="0.6", ls=":", lw=1)
    ax.set_xlabel(r"apparent magnitude $m_{F150W2}$")
    ax.set_ylabel("detection probability")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title("Sample A selection (.eff, three epochs)")
    ax.legend(loc="lower left", frameon=False, fontsize=8)

    ax = axes[1]
    eff = math.sqrt(float(np.sum(meas.mixture / np.arange(1, 4))))
    ax.semilogy(m, meas.sigma(m), color="#1f4e79", lw=2.0, label=r"one epoch $\sigma(m)$")
    ax.semilogy(m, meas.sigma(m) * eff, color="#c45c26", lw=2.0,
                label=r"catalogue (mean of 3 epochs)")
    ax.axvline(sample.m.max(), color="0.3", lw=1.0, ls="-.")
    ax.set_xlabel(r"true magnitude $m_t$ (F150W2)")
    ax.set_ylabel("magnitude scatter")
    ax.set_title("Photometric model from .eff")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "selection_and_scatter")


def fig_likelihood_cartoon(sample, selection, alpha):
    """One cold detection: numerator kernel vs denominator selection on H."""
    k = int(np.flatnonzero(sample.names == "JPB07")[0])
    one = sample.subset(np.arange(len(sample)) == k)
    like = hfit.ConditionalLikelihood(one, selection, float(one.h[0]) - 4.0,
                                      measurement="model")
    h = like.grid
    num, den = like.num[0], like.den[0]
    lam = np.exp(hfit.log_differential_exponential(h, alpha))
    keep = h <= one.h[0] + 2.5

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.fill_between(h[keep], 0, (den / den.max())[keep], color="#c45c26", alpha=0.25,
                    label=r"denominator weight $\bar S(H+\mu_k)$")
    ax.plot(h[keep], (den / den.max())[keep], color="#c45c26", lw=1.5)
    ax.fill_between(h[keep], 0, (num / num.max())[keep], color="#1f4e79", alpha=0.35,
                    label=r"numerator $\eta(m_t)\,p(\hat m_k\mid m_t)$")
    ax.plot(h[keep], (num / num.max())[keep], color="#1f4e79", lw=1.8)
    ax.plot(h[keep], (lam / lam[keep].max())[keep], color="#2a9d8f", lw=1.8, ls="--",
            label=fr"population $\lambda(H)$, $\alpha={alpha:.2f}$")
    ax.axvline(one.h[0], color="0.3", ls=":", lw=1.2,
               label=fr"catalogue $H_r={one.h[0]:.2f}$")
    ax.set_xlabel(r"true absolute magnitude $H_r$")
    ax.set_ylabel("relative weight")
    ax.set_title(fr"JPB07 ($m={one.m[0]:.2f}$, $d={one.d[0]:.1f}$ au)")
    ax.set_xlim(h[keep].min(), h[keep].max())
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    save(fig, "likelihood_cartoon")


def fig_distance_nodes():
    """Explain distance quadrature without Gauss-Hermite jargon."""
    d0, rel = 45.0, DIST_REL_ERR
    x, w = np.polynomial.hermite.hermgauss(7)
    w = w / math.sqrt(math.pi)
    s = math.log1p(rel)
    ln_ratio = math.sqrt(2.0) * s * x
    d = d0 * np.exp(ln_ratio)
    shift = 10.0 * ln_ratio / LN10

    d_grid = np.linspace(d0 * 0.75, d0 * 1.3, 400)
    pdf = (1.0 / (d_grid * s * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * ((np.log(d_grid) - math.log(d0)) / s) ** 2
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    ax.plot(d_grid, pdf, color="#1f4e79", lw=2.0, label="distance PDF")
    ax.vlines(d, 0, np.interp(d, d_grid, pdf), colors="#c45c26", lw=1.5)
    ax.scatter(d, np.interp(d, d_grid, pdf), s=40 + 400 * w / w.max(),
               color="#c45c26", zorder=3, label="quadrature nodes (size ∝ weight)")
    ax.axvline(d0, color="0.4", ls=":", label=r"nominal $d_{\mathrm{bary}}$")
    ax.set_xlabel(r"barycentric distance $d$ (au)")
    ax.set_ylabel("probability density")
    ax.set_title(r"5% distance error at $d=45$ au")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ax.stem(shift, w, linefmt="#1f4e79", markerfmt="o", basefmt=" ")
    ax.set_xlabel(r"shift in $m-H$ (mag)")
    ax.set_ylabel("node weight")
    ax.set_title("How distance error moves absolute magnitude")
    ax.axvline(0, color="0.5", ls=":")
    save(fig, "distance_nodes")


def fig_flowchart():
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                               facecolor=color, edgecolor="#333333", lw=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    box(0.2, 1.6, 2.2, 1.0, ".eff file per epoch\nη, track, mag_lim\nmag_error, phot_frac", "#d6eaf8")
    box(2.8, 1.6, 2.2, 1.0, "measurement\nmodel\nσ(m), offset, noise", "#fdebd0")
    box(5.4, 1.6, 2.0, 1.0, "numerator\nkernel\np(m̂ | m_t)", "#d5f5e3")
    box(5.4, 0.3, 2.0, 1.0, "denominator\nS̄(m_t), product\nover epochs", "#fadbd8")
    box(7.8, 0.9, 1.9, 1.2, "conditional\nshape\nlikelihood", "#e8daef")
    box(2.8, 0.3, 2.2, 1.0, "distance nodes\n(optional)\nfrom d ± σ_d", "#fcf3cf")

    for a, b in [((2.4, 2.1), (2.8, 2.1)), ((5.0, 2.1), (5.4, 2.1)),
                 ((5.0, 0.8), (5.4, 0.8)), ((7.4, 2.1), (7.8, 1.6)),
                 ((7.4, 0.8), (7.8, 1.4))]:
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
    save(fig, "flowchart")


SYNTH_TRUTH = (0.5, 0.4, 11.5)
SYNTH_N = 1000


def synthetic_cold_sample(selection, rng, truth=SYNTH_TRUTH, n_keep=SYNTH_N):
    """Detections of a tapered population at cold-belt distances.

    H is drawn over 4-16, past the faint limit of the selection at every
    distance, so the sample is cut by the survey and not by the draw.
    Detection uses p_det^3 on the true magnitude; the catalogue magnitude is
    the mean of three measurements, as in the .eff model.
    """
    meas = selection.measurement
    h_grid = np.linspace(4.0, 16.0, 40001)
    cum = np.exp(hfit.log_cumulative(h_grid, *truth))
    cum = (cum - cum[0]) / (cum[-1] - cum[0])
    m_hat, d_all = [], []
    while len(m_hat) < n_keep:
        h = np.interp(rng.random(20000), cum, h_grid)
        d = rng.uniform(42.0, 48.0, size=h.size)
        mu = np.array([mi - apparent_to_Hr(mi, di, colour=COLOUR)
                       for mi, di in zip(np.full(h.size, 28.0), d)])
        m_t = h + mu
        rate = hfit.opposition_rate(d)
        keep = rng.random(h.size) < selection.eta(m_t, rate)
        m_obs = meas.sample(m_t[keep], rng)
        ok = selection.track_char(m_obs, rate[keep]) > 0
        m_hat.extend(m_obs[ok])
        d_all.extend(d[keep][ok])
    m_hat = np.array(m_hat[:n_keep])
    d_all = np.array(d_all[:n_keep])
    names = [f"S{i:03d}" for i in range(n_keep)]
    return make_sample(names, m_hat, d_all, "synthetic")


def fig_error_bias_demo(selection):
    """Synthetic cold-like sample: with and without the scatter model."""
    rng = np.random.default_rng(7)
    sample = synthetic_cold_sample(selection, rng)
    fits = {}
    for mode in ("none", "model"):
        like, priors = build(sample, selection, mode)
        fits[mode] = hfit.fit_mle(like, priors)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    hg = np.linspace(np.floor(sample.h.min()), np.ceil(sample.h.max()), 400)
    curves = [(SYNTH_TRUTH, "truth", "#333333", "-")]
    for mode, lab, col, ls in (("none", "ignore scatter", "#c45c26", "--"),
                               ("model", "with scatter model", "#1f4e79", "-")):
        t = fits[mode].theta
        curves.append((t, fr"{lab} ($\alpha={t[0]:.2f}$, $\beta={t[1]:.2f}$, "
                          fr"$H_B={t[2]:.2f}$)", col, ls))
    for th, lab, col, ls in curves:
        ax.plot(hg, local_slope(hg, *th), color=col, ls=ls, lw=2.0, label=lab)
        ax.axhline(th[0], color=col, ls=":", lw=0.8)
    ax.set_xlim(hg[0], hg[-1])
    ax.set_xlabel(r"$H_r$ (range of the detections)")
    ax.set_ylabel(r"local slope $\mathrm{d}\log_{10}N/\mathrm{d}H$")
    ax.set_title(f"Synthetic cold-like sample ({SYNTH_N} detections)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    save(fig, "synthetic_bias")

    lines = [f"truth (alpha, beta, H_B): {SYNTH_TRUTH}",
             f"detections: {SYNTH_N}; d uniform 42-48 au; selection p_det^3 and the "
             "Sample A photometric model from the .eff files",
             f"catalogue H range: {sample.h.min():.2f} to {sample.h.max():.2f}"]
    for mode in ("none", "model"):
        f = fits[mode]
        lines.append(f"{mode}: alpha={f.theta[0]:.4f} beta={f.theta[1]:.4f} "
                     f"H_B={f.theta[2]:.4f} lnL={f.loglike:.3f}; "
                     f"single exponential alpha={f.alpha_exp:.4f}")
    (RES / "synthetic_recovery.txt").write_text("\n".join(lines) + "\n")
    return fits


def fig_jwst_results(cases):
    ref = cases["model"]
    like, mle = ref["like"], ref["mle"]
    sample = like.sample
    grid = like.grid

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1))
    ax = axes[0]
    bins = np.arange(math.floor(sample.h.min()) - 0.25, math.ceil(sample.h.max()) + 0.75, 0.5)
    ax.hist(sample.h, bins=bins, color="#1f4e79", alpha=0.55, edgecolor="white",
            label=f"Sample A cold ($n={len(sample)}$)")
    scale = 0.5
    a, b, hb = ref["theta"]
    ax.plot(grid, like.predicted_observed(a, b, hb) * scale, color="#c45c26", lw=2.0,
            label=fr"tapered MLE ($\alpha={a:.2f}$, $\beta={b:.2f}$, $H_B={hb:.2f}$)")
    a2, b2, hb2 = ref["grid_mle"]
    ax.plot(grid, like.predicted_observed(a2, b2, hb2) * scale, color="#7b3294", lw=1.4,
            ls=":", label=fr"second ridge ($\alpha={a2:.2f}$, $\beta={b2:.2f}$, "
                          fr"$H_B={hb2:.2f}$)")
    ax.plot(grid, like.predicted_observed_exponential(mle.alpha_exp) * scale,
            color="#2a9d8f", lw=1.8, ls="--",
            label=fr"single exponential ($\alpha={mle.alpha_exp:.2f}$)")
    ax.set_xlim(bins[0], bins[-1] + 0.5)
    ax.set_xlabel(r"$H_r$")
    ax.set_ylabel("detections per 0.5 mag")
    ax.set_title("Predicted vs observed detections")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    ax = axes[1]
    styles = {"none": ("#333333", "-"), "model": ("#1f4e79", "-"),
              "model_d": ("#c45c26", "-"), "old": ("0.5", ":")}
    for key, c in cases.items():
        col, ls = styles[key]
        prof = c["profile"]
        ax.plot(prof.alpha, prof.loglike - prof.loglike.max(), color=col, ls=ls, lw=2.0,
                label=c["label"])
    ax.axhline(-0.5, color="0.6", ls=":", lw=1)
    ax.set_xlim(0.0, 0.9)
    ax.set_xlabel(r"faint-end slope $\alpha$")
    ax.set_ylabel(r"profile $\ln L - \max\ln L$")
    ax.set_ylim(-6, 0.4)
    ax.set_title(r"Profile likelihood in $\alpha$ ($\beta$, $H_B$ free)")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    save(fig, "jwst_results")


def fig_local_slopes(cases):
    """Local slope of each solution across the H range of the cold detections."""
    sample = cases["model"]["like"].sample
    hg = np.linspace(sample.h.min() - 0.3, sample.h.max() + 0.3, 400)
    model, old = cases["model"], cases["old"]
    curves = [
        (model["theta"], "three-epoch S, best tapered", "#c45c26", "-"),
        (model["grid_mle"], "three-epoch S, second ridge", "#7b3294", ":"),
        ((model["mle"].alpha_exp, 0.0, 0.0), "three-epoch S, single exponential",
         "#2a9d8f", "--"),
        (old["theta"], "old single-epoch S, tapered", "0.5", "-."),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for th, lab, col, ls in curves:
        y = local_slope(hg, *th) if th[1] > 0 else np.full_like(hg, th[0])
        ax.plot(hg, y, color=col, ls=ls, lw=2.0,
                label=fr"{lab} ($\alpha={th[0]:.2f}$)")
    ax.plot(sample.h, np.full(len(sample), 0.02), "|", color="#1f4e79", ms=12,
            label="cold detections")
    ax.set_ylim(0, 1.5)
    ax.set_xlim(hg[0], hg[-1])
    ax.set_xlabel(r"$H_r$ (larger objects to the left)")
    ax.set_ylabel(r"local slope $\mathrm{d}\log_{10}N/\mathrm{d}H$")
    ax.set_title("Where each solution puts the slope")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    save(fig, "local_slopes")


def fig_jwst_sample(sample, selection):
    rate = float(np.median(sample.rate))
    m = np.linspace(26.0, 31.0, 5001)
    s = selection(m, rate)
    m50 = m[np.flatnonzero(s >= 0.5)[-1]]
    m40 = m[np.flatnonzero(s >= 0.4)[-1]]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    sc = ax.scatter(sample.h, sample.m, c=sample.d, cmap="viridis", s=50)
    for name, h, mm in zip(sample.names, sample.h, sample.m):
        ax.annotate(name, (h, mm), textcoords="offset points", xytext=(4, 2),
                    fontsize=7, color="0.3")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$d_{\mathrm{bary}}$ (au)")
    ax.axhline(m50, color="#c45c26", ls="--", lw=1,
               label=fr"$p_{{\rm det}}^3=0.5$ at {m50:.2f}")
    ax.axhline(m40, color="#c45c26", ls=":", lw=1,
               label=fr"$p_{{\rm det}}^3=0.4$ at {m40:.2f}")
    mag_lim = selection.epochs[0].blocks[0].mag_lim
    ax.axhline(mag_lim, color="0.4", ls="-.", lw=1, label=f"mag_lim = {mag_lim:.1f}")
    ax.set_ylim(sample.m.min() - 0.5, mag_lim + 0.3)
    ax.invert_yaxis()
    ax.set_xlabel(r"catalogue $H_r$")
    ax.set_ylabel(r"apparent $m_{F150W2}$")
    ax.set_title("JWST Sample A cold detections")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    save(fig, "jwst_sample")
    return m50, m40


# ---------------------------------------------------------------------------
# Results


def run_jwst(sample, selection, old):
    cases = {
        "none": fit_case(sample, selection, "none", "no measurement error"),
        "model": fit_case(sample, selection, "model", "photometric model from .eff"),
        "model_d": fit_case(with_distance_error(sample, DIST_REL_ERR), selection, "model",
                            "photometry + 5% distance error"),
        "old": fit_case(sample, old, "none", "old single-epoch tanh selection"),
    }
    lines = ["label,alpha,beta,h_b,loglike,params_at_prior_edge,"
             "grid_alpha,grid_beta,grid_h_b,grid_loglike,"
             "alpha_exp,alpha_exp_lo,alpha_exp_hi,loglike_exp"]
    for c in cases.values():
        t, g, f = c["theta"], c["grid_mle"], c["mle"]
        lines.append(f"\"{c['label']}\",{t[0]:.4f},{t[1]:.4f},{t[2]:.4f},{c['loglike']:.4f},"
                     f"\"{' '.join(c['at_edge'])}\","
                     f"{g[0]:.4f},{g[1]:.4f},{g[2]:.4f},{c['grid_loglike']:.4f},"
                     f"{f.alpha_exp:.4f},{c['alpha_exp_lo']:.4f},{c['alpha_exp_hi']:.4f},"
                     f"{f.loglike_exp:.4f}")
    (RES / "jwst_shape_mle.csv").write_text("\n".join(lines) + "\n")

    h = sample.h
    notes = [
        f"JWST Sample A cold objects (n={len(sample)}, H_r {h.min():.2f} to {h.max():.2f}),",
        "shape-only MLE with survey_debias.hfit (docs demo).",
        "alpha is the faint-end slope (H >> H_B); the taper suppresses the bright end",
        "(H < H_B). Local slope d log10 N / dH = alpha + beta 10^(-beta (H - H_B)).",
        "hfit.fit_mle keeps the better of the maximum from its fixed starting grid",
        "('grid' columns) and a refit from the peak of the profile in alpha.",
        "",
    ]
    for c in cases.values():
        t, g, f = c["theta"], c["grid_mle"], c["mle"]
        edge = f" (at prior edge: {', '.join(c['at_edge'])})" if c["at_edge"] else ""
        notes.append(f"{c['label']}: alpha={t[0]:.4f} beta={t[1]:.4f} Hb={t[2]:.4f} "
                     f"lnL={c['loglike']:.4f}{edge}; grid start alpha={g[0]:.4f} "
                     f"beta={g[1]:.4f} Hb={g[2]:.4f} lnL={c['grid_loglike']:.4f}; "
                     f"single exponential alpha={f.alpha_exp:.4f} "
                     f"[{c['alpha_exp_lo']:.4f}, {c['alpha_exp_hi']:.4f}] "
                     f"lnL={f.loglike_exp:.4f}")
    notes.append("")
    for c in cases.values():
        prof = c["profile"]
        rel = prof.loglike - prof.loglike.max()
        lo, hi = prof.interval(0.5)
        notes.append(f"profile {c['label']}: peak alpha={prof.peak[0]:.3f}; "
                     f"Delta lnL >= -0.5 for alpha {lo:.3f} to {hi:.3f}; "
                     f"Delta lnL at 0.05 = {rel[0]:.2f}, "
                     f"at 0.6 = {np.interp(0.6, prof.alpha, rel):.2f}")
    (RES / "jwst_shape_notes.txt").write_text("\n".join(notes) + "\n")
    return cases


def main():
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    selection, old = load_selection(), old_selection()
    sample = load_cold()

    cases = run_jwst(sample, selection, old)
    fig_selection_and_scatter(selection, old, sample)
    fig_likelihood_cartoon(sample, selection, cases["model"]["mle"].alpha_exp)
    fig_distance_nodes()
    fig_flowchart()
    synth = fig_error_bias_demo(selection)
    fig_jwst_results(cases)
    fig_local_slopes(cases)
    m50, m40 = fig_jwst_sample(sample, selection)

    print("Wrote figures to", FIG)
    print(f"three-epoch S = 0.5 at {m50:.2f}, 0.4 at {m40:.2f} (F150W2)")
    for c in cases.values():
        f = c["mle"]
        print(f"  {c['label']}: tapered {np.round(c['theta'], 3)} lnL={c['loglike']:.3f} "
              f"edge={c['at_edge']}; grid {np.round(c['grid_mle'], 3)} "
              f"lnL={c['grid_loglike']:.3f}; exp alpha={f.alpha_exp:.3f} "
              f"[{c['alpha_exp_lo']:.3f}, {c['alpha_exp_hi']:.3f}] lnL={f.loglike_exp:.3f}")
    for mode, f in synth.items():
        print(f"  synthetic {mode}: {np.round(f.theta, 3)} exp alpha={f.alpha_exp:.3f}")


if __name__ == "__main__":
    main()

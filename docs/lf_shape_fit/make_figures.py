#!/usr/bin/env python3
"""Generate figures for docs/lf_shape_fit/lf_shape_fit.tex.

Run from the repository root::

    python docs/lf_shape_fit/make_figures.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
RES = HERE / "results"
JWST_ROOT = HERE.parents[2] / "jwst-tno-followup"
if not JWST_ROOT.is_dir():
    JWST_ROOT = Path("/tmp/jwst-tno-followup")

LN10 = math.log(10.0)
COLOR_OFFSET = 1.0  # m_r = m_F150W2 + 1
# JWST Sample A .eff (r band)
ETA_A, ETA_M0, ETA_W = 0.96, 29.92, 0.61
MAG_LIM = 30.50
# mag_error= 0.006 0.16 29.9 0.05 29.9 -0.04  (p2 already slope-ready in file? )
# getsur transforms p2 := log10(p2/p1)/(p3-21). The .eff stores the raw p2=0.16
# as the mid-regime sigma at p3? Looking at OSSSSim: the file stores values that
# getsur converts. JWST comment says "numbers shifted". We'll apply the same
# transform as getsur when p2 looks like a sigma (0.16) rather than a slope.
MAG_ERROR = (0.006, 0.16, 29.9, 0.05, 29.9, -0.04)
NOISE_HW = math.sqrt(6.0)


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


# ---------------------------------------------------------------------------
# Photometric model (OSSSSim magran / getsur)


def mag_error_slope(params=MAG_ERROR) -> float:
    p1, p2, p3 = params[0], params[1], params[2]
    # File stores the mid-point sigma; convert to log-slope like getsur.f95.
    if p1 > 0 and p2 > 0 and p3 != 21.0 and p2 > p1:
        return math.log10(p2 / p1) / (p3 - 21.0)
    return float(p2)


def sigma_m(m, params=MAG_ERROR):
    m = np.asarray(m, dtype=float)
    p1, _p2, p3, p4, _p5, _p6 = params
    slope = mag_error_slope(params)
    mid = p1 * 10.0 ** (slope * (m - 21.0))
    faint = np.maximum(p1 * 10.0 ** (slope * (p3 - 21.0)) - (m - p3) * p4, 0.0)
    return np.where(m <= 21.0, p1, np.where(m <= p3, mid, faint))


def offset_m(m, params=MAG_ERROR):
    m = np.asarray(m, dtype=float)
    return np.where(m > params[4], (m - params[4]) * params[5], 0.0)


def eta(m):
    m = np.asarray(m, dtype=float)
    raw = 0.5 * ETA_A * (1.0 - np.tanh((m - ETA_M0) / ETA_W))
    return np.where((m > MAG_LIM) | (raw < 0.01), 0.0, raw)


def track_char(m):
    """Tracking × characterisation for Sample A (track=1; char = mag_lim cut)."""
    m = np.asarray(m, dtype=float)
    return (m <= MAG_LIM).astype(float)


def s_point(m):
    return eta(m) * track_char(m)


def s_bar(m_t, sigma_floor=0.02):
    m_t = np.asarray(m_t, dtype=float)
    sig = np.maximum(sigma_m(m_t), sigma_floor)
    nodes = np.linspace(-NOISE_HW, NOISE_HW, 161)
    dens = np.maximum(1.0 - np.abs(nodes) / NOISE_HW, 0.0) / NOISE_HW
    w = dens * (nodes[1] - nodes[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    w = w / w.sum()
    centre = m_t + offset_m(m_t)
    m_hat = centre[:, None] + sig[:, None] * nodes[None, :]
    return eta(m_t) * (track_char(m_hat) @ w)


# ---------------------------------------------------------------------------
# Absolute magnitude helper (matches grid_bias.apparent_to_Hr)


def bowell(alpha, g=-0.12):
    if alpha <= 0:
        return 0.0
    ta = math.tan(alpha / 2.0)
    phi = (1 - g) * math.exp(-3.33 * ta ** 0.63) + g * math.exp(-1.87 * ta ** 1.22)
    return 2.5 * math.log10(phi) if phi > 0 else 0.0


def apparent_to_Hr(m_survey, d_au, color_offset=COLOR_OFFSET):
    m_r = m_survey + color_offset
    denom = 2.0 * d_au * d_au
    cos_a = max(-1.0, min(1.0, (-1.0 + 2.0 * d_au ** 2) / denom))
    return m_r - 5.0 * math.log10(d_au * d_au) + bowell(math.acos(cos_a))


def log_cumulative(h, alpha, beta, h_b, h_o=0.0):
    """ln N(<H) for the exponentially tapered form."""
    h = np.asarray(h, dtype=float)
    with np.errstate(over="ignore"):
        return LN10 * alpha * (h - h_o) - 10.0 ** (-beta * (h - h_b))


def log_differential(h, alpha, beta, h_b, h_o=0.0):
    """ln dN/dH = ln N + ln d(ln N)/dH."""
    h = np.asarray(h, dtype=float)
    with np.errstate(over="ignore"):
        taper = 10.0 ** (-beta * (h - h_b))
        dln = LN10 * alpha + LN10 * beta * taper
        ln_n = LN10 * alpha * (h - h_o) - taper
        out = ln_n + np.log(dln)
        out[~np.isfinite(out) | (dln <= 0)] = -np.inf
        return out


# ---------------------------------------------------------------------------
# Figures


def fig_selection_and_scatter():
    m = np.linspace(24.0, 31.5, 800)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    ax = axes[0]
    ax.plot(m, eta(m), color="#1f4e79", lw=2.0, label=r"$\eta(m)$ detection")
    ax.plot(m, s_point(m), color="#c45c26", lw=2.0, ls="--",
            label=r"$S(m)=\eta\cdot\mathrm{track}\cdot\mathrm{char}$")
    ax.plot(m, s_bar(m), color="#2a9d8f", lw=2.0,
            label=r"$\bar S(m_t)$ with phot. scatter")
    ax.axvline(MAG_LIM, color="0.5", ls=":", lw=1)
    ax.set_xlabel(r"apparent magnitude $m_r$")
    ax.set_ylabel("probability")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Survey selection (JWST Sample A .eff)")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ax.plot(m, sigma_m(m), color="#1f4e79", lw=2.0, label=r"$\sigma(m)$")
    ax.plot(m, offset_m(m), color="#c45c26", lw=2.0, label=r"offset$(m)$")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel(r"true magnitude $m_t$")
    ax.set_ylabel("magnitude")
    ax.set_title("Photometric error model from .eff")
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(FIG / "selection_and_scatter.pdf")
    fig.savefig(FIG / "selection_and_scatter.png")
    plt.close(fig)


def fig_likelihood_cartoon():
    """One detection: numerator kernel vs denominator selection on H."""
    m_hat = 27.5
    mu = 14.5  # m - H
    h = np.linspace(4.0, 14.0, 1000)
    m_t = h + mu
    # Approximate kernel as Gaussian for the cartoon (OSSSSim triangular is similar).
    sig = np.maximum(sigma_m(m_t), 0.02)
    centre = m_t + offset_m(m_t)
    kernel = np.exp(-0.5 * ((m_hat - centre) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    num = eta(m_t) * kernel
    den = s_bar(m_t)
    # Population shape for illustration
    lam = np.exp(log_differential(h, 0.6, 0.4, 8.5))
    lam = lam / np.nanmax(lam[np.isfinite(lam)])

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.fill_between(h, 0, den / np.max(den), color="#c45c26", alpha=0.25,
                     label=r"denominator weight $\bar S(m_t)$")
    ax.plot(h, den / np.max(den), color="#c45c26", lw=1.5)
    ax.fill_between(h, 0, num / np.max(num), color="#1f4e79", alpha=0.35,
                     label=r"numerator $\eta(m_t)\,p(\hat m\mid m_t)$")
    ax.plot(h, num / np.max(num), color="#1f4e79", lw=1.8)
    ax.plot(h, lam, color="#2a9d8f", lw=1.8, ls="--",
            label=r"population $\lambda(H)$ (illustrative)")
    h_obs = m_hat - mu
    ax.axvline(h_obs, color="0.3", ls=":", lw=1.2,
               label=fr"no-error $H=\hat m-\mu={h_obs:.1f}$")
    ax.set_xlabel(r"true absolute magnitude $H$")
    ax.set_ylabel("relative weight")
    ax.set_title(r"One detection: what the conditional likelihood averages over")
    ax.set_xlim(h.min(), h.max())
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "likelihood_cartoon.pdf")
    fig.savefig(FIG / "likelihood_cartoon.png")
    plt.close(fig)


def fig_distance_nodes():
    """Explain distance quadrature without Gauss-Hermite jargon."""
    d0, rel = 45.0, 0.05
    # Gauss-Hermite for N(0,1) in the variable √2 σ x for ln d
    x, w = np.polynomial.hermite.hermgauss(7)
    w = w / math.sqrt(math.pi)
    s = math.log1p(rel)
    ln_ratio = math.sqrt(2.0) * s * x
    d = d0 * np.exp(ln_ratio)
    # Magnitude shift if m-H scales as 10 log10(d)
    shift = 10.0 * ln_ratio / LN10

    d_grid = np.linspace(d0 * 0.75, d0 * 1.3, 400)
    # Lognormal density in d: ln d ~ N(ln d0, s^2)
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
    ax.set_xlabel(r"heliocentric distance $d$ (au)")
    ax.set_ylabel("probability density")
    ax.set_title(r"5% distance error at $d=45$ au")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ax.stem(shift, w, linefmt="#1f4e79", markerfmt="o", basefmt=" ")
    ax.set_xlabel(r"shift in $m-H$ (mag)")
    ax.set_ylabel("node weight")
    ax.set_title("How distance error moves absolute magnitude")
    ax.axvline(0, color="0.5", ls=":")
    fig.tight_layout()
    fig.savefig(FIG / "distance_nodes.pdf")
    fig.savefig(FIG / "distance_nodes.png")
    plt.close(fig)


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

    box(0.2, 1.6, 2.2, 1.0, ".eff file\nη, track, mag_lim\nmag_error, phot_frac", "#d6eaf8")
    box(2.8, 1.6, 2.2, 1.0, "measurement\nmodel\nσ(m), offset, noise", "#fdebd0")
    box(5.4, 1.6, 2.0, 1.0, "numerator\nkernel\np(m̂ | m_t)", "#d5f5e3")
    box(5.4, 0.3, 2.0, 1.0, "denominator\nS̄(m_t)", "#fadbd8")
    box(7.8, 0.9, 1.9, 1.2, "conditional\nshape\nlikelihood", "#e8daef")
    box(2.8, 0.3, 2.2, 1.0, "distance nodes\n(optional)\nfrom d ± σ_d", "#fcf3cf")

    for a, b in [((2.4, 2.1), (2.8, 2.1)), ((5.0, 2.1), (5.4, 2.1)),
                 ((5.0, 0.8), (5.4, 0.8)), ((7.4, 2.1), (7.8, 1.6)),
                 ((7.4, 0.8), (7.8, 1.4)), ((5.0, 0.8), (3.9, 0.8))]:
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
    # fix last arrow direction from distance to denominator path - redraw properly
    ax.annotate("", xy=(5.4, 0.8), xytext=(5.0, 0.8),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))

    fig.tight_layout()
    fig.savefig(FIG / "flowchart.pdf")
    fig.savefig(FIG / "flowchart.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Minimal JWST shape fit for the results section


def load_jwst():
    path = JWST_ROOT / "data" / "jwst_sampleA.csv"
    rows = []
    with path.open() as f:
        header = f.readline().strip().split(",")
        idx = {n: i for i, n in enumerate(header)}
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            name = p[idx["name"]]
            m = float(p[idx["m_f150w2"]])
            d = float(p[idx["d_bary"]])
            m_r = m + COLOR_OFFSET
            h = apparent_to_Hr(m, d)
            rows.append((name, m_r, d, h))
    names, m, d, h = zip(*rows)
    return {
        "names": np.array(names),
        "m": np.asarray(m),
        "d": np.asarray(d),
        "h": np.asarray(h),
    }


def build_like(sample, measurement="none", distance_nodes=1, d_rel_err=0.0, dh=0.02):
    """Tabulate num/den on an H grid for each detection (and distance node)."""
    n = len(sample["h"])
    m_hat = sample["m"]
    h_cat = sample["h"]
    d = sample["d"]
    mu0 = m_hat - h_cat

    if distance_nodes > 1 and d_rel_err > 0:
        x, w = np.polynomial.hermite.hermgauss(distance_nodes)
        w = w / math.sqrt(math.pi)
        s = math.log1p(d_rel_err)
        ln_ratio = math.sqrt(2.0) * s * x[None, :]
        shift = 10.0 * ln_ratio / LN10  # (n, J)
        wq = np.broadcast_to(w, (n, distance_nodes)).copy()
    else:
        shift = np.zeros((n, 1))
        wq = np.ones((n, 1))
        distance_nodes = 1

    mu = (mu0[:, None] + shift).ravel()
    det = np.repeat(np.arange(n), distance_nodes)
    m_rows = m_hat[det]

    # Faint limit from selection
    m_grid = np.arange(24.0, 32.0, 0.01)
    if measurement == "none":
        curve = s_point(m_grid)
    else:
        curve = s_bar(m_grid)
    nz = np.flatnonzero(curve > 0)
    m_faint = m_grid[min(nz[-1] + 1, len(m_grid) - 1)]
    top = float((m_faint - mu).max())
    h_min = 2.0
    n_grid = int(math.ceil((top - h_min) / dh)) + 1
    grid = np.linspace(h_min, h_min + (n_grid - 1) * dh, n_grid)
    trap = np.full(n_grid, dh)
    trap[[0, -1]] = 0.5 * dh

    n_rows = len(det)
    den = np.empty((n_rows, n_grid))
    num = np.zeros((n_rows, n_grid))
    h_pt = m_rows - mu
    log_s_pt = np.log(np.maximum(s_point(m_rows), 1e-300))
    point = np.full(n_rows, measurement == "none")

    for r in range(n_rows):
        m_t = grid + mu[r]
        if measurement == "none":
            den[r] = s_point(m_t)
        else:
            den[r] = s_bar(m_t)
            sig = np.maximum(sigma_m(m_t), 2 * dh)
            centre = m_t + offset_m(m_t)
            # cell-averaged triangular CDF approx via erf-like interp of noise CDF
            nodes = np.linspace(-NOISE_HW, NOISE_HW, 161)
            dens = np.maximum(1.0 - np.abs(nodes) / NOISE_HW, 0.0) / NOISE_HW
            cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) *
                                                   (nodes[1] - nodes[0]))])
            cdf = cdf / cdf[-1]
            half = 0.5 * dh
            slope = np.where(m_t > MAG_ERROR[4], 1.0 + MAG_ERROR[5], 1.0)
            lo = np.interp((m_rows[r] - centre - slope * half) / sig, nodes, cdf,
                           left=0.0, right=1.0)
            hi = np.interp((m_rows[r] - centre + slope * half) / sig, nodes, cdf,
                           left=0.0, right=1.0)
            kernel = (hi - lo) / dh
            num[r] = eta(m_t) * kernel  # tc(m_hat) cancels in ratio but keep scale

    return {
        "grid": grid, "trap": trap, "den": den, "num": num, "h_pt": h_pt,
        "log_s_pt": log_s_pt, "point": point, "log_wq": np.log(wq),
        "n": n, "n_nodes": distance_nodes, "sample": sample,
        "measurement": measurement,
    }


def loglike(like, alpha, beta, h_b):
    if alpha <= 0 or beta <= 0:
        return -np.inf
    lg = log_differential(like["grid"], alpha, beta, h_b)
    if not np.any(np.isfinite(lg)):
        return -np.inf
    c = float(np.nanmax(lg[np.isfinite(lg)]))
    lam = np.exp(lg - c)
    lam_w = lam * like["trap"]
    den = like["den"] @ lam_w
    if np.any(den <= 0) or not np.all(np.isfinite(den)):
        return -np.inf
    log_ratio = np.empty(len(den))
    p = like["point"]
    ld = log_differential(like["h_pt"], alpha, beta, h_b)
    log_ratio[p] = ld[p] - c + like["log_s_pt"][p] - np.log(den[p])
    if np.any(~p):
        num = like["num"][~p] @ lam_w
        if np.any(num <= 0):
            return -np.inf
        log_ratio[~p] = np.log(num) - np.log(den[~p])
    if like["n_nodes"] == 1:
        return float(np.sum(log_ratio))
    terms = log_ratio.reshape(like["n"], like["n_nodes"]) + like["log_wq"]
    return float(np.sum(logsumexp(terms, axis=1)))


def fit_mle(like, start=(0.6, 0.4, 8.0)):
    def nll(x):
        v = -loglike(like, *x)
        return 1e30 if not np.isfinite(v) else v

    bounds = [(0.05, 2.0), (0.02, 2.0), (2.0, 14.0)]
    best = None
    for s in (start, (0.4, 0.3, 7.0), (0.8, 0.5, 9.0), (0.5, 0.2, 6.5)):
        res = minimize(nll, s, method="L-BFGS-B", bounds=bounds)
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("MLE failed")
    return best.x, -best.fun


def fig_jwst_results():
    sample = load_jwst()
    RES.mkdir(parents=True, exist_ok=True)

    cases = [
        ("none", 1, 0.0, "no measurement error"),
        ("model", 1, 0.0, "photometric model from .eff"),
        ("model", 7, 0.05, "photometry + 5% distance error"),
    ]
    rows = []
    likes = {}
    for measurement, nodes, drel, label in cases:
        key = f"{measurement}_n{nodes}_d{drel:.2f}"
        like = build_like(sample, measurement=measurement,
                          distance_nodes=nodes, d_rel_err=drel)
        theta, ll = fit_mle(like)
        likes[key] = (like, theta, ll, label)
        rows.append((label, *theta, ll))

    # Write summary table
    lines = [
        "label,alpha,beta,h_b,loglike",
    ]
    for label, a, b, hb, ll in rows:
        lines.append(f"\"{label}\",{a:.4f},{b:.4f},{hb:.4f},{ll:.4f}")
    (RES / "jwst_shape_mle.csv").write_text("\n".join(lines) + "\n")

    # Detected H histogram vs model prediction (no-error case)
    like, theta, ll, label = likes["none_n1_d0.00"]
    grid = like["grid"]
    # predicted detected density: average of den * lambda / norm
    lg = log_differential(grid, *theta)
    lam = np.exp(lg - np.nanmax(lg[np.isfinite(lg)]))
    # Use first node's den rows for each detection
    dens = []
    for k in range(like["n"]):
        r = k * like["n_nodes"]
        w = like["den"][r] * lam * like["trap"]
        dens.append(w / w.sum())
    pred = np.mean(dens, axis=0) / like["trap"]  # per mag

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax = axes[0]
    bins = np.arange(math.floor(sample["h"].min()) - 0.25,
                     math.ceil(sample["h"].max()) + 0.5, 0.5)
    ax.hist(sample["h"], bins=bins, color="#1f4e79", alpha=0.55,
            edgecolor="white", label="JWST Sample A (catalogue $H$)")
    scale = len(sample["h"]) * 0.5
    ax.plot(grid, pred * scale, color="#c45c26", lw=2.0,
            label=fr"shape MLE ($\alpha={theta[0]:.2f}$, "
                  fr"$\beta={theta[1]:.2f}$, $H_B={theta[2]:.1f}$)")
    ax.set_xlabel(r"$H_r$")
    ax.set_ylabel(f"count per {0.5:.1f} mag")
    ax.set_title("Shape-only fit, no measurement error")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    # Likelihood slice in alpha at the no-error MLE's (beta, H_B).
    like0, th0, _ll0, _ = likes["none_n1_d0.00"]
    alphas = np.linspace(0.05, 1.2, 80)
    ll_none = [loglike(like0, a, th0[1], th0[2]) for a in alphas]
    like1 = likes["model_n1_d0.00"][0]
    ll_model = [loglike(like1, a, th0[1], th0[2]) for a in alphas]
    like2 = likes["model_n7_d0.05"][0]
    ll_dist = [loglike(like2, a, th0[1], th0[2]) for a in alphas]
    for y, lab, col in [
        (ll_none, "no error", "#333333"),
        (ll_model, "photometric model", "#1f4e79"),
        (ll_dist, "photometry + 5% distance", "#c45c26"),
    ]:
        y = np.asarray(y)
        y = y - np.nanmax(y[np.isfinite(y)])
        ax.plot(alphas, y, lw=2.0, color=col, label=lab)
    ax.axhline(-0.5 * 1.0, color="0.6", ls=":", lw=1)  # rough 1σ for 1 param
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\ln L - \max\ln L$")
    ax.set_ylim(-8, 0.5)
    ax.set_title(r"Likelihood slice in $\alpha$ (Sample A, $n=20$)")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "jwst_results.pdf")
    fig.savefig(FIG / "jwst_results.png")
    plt.close(fig)

    # Compact comparison table figure not needed; write notes for the tex.
    notes = [
        "JWST Sample A shape-only MLE (docs demo script).",
        "alpha hits the lower bound 0.05 in all three treatments:",
        "with n=20 and most objects near the efficiency drop, the bright-end",
        "slope is not constrained by the conditional likelihood alone.",
        "",
    ]
    for label, a, b, hb, ll in rows:
        notes.append(f"{label}: alpha={a:.4f} beta={b:.4f} Hb={hb:.4f} lnL={ll:.4f}")
    (RES / "jwst_shape_notes.txt").write_text("\n".join(notes) + "\n")

    # Also a magnitude vs H plot
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    sc = ax.scatter(sample["h"], sample["m"], c=sample["d"], cmap="viridis", s=50)
    for name, h, m in zip(sample["names"], sample["h"], sample["m"]):
        ax.annotate(name, (h, m), textcoords="offset points", xytext=(4, 2),
                    fontsize=7, color="0.3")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$d_{\mathrm{bary}}$ (au)")
    ax.set_xlabel(r"catalogue $H_r$")
    ax.set_ylabel(r"apparent $m_r = m_{F150W2}+1$")
    ax.set_title("JWST Sample A detections used in the shape fit")
    # efficiency contour in m
    mline = np.linspace(sample["m"].min() - 0.5, MAG_LIM + 0.2, 200)
    ax.axhline(ETA_M0, color="#c45c26", ls="--", lw=1,
               label=fr"$\eta=50\%$ at $m_r={ETA_M0}$")
    ax.axhline(MAG_LIM, color="0.4", ls=":", lw=1, label=fr"mag_lim $={MAG_LIM}$")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "jwst_sample.pdf")
    fig.savefig(FIG / "jwst_sample.png")
    plt.close(fig)

    return rows


def fig_error_bias_demo():
    """Synthetic: ignoring scatter biases the faint slope."""
    rng = np.random.default_rng(7)
    true = (0.7, 0.35, 8.0)
    # Draw population, apply selection with magran-like noise
    h_grid = np.linspace(4, 12, 20001)
    cum = np.exp(log_cumulative(h_grid, *true))
    cum = (cum - cum[0]) / (cum[-1] - cum[0])
    h_all, m_hat_all = [], []
    while len(h_all) < 800:
        h = np.interp(rng.random(20000), cum, h_grid)
        mu = rng.uniform(13.5, 15.0, size=h.size)
        m_t = h + mu
        # detect on true
        keep = rng.random(h.size) < eta(m_t)
        m_t = m_t[keep]
        h = h[keep]
        # measure
        tri = rng.triangular(-NOISE_HW, 0.0, NOISE_HW, size=m_t.size)
        m_hat = m_t + offset_m(m_t) + sigma_m(m_t) * tri
        keep2 = track_char(m_hat) > 0
        h_all.extend(h[keep2])
        m_hat_all.extend(m_hat[keep2])
    h = np.array(h_all[:800])
    m = np.array(m_hat_all[:800])
    sample = {"names": np.array([str(i) for i in range(len(h))]),
              "m": m, "d": np.full(len(h), 42.0), "h": h}

    like_none = build_like(sample, measurement="none")
    like_model = build_like(sample, measurement="model")
    th_none, ll_none = fit_mle(like_none, start=true)
    th_model, ll_model = fit_mle(like_model, start=true)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    hg = np.linspace(5, 11, 400)
    for th, lab, col, ls in [
        (true, "truth", "#333333", "-"),
        (th_none, fr"ignore scatter ($\alpha={th_none[0]:.2f}$)", "#c45c26", "--"),
        (th_model, fr"with scatter model ($\alpha={th_model[0]:.2f}$)", "#1f4e79", "-"),
    ]:
        y = np.exp(log_differential(hg, *th))
        y = y / np.nanmax(y[np.isfinite(y)])
        ax.plot(hg, y, color=col, ls=ls, lw=2.0, label=lab)
    ax.set_xlabel(r"$H$")
    ax.set_ylabel(r"relative $\lambda(H)$")
    ax.set_title("Synthetic recovery: why the scatter model matters")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "synthetic_bias.pdf")
    fig.savefig(FIG / "synthetic_bias.png")
    plt.close(fig)

    (RES / "synthetic_recovery.txt").write_text(
        f"truth: {true}\n"
        f"none:  {th_none}  logL={ll_none:.3f}\n"
        f"model: {th_model}  logL={ll_model:.3f}\n"
    )


def main():
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    fig_selection_and_scatter()
    fig_likelihood_cartoon()
    fig_distance_nodes()
    fig_flowchart()
    fig_error_bias_demo()
    rows = fig_jwst_results()
    print("Wrote figures to", FIG)
    print("JWST MLE rows:")
    for r in rows:
        print(" ", r)


if __name__ == "__main__":
    main()

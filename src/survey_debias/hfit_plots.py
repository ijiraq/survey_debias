"""Plots for ``survey_debias_hfit``: corner, detected-H check, debiased H."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import hfit

BAND_DRAWS = 200
# Per-survey PIT curves are drawn for this many surveys (inclusive).
MIN_SURVEY_CURVES = 2
MAX_SURVEY_CURVES = 8


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _band_thetas(post, mle_theta, h_o_mle, rng):
    """Posterior draws (alpha, beta, h_b, h_o), or the MLE alone."""
    if post is None:
        return np.array([[*mle_theta, h_o_mle]])
    idx = rng.choice(len(post), size=min(BAND_DRAWS, len(post)), replace=False)
    return post[idx]


def _band(curves):
    curves = np.asarray(curves)
    if len(curves) == 1:
        return None
    return np.percentile(curves, [16, 50, 84], axis=0)


def _shared_ranges(*sets):
    data = np.vstack([s for s in sets if s is not None and len(s)])
    lo, hi = np.percentile(data, [0.5, 99.5], axis=0)
    pad = 0.05 * np.maximum(hi - lo, 1e-9)
    return [(a - p, b + p) for a, b, p in zip(lo, hi, pad)]


def corner_plot(path: Path, post, mle_theta, h_o_mle, boot=None) -> Path:
    """Posterior corner plot; bootstrap replicates overlaid in orange."""
    import corner

    plt = _pyplot()
    names = [*hfit.PARAM_NAMES, "h_o"]
    labels = [hfit.PARAM_LABELS[n] for n in names]
    main = post if post is not None else boot
    has_boot = boot is not None and len(boot) > len(names)
    ranges = _shared_ranges(main, boot if has_boot else None)
    # corner warns through the root logger when a small set has no contours.
    root_log = logging.getLogger()
    level = root_log.level
    root_log.setLevel(logging.ERROR)
    try:
        fig = _corner_figure(corner, post, boot, main, has_boot, labels, ranges,
                             mle_theta, h_o_mle)
    finally:
        root_log.setLevel(level)
    title = "posterior (red: maximum likelihood)" if post is not None else \
        "bootstrap (red: maximum likelihood)"
    if has_boot and post is not None:
        title = ("posterior (black) and bootstrap (orange); "
                 "red: maximum likelihood")
    fig.suptitle(title, y=1.02)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _corner_figure(corner, post, boot, main, has_boot, labels, ranges,
                   mle_theta, h_o_mle):
    # Unit total weight puts both histograms on the same scale.
    fig = corner.corner(
        main, labels=labels, range=ranges, bins=30,
        weights=np.full(len(main), 1.0 / len(main)),
        truths=[*mle_theta, h_o_mle], truth_color="tab:red",
        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3f",
        color="k" if post is not None else "C1",
    )
    if has_boot and post is not None:
        corner.corner(boot, fig=fig, range=ranges, bins=30, color="C1",
                      weights=np.full(len(boot), 1.0 / len(boot)),
                      plot_datapoints=False, plot_density=False)
    return fig


def detected_plot(path: Path, like, mle, thetas) -> Path:
    """Observed H of the detections against the model's prediction for them.

    The prediction is sum_k f_k(H): for each detection's own geometry, the
    H distribution of detected objects implied by the model.
    """
    from scipy import stats

    plt = _pyplot()
    h = like.sample.h
    grid = like.grid
    pred_mle = like.detected_density(*mle.theta).sum(axis=0)
    pred_exp = like.detected_density_exponential(mle.alpha_exp).sum(axis=0)
    band = _band([like.detected_density(*t[:3]).sum(axis=0) for t in thetas])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axes[0]
    width = 0.25
    edges = np.arange(np.floor(h.min() / width) * width,
                      h.max() + width, width)
    counts, _ = np.histogram(h, edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.errorbar(centres, counts / width, yerr=np.sqrt(np.maximum(counts, 1)) / width,
                fmt="o", color="k", ms=4, label=f"detections ({like.n})")
    if band is not None:
        ax.fill_between(grid, band[0], band[2], color="tab:blue", alpha=0.3,
                        label="tapered, 68% posterior")
    ax.plot(grid, pred_mle, color="tab:blue", label="tapered, max likelihood")
    ax.plot(grid, pred_exp, color="tab:orange", ls="--",
            label=f"single exponential (alpha={mle.alpha_exp:.3f})")
    ax.set_xlim(edges[0] - 0.5, max(edges[-1], grid[pred_mle > 1e-3 * pred_mle.max()].max()))
    ax.set_xlabel(f"H ({like.sample.h_column})")
    ax.set_ylabel("detections per mag")
    ax.set_title("detected H versus model prediction")
    ax.legend(fontsize=8)

    pit = like.pit(*mle.theta)
    ax = axes[1]
    ax.hist(pit, bins=10, range=(0, 1), color="0.7", edgecolor="k")
    expect = like.n / 10
    ax.axhline(expect, color="tab:blue")
    ax.axhspan(expect - np.sqrt(expect), expect + np.sqrt(expect),
               color="tab:blue", alpha=0.2)
    ax.set_xlabel("F_k(H_k)  (probability integral transform)")
    ax.set_ylabel("detections")
    ax.set_title("PIT: flat if the model fits")

    ax = axes[2]
    u = np.sort(pit)
    ecdf = np.arange(1, len(u) + 1) / len(u)
    d, p = hfit.ks_uniform(pit)
    crit = stats.kstwo.ppf(0.95, len(u))
    ax.plot([0, 1], [0, 1], color="tab:blue")
    ax.fill_between([0, 1], [-crit, 1 - crit], [crit, 1 + crit],
                    color="tab:blue", alpha=0.2, label="95% KS band")
    ax.step(u, ecdf, where="post", color="k", label=f"all (D={d:.3f}, p={p:.2g})")
    surveys = like.sample.survey
    names = sorted(set(surveys))
    if MIN_SURVEY_CURVES <= len(names) <= MAX_SURVEY_CURVES:
        for j, name in enumerate(names):
            part = np.sort(pit[surveys == name])
            if len(part) < 2:
                continue
            dj, pj = hfit.ks_uniform(part)
            ax.step(part, np.arange(1, len(part) + 1) / len(part), where="post",
                    color=f"C{(j + 1) % 10}", lw=1.0,
                    label=f"{name} ({len(part)}; D={dj:.3f}, p={pj:.2g})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("F_k(H_k)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("PIT empirical CDF")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _parameter_text(mle_t, post) -> str:
    """Max-likelihood values and, with a posterior, medians and 68% ranges."""
    names = [*hfit.PARAM_NAMES, "h_o"]
    if post is None:
        lines = ["max likelihood"]
        lines += [f"{hfit.PARAM_LABELS[n]} = {v:.3f}" for n, v in zip(names, mle_t)]
        return "\n".join(lines)
    lines = ["max likelihood; posterior median, 68%"]
    for j, (n, v) in enumerate(zip(names, mle_t)):
        med, lo, hi = hfit.summarize(post[:, j])
        label = hfit.PARAM_LABELS[n].strip("$")
        lines.append(
            rf"${label} = {v:.3f}$;  ${med:.3f}^{{+{hi:.3f}}}_{{-{lo:.3f}}}$"
        )
    return "\n".join(lines)


def debiased_plot(path: Path, like, ht, mle, h_o_mle, thetas, h_min,
                  post=None) -> Path:
    """Horvitz-Thompson H distribution against the fitted model."""
    plt = _pyplot()
    s = like.sample
    order = np.argsort(s.h)
    h = s.h[order]
    w = 1.0 / s.bias[order]
    cum = np.cumsum(w)
    cum_err = np.sqrt(np.cumsum(w * w))

    h_model = np.linspace(h_min, max(h.max(), ht.h_norm) + 0.5, 400)

    def model_cum(t):
        return (hfit.cumulative(h_model, *t)
                - hfit.cumulative(h_min, *t))

    def model_diff(t):
        return np.exp(hfit.log_differential(h_model, *t))

    mle_t = (*mle.theta, h_o_mle)
    cum_band = _band([model_cum(t) for t in thetas])
    diff_band = _band([model_diff(t) for t in thetas])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.step(h, cum, where="post", color="k", label="HT estimate, sum of 1/bias")
    ax.fill_between(h, np.maximum(cum - cum_err, 0.1 * cum), cum + cum_err,
                    step="post", color="0.6", alpha=0.4)
    if cum_band is not None:
        ax.fill_between(h_model, cum_band[0], cum_band[2], color="tab:blue",
                        alpha=0.3, label="tapered, 68% posterior")
    ax.plot(h_model, model_cum(mle_t), color="tab:blue", label="tapered, max likelihood")
    ax.set_yscale("log")
    ax.set_ylabel(f"N({h_min:.2f} < H' < H)")
    ax.set_title("debiased cumulative H distribution")

    ax = axes[1]
    width = 0.25
    edges = np.arange(np.floor(h.min() / width) * width, h.max() + width, width)
    idx = np.digitize(h, edges) - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    sums = np.bincount(idx, weights=w, minlength=len(centres))[: len(centres)]
    errs = np.sqrt(np.bincount(idx, weights=w * w, minlength=len(centres))[: len(centres)])
    ok = sums > 0
    lower = np.minimum(errs[ok], 0.9 * sums[ok])
    ax.errorbar(centres[ok], sums[ok] / width,
                yerr=np.vstack([lower, errs[ok]]) / width, fmt="o",
                color="k", ms=4, label="HT estimate per mag")
    if diff_band is not None:
        ax.fill_between(h_model, diff_band[0], diff_band[2], color="tab:blue",
                        alpha=0.3, label="tapered, 68% posterior")
    ax.plot(h_model, model_diff(mle_t), color="tab:blue", label="tapered, max likelihood")
    ax.set_yscale("log")
    ax.set_ylabel("dN/dH")
    ax.set_title("debiased differential H distribution")
    # Extra decades at the bottom leave room for the parameter inset.
    axes[0].set_ylim(0.01 * cum[0], 3.0 * cum[-1])
    axes[1].set_ylim(1e-3 * sums[ok].min() / width, 5.0 * sums[ok].max() / width)

    x_lo = max(h_min, h[0] - 0.5)
    inset = _parameter_text(mle_t, post)
    for ax in axes:
        ax.axvspan(ht.h_norm, h_model[-1], color="tab:red", alpha=0.08,
                   label=f"HT incomplete (H > {ht.h_norm:.2f})")
        ax.set_xlabel(f"H ({s.h_column})")
        ax.set_xlim(x_lo, h_model[-1])
        ax.legend(fontsize=8, loc="upper left")
        ax.text(0.97, 0.04, inset, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8.5, linespacing=1.5,
                bbox={"boxstyle": "round", "facecolor": "white",
                      "edgecolor": "0.6", "alpha": 0.9})
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_all(out_dir: Path, like, ht, mle, h_o_mle, post, rng, h_min, boot=None):
    thetas = _band_thetas(post, mle.theta, h_o_mle, rng)
    written = []
    if post is not None or (boot is not None and len(boot) > 4):
        written.append(corner_plot(out_dir / "hfit_corner.png", post, mle.theta,
                                   h_o_mle, boot))
    written.append(detected_plot(out_dir / "hfit_detected.png", like, mle, thetas))
    written.append(debiased_plot(out_dir / "hfit_debiased.png", like, ht, mle,
                                 h_o_mle, thetas, h_min, post))
    return written

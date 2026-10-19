"""``survey_debias_joint``: joint cold-LF fit of several surveys.

Configuration (TOML, paths relative to the file)::

    [joint]
    name = "JWST Sample A + OSSOS_v11, cold"
    root = "../SurveyCharacterizations"    # holds characterization/ and colour.toml
    colour_file = "colour.toml"
    colour_group = "cold"
    model_band = "r"
    eps_tag = "cold"                       # characterization/{survey}/eps_{tag}.csv
    sigma_delta = 0.05                     # prior width of each filter's mean colour
    sigma_c = 0.2                          # intrinsic colour spread
    priority = ["OSSOS_v11", "jwst_sample_a"]
    duplicates = "Duplicates"              # optional, relative to root

    [[survey]]
    name = "jwst_sample_a"
    format = "csv"
    detections = "characterization/jwst_sample_a/jwst_sample_a.csv"
    columns = { mag = "m_f150w2", d_bary = "d_bary", name = "name" }
    select = { comp = "cold" }

    [[survey]]
    name = "OSSOS_v11"
    format = "release"                     # {survey}.detections via detections_io
    cold = { classes = ["cla"], parts = ["m"], i_max = 5.0 }
    exclude_blocks = ["2015BS"]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tomllib
from pathlib import Path

import numpy as np

from . import __version__, detections_io, hfit
from .colours import ColourMap, load_colours
from .grid_bias import geometric_h
from .joint import SIGMA_C, SIGMA_DELTA, JointLikelihood, SurveyTerm, fit_joint, run_mcmc
from .pointings import load_survey_blocks, observer_icrf
from .survey_efficiency import EfficiencyTable


def load_config(path) -> tuple[dict, Path]:
    path = Path(path)
    with path.open("rb") as fh:
        return tomllib.load(fh), path.resolve().parent


def _read_csv(path) -> list[dict]:
    lines = [x for x in Path(path).read_text().splitlines() if x.strip() and not x.startswith("#")]
    return list(csv.DictReader(lines))


def survey_rows(spec: dict, root: Path, char_root: Path, blocks: dict) -> list[dict]:
    """(name, block, mag, d_bary, d_err, h, i) rows of one survey's sample."""
    name = spec["name"]
    fmt = spec.get("format", "csv")
    if fmt == "release":
        cold = dict(spec.get("cold", {}))
        cut = detections_io.ColdCut(tuple(cold.get("classes", ("cla",))),
                                    tuple(cold.get("parts", ("m",))),
                                    float(cold.get("i_max", 5.0)),
                                    bool(cold.get("secure_only", False)))
        raw = detections_io.convert(char_root, name, cut, spec.get("detections") and
                                    root / spec["detections"])
        rows = [{"name": r["name"], "block": r["block"], "mag": float(r["mag"]),
                 "d_bary": float(r["d_bary"]), "d_err": math.nan, "h": float(r["Hsur"]),
                 "i": float(r["i"])}
                for r in raw if r["comp"] == "cold"]
    elif fmt == "csv":
        cols = {"mag": "mag", "d_bary": "d_bary", "name": "name", "block": "block",
                "d_bary_err": "d_bary_err", "h": None, "i": "i", **spec.get("columns", {})}
        select = spec.get("select", {})
        rows = []
        for r in _read_csv(root / spec["detections"]):
            if any(str(r.get(k, "")) != str(v) for k, v in select.items()):
                continue
            block = r.get(cols["block"]) or spec.get("block") or (
                next(iter(blocks)) if len(blocks) == 1 else None)
            if block is None:
                raise ValueError(f"{name}: no block column and several blocks")
            err = r.get(cols["d_bary_err"], "")
            h = r.get(cols["h"], "") if cols["h"] else ""
            inc = r.get(cols["i"], "")
            rows.append({"name": r[cols["name"]], "block": block,
                         "mag": float(r[cols["mag"]]), "d_bary": float(r[cols["d_bary"]]),
                         "d_err": float(err) if err not in ("", None) else math.nan,
                         "h": float(h) if h not in ("", None) else math.nan,
                         "i": float(inc) if inc not in ("", None) else math.nan})
    else:
        raise ValueError(f"{name}: format must be 'csv' or 'release', not {fmt!r}")
    excluded = set(spec.get("exclude_blocks", []))
    return [r for r in rows if r["block"] not in excluded]


def _pointing_h(row: dict, block) -> float:
    """H in the block's filter from its first pointing's direction, epoch and observer."""
    tile = block.tiles[0][0]
    obs = observer_icrf(block.char_dirs[0], tile.observer, tile.jd)
    return geometric_h(row["mag"], row["d_bary"], tile.ra_deg, tile.dec_deg, obs)


def build_terms(cfg: dict, base: Path, sigma_c=None, sigma_delta=None
                ) -> tuple[JointLikelihood, dict]:
    j = cfg["joint"]
    root = (base / j.get("root", ".")).resolve()
    char_root = root / "characterization"
    model_band = j.get("model_band", "r")
    photspec, _ = load_colours(root / j.get("colour_file", "colour.toml"))
    colours = ColourMap(photspec, model_band)
    group = j.get("colour_group", "cold")
    tag = j.get("eps_tag", "cold")
    specs = cfg["survey"]
    priority = list(j.get("priority", [s["name"] for s in specs]))

    blocks, rows = {}, {}
    for spec in specs:
        name = spec["name"]
        b = load_survey_blocks(char_root, name, use_ossssim=False)
        blocks[name] = {k: v for k, v in b.items() if k not in set(spec.get("exclude_blocks", []))}
        rows[name] = survey_rows(spec, root, char_root, blocks[name])
    dropped = {}
    if j.get("duplicates"):
        groups = detections_io.read_duplicates(root / j["duplicates"])
        drop = detections_io.drop_duplicates({s: [r["name"] for r in v] for s, v in rows.items()},
                                             groups, priority)
        for s, names in drop.items():
            if names:
                rows[s] = [r for r in rows[s] if r["name"] not in names]
                dropped[s] = sorted(names)

    terms, info = [], {"root": str(root), "dropped_duplicates": dropped, "surveys": {}}
    for spec in specs:
        name = spec["name"]
        rs = rows[name]
        if not rs:
            raise ValueError(f"{name}: no detections in the sample")
        filt = {v.filter for v in blocks[name].values()}
        if len(filt) != 1:
            raise ValueError(f"{name}: blocks use several filters {sorted(filt)}")
        filt = filt.pop()
        m = np.array([r["mag"] for r in rs])
        d = np.array([r["d_bary"] for r in rs])
        # H in the survey's filter: the catalogue's own H (release Hsur, or a CSV h
        # column) where given, else from the block's pointing and observer. Not
        # apparent_to_Hr: r = Delta = d (phase ~1/d rad) is ~0.2 mag off near opposition.
        h_cat = np.array([r["h"] for r in rs])
        h = np.array([hc if np.isfinite(hc) else _pointing_h(r, blocks[name][r["block"]])
                      for hc, r in zip(h_cat, rs)])
        sample = hfit.HSample(
            names=np.array([r["name"] for r in rs]), h=h, m=m, bias=np.ones(len(rs)),
            rate=hfit.opposition_rate(d), h_column=f"H_{filt}", source=name,
            group=np.array([f"{name}/{r['block']}" for r in rs], dtype=object), d=d,
            d_err=np.array([r["d_err"] for r in rs]),
        )
        selections = {f"{name}/{b}": blk.selection for b, blk in blocks[name].items()}
        like = hfit.ConditionalLikelihood(sample, selections, float(h.min()) - 1.0,
                                          measurement=spec.get("measurement", "model"))
        eps = EfficiencyTable.load(char_root / name / f"eps_{tag}.csv")
        if eps.filter != filt:
            raise ValueError(f"{name}: eps_{tag}.csv is in {eps.filter!r}, blocks in {filt!r}")
        excl = set(spec.get("exclude_blocks", []))
        eps_excl = set(x for x in eps.meta.get("excluded_blocks", "none").split(",") if x != "none")
        if excl != eps_excl:
            raise ValueError(f"{name}: fit excludes {sorted(excl)} but eps_{tag}.csv excludes "
                             f"{sorted(eps_excl)}; rerun survey_debias_eff --exclude-block")
        term = SurveyTerm(name, filt, like, eps, colours.colour(filt, group),
                          inc=np.array([r["i"] for r in rs])[like.index])
        terms.append(term)
        info["surveys"][name] = {"filter": filt, "n": like.n, "colour": term.colour,
                                 "h_source": "catalogue" if np.all(np.isfinite(h_cat))
                                 else "geometric (block pointing and observer)",
                                 "h_range": [float(h.min()), float(h.max())],
                                 "blocks": sorted(blocks[name]), "eps": eps.meta}
    rank = {s: k for k, s in enumerate(priority)}
    for term in terms:
        for other in terms:
            if rank.get(other.name, 99) >= rank.get(term.name, 99):
                continue
            for a, b in ((other.name, term.name), (term.name, other.name)):
                path = char_root / a / f"eps_{tag}_x_{b}.csv"
                if path.is_file():
                    term.overlaps.append((other.name, EfficiencyTable.load(path)))
                    info["surveys"][term.name].setdefault("overlaps", []).append(str(path))
                    break
    like = JointLikelihood(
        terms,
        sigma_delta=float(sigma_delta if sigma_delta is not None else j.get("sigma_delta", SIGMA_DELTA)),
        sigma_c=float(sigma_c if sigma_c is not None else j.get("sigma_c", SIGMA_C)),
        model_band=model_band,
    )
    return like, info


def summarize_counts(like: JointLikelihood, theta) -> dict:
    shape, h_o, delta = like.unpack(theta)
    return {t.name: {"n": t.n, "expected": like.expected(t, shape, h_o, delta)[0]}
            for t in like.surveys}


def _quantiles(x):
    q = np.percentile(x, [16, 50, 84])
    return {"median": float(q[1]), "minus": float(q[1] - q[0]), "plus": float(q[2] - q[1])}


HT_DR, HT_DI = 1.0, 0.5   # (r, i) window around a detection for its bias [au, deg]


def _cut_from_meta(text: str):
    """SampleCut from an eps table's ``cut`` description (survey_efficiency.SampleCut)."""
    import re

    from .survey_efficiency import SampleCut

    cut = SampleCut()
    m = re.search(r"i < ([0-9.]+) deg", text or "")
    if m:
        cut.i_max = float(m.group(1))
    m = re.search(r"([0-9.]+) < r < ([0-9.]+) au", text or "")
    if m:
        cut.r_range = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"([0-9.]+) < a < ([0-9.]+) au", text or "")
    if m:
        cut.a_range = (float(m.group(1)), float(m.group(2)))
    return cut


def debiased_cumulative(like: JointLikelihood, char_root: Path, theta, tag: str = "cold"
                        ) -> dict:
    """Per survey: (H_r, weight) with weight = 1 / (union bias x cut fraction).

    A detection's bias is P(found by any block of its survey | its r, i, H) for
    the cold model, from the survey's plant table (plants within HT_DR au and
    HT_DI deg, weighted by w 1(in field) S_bar). sum 1/bias counts the cold
    objects inside the survey's cut; dividing by the model fraction of cold
    objects that pass the cut makes each survey estimate the whole population,
    like N(<H). H_r is the catalogue H less the survey's fitted mean colour.
    """
    from .survey_efficiency import PlantTable

    _, _, delta = like.unpack(theta)
    out = {}
    for term in like.surveys:
        path = char_root / term.name / f"plants_{tag}.npz"
        if not path.is_file() or term.inc is None or not np.all(np.isfinite(term.inc)):
            continue
        plants = PlantTable.load(path)
        frac = float(_cut_from_meta(term.eps.meta.get("cut", "")).mask(plants).mean())
        s = term.like.sample
        groups = list(term.like.selections.items())
        by_block = {g: plants.block == g.split("/", 1)[1] for g, _ in groups}
        weights = np.full(len(s), np.nan)
        for k in range(len(s)):
            near = (np.abs(plants.r - s.d[k]) < HT_DR) & (np.abs(plants.i - term.inc[k]) < HT_DI)
            bias = 0.0
            for g, sel in groups:
                window = near & by_block[g]
                n = int(window.sum())
                use = window & plants.in_field
                if n == 0 or not use.any():
                    continue
                m_t = s.h[k] + plants.mu[use]
                rate = np.round(plants.rate[use], 1)
                total = 0.0
                for rv in np.unique(rate):
                    pick = rate == rv
                    total += float(np.sum(plants.w[use][pick]
                                          * sel.measured_selection(m_t[pick], float(rv))))
                bias += total / n
            if bias > 0.0:
                weights[k] = 1.0 / (bias * frac)
        colour = like.colour_of(term, delta)
        out[term.name] = (s.h - (0.0 if term.filter == like.model_band else colour), weights)
    return out


def write_plots(out_dir: Path, like: JointLikelihood, fit, samples, debiased=None
                ) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    n = len(like.surveys)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.8), squeeze=False)
    for ax, term in zip(axes[0], like.surveys):
        h = term.like.sample.h
        bins = np.arange(math.floor(h.min()) - 0.5, math.ceil(h.max()) + 1.0, 0.5)
        ax.hist(h, bins=bins, color="#1f4e79", alpha=0.55, edgecolor="white",
                label=f"{term.name} (n={term.n})")
        hh, dens = like.detected_density(term, fit.theta)
        per_bin = [np.trapezoid(dens[(hh >= lo) & (hh < hi)], hh[(hh >= lo) & (hh < hi)])
                   for lo, hi in zip(bins[:-1], bins[1:])]
        ax.step(bins[:-1], per_bin, where="post", color="#c45c26", lw=2.0, label="expected (MLE)")
        ax.set_xlabel(f"catalogue $H_{{{term.filter}}}$")
        ax.set_ylabel("detections per 0.5 mag")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = out_dir / "joint_detected.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    prof = np.asarray(fit.profile_loglike, dtype=float)
    ok = np.isfinite(prof) & (prof > -1e299)
    if like.form == "tapered" and ok.any():
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        alpha = np.asarray(fit.profile_alpha, dtype=float)
        rel = prof - prof[ok].max()
        if not ok.all():
            # Profiles from fits with the old H_o box (-30, 40): the small-alpha
            # points next to the failed ones are pulled down by the box too.
            ok &= np.cumsum(ok & (rel > -3.0)) > 0
            ax.axvspan(-0.03, alpha[ok].min(), color="0.85", lw=0,
                       label=r"cut by the H$_o$ bound of this fit; refit to fill")
        ax.plot(alpha[ok], rel[ok], "o-", color="#1f4e79", lw=2.0, ms=3)
        ax.axvline(like.bounds()["alpha"][0], color="0.4", lw=1, ls="--",
                   label=r"$\alpha$ lower bound (current code)")
        ax.axhline(-0.5, color="0.6", ls=":")
        ax.set_xlim(-0.03, 0.7)
        ax.set_ylim(-3.0, 0.5)
        ax.set_xlabel(r"faint-end slope $\alpha$")
        ax.set_ylabel(r"profile $\ln L - \max\ln L$")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        path = out_dir / "joint_profile_alpha.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    lo, hi = like.h_r_range
    hg = np.linspace(lo - 1.0, hi + 0.5, 300)

    def cum(theta):
        shape, h_o, _ = like.unpack(theta)
        if like.form == "exponential":
            return 10.0 ** (shape[0] * (hg - h_o))
        return hfit.cumulative(hg, *shape, h_o)

    if samples is not None and len(samples):
        pick = samples[np.random.default_rng(0).choice(len(samples), min(400, len(samples)),
                                                       replace=False)]
        curves = np.array([cum(t) for t in pick])
        q = np.percentile(curves, [16, 84], axis=0)
        ax.fill_between(hg, q[0], q[1], color="#1f4e79", alpha=0.25, label="68% posterior")
    ax.plot(hg, cum(fit.theta), color="#1f4e79", lw=2.0, label="MLE")
    markers = "osD^v<>ph"
    for k, (name, (h, w)) in enumerate((debiased or {}).items()):
        good = np.isfinite(w)
        if not good.any():
            continue
        order = np.argsort(h[good])
        ax.step(h[good][order], np.cumsum(w[good][order]), where="post", lw=1.0,
                marker=markers[k % len(markers)], ms=3, alpha=0.8,
                label=f"{name} (sum 1/bias, n={int(good.sum())})")
    ax.set_yscale("log")
    ax.set_xlabel(f"$H_{{{like.model_band}}}$")
    ax.set_ylabel(r"$N(<H)$, whole cold population")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    path = out_dir / "joint_cumulative.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="survey_debias_joint",
        description="Joint fit of the cold H distribution to several surveys: conditional "
                    "shape terms per detection plus Poisson counts per survey from "
                    "population efficiencies (survey_debias_eff), with per-filter colour "
                    "offsets and an intrinsic colour spread.",
    )
    p.add_argument("--config", required=True, help="joint TOML file")
    p.add_argument("--out-dir", default=None, help="default: <config dir>/joint_<stem>")
    p.add_argument("--form", choices=("tapered", "exponential"), default="tapered")
    p.add_argument("--sigma-c", type=float, default=None, help="override the colour spread")
    p.add_argument("--sigma-delta", type=float, default=None,
                   help="override the prior width of the mean colours")
    p.add_argument("--walkers", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mcmc", action="store_true")
    p.add_argument("--replot", action="store_true",
                   help="only redraw the plots from joint_summary.json and joint_samples.csv")
    p.add_argument("--no-debiased", action="store_true",
                   help="leave the sum 1/bias points off the cumulative plot")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)

    cfg, base = load_config(args.config)
    like, info = build_terms(cfg, base, args.sigma_c, args.sigma_delta)
    like.form = args.form
    out_dir = Path(args.out_dir) if args.out_dir else base / f"joint_{Path(args.config).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    char_root = Path(info["root"]) / "characterization"
    tag = cfg["joint"].get("eps_tag", "cold")
    if args.replot:
        return replot(out_dir, like, char_root, tag, not args.no_debiased)
    for t in like.surveys:
        print(f"{t.name}: {t.n} detections in {t.filter}, colour {t.colour:+.2f}, "
              f"H_{t.filter} {t.like.sample.h.min():.2f}-{t.like.sample.h.max():.2f}"
              + (f", overlaps with {[o for o, _ in t.overlaps]}" if t.overlaps else ""),
              flush=True)

    fit = fit_joint(like)
    other = JointLikelihood(like.surveys, like.sigma_delta, like.sigma_c,
                            "exponential" if like.form == "tapered" else "tapered",
                            like.model_band)
    alt = fit_joint(other)
    taper, expo = (fit, alt) if like.form == "tapered" else (alt, fit)
    d_lnl = taper.loglike - expo.loglike
    print(f"MLE ({like.form}): " + ", ".join(f"{n}={v:.4f}" for n, v in zip(like.names, fit.theta))
          + f"  lnL={fit.loglike:.3f}", flush=True)

    summary = {
        "name": cfg["joint"].get("name", Path(args.config).stem), "form": like.form,
        "config": str(Path(args.config).resolve()), **info,
        "sigma_c": like.sigma_c, "sigma_delta": like.sigma_delta,
        "parameters": like.names,
        "mle": dict(zip(like.names, map(float, fit.theta))), "loglike": fit.loglike,
        "counts_mle": summarize_counts(like, fit.theta),
        "tapered_vs_exponential": {
            "tapered": dict(zip(like.names, map(float, taper.theta))),
            "exponential": {"alpha": float(expo.theta[0]), "h_o": float(expo.theta[3]),
                            **{n: float(v) for n, v in zip(like.names[4:], expo.theta[4:])}},
            "delta_loglike": d_lnl, "delta_aic_exp_minus_tapered": 2 * (-2) + 2 * d_lnl,
        },
    }
    if like.form == "tapered":
        rel = fit.profile_loglike - np.nanmax(fit.profile_loglike)
        inside = fit.profile_alpha[rel >= -0.5]
        summary["alpha_profile"] = {"alpha": fit.profile_alpha.tolist(),
                                    "loglike": fit.profile_loglike.tolist(),
                                    "within_0p5": [float(inside.min()), float(inside.max())]}
    samples = None
    if not args.no_mcmc:
        samples, logp, acc = run_mcmc(like, fit.theta, args.walkers, args.steps, args.burn,
                                      args.seed)
        summary["posterior"] = {n: _quantiles(samples[:, k]) for k, n in enumerate(like.names)
                                if not (like.form == "exponential" and n in ("beta", "h_b"))}
        summary["mcmc"] = {"walkers": args.walkers, "steps": args.steps, "burn": args.burn,
                           "acceptance": acc, "n_samples": int(len(samples))}
        np.savetxt(out_dir / "joint_samples.csv", np.column_stack([samples, logp]),
                   delimiter=",", header=",".join(like.names + ["loglike"]), comments="")
    (out_dir / "joint_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = summary_text(summary)
    (out_dir / "joint_summary.txt").write_text(text)
    print(text)
    debiased = None if args.no_debiased else debiased_cumulative(like, char_root, fit.theta, tag)
    for path in write_plots(out_dir, like, fit, samples, debiased):
        print(f"wrote {path}")
    return 0


def replot(out_dir: Path, like: JointLikelihood, char_root: Path, tag: str,
           with_debiased: bool = True) -> int:
    """Redraw the plots of an earlier run without fitting."""
    from types import SimpleNamespace

    summary = json.loads((out_dir / "joint_summary.json").read_text())
    if summary["parameters"] != like.names:
        raise ValueError(f"{out_dir}: parameters {summary['parameters']} do not match the "
                         f"configuration's {like.names}")
    like.form = summary.get("form", like.form)
    theta = np.array([summary["mle"][n] for n in like.names])
    prof = summary.get("alpha_profile", {})
    fit = SimpleNamespace(theta=theta, profile_alpha=np.array(prof.get("alpha", [])),
                          profile_loglike=np.array(prof.get("loglike", [])))
    samples = None
    path = out_dir / "joint_samples.csv"
    if path.is_file():
        samples = np.loadtxt(path, delimiter=",", skiprows=1)[:, : len(like.names)]
    debiased = debiased_cumulative(like, char_root, theta, tag) if with_debiased else None
    for name, (_, w) in (debiased or {}).items():
        print(f"{name}: sum 1/bias over {int(np.isfinite(w).sum())} detections = "
              f"{np.nansum(w):.4g}", flush=True)
    for p in write_plots(out_dir, like, fit, samples, debiased):
        print(f"wrote {p}")
    return 0


def summary_text(s: dict) -> str:
    lines = [f"Joint cold H-distribution fit: {s['name']} ({s['form']})",
             f"  sigma_c = {s['sigma_c']:g}, sigma_delta = {s['sigma_delta']:g}", ""]
    for name, v in s["surveys"].items():
        c = s["counts_mle"][name]
        lines.append(f"  {name}: {v['n']} detections in {v['filter']} (colour {v['colour']:+.2f}), "
                     f"H {v['h_range'][0]:.2f}-{v['h_range'][1]:.2f}; expected at MLE "
                     f"{c['expected']:.1f}")
    if s["dropped_duplicates"]:
        lines.append(f"  duplicates dropped: {s['dropped_duplicates']}")
    lines += ["", "maximum likelihood:",
              "  " + "  ".join(f"{k} = {v:.4f}" for k, v in s["mle"].items()),
              f"  ln L = {s['loglike']:.3f}"]
    t = s["tapered_vs_exponential"]
    lines.append(f"  single exponential: alpha = {t['exponential']['alpha']:.4f}, "
                 f"H_o = {t['exponential']['h_o']:.3f}; delta ln L (tapered - exp) = "
                 f"{t['delta_loglike']:.2f}, delta AIC (exp - tapered) = "
                 f"{t['delta_aic_exp_minus_tapered']:.2f}")
    if "alpha_profile" in s:
        a, b = s["alpha_profile"]["within_0p5"]
        lines.append(f"  alpha profile: ln L within 0.5 of the peak for alpha {a:.3f} to {b:.3f}")
    if "posterior" in s:
        lines += ["", "posterior (median, 68%):"]
        for k, q in s["posterior"].items():
            lines.append(f"  {k} = {q['median']:.4f} -{q['minus']:.4f} +{q['plus']:.4f}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

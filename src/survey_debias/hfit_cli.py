"""``survey_debias_hfit``: fit the tapered H distribution to debiased detections.

Reads the ``bias_results.csv`` written by ``survey_debias``, each
detection's ``characterization/{survey}/{block}.eff`` selection, and the
per-survey bias files; finds the maximum-likelihood (alpha, beta, H_B),
samples the posterior with emcee, sets H_o from the Horvitz-Thompson count,
optionally bootstraps, and writes a summary, the samples, a corner plot, and
data-versus-model plots. See :mod:`survey_debias.hfit` for the likelihood.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import __version__
from . import hfit
from .survey_config import load_survey


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="survey_debias_hfit",
        description="Maximum-likelihood and emcee fit of the exponentially "
                    "tapered H distribution, N(<H) = 10^(alpha (H - H_o)) "
                    "exp(-10^(-beta (H - H_B))), to debiased detections.",
    )
    p.add_argument("--survey", required=True,
                   help="Survey definition (same as survey_debias --survey).")
    p.add_argument("--root", default=".",
                   help="Survey directory (default: current directory).")
    p.add_argument("--results", default=None,
                   help="Results CSV (default: <root>/<results_name>).")
    p.add_argument("--h-column", default=None,
                   help="Column holding H (default: Hx, or Hx_computed).")
    p.add_argument("--h-min", type=float, default=None,
                   help="Bright limit of the fit (default: 1 mag brighter "
                        "than the brightest detection).")
    p.add_argument("--h-max", type=float, default=None,
                   help="Faint limit of the fit (default: none; the "
                        "survey selection sets it).")
    p.add_argument("--h-norm", type=float, default=None,
                   help="Normalise H_o with detections brighter than this "
                        "(default: faintest H every detected geometry sees "
                        "at half of peak selection).")
    p.add_argument("--eff-file", action="append", default=None,
                   help="Efficiency file for every block; repeat once per epoch "
                        "(default: characterization/{survey}/[epoch{i}/]{block}.eff "
                        "for each detection).")
    p.add_argument("--efficiency", default=None,
                   help="Efficiency for every block and epoch instead of files, "
                        "e.g. single:0.95,29.0,0.2[,mag_lim].")
    p.add_argument("--bootstrap", type=int, default=0, metavar="N",
                   help="Bootstrap replicates: resample within each block, "
                        "perturb every bias estimate by its MC error, refit "
                        "(default 0: none).")
    p.add_argument("--bias-rel-err", type=float, default=None,
                   help="Relative error for bias estimates with no recorded MC "
                        "error or detection count (legacy bias files; default: "
                        "treat them as exact).")
    p.add_argument("--rate", type=float, default=None,
                   help="On-sky rate [\"/hr] for rate-dependent efficiencies "
                        "(default: opposition rate from d_bary).")
    p.add_argument("--prior", action="append", default=[], metavar="NAME=LO,HI",
                   help="Uniform prior bounds for alpha, beta, or h_b.")
    p.add_argument("--walkers", type=int, default=32)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--thin", type=int, default=5)
    p.add_argument("--ppc-draws", type=int, default=300,
                   help="Posterior draws for the predictive check (default 300).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-mcmc", action="store_true",
                   help="Maximum likelihood only; no emcee, no corner plot.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <root>/hfit).")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _parse_priors(items) -> dict:
    out = {}
    for item in items:
        name, _, rng = item.partition("=")
        try:
            lo, hi = (float(v) for v in rng.split(","))
        except ValueError:
            raise ValueError(f"--prior {item!r}: use NAME=LO,HI") from None
        out[name.strip()] = (lo, hi)
    return out


def _fmt(med, lo, hi) -> str:
    return f"{med:.3f} -{lo:.3f} +{hi:.3f}"


def run(args) -> int:
    survey = load_survey(args.survey)
    root = Path(args.root)
    results = Path(args.results) if args.results else root / survey.results_name
    out_dir = Path(args.out_dir) if args.out_dir else root / "hfit"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from .pointings import Project

    project = Project.for_survey(root, survey, use_ossssim=False)
    sample = hfit.load_hsample(results, survey, args.h_column, args.rate)
    selections = hfit.block_selections(project, sample, args.eff_file, args.efficiency)
    h_min = args.h_min
    if h_min is None:
        h_min = math.floor(10.0 * (sample.h.min() - 1.0)) / 10.0
    like = hfit.ConditionalLikelihood(sample, selections, h_min, args.h_max)
    priors = hfit.default_priors(like.sample.h, h_min, _parse_priors(args.prior))
    h_norm = args.h_norm if args.h_norm is not None else float(like.h_half_k.min())
    table = hfit.bias_table(sample, project, survey.bias_method, survey.surveys,
                            args.bias_rel_err)
    ht = hfit.ht_count(like.sample, h_min, h_norm, rel_err_fallback=args.bias_rel_err)
    notes = []
    if ht.n_used < 5:
        notes.append(
            f"only {ht.n_used} detections are brighter than h_norm={h_norm:.2f}; "
            "the H_o normalisation is poorly determined (set --h-norm)"
        )
    if "no MC error" in table.note:
        notes.append(table.note.split("; ", 1)[-1])

    print(f"{like.n} detections in H [{h_min:.2f}, "
          f"{args.h_max if args.h_max is not None else 'selection limit'}] "
          f"({like.excluded} outside), H column {sample.h_column}, "
          f"bias mode {sample.bias_mode}", flush=True)
    for line in like.describe_selection():
        print(f"  {line}", flush=True)

    mle = hfit.fit_mle(like, priors)
    a, b, hb = mle.theta
    h_o_mle = float(hfit.h_o_for((a, b, hb), h_min, h_norm, ht.count))
    k_taper, k_exp = 3, 1
    d_lnl = mle.loglike - mle.loglike_exp
    aic = (2 * k_taper - 2 * mle.loglike, 2 * k_exp - 2 * mle.loglike_exp)
    bic = (k_taper * math.log(like.n) - 2 * mle.loglike,
           k_exp * math.log(like.n) - 2 * mle.loglike_exp)
    from scipy import stats
    lrt_p = float(stats.chi2.sf(max(2.0 * d_lnl, 0.0), df=k_taper - k_exp))
    ks_d, ks_p = hfit.ks_uniform(like.pit(a, b, hb))
    groups = hfit.group_diagnostics(like, mle.theta)
    print(f"MLE: alpha={a:.4f} beta={b:.4f} H_B={hb:.3f} H_o={h_o_mle:.3f} "
          f"lnL={mle.loglike:.2f}", flush=True)

    summary = {
        "results": str(results),
        "h_column": sample.h_column,
        "n_detections": like.n,
        "n_excluded": like.excluded,
        "h_min": h_min,
        "h_max": args.h_max,
        "bias_mode": sample.bias_mode,
        "selection": like.describe_selection(),
        "groups": groups,
        "rate": sample.rate_note if like.rate_dependent else "",
        "priors": {k: list(v) for k, v in priors.bounds.items()},
        "mle": {"alpha": a, "beta": b, "h_b": hb, "h_o": h_o_mle,
                "loglike": mle.loglike},
        "exponential_mle": {"alpha": mle.alpha_exp, "loglike": mle.loglike_exp},
        "model_comparison": {
            "delta_loglike": d_lnl,
            "aic_tapered": aic[0], "aic_exponential": aic[1],
            "bic_tapered": bic[0], "bic_exponential": bic[1],
            "lrt_p_value": lrt_p,
        },
        "ks_pit_mle": {"statistic": ks_d, "p_value": ks_p},
        "normalisation": {"h_norm": h_norm, "ht_count": ht.count,
                          "ht_sigma": ht.sigma, "n_used": ht.n_used,
                          "ht_sigma_sampling": ht.sigma_sampling,
                          "ht_sigma_bias": ht.sigma_bias,
                          "mode": ht.mode, "per_block": ht.per_block,
                          "bias_estimates": table.note},
    }

    mcmc = None
    post = None
    if not args.no_mcmc:
        print(f"emcee: {args.walkers} walkers x {args.steps} steps "
              f"(burn {args.burn}, thin {args.thin})", flush=True)
        mcmc = hfit.run_mcmc(like, priors, mle.theta, args.walkers,
                             args.steps, args.burn, args.thin, args.seed)
        counts = hfit.sample_counts(ht, len(mcmc.samples), rng)
        h_o = hfit.h_o_for(mcmc.samples.T, h_min, h_norm, counts)
        post = np.column_stack([mcmc.samples, h_o])
        ppp = hfit.posterior_predictive_ks(like, mcmc.samples, args.ppc_draws, rng)
        notes += hfit.prior_edge_warnings(mcmc.samples, priors)
        notes += hfit.taper_warnings(mcmc.samples, like.sample.h)
        notes += hfit.warn_if_short(mcmc)
        names = list(hfit.PARAM_NAMES) + ["h_o"]
        summary["posterior"] = {
            n: dict(zip(("median", "minus", "plus"), hfit.summarize(post[:, j])))
            for j, n in enumerate(names)
        }
        summary["mcmc"] = {
            "walkers": args.walkers, "steps": args.steps, "burn": args.burn,
            "thin": args.thin, "n_samples": int(len(post)),
            "acceptance": mcmc.acceptance,
            "autocorr": None if mcmc.autocorr is None else list(map(float, mcmc.autocorr)),
        }
        summary["posterior_predictive_ks_p"] = ppp
        np.savetxt(
            out_dir / "hfit_samples.csv",
            np.column_stack([post, mcmc.log_like]), delimiter=",",
            header="alpha,beta,h_b,h_o,loglike", comments="",
        )
    boot = None
    if args.bootstrap > 0:
        print(f"bootstrap: {args.bootstrap} replicates (resample within block, "
              "lognormal bias perturbations, warm-started MLE)", flush=True)
        boot = hfit.bootstrap(like, sample, table, priors, mle.theta, args.bootstrap,
                              rng, h_min, h_norm, progress=max(args.bootstrap // 10, 1))
        names = list(hfit.PARAM_NAMES) + ["h_o"]
        if len(boot.samples):
            summary["bootstrap"] = {
                "replicates": int(len(boot.samples)),
                "failed": boot.n_failed,
                "ht_count": dict(zip(("median", "minus", "plus"),
                                     hfit.summarize(boot.count))),
                **{n: dict(zip(("median", "minus", "plus"),
                               hfit.summarize(boot.samples[:, j])))
                   for j, n in enumerate(names)},
            }
            np.savetxt(
                out_dir / "hfit_bootstrap.csv",
                np.column_stack([boot.samples, boot.count]), delimiter=",",
                header="alpha,beta,h_b,h_o,ht_count", comments="",
            )
        if boot.n_failed:
            notes.append(f"bootstrap: {boot.n_failed} of {boot.n_requested} "
                         "replicates failed and were dropped")
        if post is not None and len(boot.samples):
            notes += hfit.bootstrap_width_warnings(boot.samples, post)
    summary["notes"] = notes

    (out_dir / "hfit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = _summary_text(summary, survey.name)
    (out_dir / "hfit_summary.txt").write_text(text)
    print(text)

    from . import hfit_plots
    written = hfit_plots.write_all(out_dir, like, ht, mle, h_o_mle, post, rng, h_min,
                                   boot=None if boot is None else boot.samples)
    for path in written:
        print(f"wrote {path}")
    return 0


def _summary_text(s: dict, survey_name: str) -> str:
    m = s["mle"]
    c = s["model_comparison"]
    lines = [
        f"Tapered H-distribution fit: {survey_name}",
        "  N(<H) = 10^(alpha (H - H_o)) exp(-10^(-beta (H - H_B)))",
        f"data: {s['results']} ({s['n_detections']} detections, "
        f"{s['n_excluded']} outside the fit range), H column {s['h_column']}",
        f"fit range: H >= {s['h_min']:.2f}"
        + (f", H <= {s['h_max']:.2f}" if s["h_max"] is not None else ""),
        "selection by survey/block (per epoch, multiplied):",
        *[f"  {line}" for line in s["selection"]],
    ]
    if s["rate"]:
        lines.append(f"  {s['rate']}")
    lines += [
        "",
        "maximum likelihood (shape from the conditional likelihood):",
        f"  alpha = {m['alpha']:.4f}  beta = {m['beta']:.4f}  "
        f"H_B = {m['h_b']:.3f}  H_o = {m['h_o']:.3f}",
        f"  ln L = {m['loglike']:.2f}",
        "",
        "tapered versus single exponential N(<H) = 10^(alpha (H - H_o)):",
        f"  exponential alpha = {s['exponential_mle']['alpha']:.4f}  "
        f"ln L = {s['exponential_mle']['loglike']:.2f}",
        f"  delta ln L = {c['delta_loglike']:.2f}; "
        f"delta AIC (exp - tapered) = {c['aic_exponential'] - c['aic_tapered']:.2f}; "
        f"delta BIC = {c['bic_exponential'] - c['bic_tapered']:.2f}",
        f"  likelihood-ratio p (chi2, 2 dof; approximate) = {c['lrt_p_value']:.3g}",
        "",
        "goodness of fit (probability integral transform of each detection):",
        f"  KS at the MLE: D = {s['ks_pit_mle']['statistic']:.4f}, "
        f"p = {s['ks_pit_mle']['p_value']:.3g} (conservative; parameters fitted)",
    ]
    if "posterior_predictive_ks_p" in s:
        lines.append(
            f"  posterior predictive p (KS) = {s['posterior_predictive_ks_p']:.3f}"
        )
    if len(s["groups"]) > 1:
        lines.append("  by survey/block at the MLE:")
        for g in s["groups"]:
            lines.append(
                f"    {g['group']}: {g['n']} detections, KS D = "
                f"{g['ks_statistic']:.3f}, p = {g['ks_p_value']:.3g}"
            )
    n = s["normalisation"]
    lines += [
        "",
        f"normalisation ({s['bias_mode']} bias): HT count of {n['n_used']} detections in "
        f"[{s['h_min']:.2f}, {n['h_norm']:.2f}] = {n['ht_count']:.4g} "
        f"+/- {n['ht_sigma']:.3g} (sampling {n['ht_sigma_sampling']:.3g}, "
        f"bias MC {n['ht_sigma_bias']:.3g})",
        f"  {n['bias_estimates']}",
    ]
    for g, v in n["per_block"].items():
        lines.append(f"  {g}: {v['n']} detections, {v['count']:.4g} +/- {v['sigma']:.3g}")
    if "posterior" in s:
        p = s["posterior"]
        mc = s["mcmc"]
        tau = mc["autocorr"]
        lines += [
            "",
            f"posterior (median and 68% interval; {mc['n_samples']} samples, "
            f"acceptance {mc['acceptance']:.2f}"
            + (f", autocorr {max(tau):.0f} steps" if tau else "") + "):",
            f"  alpha = {_fmt(*p['alpha'].values())}",
            f"  beta  = {_fmt(*p['beta'].values())}",
            f"  H_B   = {_fmt(*p['h_b'].values())}",
            f"  H_o   = {_fmt(*p['h_o'].values())}",
        ]
    if "bootstrap" in s:
        bt = s["bootstrap"]
        lines += [
            "",
            f"bootstrap (median and 68% interval; {bt['replicates']} replicates, "
            "resampled within block, bias estimates perturbed by their MC errors):",
            f"  alpha = {_fmt(*bt['alpha'].values())}",
            f"  beta  = {_fmt(*bt['beta'].values())}",
            f"  H_B   = {_fmt(*bt['h_b'].values())}",
            f"  H_o   = {_fmt(*bt['h_o'].values())}",
            f"  HT count = {_fmt(*bt['ht_count'].values())}",
        ]
    if s["notes"]:
        lines += ["", "notes:", *[f"  - {x}" for x in s["notes"]]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError) as ex:
        print(f"survey_debias_hfit: {ex}", file=sys.stderr)
        return 2
    except ImportError as ex:
        print(f"survey_debias_hfit: {ex}. Install the fit extras: "
              "pip install 'survey_debias[fit]'", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

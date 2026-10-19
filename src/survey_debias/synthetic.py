"""Synthetic detection catalogues from a known H distribution, through OSSSSim.

A survey's catalogue is a Poisson realisation of the population it would
detect. Plants are aimed into each block as in ``survey_efficiency`` (model
orbit rows, weight ``w = P_geom``) but with H drawn from the true
distribution and a colour drawn per object, and the *real* characterization
decides detection (efficiency, tracking, photometric noise, mag_lim). Each
detected plant enters the catalogue with probability
``w N_range / N_block``, ``N_range`` the true number of objects in the
drawn H range, so the expected count is the survey's true expectation.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from . import hfit
from .grid_bias import geometric_prob_for_aimed, sample_aimed_elements_at_i
from .pointings import load_survey_blocks
from .survey_efficiency import ModelPopulation, SampleCut


def draw_h(rng, n, theta, h_range, form="tapered"):
    """H_r from N(<H) on [h_lo, h_hi] (inverse CDF on a fine grid)."""
    alpha, beta, h_b, h_o = theta
    grid = np.linspace(h_range[0], h_range[1], 20001)
    if form == "exponential":
        cum = 10.0 ** (alpha * (grid - h_o))
    else:
        cum = hfit.cumulative(grid, alpha, beta, h_b, h_o)
    cdf = (cum - cum[0]) / (cum[-1] - cum[0])
    return np.interp(rng.random(n), cdf, grid), float(cum[-1] - cum[0])


def inject(char_root, survey: str, model: ModelPopulation, theta, colour: float,
           sigma_c: float, h_range, n_plants: int, seed: int, cut: SampleCut | None = None,
           colour_file=None, form: str = "tapered", exclude_blocks=(), model_band: str = "r"
           ) -> list[dict]:
    """Synthetic catalogue rows (name, block, mag, d_bary, H, i, H_r, colour) for one survey.

    ``H`` is the catalogue H in the survey's filter; ``H_r`` and ``colour`` are the truth.
    """
    from .colours import load_colours
    from .grid_bias_run import GridBiasSimulator, _TileSampler, attributed_to

    cut = cut or SampleCut()
    rng = np.random.default_rng(seed)
    blocks = {k: v for k, v in load_survey_blocks(char_root, survey).items()
              if k not in set(exclude_blocks)}
    filt = next(iter(blocks.values())).filter
    photspec, _ = load_colours(colour_file) if colour_file else (None, None)
    if photspec is None:
        from ossssim.color import PhotSpec
        photspec = PhotSpec()
    sim = GridBiasSimulator(survey, blocks, photspec, filt, seed=seed)
    out = []
    for name, block in blocks.items():
        sampler = _TileSampler(block)
        area = block.aim_area_deg2
        h_r, n_range = draw_h(rng, n_plants, theta, h_range, form)
        scatter = 0.0 if filt == model_band else sigma_c
        c = colour + scatter * rng.standard_normal(n_plants)
        q_max = 0.0
        for j in range(n_plants):
            k = int(rng.integers(len(model)))
            a, e, inc, r = (float(model.a[k]), float(model.e[k]), float(model.i[k]),
                            float(model.dist[k]))
            tile, view, obs = sampler.draw(rng)
            el = sample_aimed_elements_at_i(a, e, inc, obs, rng, r, max_tries=1, survey=view)
            if el is None:
                continue
            w = geometric_prob_for_aimed(a, e, *el, area_deg2=area)
            if w <= 0.0:
                continue
            ok = cut.i_max is None or inc < cut.i_max
            ok &= cut.r_range is None or cut.r_range[0] < r < cut.r_range[1]
            ok &= cut.a_range is None or cut.a_range[0] < a < cut.a_range[1]
            if not ok:
                continue
            h_f = float(h_r[j] + c[j])
            rows = sim.epoch_rows(a, e, *el, h_f, epoch_jd=tile.jd)
            if not attributed_to(rows, block):
                continue
            q = w * n_range / n_plants
            q_max = max(q_max, q)
            for _ in range(rng.poisson(q)):
                row = rows[0]
                mag = float(row["m_rand"].value)
                # Catalogue H as the surveys compute it: the measured magnitude less
                # the true distance and phase term (m_int - H).
                h_cat = mag - (float(row["m_int"].value) - h_f)
                out.append({
                    "name": f"S{survey[:4]}{len(out):05d}", "block": name,
                    "mag": mag, "d_bary": float(row["r"].value), "H": h_cat,
                    "i": inc, "H_r": float(h_r[j]), "colour": float(c[j]), "comp": "cold",
                })
        print(f"  {survey}/{name}: {sum(o['block'] == name for o in out)} synthetic "
              f"detections (max per-plant rate {q_max:.3f})", flush=True)
        if q_max > 1.0:
            print(f"  WARNING {survey}/{name}: a plant stands for {q_max:.1f} detections, so "
                  "the catalogue has clones; use more plants", flush=True)
    return out


def write_catalogue(rows: list[dict], path, note: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["name", "block", "mag", "d_bary", "H", "i", "H_r", "colour", "comp"]
    with path.open("w", newline="") as fh:
        for line in note.splitlines():
            fh.write(f"# {line}\n")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.5f}" if isinstance(r[k], float) else r[k]) for k in cols})


def expected_count(eps_table, theta, colour: float, sigma_c: float, in_model_band: bool,
                   form: str = "tapered") -> float:
    """int lambda_f eps dH for one survey (no overlaps), theta = (alpha, beta, h_b, h_o)."""
    from .joint import COLOUR_NODES, _gh

    alpha, beta, h_b, h_o = theta
    if in_model_band:
        shifts, weights = np.zeros(1), np.ones(1)
    elif sigma_c <= 0.0:
        shifts, weights = np.array([colour]), np.ones(1)
    else:
        x, weights = _gh(COLOUR_NODES)
        shifts = colour + sigma_c * x
    h = eps_table.h
    lam = np.zeros(len(h))
    for s, w in zip(shifts, weights):
        if form == "exponential":
            lam += w * np.exp(hfit.log_differential_exponential(h - s, alpha, h_o))
        else:
            lam += w * np.exp(hfit.log_differential(h - s, alpha, beta, h_b, h_o))
    return float(np.sum(lam * eps_table.eps * np.gradient(h)))


def h_o_for_count(eps_table, shape, target: float, colour: float, sigma_c: float,
                  in_model_band: bool, form: str = "tapered") -> float:
    """H_o giving ``target`` expected detections in a survey."""
    k = expected_count(eps_table, (*shape, 0.0), colour, sigma_c, in_model_band, form)
    return -math.log10(target / k) / shape[0]

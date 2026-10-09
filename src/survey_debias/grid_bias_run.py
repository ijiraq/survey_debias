"""Grid-cell Horvitz–Thompson runner for any :class:`GridSurvey`.

Each survey is ``characterization/{survey}/``; its ``pointings.list`` lines
are grouped into blocks by their ``{block}.eff`` tag. For every bias cell
occupied by a detection, the runner estimates P(detect in block | cell) for
each block of the project by FoV-aimed draws: a tile is chosen with
probability proportional to its area, an orbit is aimed into it, and the
draw counts when Detos1 attributes the detection to that block
(``row['Survey'] == {block}.eff``) at every epoch. The draw carries the
single-epoch geometric probability for the block's total area, so the mean
over draws is the block bias.

The bias attached to a detection is the union over every block of the
project, sum_b P_b(cell): the probability the object would have been found
anywhere. ``--per-block-bias`` keeps each detection's own block instead.
Per-survey estimates are written to ``characterization/{survey}/bias_grid[_rih].csv``.
"""
from __future__ import annotations

import argparse
import math
import zlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from astropy import units as u

from ossssim import OSSSSim

from .grid_bias import (
    TARGET_DETECTIONS,
    GridSurvey,
    OrbitModelCatalog,
    as_check_arrays,
    bounds_from_key,
    can_write_detections_full,
    check_plot_tag,
    default_orbit_model_path,
    empty_check_samples,
    epoch_geometry,
    geometric_detection_prob,
    geometric_prob_for_aimed,
    icrs_to_ecliptic,
    load_detections,
    los_circular_elements,
    record_check_sample,
    rih_bounds_from_key,
    sample_aimed_elements,
    sample_aimed_elements_at_i,
    sample_aq,
    write_bias_check_plots,
    write_bias_results,
    write_detections_full,
)
from .bias_files import (  # noqa: F401 - re-exported
    CELL_FIELDS,
    BiasEstimate,
    bias_file_name,
    load_bias_file,
    load_legacy_cache,
    mc_bias,
    save_bias_file,
)
from .pointings import Block, Project, observer_icrf

# Draws per cell for a block that never detects it, as a multiple of target.
MAX_AIMED_FACTOR = 20


class GridBiasSimulator:
    """One OSSSSim and RNG stream over one survey's characterization epochs.

    Detos1 reloads characterization when the directory changes without
    resetting ran3. A detection requires flag ≥ 4 at every epoch directory
    and, to count for a block, Detos1 must attribute it to that block's
    ``.eff`` at every epoch.
    """

    def __init__(self, survey_name: str, blocks: dict, colours, model_band: str,
                 seed: int = 42):
        self.name = survey_name
        self.blocks = blocks
        first = next(iter(blocks.values()))
        self.epoch_dirs = [str(Path(d).resolve()) for d in first.char_dirs]
        self.element_epoch = first.tiles[0][0].jd
        self.colours = colours
        self.model_band = model_band
        self.sim = OSSSSim(self.epoch_dirs[0], seed=seed)
        self._prime_surveys()

    def _prime_surveys(self) -> None:
        for epoch_dir in self.epoch_dirs:
            self.sim.characterization_directory = epoch_dir
            self.sim.simulate(self._row(44, 0.0, 20, 0, 0, 0, 8, self.element_epoch),
                              colors=self.colours, model_band=self.model_band)

    @staticmethod
    def _row(a, e, inc, node, peri, M, H, epoch_jd, comp="default") -> dict:
        return dict(a=a * u.au, e=e, inc=inc * u.deg, node=node * u.deg,
                    peri=peri * u.deg, M=M * u.deg, H=H * u.mag,
                    epoch=epoch_jd * u.day, comp=comp)

    def _simulate(self, epoch_dir, row, debug=False):
        self.sim.characterization_directory = epoch_dir
        return self.sim.simulate(row, colors=self.colours, model_band=self.model_band,
                                 debug=debug)

    def epoch_rows(self, a, e, inc, node, peri, M, H, epoch_jd=None,
                   comp="default", debug=False) -> list[dict]:
        jd = self.element_epoch if epoch_jd is None else epoch_jd
        return [
            self._simulate(d, self._row(a, e, inc, node, peri, M, H, jd, comp), debug=debug)
            for d in self.epoch_dirs
        ]


def _survey_name(row: dict) -> str:
    name = row.get("Survey", "")
    if isinstance(name, bytes):
        name = name.decode("utf-8", "replace")
    return str(name).strip()


def attributed_to(rows: list[dict], block: Block) -> bool:
    """flag ≥ 4 at every epoch, each time in a pointing tagged ``{block}.eff``.

    Detos1 reports ``{directory}/{block}`` (current OSSSSim) or the first 10
    characters of ``{block}.eff`` (older OSSSSim).
    """
    keys = block.detos_keys()
    legacy = block.detos_name.strip()
    for i, r in enumerate(rows):
        if int(r["flag"]) < 4:
            return False
        name = _survey_name(r)
        expected = keys[i] if "/" in name and i < len(keys) else legacy
        if name != expected:
            return False
    return True


def _jd_utc(jd: float) -> str:
    mjd = jd - 2400000.5
    return (datetime(1858, 11, 17) + timedelta(days=mjd)).strftime("%Y-%m-%d %H:%M")


def _row_radec(row: dict) -> tuple[float, float]:
    """Detos1 ICRS RA/Dec in degrees (available even when flag=0 after rebuild)."""
    try:
        ra = float(row["RA"].to(u.deg).value)
        dec = float(row["DEC"].to(u.deg).value)
    except Exception:
        ra = math.degrees(float(row["RA"]))
        dec = math.degrees(float(row["DEC"]))
    return ra, dec


def _detos_sky(row: dict) -> str:
    ra, dec = _row_radec(row)
    dra = float(row["d_ra"])
    ddec = float(row["d_dec"])
    rate = math.degrees(math.hypot(dra, ddec)) * 3600.0 / 24.0
    r_au = float(row["r"].to(u.au).value) if hasattr(row["r"], "to") else float(row["r"])
    dlt = float(row["delta"].to(u.au).value) if hasattr(row["delta"], "to") else float(row["delta"])
    return (
        f"flag={int(row['flag'])}  RA,Dec={ra:.5f},{dec:.5f}  "
        f"rate={rate:.3f}\"/hr  r={r_au:.3f} Δ={dlt:.3f}  survey={_survey_name(row)!r}"
    )


def sanity_check_simulator(sim: GridBiasSimulator, block: Block, survey: GridSurvey) -> None:
    """Fail fast if a bright object on a tile's line of sight is not detected there."""
    tile = block.tiles[0][0]
    view = block.field_view(tile, survey)
    char_dir = Path(block.char_dirs[0])
    jd0 = tile.jd
    obs = observer_icrf(char_dir, tile.observer, jd0)
    a, e, inc, node, peri, M = _los_elements(tile, char_dir, obs, jd0)
    half_ra = 0.5 * view.mosaic_width_deg / math.cos(math.radians(view.field_dec_deg))
    half_h = 0.5 * view.mosaic_height_deg
    geom = []
    jpl = char_dir / tile.observer
    if jpl.is_file():
        for i, jd in enumerate(view.epoch_jd, start=1):
            ra, dec, sep, rate = epoch_geometry(
                a, e, inc, node, peri, M, jpl, jd0, jd, survey=view
            )
            geom.append((i, jd, ra, dec, sep, rate))
            in_fov = (abs(ra - view.field_ra_deg) <= half_ra + 1e-3
                      and abs(dec - view.field_dec_deg) <= half_h + 1e-3)
            lo = view.rate_cut_min_arcsec_hr
            hi = view.rate_cut_max_arcsec_hr
            if (lo, hi) == (0.0, 1e9):
                lo, hi = survey.rate_cut_min_arcsec_hr, survey.rate_cut_max_arcsec_hr
            print(
                f"sanity {block.survey}/{block.name} epoch{i} {_jd_utc(jd)} JD={jd:.5f}  "
                f"RA,Dec={ra:.5f},{dec:.5f}  sep={sep * 60:.3f}'  "
                f"rate={rate:.3f}\"/hr  FoV={in_fov}  rate_cut={lo <= rate <= hi}",
                flush=True,
            )
    rows = sim.epoch_rows(a, e, inc, node, peri, M, 8.0, epoch_jd=jd0, debug=True)
    for i, row in enumerate(rows, start=1):
        print(f"sanity Detos1 epoch{i} {_detos_sky(row)}", flush=True)
    if attributed_to(rows, block):
        print(f"sanity: LOS-planted object is a {block.n_epochs}-epoch detection "
              f"in {block.eff_name}", flush=True)
        return
    if block.n_epochs > 1:
        retried = sim.epoch_rows(a, e, inc, node, peri, M, 8.0, epoch_jd=jd0)
        if attributed_to(retried, block):
            print("sanity: first Detos1 call missed, retry is a detection", flush=True)
            return
    for dM in (-0.15, -0.10, -0.05, 0.05, 0.10, 0.15):
        shifted = sim.epoch_rows(a, e, inc, node, peri, M + dM, 8.0, epoch_jd=jd0)
        if attributed_to(shifted, block):
            print(f"sanity: LOS-planted object detected with ΔM={dM:.2f}°", flush=True)
            return
    flags = [int(r["flag"]) for r in rows]
    names = [_survey_name(r) for r in rows]
    detail = "; ".join(
        f"e{i} sep={sep * 60:.3f}' rate={rate:.3f}\"/hr"
        for i, _jd, _ra, _dec, sep, rate in geom
    )
    raise RuntimeError(
        f"LOS-planted object at {block.survey}/{block.name} line {tile.line} was "
        f"not a {block.n_epochs}-epoch detection in {block.eff_name} "
        f"(flags={flags}, surveys={names}{'; ' + detail if detail else ''})."
    )


def _los_elements(tile, char_dir, obs, jd):
    """Circular orbit at 44 au on the tile's line of sight."""
    jpl = Path(char_dir) / tile.observer
    if jpl.is_file():
        return los_circular_elements(tile.ra_deg, tile.dec_deg, 44.0, jpl, jd)
    from .grid_bias import barycentric_on_icrs_los, circular_elements_through_ecliptic_xyz

    pos = barycentric_on_icrs_los(obs, tile.ra_deg, tile.dec_deg, 44.0)
    return circular_elements_through_ecliptic_xyz(*pos)


class _TileSampler:
    """Chooses a block's tile with probability ∝ area; caches observer vectors."""

    def __init__(self, block: Block, survey: GridSurvey | None = None):
        self.block = block
        self.tiles = list(block.tiles[0])
        areas = np.array([t.box_area_deg2 for t in self.tiles], dtype=float)
        self.prob = areas / areas.sum()
        self.views = [block.field_view(t, survey) for t in self.tiles]
        self._obs = {}

    def draw(self, rng):
        k = int(rng.choice(len(self.tiles), p=self.prob)) if len(self.tiles) > 1 else 0
        tile = self.tiles[k]
        if k not in self._obs:
            self._obs[k] = observer_icrf(self.block.char_dirs[0], tile.observer, tile.jd)
        return tile, self.views[k], self._obs[k]


def _bias_loop(sim, block: Block, draw_elements, seed: int, target: int,
               group: str, must_detect: bool, max_aimed: int | None,
               label: str) -> BiasEstimate:
    """Shared FoV-aimed loop: draw_elements(rng, view, obs) -> (a, e, inc, node, peri, M, H)."""
    rng = np.random.default_rng(seed)
    sampler = _TileSampler(block)
    area = block.aim_area_deg2
    sampled = empty_check_samples()
    detected = empty_check_samples()
    n_detected = n_aimed = n_fail = 0
    sum_w = sum_w2 = 0.0
    max_tries = max(target * 1000, 10000)
    max_aimed = max_aimed or MAX_AIMED_FACTOR * target
    while n_detected < target and (n_aimed + n_fail) < max_tries and n_aimed < max_aimed:
        tile, view, obs = sampler.draw(rng)
        el = draw_elements(rng, view, obs)
        if el is None:
            n_fail += 1
            continue
        a, e, inc, node, peri, M, H = el
        n_aimed += 1
        p_geom = geometric_prob_for_aimed(a, e, inc, node, peri, M, area_deg2=area)
        rows = sim.epoch_rows(a, e, inc, node, peri, M, H, epoch_jd=tile.jd, comp=group)
        ra, dec = _row_radec(rows[0])
        record_check_sample(sampled, ra, dec, a, e, inc, node, peri, M)
        if attributed_to(rows, block):
            n_detected += 1
            sum_w += p_geom
            sum_w2 += p_geom * p_geom
            record_check_sample(detected, ra, dec, a, e, inc, node, peri, M)
        if n_aimed % 500 == 0:
            bias_so_far, _ = mc_bias(n_aimed, sum_w, sum_w2)
            print(
                f"    ... {label}: {n_aimed} aimed ({n_fail} fail), "
                f"{n_detected}/{target} detections  "
                f"P(det|FoV)={n_detected / n_aimed:.3g}  bias~{bias_so_far:.3g}",
                flush=True,
            )
    if n_detected < target:
        msg = (f"{label}: {n_detected}/{target} detections after {n_aimed} aimed "
               f"plants ({n_fail} fail)")
        if must_detect and n_detected == 0:
            raise RuntimeError(msg + "; a detection in this block has no simulated twin")
        print(f"    {msg}; keeping the estimate and its MC error", flush=True)
    bias, se = mc_bias(n_aimed, sum_w, sum_w2)
    return BiasEstimate(bias, n_aimed, n_detected, se,
                        as_check_arrays(sampled), as_check_arrays(detected))


def compute_cell_bias(sim, block: Block, cell_bounds: dict, seed: int, target: int,
                      group: str = "default", must_detect: bool = True,
                      max_aimed: int | None = None) -> BiasEstimate:
    """aq_grid P(detect in block | a, q, sin i_free, H)."""
    si0, si1 = cell_bounds["sin_ifree"]
    h0, h1 = cell_bounds["Hx"]

    def draw(rng, view, obs):
        a, q = sample_aq(rng, cell_bounds["a"], cell_bounds["q"])
        e = 1.0 - q / a
        sin_ifree = float(rng.uniform(si0, si1))
        ifree = math.degrees(math.asin(max(0.0, min(1.0, sin_ifree))))
        H = float(rng.uniform(h0, h1))
        el = sample_aimed_elements(a, e, ifree, obs, rng, survey=view)
        return None if el is None else (a, e, *el, H)

    return _bias_loop(sim, block, draw, seed, target, group, must_detect, max_aimed,
                      f"{block.survey}/{block.name}")


def compute_model_ae_bias(sim, block: Block, cell_bounds: dict, model: OrbitModelCatalog,
                          seed: int, target: int, group: str = "default",
                          must_detect: bool = True, max_aimed: int | None = None
                          ) -> BiasEstimate:
    """P(detect in block | r, i, H) with (a, e) ~ OSSOS model p(a,e|r,i)."""
    r0, r1 = cell_bounds["r"]
    i0, i1 = cell_bounds["i"]
    h0, h1 = cell_bounds["Hx"]
    candidates, dr, di = model.select_expanding(r0, r1, i0, i1)
    fracs = candidates.component_fractions()
    mix = ", ".join(f"{k}={v:.2f}" for k, v in sorted(fracs.items()))
    print(
        f"  model prior: {len(candidates)} objects in "
        f"r={0.5 * (r0 + r1):.2f}±{dr:.2f}, i={0.5 * (i0 + i1):.2f}±{di:.2f} [{mix}]",
        flush=True,
    )

    def draw(rng, view, obs):
        r_au = float(rng.uniform(r0, r1))
        inc_cell = float(rng.uniform(i0, i1))
        H = float(rng.uniform(h0, h1))
        try:
            a, e, _comp = candidates.sample_ae(rng, r_au=r_au)
        except RuntimeError:
            return None
        el = sample_aimed_elements_at_i(a, e, inc_cell, obs, rng, r_au, survey=view)
        return None if el is None else (a, e, *el, H)

    return _bias_loop(sim, block, draw, seed, target, group, must_detect, max_aimed,
                      f"{block.survey}/{block.name}")


def _write_plots(plot_dir: Path, block: Block, survey: GridSurvey, est: BiasEstimate,
                 tags) -> None:
    if plot_dir is None or est.sampled is None or not est.sampled["ra"].size:
        return
    out = plot_dir / block.survey / block.name
    view = block.field_view(None, survey)
    for tag in tags:
        for path in write_bias_check_plots(out, est.sampled, est.detected,
                                           check_plot_tag(tag), survey=view):
            print(f"    wrote {path}", flush=True)


def _union(entries: dict, blocks: list[Block], group: str, cell) -> tuple[float, float | None]:
    bias = 0.0
    var = 0.0
    known = True
    for b in blocks:
        est = entries[(b.survey, b.name, group, cell)]
        bias += est.bias
        if est.bias_se is None:
            known = False
        else:
            var += est.bias_se ** 2
    return bias, (math.sqrt(var) if known else None)


def run_grid_bias(survey: GridSurvey, root: Path, target: int = TARGET_DETECTIONS,
                  seed: int = 42, check_plots_dir: Path | None = None,
                  no_check_plots: bool = False,
                  extra_header: str | None = None,
                  model_path: Path | None = None,
                  per_block_bias: bool = False) -> Path:
    root = Path(root)
    project = Project.for_survey(root, survey)
    names = project.survey_names(survey.surveys)
    if not names:
        raise ValueError(
            f"{project.char_root}: no characterization/{{survey}}/pointings.list"
        )
    # ossssim's SurveyCharacterization resets the simulator: load all first.
    blocks_by_survey = {s: project.blocks(s) for s in names}
    detections = load_detections(root / survey.detections_relpath, survey,
                                 colour_for=project.colour_for)
    for d in detections:
        if d["survey"] not in blocks_by_survey:
            raise ValueError(
                f"{d['name']}: survey {d['survey']!r} is not in the project "
                f"surveys ({', '.join(names)})"
            )
    colours = project.colours
    print(colours.note, flush=True)
    colours.validate({b.filter for bs in blocks_by_survey.values() for b in bs.values()})
    all_blocks = [b for s in names for b in blocks_by_survey[s].values()]
    method = survey.bias_method

    model = None
    if method == "model_ae":
        model_path = Path(model_path or default_orbit_model_path(root))
        print(f"loading orbit model prior from {model_path}", flush=True)
        model = OrbitModelCatalog.from_path(model_path)
        fracs = model.component_fractions()
        mix = ", ".join(f"{k}={v:.3f}" for k, v in sorted(fracs.items()))
        print(f"  {len(model)} model objects [{mix}]", flush=True)

    keys = sorted({(d["colour_group"], d["cell"]) for d in detections})
    if per_block_bias:
        wanted = sorted({(d["survey"], d["block"], d["colour_group"], d["cell"])
                         for d in detections})
    else:
        wanted = [(b.survey, b.name, g, c) for b in all_blocks for g, c in keys]
    print(f"{len(keys)} cells x {len(all_blocks)} block(s) in {len(names)} survey(s); "
          f"{len(wanted)} block-cell estimates, target={target}, method={method}, "
          f"bias={'own block' if per_block_bias else 'union over blocks'}", flush=True)
    for b in all_blocks:
        _, lat = icrs_to_ecliptic(b.tiles[0][0].ra_deg, b.tiles[0][0].dec_deg)
        p_geo = geometric_detection_prob(b.aim_area_deg2, 7.0, lat)
        print(
            f"  {b.survey}/{b.name}: {len(b.tiles[0])} tile(s), "
            f"{b.area_deg2:.4f} deg² footprint ({b.aim_area_deg2:.4f} sampled), "
            f"filter {b.filter}, {b.n_epochs} epoch(s) "
            f"JD {', '.join(f'{jd:.5f}' for jd in b.epoch_jds())}; "
            f"P_geom(i=7°) ~ {p_geo:.2e}",
            flush=True,
        )

    entries: dict = {}
    files = {}
    for s in names:
        path = project.char_root / s / bias_file_name(method)
        files[s] = path
        for (blk, g, c), est in load_bias_file(path, method).items():
            entries[(s, blk, g, c)] = est
    legacy = root / bias_file_name(method)
    if len(all_blocks) == 1 and not files[all_blocks[0].survey].exists():
        b = all_blocks[0]
        old = load_legacy_cache(legacy, method)
        if old:
            print(f"importing {len(old)} cell(s) from legacy {legacy} into "
                  f"{files[b.survey]} (block {b.name}, colour group default; "
                  "no MC error recorded)", flush=True)
            for c, est in old.items():
                entries.setdefault((b.survey, b.name, "default", c), est)
            save_bias_file(files[b.survey], {
                k[1:]: v for k, v in entries.items() if k[0] == b.survey}, method)

    plot_dir = None if no_check_plots else Path(check_plots_dir or (root / "check_plots"))
    if plot_dir is not None:
        print(f"check plots → {plot_dir}/{{survey}}/{{block}}/", flush=True)

    todo = [k for k in wanted if k not in entries]
    by_survey: dict[str, list] = {}
    for k in todo:
        by_survey.setdefault(k[0], []).append(k)
    for s, keys_s in by_survey.items():
        sim = GridBiasSimulator(s, blocks_by_survey[s], colours.photspec,
                                survey.model_band, seed=seed)
        checked = set()
        for idx, key in enumerate(keys_s):
            _, blk, group, cell = key
            block = blocks_by_survey[s][blk]
            if blk not in checked:
                sanity_check_simulator(sim, block, survey)
                checked.add(blk)
            members = [str(d["name"]) for d in detections
                       if (d["survey"], d["block"], d["colour_group"], d["cell"]) == key]
            print(f"{s}/{blk} [{group}] cell {cell} ({idx + 1}/{len(keys_s)})"
                  + (f": {', '.join(members)}" if members else ""), flush=True)
            cell_seed = seed + zlib.crc32(repr(key).encode())
            if method == "model_ae":
                est = compute_model_ae_bias(sim, block, rih_bounds_from_key(cell), model,
                                            cell_seed, target, group, bool(members))
            else:
                est = compute_cell_bias(sim, block, bounds_from_key(cell), cell_seed,
                                        target, group, bool(members))
            print(f"  bias={est.bias:.6g} ± {est.bias_se:.2g} "
                  f"({est.n_detected} detected of {est.n_drawn})", flush=True)
            if members:
                _write_plots(plot_dir, block, survey, est, members)
            entries[key] = est
            save_bias_file(files[s], {
                k[1:]: v for k, v in entries.items() if k[0] == s}, method)

    for d in detections:
        own = entries[(d["survey"], d["block"], d["colour_group"], d["cell"])]
        d["block_bias"] = own.bias
        if per_block_bias:
            d["bias"], d["bias_se"] = own.bias, own.bias_se
        else:
            d["bias"], d["bias_se"] = _union(entries, all_blocks, d["colour_group"], d["cell"])
        block = blocks_by_survey[d["survey"]][d["block"]]
        view = block.field_view(None, survey)
        d["field_ra"], d["field_dec"] = view.field_ra_deg, view.field_dec_deg
        d["field_jd"] = view.epoch_jd[0]
        d["n_epochs"] = block.n_epochs
        if d["bias"] <= 0.0:
            raise RuntimeError(f"{d['name']}: bias is zero in cell {d['cell']}")
    results = root / survey.results_name
    write_bias_results(results, detections, survey,
                       bias_mode="block" if per_block_bias else "union",
                       colour_note=colours.note)
    print(
        f"Wrote {results}; sum 1/bias = {sum(1 / d['bias'] for d in detections):.1f}",
        flush=True,
    )
    if can_write_detections_full(detections):
        out = root / survey.detections_full_name
        write_detections_full(out, detections, survey, header_lines=extra_header)
        print(f"Wrote {out}", flush=True)
    else:
        print(
            f"{survey.detections_full_name} is written when the catalog "
            "includes a, e, and i.",
            flush=True,
        )
    return results


def build_arg_parser(survey: GridSurvey, default_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Grid-cell debiasing for {survey.name} (follows Kavelaars et al. 2022 ac2c72)."
    )
    parser.add_argument(
        "--root", default=str(default_root),
        help="Project directory with the detections, characterization/{survey}/, "
             "and models; debiasing outputs are written here.",
    )
    parser.add_argument("--target", type=int, default=TARGET_DETECTIONS,
                        help="Target number of detections per block and cell")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--per-block-bias", action="store_true",
        help="Use each detection's own block bias instead of the union over blocks",
    )
    parser.add_argument(
        "--check-plots-dir", default=None,
        help="Directory for sampled-vs-detected check plots "
             "(default: <root>/check_plots/{survey}/{block}/)",
    )
    parser.add_argument(
        "--no-check-plots", action="store_true",
        help="Skip writing sampled-vs-detected check plots",
    )
    if survey.bias_method == "model_ae":
        parser.add_argument(
            "--model", default=None,
            help="Orbit model file or directory for p(a,e|r,i). Accepts "
                 "OSSOS Models or CFEPS L7 catalog file formats. "
                 "Default: $SURVEY_DEBIAS_MODEL, else <root>/Models/OSSOS",
        )
    return parser


def main(survey: GridSurvey, default_root: Path | None = None,
         extra_header: str | None = None) -> None:
    """Command-line entry for a survey defined by the calling project."""
    default_root = default_root or Path.cwd()
    args = build_arg_parser(survey, default_root).parse_args()
    model_path = Path(args.model) if getattr(args, "model", None) else None
    run_grid_bias(
        survey, Path(args.root), target=args.target, seed=args.seed,
        check_plots_dir=Path(args.check_plots_dir) if args.check_plots_dir else None,
        no_check_plots=args.no_check_plots,
        extra_header=extra_header,
        model_path=model_path,
        per_block_bias=args.per_block_bias,
    )

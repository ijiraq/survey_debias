"""Population efficiency of a survey: the fraction of a model population it detects.

``eps_s(H_f)`` is the fraction of all objects of a population (the whole
sky) with absolute magnitude ``H_f`` in the survey's own filter that survey
``s`` detects and characterises. With the population's differential H
distribution ``lambda_f`` the expected number of detections is
``Lambda_s = int lambda_f(H) eps_s(H) dH``.

The H dependence is analytic. OSSSSim is used only for geometry: on a copy
of the characterization with unit efficiency (same pointings, epochs, fill
factors and rate cuts) a plant is "in the field" when Detos1 returns flag 4
in its block. Each plant records its geometric weight and ``mu = m - H``
(distance and phase), and

    eps_b(H) = mean_j[ w_j 1(in field)_j S_bar_b(H + mu_j, rate_j) ]

with ``S_bar_b`` the block's selection from its real ``.eff`` files
(efficiency on the true magnitude, tracking and characterisation over the
photometric noise, multiplied over epochs).

Plants draw (a, e, i, r) from model rows (``r`` is the row's distance at
the model epoch, so r follows each orbit's time-averaged distribution). The
plant is aimed at a uniform position in a tile's bounding box (tiles chosen
by box area) and carries ``w = P_geom`` for the block's box area at its own
ecliptic latitude; a position the orbit cannot reach has weight zero.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid_bias import OrbitModelCatalog, geometric_prob_for_aimed, sample_aimed_elements_at_i
from .pointings import POINTINGS_NAME, SURVEY_CONF_NAME, Block, load_survey_blocks

COLD_COMPONENTS = ("coldl", "coldh", "kernel")
H_GRID = (0.0, 18.0, 0.02)
UNIT_EFF = """\
# Unit efficiency copy of {source} (geometry pass, survey_efficiency.py)
{keep}track_frac= 1.0 99.0 0.0
{blocks}"""
# A flat lookup table: Detos1 does not detect anything with a step-like
# single_param (tanh width ~0), so do not use one here.
UNIT_BLOCK = ("rates= {lo} {hi}\nfunction= lookup\nlookup_param= 0.0 1.0\n"
              "lookup_param= 40.0 1.0\nmag_lim= 40.0\n")
KEEP_KEYS = ("rate_cut", "filter")


# ---------------------------------------------------------------------------
# Model population


@dataclass
class ModelPopulation:
    """Rows of an orbit model: (a, e, i, dist, H, comp)."""

    a: np.ndarray
    e: np.ndarray
    i: np.ndarray
    dist: np.ndarray
    H: np.ndarray
    comp: np.ndarray
    source: str = ""

    def __len__(self) -> int:
        return int(self.a.size)

    @classmethod
    def from_file(cls, path, components=COLD_COMPONENTS) -> "ModelPopulation":
        """OSSOS 1.0 ``ModelUsed.dat`` rows whose component is in ``components``.

        Columns: a e i Omega omega M H epoch dist comment; the component is the
        comment without its trailing underscore (coldl_, coldh_, kernel_, hot_).
        """
        path = Path(path)
        keep = set(components) if components else None
        cols = {k: [] for k in ("a", "e", "i", "dist", "H", "comp")}
        with path.open() as fh:
            for line in fh:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                comp = parts[9].rstrip("_")
                if keep is not None and comp not in keep:
                    continue
                cols["a"].append(float(parts[0]))
                cols["e"].append(float(parts[1]))
                cols["i"].append(float(parts[2]))
                cols["H"].append(float(parts[6]))
                cols["dist"].append(float(parts[8]))
                cols["comp"].append(comp)
        if not cols["a"]:
            raise ValueError(f"{path}: no rows in components {sorted(keep or [])}")
        arrays = {k: np.asarray(v) for k, v in cols.items()}
        return cls(**arrays, source=str(path))

    def catalog(self) -> OrbitModelCatalog:
        return OrbitModelCatalog(self.a, self.e, self.i, self.dist, self.comp)


# ---------------------------------------------------------------------------
# Unit-efficiency characterization


def unit_efficiency_text(path: Path) -> str:
    """The ``.eff`` file with efficiency 1 in each rate block and no limits."""
    keep, blocks = [], []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        if key in KEEP_KEYS:
            keep.append(f"{key}= {value.strip()}\n")
        elif key == "rates":
            lo, hi = value.split()[:2]
            blocks.append(UNIT_BLOCK.format(lo=lo, hi=hi))
    if not blocks:
        raise ValueError(f"{path}: no rates= block")
    return UNIT_EFF.format(source=path, keep="".join(keep), blocks="".join(blocks))


def make_unit_characterization(char_root, survey: str, work_dir) -> Path:
    """Copy ``characterization/{survey}`` to ``work_dir/{survey}`` with unit .eff files.

    Returns the new characterization root. Observer files, survey.conf and the
    epoch layout are copied as they are.
    """
    src = Path(char_root) / survey
    dst_root = Path(work_dir)
    dst = dst_root / survey
    dirs = [src] if (src / POINTINGS_NAME).is_file() else sorted(
        p for p in src.iterdir() if p.is_dir() and (p / POINTINGS_NAME).is_file())
    for d in dirs:
        out = dst / d.relative_to(src)
        out.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if not f.is_file():
                continue
            if f.suffix == ".eff":
                (out / f.name).write_text(unit_efficiency_text(f))
            elif f.name == POINTINGS_NAME or f.suffix.lower() in (".csv", ".txt") \
                    or f.name.endswith(".jpl_batch"):
                shutil.copy2(f, out / f.name)
    if (src / SURVEY_CONF_NAME).is_file():
        shutil.copy2(src / SURVEY_CONF_NAME, dst / SURVEY_CONF_NAME)
    return dst_root


# ---------------------------------------------------------------------------
# Geometry pass


PLANT_FIELDS = ("block", "comp", "a", "e", "i", "r", "beta", "w", "in_field", "mu", "rate")


@dataclass
class PlantTable:
    """Aimed plants of one survey (all blocks), with what eps needs."""

    survey: str
    block: np.ndarray
    comp: np.ndarray
    a: np.ndarray
    e: np.ndarray
    i: np.ndarray
    r: np.ndarray
    beta: np.ndarray
    w: np.ndarray            # P_geom for the block's aimed box (0 if not reachable)
    in_field: np.ndarray     # Detos1 flag 4 in the block on the unit-efficiency copy
    mu: np.ndarray           # m - H at detection (NaN when not in the field)
    rate: np.ndarray         # on-sky rate ["/hr] (NaN when not in the field)

    def __len__(self) -> int:
        return int(self.w.size)

    def save(self, path) -> None:
        arrays = {k: getattr(self, k) for k in PLANT_FIELDS}
        for k in ("block", "comp"):
            arrays[k] = np.asarray(arrays[k], dtype=str)
        np.savez_compressed(path, survey=self.survey, **arrays)

    @classmethod
    def load(cls, path) -> "PlantTable":
        z = np.load(path, allow_pickle=False)
        return cls(survey=str(z["survey"]), **{k: z[k] for k in PLANT_FIELDS})

    @classmethod
    def concatenate(cls, tables) -> "PlantTable":
        tables = list(tables)
        return cls(survey=tables[0].survey,
                   **{k: np.concatenate([getattr(t, k) for t in tables]) for k in PLANT_FIELDS})

    def for_block(self, name: str) -> "PlantTable":
        m = self.block == name
        return PlantTable(self.survey, **{k: getattr(self, k)[m] for k in PLANT_FIELDS})


def _row_rate(row) -> float:
    return math.degrees(math.hypot(float(row["d_ra"]), float(row["d_dec"]))) * 3600.0 / 24.0


def plant_block(sim, block: Block, model: ModelPopulation, n_plants: int, seed: int,
                progress: int = 0) -> PlantTable:
    """``n_plants`` aimed model plants through one block of the unit-efficiency copy."""
    from .grid_bias_run import _TileSampler, attributed_to

    rng = np.random.default_rng(seed)
    sampler = _TileSampler(block)
    area = block.aim_area_deg2
    out = {k: np.full(n_plants, np.nan) for k in ("a", "e", "i", "r", "beta", "w", "mu", "rate")}
    comp = np.empty(n_plants, dtype=object)
    in_field = np.zeros(n_plants, dtype=bool)
    H = 5.0
    for j in range(n_plants):
        k = int(rng.integers(len(model)))
        a, e, inc, r = (float(model.a[k]), float(model.e[k]), float(model.i[k]),
                        float(model.dist[k]))
        out["a"][j], out["e"][j], out["i"][j], out["r"][j] = a, e, inc, r
        comp[j] = model.comp[k]
        tile, view, obs = sampler.draw(rng)
        el = sample_aimed_elements_at_i(a, e, inc, obs, rng, r, max_tries=1, survey=view)
        if el is None:
            out["w"][j] = 0.0
            continue
        inc_el, node, peri, M = el
        w = geometric_prob_for_aimed(a, e, inc_el, node, peri, M, area_deg2=area)
        out["w"][j] = w
        if w <= 0.0:
            continue
        rows = sim.epoch_rows(a, e, inc_el, node, peri, M, H, epoch_jd=tile.jd)
        if attributed_to(rows, block):
            in_field[j] = True
            row = rows[0]
            out["mu"][j] = float(row["m_int"].value) - H
            out["rate"][j] = _row_rate(row)
        if progress and (j + 1) % progress == 0:
            print(f"    {block.survey}/{block.name}: {j + 1}/{n_plants} plants, "
                  f"{int(in_field[: j + 1].sum())} in the field", flush=True)
    return PlantTable(block.survey, np.full(n_plants, block.name, dtype=object), comp,
                      in_field=in_field, **out)


@dataclass
class SampleCut:
    """Orbital cut defining a survey's sample, applied to model plants as to detections.

    ``i_max``: ecliptic inclination [deg] (strict); ``r_range``: distance at
    detection [au]; ``a_range``: semi-major axis [au]. ``None`` skips a cut.
    """

    i_max: float | None = None
    r_range: tuple | None = None
    a_range: tuple | None = None

    def mask(self, plants: PlantTable) -> np.ndarray:
        m = np.ones(len(plants), dtype=bool)
        if self.i_max is not None:
            m &= plants.i < self.i_max
        if self.r_range is not None:
            m &= (plants.r > self.r_range[0]) & (plants.r < self.r_range[1])
        if self.a_range is not None:
            m &= (plants.a > self.a_range[0]) & (plants.a < self.a_range[1])
        return m

    def describe(self) -> str:
        parts = []
        if self.i_max is not None:
            parts.append(f"i < {self.i_max:g} deg")
        if self.r_range is not None:
            parts.append(f"{self.r_range[0]:g} < r < {self.r_range[1]:g} au")
        if self.a_range is not None:
            parts.append(f"{self.a_range[0]:g} < a < {self.a_range[1]:g} au")
        return ", ".join(parts) or "none"


@dataclass
class EfficiencyTable:
    """eps(H) of one survey in its own filter, with bootstrap replicates."""

    survey: str
    filter: str
    h: np.ndarray
    eps: np.ndarray
    boot: np.ndarray          # (n_boot, n_h)
    meta: dict

    @property
    def se(self) -> np.ndarray:
        return self.boot.std(axis=0, ddof=1) if len(self.boot) > 1 else np.zeros_like(self.eps)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            fh.write(f"# survey: {self.survey}\n# filter: {self.filter}\n")
            for k, v in self.meta.items():
                fh.write(f"# {k}: {v}\n")
            w = csv.writer(fh)
            w.writerow(["H", "eps", "eps_se"] + [f"boot{k}" for k in range(len(self.boot))])
            se = self.se
            for j in range(len(self.h)):
                w.writerow([f"{self.h[j]:.4f}", f"{self.eps[j]:.6e}", f"{se[j]:.3e}"]
                           + [f"{b:.6e}" for b in self.boot[:, j]])

    @classmethod
    def load(cls, path) -> "EfficiencyTable":
        meta, rows = {}, []
        with Path(path).open() as fh:
            for line in fh:
                if line.startswith("#"):
                    key, _, value = line[1:].partition(":")
                    meta[key.strip()] = value.strip()
                    continue
                rows.append(line)
        reader = csv.reader(rows)
        header = next(reader)
        data = np.array([[float(x) for x in r] for r in reader])
        n_boot = len(header) - 3
        survey, filt = meta.pop("survey", ""), meta.pop("filter", "")
        return cls(survey, filt, data[:, 0], data[:, 1],
                   data[:, 3:].T if n_boot else np.zeros((0, len(data))), meta)


def survey_efficiency(plants: PlantTable, blocks: dict, cut: SampleCut | None = None,
                      h_grid=H_GRID, n_boot: int = 32, seed: int = 0,
                      meta: dict | None = None) -> EfficiencyTable:
    """eps_s(H_f) = sum over blocks of mean_j[w_j 1(in field, cut) S_bar_b(H_f + mu_j)].

    ``blocks`` are the survey's real blocks (their ``.eff`` selections). The
    bootstrap gives each plant a Poisson(1) weight, per block.
    """
    cut = cut or SampleCut()
    h = np.arange(h_grid[0], h_grid[1] + 0.5 * h_grid[2], h_grid[2])
    rng = np.random.default_rng(seed)
    keep = plants.in_field & cut.mask(plants)
    eps = np.zeros(len(h))
    boot = np.zeros((n_boot, len(h)))
    filters = set()
    for name, block in blocks.items():
        here = plants.block == name
        n_b = int(here.sum())
        if n_b == 0:
            raise ValueError(f"{plants.survey}/{name}: no plants")
        sel = block.selection
        filters.add(block.filter)
        use = here & keep
        if not use.any():
            continue
        mu, rate, w = plants.mu[use], plants.rate[use], plants.w[use]
        m_grid = np.arange(math.floor(h[0] + mu.min()) - 1.0,
                           math.ceil(h[-1] + mu.max()) + 1.0, 0.005)
        s = np.empty((int(use.sum()), len(h)))
        if sel.rate_dependent:
            keys = np.round(rate, 2)
            for key in np.unique(keys):
                rows = keys == key
                table = sel.measured_selection(m_grid, float(key))
                s[rows] = np.interp(h[None, :] + mu[rows, None], m_grid, table)
        else:
            table = sel.measured_selection(m_grid, float(np.median(rate)))
            s[:] = np.interp(h[None, :] + mu[:, None], m_grid, table)
        eps += (w @ s) / n_b
        if n_boot:
            pw = rng.poisson(1.0, size=(n_boot, len(w))) * w[None, :]
            boot += (pw @ s) / n_b
    if len(filters) > 1:
        raise ValueError(f"{plants.survey}: blocks use several filters {sorted(filters)}")
    info = {"n_plants": len(plants), "in_field": int(plants.in_field.sum()),
            "cut": cut.describe(), **(meta or {})}
    return EfficiencyTable(plants.survey, filters.pop() if filters else "", h, eps, boot, info)


def three_epoch_area_arcmin2(plants: PlantTable, blocks: dict, cut: SampleCut | None = None
                             ) -> float:
    """Box area x fraction of reachable plants in the field (unit efficiency)."""
    cut = cut or SampleCut()
    total = 0.0
    for name, block in blocks.items():
        m = (plants.block == name) & (plants.w > 0) & cut.mask(plants)
        if m.any():
            total += block.aim_area_deg2 * 3600.0 * plants.in_field[m].mean()
    return total


def geometry_pass(char_root, survey: str, model: ModelPopulation, n_plants: int,
                  seed: int = 42, work_dir=None, progress: int = 0) -> PlantTable:
    """Plant tables for every block of a survey (``n_plants`` per block)."""
    from ossssim.color import PhotSpec

    from .grid_bias_run import GridBiasSimulator

    own = work_dir is None
    work = Path(tempfile.mkdtemp(prefix="sdeff", dir="/tmp")) if own else Path(work_dir)
    try:
        unit_root = make_unit_characterization(char_root, survey, work)
        blocks = load_survey_blocks(unit_root, survey, use_ossssim=True)
        sim = GridBiasSimulator(survey, blocks, PhotSpec(), "r", seed=seed)
        tables = []
        for name, block in blocks.items():
            block_seed = seed + zlib.crc32(f"{survey}/{name}".encode()) % 100000
            t = plant_block(sim, block, model, n_plants, block_seed, progress)
            print(f"  {survey}/{name}: {n_plants} plants, {int(t.in_field.sum())} in "
                  f"the field ({100.0 * t.in_field.mean():.1f}%)", flush=True)
            tables.append(t)
        return PlantTable.concatenate(tables)
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)


def _range(lo, hi):
    if lo is None and hi is None:
        return None
    return (-math.inf if lo is None else lo, math.inf if hi is None else hi)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="survey_debias_eff",
        description="Population efficiency eps(H) of a survey, in its own filter: the "
                    "fraction of a model population (default the OSSOS 1.0 cold "
                    "classicals) that the survey detects, for the joint LF fit.",
    )
    p.add_argument("--root", default=".", help="directory holding characterization/")
    p.add_argument("--survey", action="append", required=True,
                   help="characterization/{survey} (repeat for several)")
    p.add_argument("--model", required=True, help="OSSOS 1.0 ModelUsed.dat file")
    p.add_argument("--components", default=",".join(COLD_COMPONENTS),
                   help=f"model components (default {','.join(COLD_COMPONENTS)})")
    p.add_argument("--tag", default="cold", help="name used in the output files")
    p.add_argument("--plants", type=int, default=20000, help="aimed plants per block")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--boot", type=int, default=32, help="bootstrap replicates of eps")
    p.add_argument("--i-max", type=float, default=None, help="sample cut: i < I_MAX [deg]")
    p.add_argument("--r-min", type=float, default=None, help="sample cut: r > R_MIN [au]")
    p.add_argument("--r-max", type=float, default=None, help="sample cut: r < R_MAX [au]")
    p.add_argument("--a-min", type=float, default=None, help="sample cut: a > A_MIN [au]")
    p.add_argument("--a-max", type=float, default=None, help="sample cut: a < A_MAX [au]")
    p.add_argument("--reuse", action="store_true",
                   help="reuse an existing plant table instead of planting again")
    args = p.parse_args(argv)

    char_root = Path(args.root) / "characterization"
    components = tuple(c for c in args.components.split(",") if c)
    model = None
    cut = SampleCut(args.i_max, _range(args.r_min, args.r_max), _range(args.a_min, args.a_max))
    for survey in args.survey:
        plants_path = char_root / survey / f"plants_{args.tag}.npz"
        if args.reuse and plants_path.is_file():
            plants = PlantTable.load(plants_path)
            print(f"{survey}: reusing {plants_path} ({len(plants)} plants)")
        else:
            if model is None:
                model = ModelPopulation.from_file(args.model, components)
                print(f"model: {len(model)} rows of {', '.join(components)} from {args.model}")
            plants = geometry_pass(char_root, survey, model, args.plants, args.seed,
                                   progress=max(args.plants // 5, 1))
            plants.save(plants_path)
            print(f"wrote {plants_path}")
        blocks = load_survey_blocks(char_root, survey, use_ossssim=False)
        meta = {"model": args.model, "components": ",".join(components),
                "plants_per_block": args.plants, "seed": args.seed}
        table = survey_efficiency(plants, blocks, cut, n_boot=args.boot, seed=args.seed,
                                  meta=meta)
        out = char_root / survey / f"eps_{args.tag}.csv"
        table.save(out)
        peak = float(table.eps.max())
        half = table.h[np.flatnonzero(table.eps >= 0.5 * peak)[-1]] if peak > 0 else float("nan")
        print(f"wrote {out}: filter {table.filter}, peak eps {peak:.3e}, "
              f"half of peak at H_{table.filter} = {half:.2f}, cut {cut.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

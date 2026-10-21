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
ecliptic latitude; a position the orbit cannot reach has weight zero. A
survey found at fewer epochs than it has is aimed at every epoch's tiles,
and ``w`` divides out the boxes that hold the orbit (``_TileSampler.weight``).
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grid_bias import OrbitModelCatalog, ecliptic_xyz_from_elements, sample_aimed_elements_at_i
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
    """Rows of an orbit model: (a, e, i, dist, H, comp), with (node, peri, M, epoch) if read."""

    a: np.ndarray
    e: np.ndarray
    i: np.ndarray
    dist: np.ndarray
    H: np.ndarray
    comp: np.ndarray
    source: str = ""
    node: np.ndarray | None = None
    peri: np.ndarray | None = None
    M: np.ndarray | None = None
    epoch: np.ndarray | None = None
    _ifree: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return int(self.a.size)

    @property
    def ifree(self) -> np.ndarray:
        """Free inclination of each row (Huang et al. 2022; heliocentric rows made barycentric)."""
        if self._ifree is None:
            if self.node is None:
                raise ValueError(f"{self.source}: the model rows have no node; i_free needs "
                                 "the full elements")
            from .ifree import free_inclination_helio

            self._ifree = np.asarray(free_inclination_helio(self.a, self.e, self.i, self.node,
                                                            self.peri, self.M, self.epoch))
        return self._ifree

    def forcing_poles(self) -> np.ndarray:
        """Each row's forcing-pole unit vector (ecliptic J2000), shape (n, 3)."""
        from .ifree import forcing_pole_vector

        return forcing_pole_vector(self.a, self.e, self.i, self.peri)

    @classmethod
    def from_file(cls, path, components=COLD_COMPONENTS) -> "ModelPopulation":
        """OSSOS 1.0 ``ModelUsed.dat`` rows whose component is in ``components``.

        Columns: a e i Omega omega M H epoch dist comment; the component is the
        comment without its trailing underscore (coldl_, coldh_, kernel_, hot_).
        """
        path = Path(path)
        keep = set(components) if components else None
        cols = {k: [] for k in ("a", "e", "i", "dist", "H", "comp", "node", "peri", "M", "epoch")}
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
                cols["node"].append(float(parts[3]))
                cols["peri"].append(float(parts[4]))
                cols["M"].append(float(parts[5]))
                cols["H"].append(float(parts[6]))
                cols["epoch"].append(float(parts[7]))
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


PLANT_FIELDS = ("block", "comp", "a", "e", "i", "r", "node", "peri", "M", "epoch", "w",
                "in_field", "mu", "rate")


@dataclass
class PlantTable:
    """Aimed plants of one survey (all blocks), with what eps needs.

    (a, e, i, node, peri, M) at ``epoch`` [JD] are the aimed elements (NaN
    angles when the aimed position is unreachable), so the same plants can be
    run through another survey's characterization.
    """

    survey: str
    block: np.ndarray
    comp: np.ndarray
    a: np.ndarray
    e: np.ndarray
    i: np.ndarray
    r: np.ndarray
    node: np.ndarray
    peri: np.ndarray
    M: np.ndarray
    epoch: np.ndarray
    w: np.ndarray            # P_geom for the block's aimed box (0 if not reachable)
    in_field: np.ndarray     # Detos1 flag 4 in the block on the unit-efficiency copy
    mu: np.ndarray           # m - H at detection (NaN when not in the field)
    rate: np.ndarray         # on-sky rate ["/hr] (NaN when not in the field)
    ifree: np.ndarray | None = None   # Huang et al. I_free of the aimed orbit
    aim: str = "ecliptic"    # "ecliptic": model i kept; "ifree": model I_free kept

    def __post_init__(self):
        if self.ifree is None:
            self.ifree = np.full(len(self.w), np.nan)

    def __len__(self) -> int:
        return int(self.w.size)

    def save(self, path) -> None:
        arrays = {k: getattr(self, k) for k in PLANT_FIELDS + ("ifree",)}
        for k in ("block", "comp"):
            arrays[k] = np.asarray(arrays[k], dtype=str)
        np.savez_compressed(path, survey=self.survey, aim=self.aim, **arrays)

    @classmethod
    def load(cls, path) -> "PlantTable":
        z = np.load(path, allow_pickle=False)
        return cls(survey=str(z["survey"]), **{k: z[k] for k in PLANT_FIELDS},
                   ifree=z["ifree"] if "ifree" in z.files else None,
                   aim=str(z["aim"]) if "aim" in z.files else "ecliptic")

    @classmethod
    def concatenate(cls, tables) -> "PlantTable":
        tables = list(tables)
        return cls(survey=tables[0].survey,
                   **{k: np.concatenate([getattr(t, k) for t in tables])
                      for k in PLANT_FIELDS + ("ifree",)}, aim=tables[0].aim)

    def for_block(self, name: str) -> "PlantTable":
        m = self.block == name
        return PlantTable(self.survey, **{k: getattr(self, k)[m] for k in PLANT_FIELDS + ("ifree",)},
                          aim=self.aim)


def _row_rate(row) -> float:
    return math.degrees(math.hypot(float(row["d_ra"]), float(row["d_dec"]))) * 3600.0 / 24.0


def plant_block(sim, block: Block, model: ModelPopulation, n_plants: int, seed: int,
                progress: int = 0) -> PlantTable:
    """``n_plants`` aimed model plants through one block of the unit-efficiency copy."""
    from .grid_bias_run import _TileSampler, attributed_to

    rng = np.random.default_rng(seed)
    sampler = _TileSampler(block)
    out = {k: np.full(n_plants, np.nan)
           for k in ("a", "e", "i", "r", "node", "peri", "M", "epoch", "w", "mu", "rate")}
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
        w = sampler.weight(tile, a, e, inc_el, node, peri, M)
        out["w"][j] = w
        out["node"][j], out["peri"][j], out["M"][j], out["epoch"][j] = node, peri, M, tile.jd
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


def poles_at_angle(rhat: np.ndarray, centre: np.ndarray, angle_deg: float) -> list:
    """Unit poles P with P . rhat = 0 at ``angle_deg`` from ``centre`` (0 or 2 solutions)."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(rhat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(rhat, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(rhat, e1)
    amp_a, amp_b = float(e1 @ centre), float(e2 @ centre)
    amp = math.hypot(amp_a, amp_b)
    target = math.cos(math.radians(angle_deg))
    if amp < 1e-15 or target > amp:
        return []
    phase = math.atan2(amp_b, amp_a)
    delta = math.acos(min(1.0, target / amp))
    return [math.cos(phi) * e1 + math.sin(phi) * e2 for phi in (phase + delta, phase - delta)]


def geometric_prob_free(area_deg2: float, ifree_deg: float, beta_f_deg: float) -> float:
    """P_geom of an orbit at free inclination i_f at latitude beta_f above its forced plane.

    The orbit pole is uniform in azimuth about the forcing pole and the
    argument of latitude uniform, so the ecliptic formula holds in the forced
    frame: A / (360 x 180 x sqrt(sin^2 i_f - sin^2 beta_f)).
    """
    from .grid_bias import geometric_detection_prob

    return geometric_detection_prob(area_deg2, ifree_deg, beta_f_deg)


def aimed_free_elements(a: float, e: float, ifree_deg: float, pole_f: np.ndarray, obs, view,
                        r_au: float, rng) -> tuple | None:
    """FoV-aimed (i, node, peri, M, beta_f) at free inclination ``ifree_deg`` about ``pole_f``."""
    from .grid_bias import barycentric_on_icrs_los, peri_m_from_position, sample_mosaic_icrs

    ra, dec = sample_mosaic_icrs(rng, survey=view)
    pos = barycentric_on_icrs_los(obs, ra, dec, r_au)
    if pos is None:
        return None
    rhat = np.asarray(pos, dtype=float) / np.linalg.norm(pos)
    poles = poles_at_angle(rhat, pole_f, ifree_deg)
    if not poles:
        return None
    pole = poles[int(rng.integers(0, len(poles)))]
    inc = math.degrees(math.acos(max(-1.0, min(1.0, float(pole[2])))))
    node = math.degrees(math.atan2(float(pole[0]), -float(pole[1]))) % 360.0
    f_sign = 1.0 if rng.random() < 0.5 else -1.0
    peri, mean_anom = peri_m_from_position(a, e, inc, node, pos, f_sign)
    beta_f = math.degrees(math.asin(max(-1.0, min(1.0, float(rhat @ pole_f)))))
    return inc, node, peri, mean_anom, beta_f


def plant_block_free(sim, block: Block, model: ModelPopulation, n_plants: int, seed: int,
                     progress: int = 0) -> PlantTable:
    """Like ``plant_block``, but each plant keeps its model row's I_free, not its ecliptic i.

    The model row's I_free and forcing pole (Huang et al. 2022) fix the aimed
    orbit pole through the planted position (two solutions, one drawn); the
    weight is the geometric probability at the position's latitude above the
    forced plane. The plant's own I_free (from its aimed elements, which
    shifts its forcing pole slightly) is stored for the sample cut.
    """
    from .grid_bias_run import _TileSampler, attributed_to
    from .ifree import free_inclination

    rng = np.random.default_rng(seed)
    sampler = _TileSampler(block)
    model_ifree = model.ifree
    poles = model.forcing_poles()
    out = {k: np.full(n_plants, np.nan)
           for k in ("a", "e", "i", "r", "node", "peri", "M", "epoch", "w", "mu", "rate", "ifree")}
    comp = np.empty(n_plants, dtype=object)
    in_field = np.zeros(n_plants, dtype=bool)
    H = 5.0
    for j in range(n_plants):
        k = int(rng.integers(len(model)))
        a, e, r = float(model.a[k]), float(model.e[k]), float(model.dist[k])
        out["a"][j], out["e"][j], out["r"][j] = a, e, r
        out["i"][j] = float(model.i[k])
        comp[j] = model.comp[k]
        tile, view, obs = sampler.draw(rng)
        el = aimed_free_elements(a, e, float(model_ifree[k]), poles[k], obs, view, r, rng)
        if el is None:
            out["w"][j] = 0.0
            continue
        inc, node, peri, M, _beta_f = el
        ifree_k, pole_k = float(model_ifree[k]), poles[k]

        def prob(m_deg, area):
            xyz = np.array(ecliptic_xyz_from_elements(a, e, inc, node, peri, m_deg))
            beta = math.degrees(math.asin(max(-1.0, min(1.0, float(xyz @ pole_k)
                                                         / float(np.linalg.norm(xyz))))))
            return geometric_prob_free(area, ifree_k, beta)

        w = sampler.weight(tile, a, e, inc, node, peri, M, prob=prob)
        out["w"][j] = w
        out["i"][j], out["node"][j], out["peri"][j], out["M"][j] = inc, node, peri, M
        out["epoch"][j] = tile.jd
        out["ifree"][j] = float(free_inclination(a, e, inc, node, peri)[0])
        if w <= 0.0:
            continue
        rows = sim.epoch_rows(a, e, inc, node, peri, M, H, epoch_jd=tile.jd)
        if attributed_to(rows, block):
            in_field[j] = True
            row = rows[0]
            out["mu"][j] = float(row["m_int"].value) - H
            out["rate"][j] = _row_rate(row)
        if progress and (j + 1) % progress == 0:
            print(f"    {block.survey}/{block.name}: {j + 1}/{n_plants} plants, "
                  f"{int(in_field[: j + 1].sum())} in the field", flush=True)
    return PlantTable(block.survey, np.full(n_plants, block.name, dtype=object), comp,
                      in_field=in_field, aim="ifree", **out)


@dataclass
class SampleCut:
    """Orbital cut defining a survey's sample, applied to model plants as to detections.

    ``i_max``: ecliptic inclination [deg] (strict); ``ifree_max``: free
    inclination (Huang et al. 2022) [deg] (strict; the plants must carry
    ``ifree``, i.e. be planted with ``aim = "ifree"``); ``r_range``: distance at
    detection [au]; ``a_range``: semi-major axis [au]. ``None`` skips a cut.
    """

    i_max: float | None = None
    r_range: tuple | None = None
    a_range: tuple | None = None
    ifree_max: float | None = None

    def mask(self, plants: PlantTable) -> np.ndarray:
        m = np.ones(len(plants), dtype=bool)
        if self.i_max is not None:
            m &= plants.i < self.i_max
        if self.ifree_max is not None:
            if getattr(plants, "aim", "ifree") != "ifree":
                raise ValueError(f"{plants.survey}: an i_free cut needs plants aimed by i_free "
                                 "(plant again without --reuse)")
            m &= plants.ifree < self.ifree_max
        if self.r_range is not None:
            m &= (plants.r > self.r_range[0]) & (plants.r < self.r_range[1])
        if self.a_range is not None:
            m &= (plants.a > self.a_range[0]) & (plants.a < self.a_range[1])
        return m

    def describe(self) -> str:
        parts = []
        if self.i_max is not None:
            parts.append(f"i < {self.i_max:g} deg")
        if self.ifree_max is not None:
            parts.append(f"i_free < {self.ifree_max:g} deg")
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
                  seed: int = 42, work_dir=None, progress: int = 0,
                  aim: str = "ecliptic") -> PlantTable:
    """Plant tables for every block of a survey (``n_plants`` per block).

    ``aim = "ecliptic"`` keeps each model row's ecliptic i (node redrawn);
    ``aim = "ifree"`` keeps its free inclination about its forcing pole
    (``plant_block_free``), which an i_free cut needs.
    """
    planter = {"ecliptic": plant_block, "ifree": plant_block_free}[aim]
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
            t = planter(sim, block, model, n_plants, block_seed, progress)
            print(f"  {survey}/{name}: {n_plants} plants, {int(t.in_field.sum())} in "
                  f"the field ({100.0 * t.in_field.mean():.1f}%)", flush=True)
            tables.append(t)
        return PlantTable.concatenate(tables)
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Overlaps between surveys


COLD_DRIFT_DEG_PER_YR = 1.6   # cold classical sky motion bound (~1.3 deg/yr) with margin


def _block_centre(block: Block) -> tuple[float, float, float]:
    tiles = block.tiles[0]
    ra = float(np.mean([t.ra_deg for t in tiles]))
    dec = float(np.mean([t.dec_deg for t in tiles]))
    return ra, dec, float(np.mean([t.jd for t in tiles]))


def _block_radius(block: Block) -> float:
    ra, dec, _ = _block_centre(block)
    from .detections_io import _angular_sep_deg

    return max(_angular_sep_deg(ra, dec, t.ra_deg, t.dec_deg)
               + 0.5 * math.hypot(t.width_deg, t.height_deg) for t in block.tiles[0])


def reachable_pairs(blocks_a: dict, blocks_b: dict, margin_deg: float = 3.0) -> list:
    """Block pairs whose fields a cold object could occupy at both epochs."""
    from .detections_io import _angular_sep_deg

    pairs = []
    for na, a in blocks_a.items():
        ra_a, dec_a, jd_a = _block_centre(a)
        for nb, b in blocks_b.items():
            ra_b, dec_b, jd_b = _block_centre(b)
            reach = (COLD_DRIFT_DEG_PER_YR * abs(jd_b - jd_a) / 365.25 + margin_deg
                     + _block_radius(a) + _block_radius(b))
            if _angular_sep_deg(ra_a, dec_a, ra_b, dec_b) <= reach:
                pairs.append((na, nb))
    return pairs


@dataclass
class OverlapTable:
    """A's in-field plants through B: in B's field, B's block, mu and rate there."""

    survey_a: str
    survey_b: str
    index: np.ndarray        # rows of A's plant table that were run
    in_b: np.ndarray
    block_b: np.ndarray
    mu_b: np.ndarray
    rate_b: np.ndarray


def run_through(plants: PlantTable, char_root, survey_b: str, seed: int = 42,
                work_dir=None) -> OverlapTable:
    """Run A's in-field plants (stored elements) through B's unit-efficiency copy."""
    from ossssim.color import PhotSpec

    from .grid_bias_run import GridBiasSimulator, attributed_to

    index = np.flatnonzero(plants.in_field)
    out = OverlapTable(plants.survey, survey_b, index, np.zeros(index.size, dtype=bool),
                       np.full(index.size, "", dtype=object), np.full(index.size, np.nan),
                       np.full(index.size, np.nan))
    if index.size == 0:
        return out
    own = work_dir is None
    work = Path(tempfile.mkdtemp(prefix="sdovl", dir="/tmp")) if own else Path(work_dir)
    try:
        unit_root = make_unit_characterization(char_root, survey_b, work)
        blocks = load_survey_blocks(unit_root, survey_b, use_ossssim=True)
        sim = GridBiasSimulator(survey_b, blocks, PhotSpec(), "r", seed=seed)
        H = 5.0
        for n, j in enumerate(index):
            rows = sim.epoch_rows(plants.a[j], plants.e[j], plants.i[j], plants.node[j],
                                  plants.peri[j], plants.M[j], H, epoch_jd=plants.epoch[j])
            for name, block in blocks.items():
                if attributed_to(rows, block):
                    out.in_b[n] = True
                    out.block_b[n] = name
                    out.mu_b[n] = float(rows[0]["m_int"].value) - H
                    out.rate_b[n] = _row_rate(rows[0])
                    break
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)
    print(f"  {plants.survey} through {survey_b}: {int(out.in_b.sum())} of "
          f"{index.size} in-field plants also in {survey_b}'s fields "
          f"(blocks: {sorted(set(out.block_b[out.in_b]))})", flush=True)
    return out


def overlap_efficiency(plants: PlantTable, blocks_a: dict, overlap: OverlapTable,
                       blocks_b: dict, colour_a: float, colour_b: float,
                       cut: SampleCut | None = None, h_grid=H_GRID, n_boot: int = 32,
                       seed: int = 0) -> EfficiencyTable:
    """eps_{A and B}(H) in the model band: found by both surveys.

    ``colour_a``, ``colour_b`` are each survey's (filter - model band) mean
    colour; the product of the two selections is taken per plant.
    """
    cut = cut or SampleCut()
    h = np.arange(h_grid[0], h_grid[1] + 0.5 * h_grid[2], h_grid[2])
    rng = np.random.default_rng(seed)
    keep_a = cut.mask(plants)
    eps = np.zeros(len(h))
    boot = np.zeros((n_boot, len(h)))
    in_b = overlap.in_b & np.isin(overlap.block_b.astype(str), list(blocks_b))
    rows = overlap.index[in_b]
    sub_b = np.flatnonzero(in_b)
    for name, block in blocks_a.items():
        n_b = int((plants.block == name).sum())
        pick = (plants.block[rows] == name) & keep_a[rows]
        if not pick.any():
            continue
        j, k = rows[pick], sub_b[pick]
        sel_a = block.selection
        s = np.empty((j.size, len(h)))
        for n in range(j.size):
            sel_b = blocks_b[overlap.block_b[k[n]]].selection
            m_a = h + colour_a + plants.mu[j[n]]
            m_b = h + colour_b + overlap.mu_b[k[n]]
            s[n] = (sel_a.measured_selection(m_a, float(plants.rate[j[n]]))
                    * sel_b.measured_selection(m_b, float(overlap.rate_b[k[n]])))
        w = plants.w[j]
        eps += (w @ s) / n_b
        if n_boot:
            pw = rng.poisson(1.0, size=(n_boot, w.size)) * w[None, :]
            boot += (pw @ s) / n_b
    meta = {"overlap": f"{overlap.survey_a} and {overlap.survey_b}",
            "colours": f"{colour_a:+.3f} {colour_b:+.3f} (filter - model band)",
            "n_both": int(overlap.in_b.sum()), "cut": cut.describe()}
    return EfficiencyTable(f"{overlap.survey_a}x{overlap.survey_b}", "model", h, eps, boot, meta)


def check_detections(char_root, survey: str, n_per_block: int = 30, h_offset: float = -1.5,
                     seed: int = 42) -> dict:
    """Recovery of each block's own release detections by its characterization.

    Each object's catalogue orbit is run through the real characterization at
    its discovery epoch, ``h_offset`` mag brighter than its Hsur so that the
    efficiency is near its plateau; a block whose characterization rejects
    its own objects (a wrong rate cut, footprint or epoch) recovers few.
    Returns ``{block: (recovered, tried, median motion angle [deg])}``.
    """
    from ossssim.color import PhotSpec

    from .detections_io import BLOCK_PREFIXES, assign_block, read_detections_file
    from .grid_bias_run import GridBiasSimulator, attributed_to

    char_root = Path(char_root)
    blocks = load_survey_blocks(char_root, survey)
    sim = GridBiasSimulator(survey, blocks, PhotSpec(), next(iter(blocks.values())).filter,
                            seed=seed)
    rows = read_detections_file(char_root / survey / f"{survey}.detections")
    by_block: dict = {}
    for r in rows:
        if not (r["a"] > 0 and 0 <= r["e"] < 1):
            continue
        b = assign_block(r["object"], blocks, BLOCK_PREFIXES.get(survey),
                         r["RAdeg"], r["DEdeg"], r["JD"])
        by_block.setdefault(b, []).append(r)
    out = {}
    for b, members in sorted(by_block.items()):
        hit, angles = 0, []
        for r in members[:n_per_block]:
            n = 0.9856076686 / r["a"] ** 1.5
            M = (n * (r["JD"] - (r["tperi"] + 2400000.5))) % 360.0
            res = sim.epoch_rows(r["a"], r["e"], r["i"], r["Omega"], r["omega"], M,
                                 r["Hsur"] + h_offset, epoch_jd=r["JD"])
            hit += attributed_to(res, blocks[b])
            if float(res[0]["d_ra"]) or float(res[0]["d_dec"]):
                angles.append(math.degrees(math.atan2(float(res[0]["d_dec"]),
                                                      float(res[0]["d_ra"]))))
        out[b] = (hit, min(len(members), n_per_block),
                  float(np.median(angles)) if angles else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Complete surveys (a catalogue that has found everything below an H limit)


@dataclass
class CompleteSpec:
    """A survey complete for ``H_filter < h_lim`` over the whole sky (``format = "complete"``).

    Read from the ``[survey]`` table of its TOML (for example
    ``characterization/MPC/MPC.toml``): ``surveys`` (its name), ``filter``,
    ``h_lim``, ``sigma_h`` (scatter of the catalogue H about the true H in
    ``filter``), ``detections_relpath`` and ``duplicates_relpath``.
    """

    name: str
    filter: str
    h_lim: float
    sigma_h: float = 0.0
    detections: str = ""
    duplicates: str = ""
    columns: dict = field(default_factory=dict)
    path: str = ""

    @classmethod
    def load(cls, path) -> "CompleteSpec":
        import tomllib

        with Path(path).open("rb") as fh:
            cfg = tomllib.load(fh)
        s = cfg["survey"]
        if s.get("format", "complete") != "complete":
            raise ValueError(f"{path}: [survey] format must be 'complete'")
        return cls(name=s["surveys"][0], filter=s["filter"], h_lim=float(s["h_lim"]),
                   sigma_h=float(s.get("sigma_h", 0.0)),
                   detections=s.get("detections_relpath", ""),
                   duplicates=s.get("duplicates_relpath", ""),
                   columns=dict(cfg.get("detection_columns", {})), path=str(path))


def step_weights(h: np.ndarray, h_lim: float) -> np.ndarray:
    """Fraction of each H cell [h - dh/2, h + dh/2] below ``h_lim``.

    With these weights the rectangle sum over the grid integrates exactly to
    ``h_lim``, as ``JointLikelihood.expected`` does.
    """
    dh = np.gradient(h)
    return np.clip((h_lim - (h - 0.5 * dh)) / dh, 0.0, 1.0)


def model_cut_fraction(model: ModelPopulation, cut: SampleCut, n_boot: int = 32,
                       seed: int = 0) -> tuple[float, np.ndarray]:
    """Fraction of model rows inside ``cut`` (a, i, i_free, and r = the row's distance)."""
    m = np.ones(len(model), dtype=bool)
    if cut.i_max is not None:
        m &= model.i < cut.i_max
    if cut.ifree_max is not None:
        m &= model.ifree < cut.ifree_max
    if cut.r_range is not None:
        m &= (model.dist > cut.r_range[0]) & (model.dist < cut.r_range[1])
    if cut.a_range is not None:
        m &= (model.a > cut.a_range[0]) & (model.a < cut.a_range[1])
    rng = np.random.default_rng(seed)
    boot = rng.binomial(len(model), m.mean(), size=n_boot) / len(model)
    return float(m.mean()), boot


def complete_efficiency(spec: CompleteSpec, f_cut: float, f_boot: np.ndarray,
                        meta: dict | None = None, h_grid=H_GRID) -> EfficiencyTable:
    """eps(H_filter) = f_cut * 1(H_filter < h_lim): every object below the limit is known.

    The cut on the catalogue H is in the survey's filter; the catalogue H
    scatter (``sigma_h``) is applied to the model LF by the joint fit, not here.
    """
    h = np.arange(h_grid[0], h_grid[1] + 0.5 * h_grid[2], h_grid[2])
    w = step_weights(h, spec.h_lim)
    meta = dict(meta or {})
    meta.update({"complete": f"{spec.filter} < {spec.h_lim:g}", "f_cut": f"{f_cut:.5f}",
                 "sigma_h": f"{spec.sigma_h:g}", "spec": spec.path})
    meta.setdefault("excluded_blocks", "none")
    return EfficiencyTable(spec.name, spec.filter, h, f_cut * w, f_boot[:, None] * w[None, :],
                           meta)


def complete_sigma(filt: str, other_filt: str, sigma_c: float, sigma_h: float) -> float:
    """Scatter of the complete survey's catalogue H about an object's H in another survey.

    The catalogue scatter ``sigma_h``, plus the colour spread ``sigma_c``
    when the two surveys measure in different filters.
    """
    return math.hypot(sigma_h, sigma_c if filt != other_filt else 0.0)


def complete_overlap(spec: CompleteSpec, other: EfficiencyTable, colour_spec: float,
                     colour_other: float, sigma: float, h_grid=H_GRID) -> EfficiencyTable:
    """eps(both) in the model band: found by ``other`` and below the complete survey's limit.

    ``other`` is eps in its own filter (``H_o = H_r + colour_other``); the
    catalogue magnitude ``H_r + colour_spec`` scatters by ``sigma`` (colour
    spread and catalogue noise), so

        eps_x(H_r) = eps_o(H_r + colour_other) * Phi((h_lim - H_r - colour_spec) / sigma)
    """
    from scipy.special import ndtr

    h = np.arange(h_grid[0], h_grid[1] + 0.5 * h_grid[2], h_grid[2])
    if sigma > 0.0:
        below = ndtr((spec.h_lim - h - colour_spec) / sigma)
    else:
        below = step_weights(h, spec.h_lim - colour_spec)
    eps_o = np.interp(h + colour_other, other.h, other.eps, right=0.0)
    boot = (np.array([np.interp(h + colour_other, other.h, b, right=0.0) for b in other.boot])
            if len(other.boot) else np.zeros((0, len(h))))
    meta = {"overlap": f"{spec.name} and {other.survey}", "complete": f"{spec.filter} < "
            f"{spec.h_lim:g}", "colour_spec": f"{colour_spec:+.3f}",
            "colour_other": f"{colour_other:+.3f}", "sigma": f"{sigma:g}",
            "cut": other.meta.get("cut", "")}
    return EfficiencyTable(f"{spec.name}x{other.survey}", "model", h, eps_o * below,
                           boot * below[None, :], meta)


def complete_main(args) -> int:
    """``survey_debias_eff --complete SPEC``: analytic eps and overlap tables."""
    from .colours import ColourMap, load_colours
    from .joint import SIGMA_C

    root = Path(args.root)
    char_root = root / "characterization"
    spec = CompleteSpec.load(root / args.complete if not Path(args.complete).is_absolute()
                             else args.complete)
    components = tuple(c for c in args.components.split(",") if c)
    model = ModelPopulation.from_file(args.model, components)
    cut = SampleCut(args.i_max, _range(args.r_min, args.r_max), _range(args.a_min, args.a_max),
                    args.ifree_max)
    f_cut, f_boot = model_cut_fraction(model, cut, n_boot=args.boot, seed=args.seed)
    meta = {"model": args.model, "components": ",".join(components), "n_model": len(model),
            "cut": cut.describe(), "seed": args.seed}
    table = complete_efficiency(spec, f_cut, f_boot, meta)
    out = char_root / spec.name / f"eps_{args.tag}.csv"
    table.save(out)
    print(f"wrote {out}: {spec.name} complete for {spec.filter} < {spec.h_lim:g}, "
          f"f_cut = {f_cut:.4f} ({cut.describe()}, {len(model)} model rows)")
    if not args.overlap_with:
        return 0
    colour_path = Path(args.colour_file) if args.colour_file else root / "colour.toml"
    photspec, _ = load_colours(colour_path)
    colours = ColourMap(photspec, args.model_band)
    sigma_c = SIGMA_C if args.sigma_c is None else args.sigma_c
    c_spec = colours.colour(spec.filter, args.colour_group)
    for other in args.overlap_with:
        path = char_root / other / f"eps_{args.tag}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found: run survey_debias_eff for {other} first")
        eps_o = EfficiencyTable.load(path)
        c_other = colours.colour(eps_o.filter, args.colour_group)
        sigma = complete_sigma(spec.filter, eps_o.filter, sigma_c, spec.sigma_h)
        ov = complete_overlap(spec, eps_o, c_spec, c_other, sigma)
        out = char_root / spec.name / f"eps_{args.tag}_x_{other}.csv"
        ov.save(out)
        print(f"wrote {out}: peak eps(both) {float(ov.eps.max()):.3e}, "
              f"integral below H_r = 6: {float(np.sum(ov.eps[ov.h < 6]) * H_GRID[2]):.3e}")
    return 0


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
    p.add_argument("--survey", action="append", default=[],
                   help="characterization/{survey} (repeat for several)")
    p.add_argument("--complete", default=None, metavar="SPEC",
                   help="complete-survey TOML (relative to --root), e.g. "
                        "characterization/MPC/MPC.toml: write the analytic eps = f_cut * "
                        "1(H < h_lim) instead of planting; --overlap-with then writes "
                        "eps_{tag}_x_{SURVEY}.csv from SURVEY's own eps_{tag}.csv")
    p.add_argument("--sigma-c", type=float, default=None,
                   help="colour spread for --complete overlaps (default joint SIGMA_C)")
    p.add_argument("--model", required=True, help="OSSOS 1.0 ModelUsed.dat file")
    p.add_argument("--components", default=",".join(COLD_COMPONENTS),
                   help=f"model components (default {','.join(COLD_COMPONENTS)})")
    p.add_argument("--tag", default="cold", help="name used in the output files")
    p.add_argument("--plants", type=int, default=20000, help="aimed plants per block")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--boot", type=int, default=32, help="bootstrap replicates of eps")
    p.add_argument("--i-max", type=float, default=None, help="sample cut: i < I_MAX [deg]")
    p.add_argument("--ifree-max", type=float, default=None,
                   help="sample cut: free inclination (Huang et al. 2022) < IFREE_MAX [deg]; "
                        "plants are then aimed by i_free (--aim ifree)")
    p.add_argument("--aim", choices=("ecliptic", "ifree"), default=None,
                   help="planting: keep the model's ecliptic i or its i_free (default ifree "
                        "with --ifree-max, else ecliptic)")
    p.add_argument("--r-min", type=float, default=None, help="sample cut: r > R_MIN [au]")
    p.add_argument("--r-max", type=float, default=None, help="sample cut: r < R_MAX [au]")
    p.add_argument("--a-min", type=float, default=None, help="sample cut: a > A_MIN [au]")
    p.add_argument("--a-max", type=float, default=None, help="sample cut: a < A_MAX [au]")
    p.add_argument("--reuse", action="store_true",
                   help="reuse an existing plant table instead of planting again")
    p.add_argument("--overlap-with", action="append", default=[], metavar="SURVEY",
                   help="also write eps for objects found by both this survey and SURVEY "
                        "(eps_{tag}_x_{SURVEY}.csv, model band)")
    p.add_argument("--colour-file", default=None,
                   help="colours for the overlap (default <root>/colour.toml)")
    p.add_argument("--colour-group", default="cold", help="spectral group (default cold)")
    p.add_argument("--model-band", default="r")
    p.add_argument("--exclude-block", action="append", default=[], metavar="BLOCK",
                   help="leave BLOCK out of eps, and out of the other survey in overlaps (its "
                        "detections must be left out of the fit)")
    p.add_argument("--check-detections", action="store_true",
                   help="only report how many of each block's release detections its "
                        "characterization recovers")
    args = p.parse_args(argv)

    if args.complete:
        return complete_main(args)
    if not args.survey:
        p.error("--survey is required (or --complete SPEC)")
    char_root = Path(args.root) / "characterization"
    if args.check_detections:
        for survey in args.survey:
            for b, (hit, tried, angle) in check_detections(char_root, survey).items():
                flag = "  <-- low" if tried and hit < 0.3 * tried else ""
                print(f"{survey}/{b}: {hit}/{tried} release detections recovered, "
                      f"median motion angle {angle:.1f} deg{flag}")
        return 0
    components = tuple(c for c in args.components.split(",") if c)
    model = None
    cut = SampleCut(args.i_max, _range(args.r_min, args.r_max), _range(args.a_min, args.a_max),
                    args.ifree_max)
    aim = args.aim or ("ifree" if args.ifree_max is not None else "ecliptic")
    for survey in args.survey:
        plants_path = char_root / survey / f"plants_{args.tag}.npz"
        if args.reuse and plants_path.is_file():
            plants = PlantTable.load(plants_path)
            if plants.aim != aim:
                raise SystemExit(f"{plants_path} was aimed by {plants.aim}, not {aim}: "
                                 "plant again without --reuse")
            print(f"{survey}: reusing {plants_path} ({len(plants)} plants)")
        else:
            if model is None:
                model = ModelPopulation.from_file(args.model, components)
                print(f"model: {len(model)} rows of {', '.join(components)} from {args.model}")
            plants = geometry_pass(char_root, survey, model, args.plants, args.seed,
                                   progress=max(args.plants // 5, 1), aim=aim)
            plants.save(plants_path)
            print(f"wrote {plants_path}")
        blocks = load_survey_blocks(char_root, survey, use_ossssim=False)
        excluded = [b for b in args.exclude_block if b in blocks]
        blocks = {k: v for k, v in blocks.items() if k not in excluded}
        meta = {"model": args.model, "components": ",".join(components),
                "plants_per_block": args.plants, "seed": args.seed, "aim": plants.aim,
                "excluded_blocks": ",".join(excluded) or "none"}
        table = survey_efficiency(plants, blocks, cut, n_boot=args.boot, seed=args.seed,
                                  meta=meta)
        out = char_root / survey / f"eps_{args.tag}.csv"
        table.save(out)
        peak = float(table.eps.max())
        half = table.h[np.flatnonzero(table.eps >= 0.5 * peak)[-1]] if peak > 0 else float("nan")
        print(f"wrote {out}: filter {table.filter}, peak eps {peak:.3e}, "
              f"half of peak at H_{table.filter} = {half:.2f}, cut {cut.describe()}")
        for other in args.overlap_with:
            write_overlap(args, char_root, survey, plants, blocks, other, cut)
    return 0


def write_overlap(args, char_root, survey, plants, blocks, other, cut) -> None:
    from .colours import ColourMap, load_colours

    blocks_b = {k: v for k, v in load_survey_blocks(char_root, other, use_ossssim=False).items()
                if k not in set(args.exclude_block)}
    pairs = reachable_pairs(blocks, blocks_b)
    out = char_root / survey / f"eps_{args.tag}_x_{other}.csv"
    if not pairs:
        print(f"  {survey} and {other}: no block pair a cold object can reach at both "
              f"epochs; no overlap file")
        if out.exists():
            out.unlink()
        return
    print(f"  {survey} and {other}: {len(pairs)} reachable block pair(s)")
    colour_path = Path(args.colour_file) if args.colour_file else Path(args.root) / "colour.toml"
    photspec, _ = load_colours(colour_path)
    colours = ColourMap(photspec, args.model_band)
    group = args.colour_group
    filt_a = next(iter(blocks.values())).filter
    filt_b = next(iter(blocks_b.values())).filter
    overlap = run_through(plants, char_root, other, seed=args.seed)
    table = overlap_efficiency(plants, blocks, overlap, blocks_b,
                               colours.colour(filt_a, group), colours.colour(filt_b, group),
                               cut, n_boot=args.boot, seed=args.seed)
    table.save(out)
    print(f"wrote {out}: peak eps(both) {float(table.eps.max()):.3e}")


if __name__ == "__main__":
    raise SystemExit(main())

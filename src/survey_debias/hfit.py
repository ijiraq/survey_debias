"""Maximum-likelihood fit of the exponentially tapered H distribution.

The cumulative H distribution (Kavelaars et al. 2021; Petit et al. 2023,
ApJL 946, L4, Eq. B4) is

    N(<H) = 10**(alpha * (H - H_o)) * exp(-10**(-beta * (H - H_B)))

so the differential distribution is

    dN/dH = ln(10) * N(<H) * (alpha + beta * 10**(-beta * (H - H_B))).

Shape (alpha, beta, H_B)
    Each detection k is seen at its own geometry (distance, phase, rate).
    If the population factorises as dN = lambda(H) g(geometry), the
    probability density of its H, given its geometry and that it was
    detected, is

        f_k(H) = lambda(H) S(m_k + H - H_k) / integral lambda(H') S(m_k + H' - H_k) dH'

    where m_k is its apparent magnitude in its block's filter and S(m) is
    that block's magnitude selection (efficiency x tracking x
    characterisation limit, at every epoch), read from
    ``characterization/{survey}/{block}.eff``. Detections from different
    blocks simply use different S. This conditional likelihood is exact
    under that assumption, needs no model of the orbital distribution, and
    gives ordinary (not weight-inflated) uncertainties. H_o cancels from it.

Normalisation (H_o)
    The Horvitz-Thompson sum of 1/bias over detections brighter than
    ``h_norm`` estimates the number of objects in [h_min, h_norm]. With a
    union bias (the default) bias is P(detect in any block | cell); with
    per-block biases each block's count is an estimate of the same total
    and they are combined by inverse variance. The variance has a sampling
    term sum w(w - 1) and a Monte Carlo term
    sum_cells (sum_{k in cell} w_k)^2 (se/bias)^2 from the bias estimates.
    ``h_norm`` defaults to the faintest H at which every detected geometry
    still has at least half of its peak selection.

Bootstrap
    ``bootstrap`` resamples detections within each block, multiplies every
    (block, cell) bias estimate by an independent lognormal factor with its
    MC relative error, refits the MLE from the original optimum, and
    recomputes H_o from the resampled HT count.
"""
from __future__ import annotations

import csv
import io
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .efficiency import (  # noqa: F401 - re-exported
    ETA_CHARACTERISED,
    ETA_FLOOR,
    EpochSelection,
    RateBlock,
    SurveySelection,
    parse_efficiency_spec,
    read_efficiency_file,
)
from .grid_bias import GridSurvey

LN10 = math.log(10.0)
# Earth's orbital speed as an on-sky rate at 1 au ["/hr].
EARTH_RATE_ASPHR = 147.9

PARAM_NAMES = ("alpha", "beta", "h_b")
PARAM_LABELS = {
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "h_b": r"$H_B$",
    "h_o": r"$H_o$",
}
DEFAULT_PRIORS = {"alpha": (0.05, 2.0), "beta": (0.02, 2.0)}
MEASUREMENT_MODES = ("model", "none")
# Scatter below this many H cells is raised to it (the kernel must be resolved).
SIGMA_FLOOR_CELLS = 2


# ---------------------------------------------------------------------------
# Functional form


def log_cumulative(h, alpha, beta, h_b, h_o=0.0):
    """ln N(<H) for the exponentially tapered form."""
    h = np.asarray(h, dtype=float)
    with np.errstate(over="ignore"):
        return LN10 * alpha * (h - h_o) - 10.0 ** (-beta * (h - h_b))


def log_differential(h, alpha, beta, h_b, h_o=0.0):
    """ln dN/dH for the exponentially tapered form (-inf where it underflows)."""
    h = np.asarray(h, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        taper = 10.0 ** (-beta * (h - h_b))
        out = (
            LN10 * alpha * (h - h_o) - taper
            + math.log(LN10) + np.log(alpha + beta * taper)
        )
    return np.where(np.isnan(out), -np.inf, out)


def log_differential_exponential(h, alpha, h_o=0.0):
    """ln dN/dH for a single exponential N(<H) = 10**(alpha (H - H_o))."""
    h = np.asarray(h, dtype=float)
    return LN10 * alpha * (h - h_o) + math.log(LN10 * alpha)


def cumulative(h, alpha, beta, h_b, h_o):
    """N(<H) for the exponentially tapered form."""
    return np.exp(log_cumulative(h, alpha, beta, h_b, h_o))


# ---------------------------------------------------------------------------
# Survey magnitude selection (mirrors OSSSSim effut.f95 / surveysub.f95)


def block_selections(project, sample: "HSample", eff_files=None,
                     efficiency=None) -> dict:
    """``{"survey/block": SurveySelection}`` for every group in the sample.

    Each detection uses ``characterization/{survey}/[epoch{i}/]{block}.eff``.
    ``efficiency`` (a spec string) or ``eff_files`` (one per epoch, or one
    for all) replace the files for every block.
    """
    out = {}
    for group in np.unique(sample.group):
        survey, _, block = str(group).partition("/")
        if efficiency or eff_files:
            n = project.block(survey, block).n_epochs if project else 1
            if efficiency:
                out[group] = SurveySelection([parse_efficiency_spec(efficiency)] * n)
            else:
                files = [Path(f) for f in eff_files]
                if len(files) == 1:
                    files = files * n
                if len(files) != n:
                    raise ValueError(f"give one --eff-file, or one per epoch ({n})")
                out[group] = SurveySelection([read_efficiency_file(f) for f in files])
        else:
            out[group] = project.block(survey, block).selection
    return out


# ---------------------------------------------------------------------------
# Detections


def _object_array(values) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        out[i] = v
    return out


@dataclass
class HSample:
    names: np.ndarray
    h: np.ndarray        # model-band absolute magnitude used in the fit
    m: np.ndarray        # apparent magnitude in the block's filter
    bias: np.ndarray     # P(detect | cell): union over blocks, or own block
    rate: np.ndarray     # on-sky rate ["/hr] used to pick efficiency blocks
    h_column: str
    source: str
    rate_note: str = ""
    group: np.ndarray = None      # "survey/block"
    bias_key: np.ndarray = None   # detections sharing a key share one bias estimate
    bias_se: np.ndarray = None    # MC standard error of bias (NaN if unknown)
    bias_mode: str = "union"
    d: np.ndarray = None          # distance [au] behind h (NaN if unknown)
    d_err: np.ndarray = None      # its uncertainty [au] (NaN: exact)
    m_err: np.ndarray = None      # per-object magnitude error (NaN: use the .eff model)

    def __post_init__(self):
        n = len(self.h)
        if self.group is None:
            self.group = np.array(["all"] * n, dtype=object)
        if self.bias_key is None:
            self.bias_key = _object_array([(i,) for i in range(n)])
        if self.bias_se is None:
            self.bias_se = np.full(n, np.nan)
        for name in ("d", "d_err", "m_err"):
            if getattr(self, name) is None:
                setattr(self, name, np.full(n, np.nan))

    def __len__(self) -> int:
        return len(self.h)

    @property
    def survey(self) -> np.ndarray:
        return np.array([str(g).partition("/")[0] for g in self.group], dtype=object)

    def subset(self, mask) -> "HSample":
        return HSample(self.names[mask], self.h[mask], self.m[mask],
                       self.bias[mask], self.rate[mask], self.h_column,
                       self.source, self.rate_note, self.group[mask],
                       self.bias_key[mask], self.bias_se[mask], self.bias_mode,
                       self.d[mask], self.d_err[mask], self.m_err[mask])


def _read_results(path: Path) -> tuple[list[dict], dict]:
    """Rows of a results CSV and its ``# key: value`` header comments."""
    meta = {}
    lines = []
    for ln in path.read_text().splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            key, sep, value = stripped.lstrip("#").partition(":")
            if sep:
                meta.setdefault(key.strip(), value.strip())
            continue
        lines.append(ln)
    if not lines:
        raise ValueError(f"{path}: no header row")
    return list(csv.DictReader(io.StringIO("\n".join(lines)))), meta


def opposition_rate(d_au) -> np.ndarray:
    """Approximate on-sky rate ["/hr] at opposition for distance d (au)."""
    d = np.asarray(d_au, dtype=float)
    return EARTH_RATE_ASPHR * (1.0 - 1.0 / np.sqrt(d)) / (d - 1.0)


def _column(header, survey: GridSurvey, canon: str) -> str | None:
    """File column for a program name, or the derived column the run wrote."""
    if canon in survey.detection_columns:
        return survey.detection_columns[canon]
    for name in (f"{canon}_computed", canon):
        if name in header:
            return name
    return None


def load_hsample(path, survey: GridSurvey, h_column: str | None = None,
                 rate_asphr: float | None = None) -> HSample:
    """Read a ``bias_results.csv`` written by ``survey_debias``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"results file not found: {path} (run survey_debias first)"
        )
    rows, meta = _read_results(path)
    if not rows:
        raise ValueError(f"{path}: no detections")
    header = list(rows[0])
    if h_column is None:
        h_column = "Hx_computed" if "Hx_computed" in header else "Hx"
    mag_col = survey.detection_columns["mag"]
    dist_col = survey.detection_columns.get("d_bary")
    name_col = survey.detection_columns.get("name", "name")
    for col in (h_column, mag_col, "bias"):
        if col not in header:
            raise ValueError(
                f"{path}: column {col!r} not found. Columns: {', '.join(header)}"
            )

    def numbers(col, blank=None):
        out = []
        for n, row in enumerate(rows, start=1):
            text = (row.get(col) or "").strip()
            if not text and blank is not None:
                out.append(blank)
                continue
            try:
                out.append(float(text))
            except ValueError:
                raise ValueError(
                    f"{path}: row {n} column {col!r} value {text!r} is not a number"
                ) from None
        return np.array(out)

    def texts(canon, default):
        col = _column(header, survey, canon)
        fallback = survey.detection_defaults.get(canon, default)
        if col is None:
            return [fallback] * len(rows)
        return [(row.get(col) or "").strip() or fallback for row in rows]

    h = numbers(h_column)
    m = numbers(mag_col)
    bias = numbers("bias")
    if np.any(bias <= 0.0):
        raise ValueError(f"{path}: bias must be positive for every detection")
    se_col = _column(header, survey, "bias_se")
    bias_se = numbers(se_col, blank=np.nan) if se_col else np.full(len(h), np.nan)
    names = np.array([
        (row.get(name_col) or str(i)).strip() for i, row in enumerate(rows, start=1)
    ])
    surveys = texts("survey", "all")
    blocks = texts("block", "all")
    group = np.array([f"{s}/{b}" for s, b in zip(surveys, blocks)], dtype=object)
    cg = texts("colour_group", "default")
    cell_col = _column(header, survey, "cell")
    cells = [row.get(cell_col, "") for row in rows] if cell_col else [""] * len(rows)
    mode = meta.get("bias_mode", "union")
    if mode == "block":
        keys = [(s, b, g, c) for s, b, g, c in zip(surveys, blocks, cg, cells)]
    else:
        keys = [(g, c) for g, c in zip(cg, cells)]
    bias_key = _object_array([k if cell_col else (i,) for i, k in enumerate(keys)])
    note = ""
    d = numbers(dist_col) if dist_col and dist_col in header else np.full(len(h), np.nan)
    if rate_asphr is not None:
        rate = np.full(len(h), float(rate_asphr))
        note = f"rate {rate_asphr:g}\"/hr for every detection (--rate)"
    elif np.all(np.isfinite(d)):
        rate = opposition_rate(d)
        note = f"rate from {dist_col} at opposition"
    else:
        rate = np.full(len(h), np.nan)

    def optional(canon):
        col = survey.detection_columns.get(canon)
        if col is None:
            return np.full(len(h), np.nan)
        if col not in header:
            raise ValueError(f"{path}: column {col!r} ({canon}) not found")
        return numbers(col, blank=np.nan)

    return HSample(names, h, m, bias, rate, h_column, str(path), note,
                   group, bias_key, bias_se, mode, d,
                   optional("d_bary_err"), optional("mag_err"))


# ---------------------------------------------------------------------------
# Monte Carlo bias table (for the HT variance and the bootstrap)


@dataclass
class BiasTable:
    """Bias estimates behind each detection's bias.

    ``matrix[k, j]`` is 1 when estimate j is summed into detection k's
    bias (every block of the project for a union bias, the detection's own
    block otherwise). ``rel_err[j]`` is the estimate's relative MC error.
    """

    keys: list
    value: np.ndarray
    rel_err: np.ndarray
    matrix: np.ndarray
    note: str = ""

    def biases(self, factors=None) -> np.ndarray:
        v = self.value if factors is None else self.value * factors
        return self.matrix @ v

    def perturbation(self, rng) -> np.ndarray:
        """Lognormal factors with mean 1 and the estimates' relative errors."""
        s2 = np.log1p(self.rel_err ** 2)
        return np.exp(np.sqrt(s2) * rng.standard_normal(len(self.value)) - 0.5 * s2)

    def take(self, idx) -> "BiasTable":
        return BiasTable(self.keys, self.value, self.rel_err, self.matrix[idx], self.note)


def _estimates_for(sample: HSample, entries: dict, by_cell: dict, k: int):
    from .bias_files import parse_cell

    key = sample.bias_key[k]
    try:
        if sample.bias_mode == "block":
            s, b, g, c = key
            ek = (s, b, g, parse_cell(c))
            return [(ek, entries.get(ek))]
        g, c = key
        return by_cell.get((g, parse_cell(c)), [])
    except (TypeError, ValueError, SyntaxError):
        return []


def bias_table(sample: HSample, project=None, method: str = "model_ae",
               surveys=(), rel_err_fallback: float | None = None) -> BiasTable:
    """Per-(block, cell) estimates from the survey bias files, else the results."""
    from .bias_files import load_project_bias

    n = len(sample)
    entries = {}
    if project is not None:
        names = set(project.survey_names(surveys)) | set(sample.survey)
        entries = load_project_bias(project.char_root, sorted(names), method)
    rows: list[list[int]] = []
    keys: list = []
    values: list[float] = []
    rels: list[float] = []
    index = {}
    missing_err = 0
    note = ""
    if entries:
        by_cell: dict = {}
        for (s, b, g, c), est in entries.items():
            by_cell.setdefault((g, c), []).append(((s, b, g, c), est))
        for k in range(n):
            found = _estimates_for(sample, entries, by_cell, k)
            if not found or any(est is None for _, est in found):
                entries = {}
                break
            cols = []
            for ek, est in found:
                if ek not in index:
                    rel = est.relative_error(rel_err_fallback)
                    if rel is None:
                        missing_err += 1
                        rel = 0.0
                    index[ek] = len(keys)
                    keys.append(ek)
                    values.append(est.bias)
                    rels.append(rel)
                cols.append(index[ek])
            rows.append(cols)
        if entries:
            total = np.array([sum(values[j] for j in cols) for cols in rows])
            if not np.allclose(total, sample.bias, rtol=1e-5):
                note = ("bias files differ from the results bias; using the bias "
                        "files for the bootstrap")
            note = note or f"{len(keys)} block-cell bias estimates from the survey bias files"
    if not entries:
        rows, keys, values, rels, index = [], [], [], [], {}
        missing_err = 0
        for k in range(n):
            key = sample.bias_key[k]
            if key not in index:
                se = sample.bias_se[k]
                if np.isfinite(se):
                    rel = se / sample.bias[k]
                elif rel_err_fallback is not None:
                    rel = rel_err_fallback
                else:
                    missing_err += 1
                    rel = 0.0
                index[key] = len(keys)
                keys.append(key)
                values.append(sample.bias[k])
                rels.append(rel)
            rows.append([index[key]])
        note = "bias and bias_se from the results file"
    matrix = np.zeros((n, len(keys)))
    for k, cols in enumerate(rows):
        matrix[k, cols] = 1.0
    if missing_err:
        note += (f"; {missing_err} estimate(s) have no MC error (legacy bias file): "
                 "treated as exact, set --bias-rel-err")
    return BiasTable(keys, np.array(values), np.array(rels), matrix, note)


# ---------------------------------------------------------------------------
# Conditional likelihood


class ConditionalLikelihood:
    """Shape likelihood prod_k f_k(observed m_k | geometry_k, detected).

    Each detection uses its own block's selection (``selection`` is one
    ``SurveySelection`` or ``{"survey/block": SurveySelection}``). With
    ``measurement="model"`` the true magnitude m_t = H + mu_k is latent:

        f_k = sum_j wq_kj  int lambda(H) eta(m_t) p(m_k | m_t) dH * tc(m_k)
                           / int lambda(H) S_bar(m_t) dH

    where eta is the detection efficiency (true magnitude), tc is tracking x
    characterisation (measured magnitude), p is the OSSSSim photometric
    model of the block's .eff (or a ``mag_err`` column), S_bar(m_t) =
    prod_e eta_e(m_t) E[tc_e | m_t], and j runs over Gauss-Hermite distance
    nodes when ``d_bary_err`` is known. Without scatter this is
    lambda(H_k) S(m_k) / int lambda(H) S(m_k + H - H_k) dH, which is what
    ``measurement="none"`` computes. Everything is tabulated once on an H
    grid; a likelihood call is two matrix-vector products.
    """

    def __init__(self, sample: HSample, selection, h_min: float,
                 h_max: float | None = None, dh: float = 0.01,
                 measurement: str = "model", distance_nodes: int = 7):
        if measurement not in MEASUREMENT_MODES:
            raise ValueError(f"measurement must be one of {MEASUREMENT_MODES}")
        selections = selection if isinstance(selection, dict) else None
        groups = np.unique(sample.group)
        if selections is not None:
            absent = [g for g in groups if g not in selections]
            if absent:
                raise ValueError(f"no selection for {', '.join(map(str, absent))}")
        self.selections = selections or {g: selection for g in groups}
        if (any(self.selections[g].rate_dependent for g in groups)
                and np.any(~np.isfinite(sample.rate))):
            raise ValueError(
                "the efficiency depends on rate; give --rate or map d_bary "
                "in [detection_columns]"
            )
        rate = np.where(np.isfinite(sample.rate), sample.rate, 1.0)
        keep = sample.h >= h_min
        if h_max is not None:
            keep &= sample.h <= h_max
        self.excluded = int((~keep).sum())
        self.index = np.flatnonzero(keep)
        self.sample = sample.subset(keep)
        self.rate = rate[keep]
        if len(self.sample) == 0:
            raise ValueError("no detections inside the fit H range")
        self.measurement = measurement
        self.sigma_floor = SIGMA_FLOOR_CELLS * dh
        s = self.sample
        n = len(s)

        use_model = measurement == "model"
        shift, wq = _distance_nodes(s, distance_nodes if use_model else 1)
        n_nodes = shift.shape[1]
        self.n_nodes = n_nodes
        self.log_wq = np.log(wq)
        mu0 = s.m - s.h
        mu = (mu0[:, None] + shift).ravel()          # row r = k * n_nodes + j
        det = np.repeat(np.arange(n), n_nodes)
        m_hat = s.m[det]
        rows_rate = self.rate[det]
        self._mu, self._m_hat, self._det, self._shift = mu, m_hat, det, shift.ravel()

        # Per group: zero model (exact old formula) or a measurement model.
        self.models = {}
        for g in np.unique(s.group):
            meas = self.selections[g].measurement
            self.models[g] = None if (not use_model or meas.is_zero) else meas
        m_err = s.m_err[det] if use_model else np.full(len(det), np.nan)
        model_rows = np.array([self.models[s.group[k]] is not None for k in det])
        self.point = np.where(np.isfinite(m_err), m_err <= 0.0, ~model_rows)
        self._m_err = m_err

        # Faint limits on the selection actually applied (S_bar for models).
        m_grid = np.arange(np.floor(s.m.min()) - 1.0, 45.0, 0.005)
        self._tables = {}
        s_at_det = np.empty(n)
        self.h_faint_k = np.empty(n)
        self.h_half_k = np.empty(n)
        m_faint_rows = np.empty(len(det))
        for g in np.unique(s.group):
            ks = np.flatnonzero(s.group == g)
            sel = self.selections[g]
            s_at_det[ks] = sel(s.m[ks], self.rate[ks])
            for key, members in self._rate_groups(g, ks).items():
                curve = self._table(g, key, m_grid)
                nz = np.flatnonzero(curve > 0.0)
                if nz.size == 0:
                    raise ValueError(f"{g}: the selection is zero at every magnitude")
                m_faint = m_grid[min(nz[-1] + 1, len(m_grid) - 1)]
                m_half = m_grid[np.flatnonzero(curve >= 0.5 * curve.max())[-1]]
                self.h_faint_k[members] = s.h[members] + (m_faint - s.m[members])
                self.h_half_k[members] = s.h[members] + (m_half - s.m[members])
                rows = np.flatnonzero(np.isin(det, members))
                m_faint_rows[rows] = m_faint - mu[rows]
        tc_det = np.empty(n)
        for g in np.unique(s.group):
            ks = np.flatnonzero(s.group == g)
            tc_det[ks] = self.selections[g].track_char(s.m[ks], self.rate[ks])
        point_det = self.point.reshape(n, n_nodes)[:, 0]
        bad = (point_det & (s_at_det <= 0.0)) | (tc_det <= 0.0)
        if np.any(bad):
            names = ", ".join(s.names[bad][:5])
            raise ValueError(
                f"the selection is zero at the observed magnitude of: {names}. "
                "Check the block's .eff file and its filter."
            )

        top = float(m_faint_rows.max())
        if h_max is not None:
            top = min(top, h_max)
        self.h_min = float(h_min)
        self.h_max = h_max
        n_grid = int(math.ceil((top - h_min) / dh)) + 1
        self.grid = np.linspace(h_min, h_min + (n_grid - 1) * dh, n_grid)
        self.dh = dh
        w = np.full(n_grid, dh)
        w[0] = w[-1] = 0.5 * dh
        self.trap = w

        n_rows = len(det)
        self.den = np.empty((n_rows, n_grid))
        self.num = np.zeros((n_rows, n_grid))
        self.h_pt = m_hat - mu
        self.outside = (self.h_pt < self.grid[0]) | (self.h_pt > self.grid[-1])
        self.log_s_pt = np.zeros(n_rows)
        for g in np.unique(s.group):
            sel = self.selections[g]
            meas = self.models[g]
            ks = np.flatnonzero(s.group == g)
            rows = np.flatnonzero(np.isin(det, ks))
            m_t = self.grid[None, :] + mu[rows, None]
            r_rate = rows_rate[rows]
            if meas is None:
                self.den[rows] = sel(m_t, r_rate[:, None])
            else:
                for key, members in self._rate_groups(g, ks).items():
                    sub = rows[np.isin(det[rows], members)]
                    self.den[sub] = np.interp(self.grid[None, :] + mu[sub, None], m_grid,
                                              self._table(g, key, m_grid))
            pt = rows[self.point[rows]]
            if pt.size:
                self.log_s_pt[pt] = np.log(sel(m_hat[pt], rows_rate[pt]))
            kr = rows[~self.point[rows]]
            if kr.size:
                mt = self.grid[None, :] + mu[kr, None]
                kernel = self._kernel_cell(meas, m_hat[kr, None], mt, m_err[kr, None])
                tc = sel.track_char(m_hat[kr], rows_rate[kr])
                self.num[kr] = sel.eta(mt, rows_rate[kr, None]) * kernel * tc[:, None]
        if h_max is not None:
            self.den[:, self.grid > h_max] = 0.0
            self.num[:, self.grid > h_max] = 0.0
        self._pit_cache = None

    # -- construction helpers ------------------------------------------------

    def _rate_groups(self, g, ks) -> dict:
        """Detections of group g that share one selection table, by rate."""
        if not self.selections[g].rate_dependent:
            return {float(self.rate[ks[0]]): ks}
        out: dict = {}
        for k in ks:
            out.setdefault(round(float(self.rate[k]), 3), []).append(k)
        return {r: np.array(v) for r, v in out.items()}

    def _table(self, g, rate, m_grid) -> np.ndarray:
        """Selection on a magnitude grid: S (zero model) or S_bar."""
        key = (g, rate)
        if key not in self._tables:
            sel = self.selections[g]
            if self.models[g] is None:
                self._tables[key] = sel(m_grid, rate)
            else:
                self._tables[key] = sel.measured_selection(m_grid, rate, self.sigma_floor)
        return self._tables[key]

    def _kernel_cell(self, meas, m_hat, m_t, m_err) -> np.ndarray:
        """p(m_hat | m_t) averaged over each H cell (per unit H)."""
        half = 0.5 * self.dh
        override = np.isfinite(m_err)
        out = np.zeros(np.broadcast(m_hat, m_t).shape)
        if meas is not None:
            sigma = np.maximum(meas.sigma(m_t), self.sigma_floor)
            centre = m_t + meas.offset(m_t)
            slope = meas.offset_slope(m_t)
            lo = meas.noise_cdf((m_hat - centre - slope * half) / sigma)
            hi = meas.noise_cdf((m_hat - centre + slope * half) / sigma)
            out = (hi - lo) / self.dh
        if np.any(override):
            from scipy.special import ndtr

            sigma = np.maximum(np.where(override, m_err, 1.0), self.sigma_floor)
            gauss = (ndtr((m_hat - m_t + half) / sigma)
                     - ndtr((m_hat - m_t - half) / sigma)) / self.dh
            out = np.where(override, gauss, out)
        return out

    # -- properties ------------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.sample)

    @property
    def selection(self):
        """The single selection when every detection shares one."""
        values = list(self.selections.values())
        return values[0] if len(values) == 1 else None

    def describe_selection(self) -> list[str]:
        lines = []
        for g in sorted(self.selections):
            sel = self.selections[g]
            count = int(np.sum(self.sample.group == g))
            lines.append(f"{g} ({count} detections):")
            lines += [f"  {line}" for line in sel.describe()]
            if self.measurement == "model":
                lines.append(f"  measured magnitude: {sel.measurement.describe()}")
        return lines

    def describe_measurement(self) -> dict:
        """Summary of the measurement treatment for the fit summary."""
        s = self.sample
        sig = np.full(self.n, 0.0)
        for g, meas in self.models.items():
            ks = s.group == g
            if meas is not None:
                sig[ks] = meas.sigma(s.m[ks])
        sig = np.where(np.isfinite(s.m_err), s.m_err, sig)
        out = {
            "mode": self.measurement,
            "sigma_floor": self.sigma_floor if self.measurement == "model" else 0.0,
            "median_sigma_m": float(np.median(sig)),
            "max_sigma_m": float(np.max(sig)),
            "mag_err_column": int(np.isfinite(s.m_err).sum()),
            "distance_nodes": self.n_nodes,
            "d_err_detections": int(np.sum(np.isfinite(s.d_err) & (s.d_err > 0))),
            "blocks": {},
        }
        offsets = []
        for g in sorted(self.models, key=str):
            meas = self.selections[g].measurement
            ks = s.group == g
            off = meas.offset(s.m[ks]) if self.models[g] is not None else np.zeros(1)
            offsets.append(float(np.max(np.abs(off))))
            out["blocks"][str(g)] = {
                "model": meas.describe() if self.measurement == "model" else "not applied",
                "defaults": bool(meas.defaults),
                "zero": self.models[g] is None,
                "max_abs_offset": offsets[-1],
            }
        out["max_abs_offset"] = max(offsets) if offsets else 0.0
        return out

    @property
    def rate_dependent(self) -> bool:
        return any(s.rate_dependent for s in self.selections.values())

    def _rows_of(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=int)
        return (idx[:, None] * self.n_nodes + np.arange(self.n_nodes)[None, :]).ravel()

    def take(self, idx) -> "ConditionalLikelihood":
        """The likelihood for rows ``idx`` (a bootstrap resample), without retabulating."""
        idx = np.asarray(idx, dtype=int)
        rows = self._rows_of(idx)
        new = object.__new__(ConditionalLikelihood)
        new.__dict__.update(self.__dict__)
        new.sample = self.sample.subset(idx)
        new.index = self.index[idx]
        new.rate = self.rate[idx]
        new.log_wq = self.log_wq[idx]
        new.h_faint_k = self.h_faint_k[idx]
        new.h_half_k = self.h_half_k[idx]
        for name in ("den", "num", "h_pt", "log_s_pt", "point", "outside", "_mu", "_m_hat",
                     "_shift", "_m_err"):
            setattr(new, name, getattr(self, name)[rows])
        new._det = np.repeat(np.arange(len(idx)), self.n_nodes)
        new._pit_cache = None
        return new

    # -- likelihood ------------------------------------------------------------

    def _shape(self, log_lambda_grid):
        c = float(np.max(log_lambda_grid))
        lam = np.exp(log_lambda_grid - c)
        return c, lam

    def loglike_from(self, log_lambda_grid, log_lambda_pt) -> float:
        """``log_lambda_pt``: ln lambda at ``h_pt`` (used by point rows)."""
        c, lam = self._shape(log_lambda_grid)
        lam_w = lam * self.trap
        den = self.den @ lam_w
        if np.any(den <= 0.0) or not np.all(np.isfinite(den)):
            return -np.inf
        log_ratio = np.empty(len(den))
        p = self.point
        log_ratio[p] = log_lambda_pt[p] - c + self.log_s_pt[p] - np.log(den[p])
        if np.any(~p):
            num = self.num[~p] @ lam_w
            if np.any(num <= 0.0):
                return -np.inf
            log_ratio[~p] = np.log(num) - np.log(den[~p])
        if self.n_nodes == 1:
            return float(np.sum(log_ratio))
        terms = log_ratio.reshape(self.n, self.n_nodes) + self.log_wq
        top = terms.max(axis=1)
        if not np.all(np.isfinite(top)):
            return -np.inf
        return float(np.sum(top + np.log(np.exp(terms - top[:, None]).sum(axis=1))))

    def _point_log_lambda(self, ld) -> np.ndarray | None:
        """ln lambda at point rows; None if a usable point row underflows."""
        ld = np.where(self.outside, -np.inf, ld)
        use = self.point & ~self.outside
        if not np.all(np.isfinite(ld[use])):
            return None
        return ld

    def loglike(self, alpha, beta, h_b) -> float:
        if alpha <= 0 or beta <= 0:
            return -np.inf
        lg = log_differential(self.grid, alpha, beta, h_b)
        ld = self._point_log_lambda(log_differential(self.h_pt, alpha, beta, h_b))
        if ld is None or not np.any(np.isfinite(lg)):
            return -np.inf
        return self.loglike_from(lg, ld)

    def loglike_exponential(self, alpha) -> float:
        if alpha <= 0:
            return -np.inf
        ld = self._point_log_lambda(log_differential_exponential(self.h_pt, alpha))
        if ld is None:
            return -np.inf
        return self.loglike_from(log_differential_exponential(self.grid, alpha), ld)

    # -- predictions and goodness of fit ---------------------------------------

    def _row_densities(self, log_lambda_grid, matrix=None) -> np.ndarray:
        _c, lam = self._shape(log_lambda_grid)
        rows = (self.den if matrix is None else matrix) * lam[None, :]
        total = rows @ self.trap
        return rows / np.where(total > 0.0, total, 1.0)[:, None]

    def density_from(self, log_lambda_grid) -> np.ndarray:
        """Row k: density of detection k's true H on the grid (integrates to 1)."""
        rows = self._row_densities(log_lambda_grid)
        wq = np.exp(self.log_wq).ravel()[:, None]
        return (rows * wq).reshape(self.n, self.n_nodes, -1).sum(axis=1)

    def detected_density(self, alpha, beta, h_b) -> np.ndarray:
        return self.density_from(log_differential(self.grid, alpha, beta, h_b))

    def detected_density_exponential(self, alpha) -> np.ndarray:
        return self.density_from(log_differential_exponential(self.grid, alpha))

    def observed_from(self, log_lambda_grid) -> np.ndarray:
        """Sum over detections of the predicted density of the observed H.

        Each true-H prediction is moved by the detection's distance-node shift
        and magnitude offset and blurred by its scatter at the observed
        magnitude (a per-detection stationary approximation, for plots).
        """
        rows = self._row_densities(log_lambda_grid)
        wq = np.exp(self.log_wq).ravel()
        shift = self._shift.copy()
        sigma = np.zeros(len(rows))
        for g, meas in self.models.items():
            r = self.sample.group[self._det] == g
            if meas is not None:
                shift[r] += meas.offset(self._m_hat[r])
                eff = math.sqrt(float(np.sum(meas.mixture / np.arange(1, 4))))
                sigma[r] = np.maximum(meas.sigma(self._m_hat[r]), self.sigma_floor) * eff
        override = np.isfinite(self._m_err)
        sigma[override] = self._m_err[override]
        out = np.zeros(len(self.grid))
        width = np.round(sigma / self.dh).astype(int)
        for wpix in np.unique(width):
            sel = np.flatnonzero(width == wpix)
            part = np.zeros(len(self.grid))
            for r in sel:
                part += wq[r] * np.interp(self.grid - shift[r], self.grid, rows[r],
                                          left=0.0, right=0.0)
            if wpix > 0:
                x = np.arange(-5 * wpix, 5 * wpix + 1)
                kern = np.exp(-0.5 * (x / wpix) ** 2)
                part = np.convolve(part, kern / kern.sum(), mode="same")
            out += part
        return out

    def predicted_observed(self, alpha, beta, h_b) -> np.ndarray:
        return self.observed_from(log_differential(self.grid, alpha, beta, h_b))

    def predicted_observed_exponential(self, alpha) -> np.ndarray:
        return self.observed_from(log_differential_exponential(self.grid, alpha))

    def _pit_matrices(self):
        """(numerator, denominator) rows for P(m_hat <= m_k | detected, node)."""
        if self._pit_cache is not None:
            return self._pit_cache
        n_rows = len(self._det)
        pnum = np.zeros((n_rows, len(self.grid)))
        pden = np.zeros((n_rows, len(self.grid)))
        s = self.sample
        groups = s.group[self._det]
        for g in np.unique(groups):
            rows = np.flatnonzero(groups == g)
            sel = self.selections[g]
            meas = self.models[g]
            r_rate = self.rate[self._det[rows]]
            m_t = self.grid[None, :] + self._mu[rows, None]
            if meas is None:
                below = np.clip((self.h_pt[rows, None] - self.grid[None, :]) / self.dh + 0.5,
                                0.0, 1.0)
                pden[rows] = self.den[rows]
                pnum[rows] = self.den[rows] * below
                continue
            nodes = meas.nodes
            step = nodes[1] - nodes[0]
            for rate in np.unique(r_rate):
                sub = rows[r_rate == rate]
                mt = m_t[r_rate == rate]
                flat = mt.ravel()
                cum = sel.cumulative_track_char(flat, rate, self.sigma_floor)
                sigma = np.maximum(meas.sigma(flat), self.sigma_floor)
                z = (np.repeat(self._m_hat[sub], mt.shape[1]) - flat - meas.offset(flat)) / sigma
                pos = np.clip((z - nodes[0]) / step, 0.0, len(nodes) - 1.0)
                lo = np.floor(pos).astype(int)
                hi = np.minimum(lo + 1, len(nodes) - 1)
                frac = pos - lo
                ar = np.arange(len(flat))
                below = np.where(z < nodes[0], 0.0,
                                 cum[ar, lo] * (1.0 - frac) + cum[ar, hi] * frac)
                eta = sel.eta(flat, rate)
                pnum[sub] = (eta * below).reshape(mt.shape)
                pden[sub] = (eta * cum[:, -1]).reshape(mt.shape)
        if self.h_max is not None:
            pnum[:, self.grid > self.h_max] = 0.0
            pden[:, self.grid > self.h_max] = 0.0
        self._pit_cache = (pnum, pden)
        return self._pit_cache

    def pit(self, alpha, beta, h_b) -> np.ndarray:
        """P(observed magnitude <= m_k | geometry, detected): uniform if the model is right."""
        _c, lam = self._shape(log_differential(self.grid, alpha, beta, h_b))
        lam_w = lam * self.trap
        pnum, pden = self._pit_matrices()
        den = pden @ lam_w
        f = (pnum @ lam_w) / np.where(den > 0.0, den, 1.0)
        wq = np.exp(self.log_wq)
        return np.clip((f.reshape(self.n, self.n_nodes) * wq).sum(axis=1), 0.0, 1.0)


def _distance_nodes(sample: HSample, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """(magnitude shift of m - H, weight) at Gauss-Hermite nodes in ln d.

    The distance modulus 5 log10(r Delta) is taken to scale as
    10 log10(d) (r ~ Delta ~ d), as when Hx was computed from d_bary.
    """
    n = len(sample)
    rel = sample.d_err / sample.d
    usable = np.isfinite(rel) & (rel > 0.0)
    if n_nodes <= 1 or not usable.any():
        return np.zeros((n, 1)), np.ones((n, 1))
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    w = w / math.sqrt(math.pi)
    s = np.log1p(np.where(usable, rel, 0.0))
    ln_ratio = math.sqrt(2.0) * s[:, None] * x[None, :]
    shift = 10.0 * ln_ratio / LN10
    weights = np.where(usable[:, None], w[None, :], 0.0)
    centre = n_nodes // 2
    if n_nodes % 2 == 0:
        weights[~usable, centre - 1] = 0.5
        weights[~usable, centre] = 0.5
    else:
        weights[~usable, centre] = 1.0
    return shift, weights


# ---------------------------------------------------------------------------
# Fitting


@dataclass
class Priors:
    bounds: dict

    def contains(self, theta) -> bool:
        return all(lo < v < hi for v, (lo, hi) in
                   zip(theta, (self.bounds[p] for p in PARAM_NAMES)))

    def clip(self, theta):
        out = []
        for v, p in zip(theta, PARAM_NAMES):
            lo, hi = self.bounds[p]
            span = hi - lo
            out.append(min(max(v, lo + 1e-6 * span), hi - 1e-6 * span))
        return np.array(out)


def default_priors(sample_h: np.ndarray, h_min: float, overrides=None) -> Priors:
    bounds = dict(DEFAULT_PRIORS)
    bounds["h_b"] = (h_min - 5.0, float(np.max(sample_h)) + 5.0)
    for name, (lo, hi) in (overrides or {}).items():
        if name not in PARAM_NAMES:
            raise ValueError(f"--prior: unknown parameter {name!r}; use {PARAM_NAMES}")
        if not lo < hi:
            raise ValueError(f"--prior {name}: lower bound must be below upper")
        bounds[name] = (float(lo), float(hi))
    return Priors(bounds)


def log_posterior(theta, like: ConditionalLikelihood, priors: Priors) -> float:
    if not priors.contains(theta):
        return -np.inf
    return like.loglike(*theta)


@dataclass
class MLEResult:
    theta: np.ndarray
    loglike: float
    alpha_exp: float
    loglike_exp: float


def fit_mle(like: ConditionalLikelihood, priors: Priors, start=None,
            exponential: bool = True) -> MLEResult:
    """Tapered (and single-exponential) MLE.

    ``start`` warm-starts one Nelder-Mead run there instead of a grid of
    starting points (used by the bootstrap).
    """
    from scipy import optimize

    def neg(theta):
        v = log_posterior(theta, like, priors)
        return 1e300 if not np.isfinite(v) else -v

    if start is not None:
        scored = [priors.clip(start)]
    else:
        h = like.sample.h
        starts = []
        for alpha in (0.3, 0.5, 0.8):
            for beta in (0.15, 0.3, 0.8):
                for h_b in np.percentile(h, [10, 50, 90]):
                    starts.append(priors.clip((alpha, beta, h_b)))
        scored = sorted(starts, key=neg)[:6]
    best = None
    for start in scored:
        res = optimize.minimize(neg, start, method="Nelder-Mead",
                                options={"xatol": 1e-5, "fatol": 1e-7,
                                         "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    if not exponential:
        return MLEResult(np.asarray(best.x), -float(best.fun), float("nan"), float("nan"))
    exp_res = optimize.minimize_scalar(
        lambda a: -like.loglike_exponential(a), bounds=(1e-3, 3.0),
        method="bounded",
    )
    return MLEResult(
        theta=np.asarray(best.x), loglike=-float(best.fun),
        alpha_exp=float(exp_res.x), loglike_exp=-float(exp_res.fun),
    )


@dataclass
class MCMCResult:
    samples: np.ndarray       # (n, 3) alpha, beta, h_b
    log_like: np.ndarray
    acceptance: float
    autocorr: np.ndarray | None
    n_walkers: int
    n_steps: int
    burn: int


def run_mcmc(like, priors, start, n_walkers=32, n_steps=4000, burn=1000,
             thin=1, seed=42, progress=False) -> MCMCResult:
    import emcee

    rng = np.random.default_rng(seed)
    ndim = len(PARAM_NAMES)
    scale = np.array([0.02, 0.02, 0.1])
    p0 = []
    while len(p0) < n_walkers:
        trial = start + scale * rng.standard_normal(ndim)
        if np.isfinite(log_posterior(trial, like, priors)):
            p0.append(trial)
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_posterior, args=(like, priors),
    )
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(np.array(p0), n_steps, progress=progress)
    emcee_log = logging.getLogger("emcee")
    level = emcee_log.level
    emcee_log.setLevel(logging.ERROR)
    try:
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
    except Exception:
        tau = None
    finally:
        emcee_log.setLevel(level)
    flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
    logp = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    return MCMCResult(
        samples=flat, log_like=logp,
        acceptance=float(np.mean(sampler.acceptance_fraction)),
        autocorr=tau, n_walkers=n_walkers, n_steps=n_steps, burn=burn,
    )


# ---------------------------------------------------------------------------
# Normalisation and goodness of fit


@dataclass
class HTCount:
    h_norm: float
    count: float
    sigma: float
    n_used: int
    sigma_sampling: float = 0.0
    sigma_bias: float = 0.0
    mode: str = "union"
    per_block: dict = field(default_factory=dict)


def _ht_terms(w, keys, rel) -> tuple[float, float, float]:
    """(count, sampling variance, bias-MC variance) for weights w = 1/bias."""
    count = float(w.sum())
    var_s = float(np.sum(w * (w - 1.0)))
    totals: dict = {}
    for wk, key, r in zip(w, keys, rel):
        t, _ = totals.get(key, (0.0, r))
        totals[key] = (t + wk, r)
    var_b = float(sum((t * r) ** 2 for t, r in totals.values()))
    return count, max(var_s, 0.0), var_b


def ht_count(sample: HSample, h_min: float, h_norm: float, bias=None,
             rel_err_fallback: float | None = None) -> HTCount:
    """Horvitz-Thompson count in [h_min, h_norm] and its variance.

    ``bias`` overrides ``sample.bias`` (bootstrap perturbations). The MC
    term groups detections that share a bias estimate. A union bias gives
    one total; per-block biases give one estimate per block, combined by
    inverse variance.
    """
    bias = sample.bias if bias is None else np.asarray(bias)
    sel = (sample.h >= h_min) & (sample.h <= h_norm)
    rel = sample.bias_se / sample.bias
    if rel_err_fallback is not None:
        rel = np.where(np.isfinite(rel), rel, rel_err_fallback)
    rel = np.where(np.isfinite(rel), rel, 0.0)
    w = 1.0 / bias[sel]
    keys = sample.bias_key[sel]
    if sample.bias_mode != "block":
        count, var_s, var_b = _ht_terms(w, keys, rel[sel])
        return HTCount(h_norm, count, math.sqrt(var_s + var_b), int(sel.sum()),
                       math.sqrt(var_s), math.sqrt(var_b), "union")
    per_block = {}
    groups = sample.group[sel]
    parts = []
    for g in np.unique(groups):
        rows = groups == g
        c, vs, vb = _ht_terms(w[rows], keys[rows], rel[sel][rows])
        per_block[str(g)] = {"count": c, "sigma": math.sqrt(vs + vb), "n": int(rows.sum())}
        parts.append((c, vs, vb))
    inv = np.array([1.0 / (vs + vb) if vs + vb > 0.0 else 0.0 for _, vs, vb in parts])
    if inv.sum() == 0.0:
        f = np.full(len(parts), 1.0 / max(len(parts), 1))
    else:
        f = inv / inv.sum()
    count = float(sum(fi * c for fi, (c, _, _) in zip(f, parts)))
    var_s = float(sum(fi * fi * vs for fi, (_, vs, _) in zip(f, parts)))
    var_b = float(sum(fi * fi * vb for fi, (_, _, vb) in zip(f, parts)))
    return HTCount(h_norm, count, math.sqrt(var_s + var_b), int(sel.sum()),
                   math.sqrt(var_s), math.sqrt(var_b), "block", per_block)


def h_o_for(theta, h_min, h_norm, count) -> np.ndarray:
    """H_o such that N(<h_norm) - N(<h_min) equals ``count``."""
    alpha, beta, h_b = (np.asarray(x, dtype=float) for x in theta)
    a = log_cumulative(h_norm, alpha, beta, h_b)
    b = log_cumulative(h_min, alpha, beta, h_b)
    with np.errstate(divide="ignore"):
        log_diff = a + np.log(-np.expm1(np.minimum(b - a, -1e-12)))
    log_amp = np.log(count) - log_diff
    return -log_amp / (LN10 * alpha)


def sample_counts(ht: HTCount, size: int, rng) -> np.ndarray:
    """Lognormal draws with the HT mean and variance."""
    if ht.count <= 0:
        raise ValueError("no detections brighter than h_norm for the normalisation")
    cv2 = (ht.sigma / ht.count) ** 2
    s2 = math.log1p(cv2)
    mu = math.log(ht.count) - 0.5 * s2
    return np.exp(mu + math.sqrt(s2) * rng.standard_normal(size))


@dataclass
class BootstrapResult:
    samples: np.ndarray   # (n, 4): alpha, beta, h_b, h_o
    count: np.ndarray     # HT count of each replicate
    n_failed: int
    n_requested: int


def bootstrap(like: ConditionalLikelihood, sample: HSample, table: BiasTable,
              priors: Priors, start, n_boot: int, rng, h_min: float,
              h_norm: float, progress=None) -> BootstrapResult:
    """Block-stratified bootstrap with lognormal bias perturbations.

    Each replicate resamples detections with replacement within every
    ``survey/block`` group, multiplies every bias estimate by an independent
    lognormal factor (mean 1, its MC relative error), refits the MLE from
    ``start``, and sets H_o from the replicate's HT count. A block's
    replicate size is Poisson(n_block), so the HT count keeps the Poisson
    scatter in the number of detections that its sum w(w - 1) variance
    describes; a fixed size would remove it.
    """
    pos = np.full(len(sample), -1)
    pos[like.index] = np.arange(like.n)
    group_rows = [np.flatnonzero(sample.group == g) for g in np.unique(sample.group)]
    out, counts = [], []
    failed = 0
    for r in range(n_boot):
        idx = np.concatenate([rng.choice(rows, size=rng.poisson(len(rows)), replace=True)
                              for rows in group_rows])
        biases = table.biases(table.perturbation(rng))
        sub = sample.subset(idx)
        ht = ht_count(sub, h_min, h_norm, bias=biases[idx])
        like_idx = pos[idx]
        like_idx = like_idx[like_idx >= 0]
        if len(like_idx) < 3 or ht.count <= 0.0:
            failed += 1
            continue
        fit = fit_mle(like.take(like_idx), priors, start=start, exponential=False)
        if not np.isfinite(fit.loglike):
            failed += 1
            continue
        h_o = float(h_o_for(fit.theta, h_min, h_norm, ht.count))
        out.append([*fit.theta, h_o])
        counts.append(ht.count)
        if progress and (r + 1) % progress == 0:
            print(f"  bootstrap {r + 1}/{n_boot}", flush=True)
    return BootstrapResult(np.array(out).reshape(-1, 4), np.array(counts), failed, n_boot)


def bootstrap_width_warnings(boot_samples, post_samples, ratio: float = 0.5) -> list[str]:
    """Flag parameters whose bootstrap 68% interval is much narrower than the posterior's."""
    out = []
    names = list(PARAM_NAMES) + ["h_o"]
    for j, name in enumerate(names):
        b = np.percentile(boot_samples[:, j], [16, 84])
        p = np.percentile(post_samples[:, j], [16, 84])
        if p[1] > p[0] and (b[1] - b[0]) < ratio * (p[1] - p[0]):
            out.append(
                f"bootstrap: {name} interval is {(b[1] - b[0]) / (p[1] - p[0]):.0%} of "
                "the posterior's; the warm-started refits stay near the MLE mode, "
                "so quote the posterior for this parameter"
            )
    return out


def group_diagnostics(like: ConditionalLikelihood, theta) -> list[dict]:
    """Per survey/block: detections, PIT KS at ``theta``, efficiency source."""
    pit = like.pit(*theta)
    out = []
    for g in sorted(like.selections, key=str):
        rows = like.sample.group == g
        n = int(rows.sum())
        d, p = ks_uniform(pit[rows]) if n >= 2 else (float("nan"), float("nan"))
        out.append({
            "group": str(g), "n": n, "ks_statistic": d, "ks_p_value": p,
            "efficiency": [e.source for e in like.selections[g].epochs],
        })
    return out


def ks_uniform(u: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    res = stats.kstest(u, "uniform")
    return float(res.statistic), float(res.pvalue)


def ks_stat(u: np.ndarray) -> float:
    u = np.sort(u)
    n = len(u)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def posterior_predictive_ks(like, samples, n_draws, rng) -> float:
    """Fraction of draws where replicated data fit worse than the observed.

    A replicate drawn from the model at theta (true H, then measured
    magnitude with the photometric scatter) has exactly uniform PITs at
    theta, so its KS distance is that of n uniform variates.
    """
    idx = rng.choice(len(samples), size=min(n_draws, len(samples)), replace=False)
    worse = 0
    for j in idx:
        d_obs = ks_stat(like.pit(*samples[j]))
        if ks_stat(rng.random(like.n)) >= d_obs:
            worse += 1
    return worse / len(idx)


def summarize(values: np.ndarray) -> tuple[float, float, float]:
    lo, med, hi = np.percentile(values, [16, 50, 84])
    return float(med), float(med - lo), float(hi - med)


def prior_edge_warnings(samples, priors: Priors) -> list[str]:
    notes = []
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = priors.bounds[name]
        span = hi - lo
        col = samples[:, j]
        for edge, frac in ((lo, np.mean(col < lo + 0.02 * span)),
                           (hi, np.mean(col > hi - 0.02 * span))):
            if frac > 0.05:
                notes.append(
                    f"{name}: {100 * frac:.0f}% of samples within 2% of the "
                    f"prior edge {edge:g}; the data do not bound it there"
                )
    return notes


def taper_warnings(samples, sample_h) -> list[str]:
    frac = float(np.mean(samples[:, PARAM_NAMES.index("h_b")] < np.min(sample_h)))
    if frac > 0.16:
        return [
            f"h_b: {100 * frac:.0f}% of samples are brighter than every "
            "detection; the data do not require a taper and beta, h_b are "
            "set by the prior there (compare with the single exponential)"
        ]
    return []


def warn_if_short(mcmc: MCMCResult) -> list[str]:
    notes = []
    if mcmc.autocorr is not None and np.all(np.isfinite(mcmc.autocorr)):
        need = 50 * float(np.max(mcmc.autocorr))
        if mcmc.n_steps - mcmc.burn < need:
            notes.append(
                f"chain length after burn-in ({mcmc.n_steps - mcmc.burn}) is "
                f"under 50 autocorrelation times ({need:.0f}); run longer"
            )
    if not 0.15 < mcmc.acceptance < 0.7:
        notes.append(f"mean acceptance fraction {mcmc.acceptance:.2f} is unusual")
    return notes


__all__ = [
    "BiasTable", "BootstrapResult", "ConditionalLikelihood", "EpochSelection",
    "HSample", "HTCount", "MCMCResult", "MLEResult", "PARAM_NAMES", "Priors",
    "SurveySelection", "bias_table", "block_selections", "bootstrap",
    "cumulative", "default_priors", "fit_mle", "group_diagnostics", "h_o_for",
    "ht_count", "load_hsample", "log_cumulative", "log_differential",
    "parse_efficiency_spec", "read_efficiency_file", "run_mcmc",
]

"""OSSOS ``.eff`` efficiency files and the magnitude selection they define.

Mirrors OSSSSim ``effut.f95`` / ``surveysub.f95`` for a flag-4 (tracked and
characterised) detection, including the photometric measurement model of
``numutils.f95 magran``: detection is decided on the true magnitude, and
tracking and the characterisation limit on the measured one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# OSSSSim eta(): efficiencies below this are treated as zero.
ETA_FLOOR = 0.01
# OSSSSim characterisation threshold when a rate block has no mag_lim.
ETA_CHARACTERISED = 0.4
# getsur.f95 values when an .eff file has no mag_error= / phot_frac= line.
OSSSSIM_MAG_ERROR = (0.026, 0.33, 24.45, 0.7, 23.7, -0.3)
OSSSSIM_PHOT_FRAC = (1.0, 0.0, 0.0)
# magran noise: unit-variance triangular variates on [-sqrt(6), sqrt(6)].
NOISE_HALF_WIDTH = math.sqrt(6.0)
NOISE_NODES = 161


def _triangular_mean_densities(nodes: int = NOISE_NODES) -> tuple[np.ndarray, list]:
    """Densities of the mean of 1, 2, 3 magran variates on a common grid."""
    t = np.linspace(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, nodes)
    fine = np.linspace(-NOISE_HALF_WIDTH, NOISE_HALF_WIDTH, 4001)
    step = fine[1] - fine[0]
    f1 = np.maximum(1.0 - np.abs(fine) / NOISE_HALF_WIDTH, 0.0) / NOISE_HALF_WIDTH
    out = [np.interp(t, fine, f1)]
    total = f1
    for n in (2, 3):
        total = np.convolve(total, f1) * step        # density of the sum of n
        u = np.linspace(-n * NOISE_HALF_WIDTH, n * NOISE_HALF_WIDTH, len(total))
        out.append(n * np.interp(n * t, u, total))    # density of the mean
    return t, out


_NOISE_T, _NOISE_DENSITIES = _triangular_mean_densities()


@dataclass
class MeasurementModel:
    """OSSSSim photometric model: m_hat = m + offset(m) + sigma(m) * T.

    ``params`` are the six ``mag_error=`` values as written in the .eff file;
    T is the mean of 1, 2 or 3 unit-variance triangular variates, with the
    ``phot_frac`` probabilities.
    """

    params: tuple = OSSSSIM_MAG_ERROR
    phot_frac: tuple = OSSSSIM_PHOT_FRAC
    defaults: bool = False

    def __post_init__(self):
        p = [float(v) for v in self.params]
        if len(p) != 6:
            raise ValueError(f"mag_error needs 6 values, got {self.params}")
        self.params = tuple(p)
        p1, p2, p3 = p[0], p[1], p[2]
        if p1 > 0.0 and p2 > 0.0 and p3 != 21.0:
            self._slope = math.log10(p2 / p1) / (p3 - 21.0)
        else:
            self._slope = 0.0
        f1, f2 = (float(v) for v in self.phot_frac[:2])
        w = np.array([f1, f2, max(0.0, 1.0 - f1 - f2)])
        if w.sum() <= 0.0:
            raise ValueError(f"phot_frac {self.phot_frac} has no weight")
        self.mixture = w / w.sum()
        dens = sum(wi * d for wi, d in zip(self.mixture, _NOISE_DENSITIES))
        dt = _NOISE_T[1] - _NOISE_T[0]
        q = dens * dt
        q[0] *= 0.5
        q[-1] *= 0.5
        self.nodes = _NOISE_T
        self.weights = q / q.sum()
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * dt)])
        self._cdf = cdf / cdf[-1]

    @property
    def is_zero(self) -> bool:
        """No scatter and no offset at any magnitude."""
        p1, _p2, _p3, p4, _p5, p6 = self.params
        return p1 == 0.0 and p4 >= 0.0 and p6 == 0.0

    def sigma(self, m) -> np.ndarray:
        """magran's per-measurement uncertainty at true magnitude m."""
        m = np.asarray(m, dtype=float)
        p1, p2, p3, p4, _p5, _p6 = self.params
        mid = p1 * 10.0 ** (self._slope * (m - 21.0))
        faint = np.maximum(p1 * 10.0 ** (self._slope * (p3 - 21.0)) - (m - p3) * p4, 0.0)
        return np.where(m <= 21.0, p1, np.where(m <= p3, mid, faint))

    def offset(self, m) -> np.ndarray:
        """magran's deterministic shift for m fainter than p5."""
        m = np.asarray(m, dtype=float)
        _p1, _p2, _p3, _p4, p5, p6 = self.params
        return np.where(m > p5, (m - p5) * p6, 0.0)

    def offset_slope(self, m) -> np.ndarray:
        """d(m + offset)/dm."""
        m = np.asarray(m, dtype=float)
        return np.where(m > self.params[4], 1.0 + self.params[5], 1.0)

    def noise_cdf(self, z) -> np.ndarray:
        """P(T <= z) for the standardised mixture."""
        return np.interp(z, _NOISE_T, self._cdf, left=0.0, right=1.0)

    def sample(self, m, rng) -> np.ndarray:
        """Measured magnitudes for true magnitudes m (as magran draws them)."""
        m = np.asarray(m, dtype=float)
        n = rng.choice(3, size=m.shape, p=self.mixture) + 1
        tri = rng.triangular(-NOISE_HALF_WIDTH, 0.0, NOISE_HALF_WIDTH, size=(3,) + m.shape)
        mean = np.where(n == 1, tri[0], np.where(n == 2, tri[:2].mean(0), tri.mean(0)))
        return m + self.offset(m) + self.sigma(m) * mean

    def describe(self) -> str:
        p = ", ".join(f"{v:g}" for v in self.params)
        f = ", ".join(f"{v:g}" for v in self.phot_frac)
        src = "OSSSSim defaults" if self.defaults else "from .eff"
        return f"mag_error {p}; phot_frac {f} ({src})"


@dataclass
class RateBlock:
    """One ``rates=`` block of an OSSOS efficiency file."""

    rate_min: float
    rate_max: float
    function: str = ""
    params: list = field(default_factory=list)
    lookup_mag: list = field(default_factory=list)
    lookup_eff: list = field(default_factory=list)
    mag_lim: float = -1.0

    def contains(self, rate_asphr: np.ndarray) -> np.ndarray:
        return (rate_asphr - self.rate_min) * (rate_asphr - self.rate_max) <= 0.0

    def eta_raw(self, m: np.ndarray, rate_asphr: np.ndarray) -> np.ndarray:
        m = np.maximum(np.asarray(m, dtype=float), 0.0)
        p = self.params
        kind = self.function
        if kind == "single":
            alpha_rate = p[3] if len(p) > 3 else 0.0
            m0 = p[1] + alpha_rate * rate_asphr
            return p[0] / 2.0 * (1.0 - np.tanh((m - m0) / p[2]))
        if kind == "double":
            return (p[0] / 4.0
                    * (1.0 - np.tanh((m - p[1]) / p[2]))
                    * (1.0 - np.tanh((m - p[1]) / p[3])))
        if kind == "linear":
            out = np.where(m < p[1], p[0], (m - p[2]) * p[0] / (p[1] - p[2]))
            return np.where(m >= p[2], 0.0, out)
        if kind == "square":
            val = (p[0] - p[1] * (m - 21.0) ** 2) / (1.0 + np.exp((m - p[2]) / p[3]))
            return np.where(m < 21.0, p[0], val)
        if kind == "lookup":
            b = np.asarray(self.lookup_mag)
            e = np.asarray(self.lookup_eff)
            out = np.interp(m, b, e)
            out = np.where(m < b[0], e[0], out)
            return np.where(m > b[-1], 0.0, out)
        raise ValueError(f"unsupported efficiency function {kind!r}")

    def check(self, source) -> None:
        need = {"single": 3, "double": 4, "linear": 3, "square": 4}
        if self.function == "lookup":
            if not self.lookup_mag:
                raise ValueError(f"{source}: lookup function has no lookup_param lines")
        elif self.function in need:
            if len(self.params) < need[self.function]:
                raise ValueError(
                    f"{source}: {self.function}_param needs "
                    f"{need[self.function]} values, got {self.params}"
                )
        else:
            raise ValueError(
                f"{source}: rates {self.rate_min}-{self.rate_max} has no "
                "supported 'function=' (single, double, linear, square, lookup)"
            )


@dataclass
class EpochSelection:
    """Magnitude selection for one epoch: eta x tracking x characterisation.

    Follows OSSSSim for a flag-4 (tracked and characterised) detection:
    eta(m) is zero below 0.01; tracking is min(peak, 1 + (m - Rc) * slope);
    characterisation requires m <= mag_lim of the rate block, or eta >= 0.4
    when the block has no mag_lim. Photometric scatter is not modelled.
    ``m`` is in the file's ``filter``.
    """

    blocks: list
    track: tuple = (1.0, 99.0, 0.0)
    source: str = ""
    filter: str = ""
    rate_cut: tuple | None = None
    mag_error: tuple | None = None    # as written in the file; None if absent
    phot_frac: tuple | None = None

    @property
    def rate_dependent(self) -> bool:
        if len(self.blocks) > 1:
            return True
        b = self.blocks[0]
        return b.function == "single" and len(b.params) > 3 and b.params[3] != 0.0

    @property
    def measurement(self) -> MeasurementModel:
        """The file's photometric model, or OSSSSim's defaults."""
        return MeasurementModel(
            self.mag_error if self.mag_error is not None else OSSSSIM_MAG_ERROR,
            self.phot_frac if self.phot_frac is not None else OSSSSIM_PHOT_FRAC,
            defaults=self.mag_error is None or self.phot_frac is None,
        )

    def _parts(self, m, rate_asphr) -> tuple[np.ndarray, np.ndarray]:
        """(eta, track x char), both evaluated at magnitude m."""
        m = np.asarray(m, dtype=float)
        rate = np.broadcast_to(np.asarray(rate_asphr, dtype=float), m.shape)
        eta_out = np.zeros(m.shape)
        tc_out = np.zeros(m.shape)
        assigned = np.zeros(m.shape, dtype=bool)
        for block in self.blocks:
            sel = block.contains(rate) & ~assigned
            if not sel.any():
                continue
            assigned |= sel
            eta = block.eta_raw(m[sel], rate[sel])
            eta = np.where(eta < ETA_FLOOR, 0.0, eta)
            if block.mag_lim > 0.0:
                char = m[sel] <= block.mag_lim
            else:
                char = eta >= ETA_CHARACTERISED
            peak, rc, slope = self.track
            track = np.clip(np.minimum(peak, 1.0 + (m[sel] - rc) * slope), 0.0, 1.0)
            eta_out[sel] = eta
            tc_out[sel] = track * char
        return eta_out, tc_out

    def eta(self, m, rate_asphr) -> np.ndarray:
        """Detection efficiency (applied to the true magnitude)."""
        return self._parts(m, rate_asphr)[0]

    def track_char(self, m, rate_asphr) -> np.ndarray:
        """Tracking x characterisation (applied to the measured magnitude)."""
        return self._parts(m, rate_asphr)[1]

    def __call__(self, m, rate_asphr) -> np.ndarray:
        eta, tc = self._parts(m, rate_asphr)
        return eta * tc


def _floats(text: str, count: int | None = None) -> list[float]:
    values = []
    for token in text.split():
        try:
            values.append(float(token))
        except ValueError:
            break
        if count is not None and len(values) == count:
            break
    return values


def read_efficiency_file(path) -> EpochSelection:
    """Parse an OSSOS-format ``.eff`` file into an :class:`EpochSelection`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"efficiency file not found: {path}")
    blocks: list[RateBlock] = []
    track = (1.0, 99.0, 0.0)
    pending_mag_lim = None
    filt = ""
    rate_cut = None
    mag_error = None
    phot_frac = None
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        if key == "rates":
            lo, hi = _floats(value, 2)
            blocks.append(RateBlock(rate_min=lo, rate_max=hi))
            if pending_mag_lim is not None:
                blocks[-1].mag_lim = pending_mag_lim
        elif key == "track_frac":
            vals = _floats(value, 3)
            if len(vals) == 3:
                track = tuple(vals)
        elif key == "filter":
            filt = value.split()[0].strip() if value.split() else ""
        elif key == "rate_cut":
            vals = _floats(value, 2)
            if len(vals) == 2:
                rate_cut = tuple(vals)
        elif key == "mag_error":
            vals = _floats(value, 6)
            if len(vals) != 6:
                raise ValueError(f"{path}: mag_error needs 6 values, got {value.strip()!r}")
            mag_error = tuple(vals)
        elif key == "phot_frac":
            vals = _floats(value, 3)
            if len(vals) != 3:
                raise ValueError(f"{path}: phot_frac needs 3 values, got {value.strip()!r}")
            phot_frac = tuple(vals)
        elif key == "mag_lim":
            (lim,) = _floats(value, 1)
            if blocks:
                blocks[-1].mag_lim = lim
            else:
                pending_mag_lim = lim
        elif key in ("function",) and blocks:
            blocks[-1].function = value.split()[0].strip().lower()
        elif key.endswith("_param") and blocks:
            kind = key[: -len("_param")]
            if kind == "lookup":
                b, e = _floats(value, 2)
                blocks[-1].lookup_mag.append(b)
                blocks[-1].lookup_eff.append(e)
            elif kind == blocks[-1].function:
                blocks[-1].params = _floats(value)
    if not blocks:
        raise ValueError(f"{path}: no 'rates=' efficiency block")
    for block in blocks:
        block.check(path)
    return EpochSelection(blocks=blocks, track=track, source=str(path),
                          filter=filt, rate_cut=rate_cut,
                          mag_error=mag_error, phot_frac=phot_frac)


def parse_efficiency_spec(spec: str) -> EpochSelection:
    """``single:A,m0,w[,mag_lim]`` or ``double:A,m0,w1,w2[,mag_lim]``."""
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    values = [float(v) for v in rest.replace(",", " ").split()]
    need = {"single": 3, "double": 4, "linear": 3, "square": 4}
    if kind not in need or len(values) not in (need[kind], need[kind] + 1):
        raise ValueError(
            f"efficiency spec {spec!r}: use single:A,m0,w[,mag_lim] or "
            "double:A,m0,w1,w2[,mag_lim]"
        )
    params = values[: need[kind]]
    mag_lim = values[need[kind]] if len(values) > need[kind] else -1.0
    block = RateBlock(0.0, 1e9, kind, params, mag_lim=mag_lim)
    return EpochSelection(blocks=[block], source=spec)


@dataclass
class SurveySelection:
    """Detection requires success at every epoch: S(m) = prod_e S_e(m)."""

    epochs: list

    def __call__(self, m, rate_asphr) -> np.ndarray:
        out = np.ones(np.shape(m))
        for epoch in self.epochs:
            out = out * epoch(m, rate_asphr)
        return out

    def eta(self, m, rate_asphr) -> np.ndarray:
        out = np.ones(np.shape(m))
        for epoch in self.epochs:
            out = out * epoch.eta(m, rate_asphr)
        return out

    def track_char(self, m, rate_asphr) -> np.ndarray:
        out = np.ones(np.shape(m))
        for epoch in self.epochs:
            out = out * epoch.track_char(m, rate_asphr)
        return out

    @property
    def measurement(self) -> MeasurementModel:
        """Model for the catalogue magnitude (epoch 1's)."""
        return self.epochs[0].measurement

    def measured_selection(self, m_t, rate_asphr: float, sigma_floor: float = 0.0
                           ) -> np.ndarray:
        """S_bar(m_t) = prod_e eta_e(m_t) E[track_e char_e(m_hat_e) | m_t].

        Each epoch draws its own measurement. ``m_t`` is 1-D; scatter below
        ``sigma_floor`` is raised to it for non-zero models.
        """
        m_t = np.asarray(m_t, dtype=float)
        out = np.ones(m_t.shape)
        for epoch in self.epochs:
            meas = epoch.measurement
            if meas.is_zero:
                out = out * epoch(m_t, rate_asphr)
                continue
            sigma = np.maximum(meas.sigma(m_t), sigma_floor)
            centre = m_t + meas.offset(m_t)
            m_hat = centre[:, None] + sigma[:, None] * meas.nodes[None, :]
            expected = epoch.track_char(m_hat, rate_asphr) @ meas.weights
            out = out * epoch.eta(m_t, rate_asphr) * expected
        return out

    def cumulative_track_char(self, m_t, rate_asphr: float, sigma_floor: float = 0.0
                              ) -> np.ndarray:
        """Row i: cumsum over noise nodes of q_j prod_e track_char_e(m_hat_j).

        With ``measurement.nodes`` this gives
        E[prod_e track_char_e(m_hat) 1(m_hat <= x) | m_t] by interpolation in
        z = (x - centre) / sigma; the last column is the unrestricted mean.
        """
        meas = self.measurement
        m_t = np.asarray(m_t, dtype=float)
        sigma = np.maximum(meas.sigma(m_t), sigma_floor)
        centre = m_t + meas.offset(m_t)
        m_hat = centre[:, None] + sigma[:, None] * meas.nodes[None, :]
        return np.cumsum(self.track_char(m_hat, rate_asphr) * meas.weights[None, :], axis=1)

    @property
    def rate_dependent(self) -> bool:
        return any(e.rate_dependent for e in self.epochs)

    def describe(self) -> list[str]:
        lines = []
        for i, e in enumerate(self.epochs, start=1):
            parts = []
            for b in e.blocks:
                lim = f", mag_lim {b.mag_lim:g}" if b.mag_lim > 0 else ""
                vals = b.params if b.function != "lookup" else f"{len(b.lookup_mag)} bins"
                parts.append(
                    f"rates {b.rate_min:g}-{b.rate_max:g}\"/hr {b.function} {vals}{lim}"
                )
            filt = f" filter {e.filter}" if e.filter else ""
            lines.append(
                f"epoch {i}: {e.source}{filt}: " + "; ".join(parts)
                + f"; track_frac {tuple(e.track)}"
            )
        return lines

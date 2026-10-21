"""Free inclinations of main-belt TNOs (Huang, Gladman & Volk 2022, ApJS 259, 54).

A non-resonant TNO's orbit pole precesses about a local forcing pole at a
nearly constant angle, the free inclination I_free; a cut I_free < 4 deg
separates the cold classicals from the hot population far more cleanly
than a cut in ecliptic inclination. Huang et al. compute the forcing pole by
double-averaging the planetary Hamiltonian, which (unlike Laplace-Lagrange
theory) stays correct near the nu_18 secular resonance at a ~ 40.5 au.

Two data products of that paper are bundled in ``data/`` (from
https://yukunhuang.com/data/2022/free-inclination-data-release/):

  huang2022_qp_forced.npz  Table 2: forcing-pole (q, p) = I_forced (cos, sin)
                           Omega_forced [deg] on the grid a 39.4-47.7 au (0.1),
                           e 0-0.25 (0.01), I 0-40 deg (2), omega 0-90 deg (10)
  huang2022_ifree.csv      Table 1: barycentric elements (epoch JD 2459400.5)
                           and I_free (double average) of 2018 main-belt TNOs,
                           with dynamical class (RESO: 0 classical, -1
                           scattering, pq resonant) and OSSOS++ designations

``free_inclination`` follows the paper's ``_read_grid.py``: the forcing pole
of the nearest grid point, subtracted from the orbit's (q, p) = I (cos,
sin) Omega. Huang et al. quote ~0.1 deg accuracy against the full double
average for non-resonant orbits inside the grid and away from nu_18.
Elements must be barycentric ecliptic J2000 (``barycentric_elements``
converts heliocentric ones). Outside the grid the nearest edge is used and
``in_grid`` is False.
"""
from __future__ import annotations

import csv
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
GRID_FILE = DATA / "huang2022_qp_forced.npz"
TABLE_FILE = DATA / "huang2022_ifree.csv"

A_AXIS = np.linspace(39.4, 47.7, 84)
E_AXIS = np.linspace(0.0, 0.25, 26)
E_AXIS[0] = 0.001
I_AXIS = np.linspace(0.0, 40.0, 21)
I_AXIS[0] = 0.02
W_AXIS = np.linspace(0.0, 90.0, 10)

K2 = 0.01720209895 ** 2                  # GM_sun [au^3 / day^2]
PLANET_MASS = 1.0 / 1047.348 + 1.0 / 3497.90 + 1.0 / 22902.98 + 1.0 / 19412.24 + 6.0e-6
OBLIQUITY = math.radians(23.4392911)
NCODE = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CENTURY = {"I": 1800, "J": 1900, "K": 2000}


@lru_cache(maxsize=1)
def _grid():
    z = np.load(GRID_FILE)
    return z["q"].astype(float), z["p"].astype(float)


def _nearest(axis: np.ndarray, x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = np.searchsorted(axis, x)
    k = np.clip(k, 1, len(axis) - 1)
    left = axis[k - 1]
    right = axis[k]
    return np.where(np.abs(x - left) <= np.abs(right - x), k - 1, k)


def wrap_omega(omega) -> np.ndarray:
    """omega folded into 0-90 deg (the grid's two-fold reflection symmetries)."""
    w = np.mod(np.asarray(omega, dtype=float), 360.0)
    return np.select([w <= 90.0, w <= 180.0, w <= 270.0], [w, 180.0 - w, w - 180.0], 360.0 - w)


def in_grid(a, e, inc) -> np.ndarray:
    a, e, inc = (np.asarray(x, dtype=float) for x in (a, e, inc))
    return (a >= 39.4) & (a <= 47.7) & (e >= 0.0) & (e <= 0.25) & (inc >= 0.0) & (inc <= 40.0)


def forcing_pole(a, e, inc, omega) -> tuple[np.ndarray, np.ndarray]:
    """(q, p) [deg] of the forcing pole at the nearest grid point."""
    q, p = _grid()
    ia = _nearest(A_AXIS, a)
    ie = _nearest(E_AXIS, e)
    ii = _nearest(I_AXIS, inc)
    iw = _nearest(W_AXIS, wrap_omega(omega))
    return q[ia, ie, ii, iw], p[ia, ie, ii, iw]


def free_inclination(a, e, inc, node, omega) -> tuple[np.ndarray, np.ndarray]:
    """(I_free, Omega_free) [deg] from barycentric ecliptic elements [au, deg]."""
    inc = np.asarray(inc, dtype=float)
    node = np.radians(np.asarray(node, dtype=float))
    qf, pf = forcing_pole(a, e, inc, omega)
    dq = inc * np.cos(node) - qf
    dp = inc * np.sin(node) - pf
    return np.hypot(dq, dp), np.degrees(np.arctan2(dp, dq)) % 360.0


def forcing_pole_vector(a, e, inc, omega) -> np.ndarray:
    """Unit forcing-pole vector(s) in ecliptic J2000, shape (..., 3)."""
    qf, pf = forcing_pole(a, e, inc, omega)
    i_f = np.radians(np.hypot(qf, pf))
    o_f = np.arctan2(pf, qf)
    return np.stack([np.sin(i_f) * np.sin(o_f), -np.sin(i_f) * np.cos(o_f), np.cos(i_f)], axis=-1)


# ---------------------------------------------------------------------------
# Heliocentric -> barycentric elements


def _state(a, e, inc, node, peri, M, mu):
    """Ecliptic position [au], velocity [au/day] from elements (angles in deg)."""
    a, e = np.asarray(a, float), np.asarray(e, float)
    i, O, w, M = (np.radians(np.asarray(x, float)) for x in (inc, node, peri, M))
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    cE, sE = np.cos(E), np.sin(E)
    b = np.sqrt(1.0 - e * e)
    x, y = a * (cE - e), a * b * sE
    n = np.sqrt(mu / a ** 3)
    edot = n / (1.0 - e * cE)
    vx, vy = -a * sE * edot, a * b * cE * edot
    cO, sO, ci, si, cw, sw = np.cos(O), np.sin(O), np.cos(i), np.sin(i), np.cos(w), np.sin(w)
    r11, r12 = cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci
    r21, r22 = sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci
    r31, r32 = sw * si, cw * si
    pos = np.stack([r11 * x + r12 * y, r21 * x + r22 * y, r31 * x + r32 * y], axis=-1)
    vel = np.stack([r11 * vx + r12 * vy, r21 * vx + r22 * vy, r31 * vx + r32 * vy], axis=-1)
    return pos, vel


def _elements(pos, vel, mu):
    """(a, e, inc, node, peri, M) [au, deg] from ecliptic state vectors."""
    r = np.linalg.norm(pos, axis=-1)
    h = np.cross(pos, vel)
    hn = np.linalg.norm(h, axis=-1)
    inc = np.degrees(np.arccos(np.clip(h[..., 2] / hn, -1.0, 1.0)))
    node = np.degrees(np.arctan2(h[..., 0], -h[..., 1])) % 360.0
    v2 = np.sum(vel * vel, axis=-1)
    a = 1.0 / (2.0 / r - v2 / mu)
    evec = np.cross(vel, h) / mu - pos / r[..., None]
    e = np.linalg.norm(evec, axis=-1)
    O = np.radians(node)
    nvec = np.stack([np.cos(O), np.sin(O), np.zeros_like(O)], axis=-1)
    cos_w = np.sum(nvec * evec, axis=-1) / np.maximum(e, 1e-15)
    peri = np.degrees(np.arccos(np.clip(cos_w, -1.0, 1.0)))
    peri = np.where(evec[..., 2] < 0.0, 360.0 - peri, peri)
    cos_f = np.sum(evec * pos, axis=-1) / np.maximum(e * r, 1e-15)
    f = np.arccos(np.clip(cos_f, -1.0, 1.0))
    f = np.where(np.sum(pos * vel, axis=-1) < 0.0, 2.0 * np.pi - f, f)
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(f / 2.0), np.sqrt(1.0 + e) * np.cos(f / 2.0))
    M = np.degrees(E - e * np.sin(E)) % 360.0
    return tuple(np.asarray(x)[()] for x in (a, e, inc, node, peri, M))


def sun_barycentric(jd) -> tuple[np.ndarray, np.ndarray]:
    """Sun's barycentric ecliptic J2000 position [au] and velocity [au/day] at TDB JD(s)."""
    from astropy import units as u
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time

    t = Time(np.atleast_1d(np.asarray(jd, dtype=float)), format="jd", scale="tdb")
    pos, vel = get_body_barycentric_posvel("sun", t)
    p = pos.xyz.to(u.au).value.T
    v = vel.xyz.to(u.au / u.day).value.T
    c, s = math.cos(OBLIQUITY), math.sin(OBLIQUITY)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    return p @ rot.T, v @ rot.T


def barycentric_elements(a, e, inc, node, peri, M, epoch_jd):
    """Heliocentric osculating ecliptic J2000 elements -> barycentric (same epoch)."""
    pos, vel = _state(a, e, inc, node, peri, M, K2)
    sp, sv = sun_barycentric(epoch_jd)
    if np.ndim(a) == 0:
        sp, sv = sp[0], sv[0]
    return _elements(pos + sp, vel + sv, K2 * (1.0 + PLANET_MASS))


def free_inclination_helio(a, e, inc, node, peri, M, epoch_jd) -> np.ndarray:
    """I_free [deg] from heliocentric osculating elements (converted to barycentric)."""
    ab, eb, ib, nb, wb, _ = barycentric_elements(a, e, inc, node, peri, M, epoch_jd)
    return free_inclination(ab, eb, ib, nb, wb)[0]


# ---------------------------------------------------------------------------
# Huang et al. Table 1


def unpack_provisional(packed: str) -> str:
    """Packed provisional designation (J94E02S) -> compact 1994ES2."""
    if len(packed) != 7 or packed[0] not in CENTURY:
        return packed
    year = CENTURY[packed[0]] + int(packed[1:3])
    cycle = NCODE.index(packed[4]) * 10 + int(packed[5])
    return f"{year}{packed[3]}{packed[6]}{cycle if cycle else ''}"


@lru_cache(maxsize=1)
def huang_table() -> dict:
    """Huang et al. Table 1 rows keyed by number, compact provisional and OSSOS++ name."""
    out = {}
    with TABLE_FILE.open() as fh:
        for row in csv.DictReader(fh):
            row = dict(row)
            for k in ("a", "e", "inc", "Omega", "omega", "Ifree", "IfreeRange"):
                row[k] = float(row[k]) if row[k] not in ("-", "") else math.nan
            row["RESO"] = int(row["RESO"])
            ident = row["ID"]
            keys = []
            if ident.startswith("a"):
                keys.append(str(int(ident[1:])))
            elif ident.startswith("b"):
                keys.append(unpack_provisional(ident[1:]))
            m = re.search(r"\((\d{4}) ([A-Z]{2}\d*)\)", row["Name"])
            if m:
                keys.append(m.group(1) + m.group(2))
            if row["OSSOS"] != "x":
                keys.append(row["OSSOS"])
            for k in keys:
                out.setdefault(k, row)
    return out


def huang_lookup(*keys) -> dict | None:
    """The Table 1 row of the first key (number, compact provisional, OSSOS++ name) found."""
    table = huang_table()
    for k in keys:
        if k and k in table:
            return table[k]
    return None

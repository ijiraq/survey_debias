"""Grid-cell Horvitz–Thompson helpers for pencil-beam TNO surveys.

This library defines no surveys. Each characterization project builds its
own :class:`GridSurvey` and passes it as ``survey=`` (see ``examples/``).
"""
from __future__ import annotations

import csv
import io
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# size of aq_grid cells (a, q, sin i_free, H)
A_STEP = 0.2
Q_STEP = 0.2
SI_STEP = 0.001
H_STEP = 0.1
# size of model_ae cells (r, i)
R_STEP = 1.0
I_STEP = 1.0
# Some helpful constants.
BOWELL_G = -0.12
OBLIQUITY_J2000_DEG = 23.4392911
# rot.f95 equat_ecl; used when matching Detos1 / RADECeclXV
F95_OBLIQUITY_ARCSEC = 84381.41
TWO_HOURS_DAY = 2.0 / 24.0
# Minimum model objects retained in an (r, i) window before the window grows.
MODEL_AE_MIN_CANDIDATES = 50
MODEL_AE_MAX_EXPAND = 8
# Detections required per cell before its bias is accepted.
TARGET_DETECTIONS = 5000
# Program names a ``[detection_columns]`` table may map to file columns.
# ``mag`` is the apparent-magnitude column. Other file columns are kept as-is.
DETECTION_COLUMNS = (
    "mag", "name", "survey", "block", "a", "e", "i", "d_bary", "q", "ifree",
    "Omega", "Omfree", "omfree", "comp", "mag_err", "d_bary_err",
)
# Columns ``[detection_defaults]`` may fill when the table has none.
DETECTION_DEFAULTS = ("survey", "block", "comp")
# Derived quantities, in the order they are appended to the results table.
_DERIVED_ORDER = (
    "name", "survey", "block", "filter", "colour_group", "colour",
    "a", "e", "q", "ifree", "sin_ifree", "Hx", "cell",
    "block_bias", "bias", "bias_se",
)
_CFEPS_ELEMENTS = ("a", "e", "i", "d_bary", "Hx", "ifree")
# Fields that moved out of the survey file, and where they went.
REMOVED_FIELDS = {
    "field_ra_deg": "pointings.list",
    "field_dec_deg": "pointings.list",
    "mosaic_width_deg": "pointings.list",
    "mosaic_height_deg": "pointings.list",
    "epoch_jd": "pointings.list (one per epoch{i}/ directory)",
    "observer_csv": "pointings.list (observer column)",
    "eff_file": "pointings.list ({block}.eff column)",
    "fill_factor": "pointings.list (fill column)",
    "epoch_layout": "characterization/{survey}/ (epoch{i}/ directories are found)",
    "mag_color_offset": "colour.toml (filter - model_band colours)",
}


@dataclass(frozen=True)
class GridSurvey:
    """A debiasing project: detections, column mapping, and photometry.

    Geometry is not part of the survey file. Each survey's pointings,
    epochs, observer, and efficiency files are read from
    ``<root>/characterization/{survey}/pointings.list`` (see
    :mod:`survey_debias.pointings`), and colours from ``colour.toml``.

    Parameters
    ----------
    name : str
        The name of the project, used in log messages and output headers.
    detection_columns : dict
        Program column name → column name in the detections table, for
        every column that is read. ``mag`` is required (the apparent
        magnitude in the block's filter). ``survey`` and ``block`` select
        ``characterization/{survey}/{block}.eff``. ``model_ae`` also reads
        ``d_bary`` and ``i``. ``aq_grid`` also reads ``a``, ``e`` or ``q``,
        and ``i`` or ``ifree``. Other recognized names are ``name``,
        ``Omega``, ``Omfree``, ``omfree``, and ``comp``; ``mag_err`` and
        ``d_bary_err`` are read only by the H-distribution fit.
    detection_defaults : dict
        Values of ``survey``, ``block``, or ``comp`` for a table without
        that column (for example ``survey = "test"``, ``block = "test"``).
    model_band : str
        Band H is quoted in; one letter A–z (default ``"r"``).
    colour_file : str
        ``colour.toml`` relative to the root (``PhotSpec.COLORS`` layout).
        When it does not exist ossssim's built-in colours are used.
    surveys : tuple
        Survey directories included in the union bias. Empty: every
        ``characterization/{survey}`` with a ``pointings.list``.
    paper_reference_jd : float | None
        Optional reference JD from the survey paper, used for logging and
        as the JD column in the detections-full output.
    rate_cut_min_arcsec_hr, rate_cut_max_arcsec_hr : float
        Sky-motion range used by the start-up sanity check when the ``.eff``
        file has no ``rate_cut=`` line.
    detections_relpath : str
        Path of the input detections CSV, relative to the root.
    detections_full_name : str
        CFEPS detections file written to the root when every detection has
        ``a``, ``e``, and ``i``.
    results_name : str
        Results CSV written to the root.
    check_detected_title : str
        Label for detected objects in the check-plot titles.
    bias_method : str
        ``"aq_grid"``: cells in (a, q, sin i_free, H). ``"model_ae"``: cells
        in (r, i, H) with (a, e) drawn from an orbit model p(a, e | r, i).
    """

    name: str
    detection_columns: dict[str, str]
    detection_defaults: dict = field(default_factory=dict)
    model_band: str = "r"
    colour_file: str = "colour.toml"
    surveys: tuple = ()
    paper_reference_jd: float | None = None
    rate_cut_min_arcsec_hr: float = 0.03
    rate_cut_max_arcsec_hr: float = 8.66
    detections_relpath: str = "data/detections.csv"
    detections_full_name: str = "detections-full"
    results_name: str = "bias_results.csv"
    check_detected_title: str = "detected flag≥4"
    bias_method: str = "aq_grid"

    def __hash__(self) -> int:
        return hash((self.name, self.detections_relpath, self.bias_method))

    def __post_init__(self) -> None:
        if not isinstance(self.detection_columns, dict):
            raise ValueError(
                "detection_columns must be a table of program name = file column"
            )
        known = set(DETECTION_COLUMNS)
        canon: dict[str, str] = {}
        owners: dict[str, str] = {}
        for key, value in dict(self.detection_columns).items():
            name = str(key).strip()
            if name not in known:
                raise ValueError(
                    f"detection_columns: {key!r} is not a program column. "
                    f"Known columns: {', '.join(DETECTION_COLUMNS)}."
                )
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"detection_columns: {name} must name a file column, "
                    f"got {value!r}"
                )
            file_col = value.strip()
            if file_col in owners:
                raise ValueError(
                    f"detection_columns: file column {file_col!r} is assigned "
                    f"to both {owners[file_col]!r} and {name!r}"
                )
            owners[file_col] = name
            canon[name] = file_col
        if "mag" not in canon:
            raise ValueError(
                "detection_columns must include mag, the magnitude column "
                "in the detections table"
            )
        ordered = {name: canon[name] for name in DETECTION_COLUMNS if name in canon}
        object.__setattr__(self, "detection_columns", ordered)
        defaults = {}
        for key, value in dict(self.detection_defaults or {}).items():
            name = str(key).strip()
            if name not in DETECTION_DEFAULTS:
                raise ValueError(
                    f"detection_defaults: {key!r} cannot have a default; "
                    f"use {', '.join(DETECTION_DEFAULTS)}"
                )
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"detection_defaults: {name} must be a name, got {value!r}")
            defaults[name] = value.strip()
        object.__setattr__(self, "detection_defaults", defaults)
        for name in ("survey", "block"):
            if name not in ordered and name not in defaults:
                raise ValueError(
                    f"{name} needs a [detection_columns] entry or a "
                    f'[detection_defaults] value (for example {name} = "test")'
                )
        object.__setattr__(self, "surveys", tuple(str(s) for s in self.surveys))
        band = str(self.model_band).strip()
        if len(band) != 1 or not ("A" <= band <= "z"):
            raise ValueError(f"model_band {self.model_band!r} must be one letter A-z")
        object.__setattr__(self, "model_band", band)

    @property
    def mag_column(self) -> str:
        """File column for apparent magnitude (``detection_columns['mag']``)."""
        return self.detection_columns["mag"]


@dataclass(frozen=True)
class FieldView:
    """Geometry the sampling and plotting helpers read as ``survey=``.

    Built from one ``pointings.list`` tile (or a block's bounding box) by
    :meth:`survey_debias.pointings.Block.field_view`. ``mosaic_width_deg``
    is the on-sky width; the RA extent is that over cos(dec).
    """

    name: str
    field_ra_deg: float
    field_dec_deg: float
    mosaic_width_deg: float
    mosaic_height_deg: float
    epoch_jd: tuple
    observer_csv: str = ""
    rate_cut_min_arcsec_hr: float = 0.03
    rate_cut_max_arcsec_hr: float = 8.66
    paper_reference_jd: float | None = None
    check_detected_title: str = "detected flag≥4"

    @property
    def n_epochs(self) -> int:
        return len(self.epoch_jd)

    @property
    def mosaic_area_deg2(self) -> float:
        return self.mosaic_width_deg * self.mosaic_height_deg

    @property
    def mosaic_side_deg(self) -> float:
        return math.sqrt(self.mosaic_area_deg2)


def _require_survey(survey: GridSurvey | None, what: str) -> GridSurvey:
    if survey is None:
        raise ValueError(
            f"{what}: pass survey= (a GridSurvey) or the explicit field values"
        )
    return survey


def laplace_inclination(a_au: float) -> float:
    return 1.759 + 0.0321 * (a_au - 41.8)


def laplace_node(a_au: float) -> float:
    return 90.0 - 0.5 * (a_au - 43.0)


def compute_ifree(i_deg: float, node_deg: float, a_au: float) -> float:
    """Free inclination relative to the Laplace plane, given ecliptic (i, Ω)."""
    ip = laplace_inclination(a_au)
    om_lp = laplace_node(a_au)
    cos_ifree = (
        math.cos(math.radians(i_deg)) * math.cos(math.radians(ip))
        + math.sin(math.radians(i_deg)) * math.sin(math.radians(ip))
        * math.cos(math.radians(node_deg - om_lp))
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_ifree))))


def _orbit_pole(i_deg: float, node_deg: float) -> np.ndarray:
    i = math.radians(i_deg)
    node = math.radians(node_deg)
    return np.array([
        math.sin(i) * math.sin(node),
        -math.sin(i) * math.cos(node),
        math.cos(i),
    ])


def ecliptic_from_ifree(ifree_deg: float, a_au: float, rng: np.random.Generator) -> tuple[float, float]:
    """Draw ecliptic (i, Ω) whose free inclination is ifree_deg.

    The orbit pole is placed at angular distance i_free from the Laplace pole
    with a uniform random azimuth. Inverse of compute_ifree.
    """
    ip = laplace_inclination(a_au)
    om_lp = laplace_node(a_au)
    lp = _orbit_pole(ip, om_lp)
    ref = np.array([1.0, 0.0, 0.0]) if abs(lp[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(lp, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(lp, e1)
    psi = rng.uniform(0.0, 2.0 * math.pi)
    ifree = math.radians(ifree_deg)
    pole = math.cos(ifree) * lp + math.sin(ifree) * (math.cos(psi) * e1 + math.sin(psi) * e2)
    pole = pole / np.linalg.norm(pole)
    i_ecl = math.degrees(math.acos(max(-1.0, min(1.0, float(pole[2])))))
    si = math.sin(math.radians(i_ecl))
    if si < 1e-12:
        node_ecl = float(rng.uniform(0.0, 360.0))
    else:
        node_ecl = math.degrees(math.atan2(pole[0] / si, -pole[1] / si)) % 360.0
    return max(i_ecl, 0.05), node_ecl


def bowell_phase_correction(alpha_rad: float, g: float = BOWELL_G) -> float:
    """2.5 log10((1-G)φ1 + G φ2), the AppMag term added when inverting for H."""
    if alpha_rad <= 0.0:
        return 0.0
    ta = math.tan(alpha_rad / 2.0)
    phi1 = math.exp(-3.33 * ta ** 0.63)
    phi2 = math.exp(-1.87 * ta ** 1.22)
    phi = (1.0 - g) * phi1 + g * phi2
    if phi <= 0.0:
        return 0.0
    return 2.5 * math.log10(phi)


def apparent_to_Hr(m_survey: float, d_au: float, robs_au: float = 1.0,
                   g: float = BOWELL_G, colour: float = 0.0) -> float:
    """Model-band H inverted from AppMag with r = Δ = d_bary.

    ``colour`` is ``filter - model_band`` for the object's spectral group
    (from ``colour.toml``), the term OSSSSim's ``detos1`` adds to H before
    computing the magnitude in the block's filter. Geometry uses the same
    Bowell G=-0.12 law as OSSSSim.
    """
    denom = 2.0 * d_au * d_au
    cos_a = max(-1.0, min(1.0, (-robs_au ** 2 + 2.0 * d_au ** 2) / denom))
    alpha = math.acos(cos_a)
    return (m_survey - colour - 5.0 * math.log10(d_au * d_au)
            + bowell_phase_correction(alpha, g))


def geometric_detection_prob(area_deg2: float, inc_deg: float, beta_deg: float) -> float:
    """Single-epoch geometric probability for a small field.

    P ≈ A / (360° × 2 × sqrt(i² − β²)) when i > |β|. For a 0.05 deg² mosaic
    near the ecliptic and a ~7° inclination belt this is ~1e-5.
    """
    if abs(inc_deg) <= abs(beta_deg):
        return 0.0
    return area_deg2 / (360.0 * 2.0 * math.sqrt(inc_deg ** 2 - beta_deg ** 2))


def geometric_prob_for_aimed(a: float, e: float, inc_deg: float, node_deg: float,
                             peri_deg: float, M_deg: float,
                             area_deg2: float | None = None,
                             survey: GridSurvey | None = None) -> float:
    """P_geom at the aimed orbit's own ecliptic latitude, not the ICRS field.

    Parallax (~1°) means barycentric β of an on-LOS TNO is not icrs_to_ecliptic
    of the pointing. Using the field latitude spuriously zeros P_geom for cold
    cells where aimed i sits between |β_obj| and |β_field|.
    """
    if area_deg2 is None:
        area_deg2 = _require_survey(survey, "geometric_prob_for_aimed").mosaic_area_deg2
    x, y, z = ecliptic_xyz_from_elements(a, e, inc_deg, node_deg, peri_deg, M_deg)
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0.0:
        return 0.0
    beta = math.degrees(math.asin(max(-1.0, min(1.0, z / r))))
    return geometric_detection_prob(area_deg2, inc_deg, beta)


def icrs_to_ecliptic(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """J2000 equatorial (RA, Dec) to ecliptic (lon, lat), degrees."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    eps = math.radians(OBLIQUITY_J2000_DEG)
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)
    ye = y * math.cos(eps) + z * math.sin(eps)
    ze = -y * math.sin(eps) + z * math.cos(eps)
    lon = math.degrees(math.atan2(ye, x)) % 360.0
    lat = math.degrees(math.asin(max(-1.0, min(1.0, ze))))
    return lon, lat


def aimed_at_field(ra_deg: float | None = None, dec_deg: float | None = None,
                   survey: GridSurvey | None = None
                   ) -> tuple[float, float, float, float]:
    """(i, Ω, ω, M) that places a circular orbit on the given ICRS pointing.

    This is the barycentric sky direction, not the apparent direction from
    the observatory. Use los_circular_elements to plant in the mosaic as
    Detos1 sees it.
    """
    if ra_deg is None or dec_deg is None:
        surv = _require_survey(survey, "aimed_at_field")
        ra_deg = surv.field_ra_deg if ra_deg is None else ra_deg
        dec_deg = surv.field_dec_deg if dec_deg is None else dec_deg
    lon, lat = icrs_to_ecliptic(ra_deg, dec_deg)
    inc = max(abs(lat), 0.05)
    arglat = 90.0 if lat >= 0.0 else 270.0
    node = (lon - arglat) % 360.0
    return inc, node, arglat, 0.0


def _obliquity_rad() -> float:
    return math.radians(F95_OBLIQUITY_ARCSEC / 3600.0)


def ecliptic_to_icrf(x: float, y: float, z: float) -> tuple[float, float, float]:
    """equat_ecl(-1): ecliptic J2000 → ICRF."""
    coseps = math.cos(_obliquity_rad())
    sineps = math.sin(_obliquity_rad())
    return x, coseps * y - sineps * z, sineps * y + coseps * z


def icrf_to_ecliptic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """equat_ecl(+1): ICRF → ecliptic J2000."""
    coseps = math.cos(_obliquity_rad())
    sineps = math.sin(_obliquity_rad())
    return x, coseps * y + sineps * z, -sineps * y + coseps * z


def circular_elements_through_ecliptic_xyz(
        x: float, y: float, z: float) -> tuple[float, float, float, float, float, float]:
    """Circular (a, e, i, Ω, ω, M) whose position is ecliptic (x,y,z) AU."""
    r = math.sqrt(x * x + y * y + z * z)
    lat = math.degrees(math.asin(max(-1.0, min(1.0, z / r))))
    lon = math.degrees(math.atan2(y, x)) % 360.0
    inc = max(abs(lat), 0.05)
    arglat = 90.0 if lat >= 0.0 else 270.0
    node = (lon - arglat) % 360.0
    return r, 0.0, inc, node, arglat, 0.0


def parse_jpl_horizons_icrf(path, jd: float) -> tuple[float, float, float]:
    """Observer barycentric ICRF (AU) from a Horizons CSV, matching read_jpl_csv."""
    with open(path) as fh:
        text = fh.read()
    header, _, rest = text.partition("$$SOE")
    ecliptic_frame = False
    for line in header.splitlines():
        if "Reference frame" in line:
            ecliptic_frame = ("Ecliptic" in line) or ("ecliptic" in line)
    for line in rest.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("$$EOE"):
            break
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 8:
            continue
        ejd = float(parts[0])
        if ejd <= jd:
            continue
        x, y, z = (float(parts[i]) for i in (2, 3, 4))
        vx, vy, vz = (float(parts[i]) for i in (5, 6, 7))
        dt = jd - ejd
        pos = (x + vx * dt, y + vy * dt, z + vz * dt)
        if ecliptic_frame:
            pos = ecliptic_to_icrf(*pos)
        return pos
    raise RuntimeError(f"JD {jd} outside Horizons range in {path}")


def los_circular_elements(ra_deg: float, dec_deg: float, a_au: float,
                          jpl_path, jd: float
                          ) -> tuple[float, float, float, float, float, float]:
    """Circular orbit at a_au along the observer LOS to (ra, dec) at jd."""
    obs = parse_jpl_horizons_icrf(jpl_path, jd)
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    los = (
        math.cos(dec) * math.cos(ra),
        math.cos(dec) * math.sin(ra),
        math.sin(dec),
    )
    b = 2.0 * sum(o * u_k for o, u_k in zip(obs, los))
    c = sum(o * o for o in obs) - a_au * a_au
    disc = max(0.0, b * b - 4.0 * c)
    t = 0.5 * (-b + math.sqrt(disc))
    obj_icrf = tuple(o + t * u_k for o, u_k in zip(obs, los))
    obj_ecl = icrf_to_ecliptic(*obj_icrf)
    return circular_elements_through_ecliptic_xyz(*obj_ecl)


def ecliptic_xyz_from_elements(a: float, e: float, inc_deg: float,
                               node_deg: float, peri_deg: float, M_deg: float
                               ) -> tuple[float, float, float]:
    """Barycentric ecliptic xyz matching F95 pos_cart (e near 0 is fine)."""
    inc = math.radians(inc_deg)
    node = math.radians(node_deg)
    peri = math.radians(peri_deg)
    M = math.radians(M_deg) % (2.0 * math.pi)
    E = M
    for _ in range(20):
        f = E - e * math.sin(E) - M
        if abs(f) < 1e-14:
            break
        E -= f / (1.0 - e * math.cos(E))
    cos_i, sin_i = math.cos(inc), math.sin(inc)
    c_w, s_w = math.cos(peri), math.sin(peri)
    c_o, s_o = math.cos(node), math.sin(node)
    q0 = a * (math.cos(E) - e)
    q1 = a * math.sqrt(max(0.0, 1.0 - e * e)) * math.sin(E)
    x = (c_o * c_w - cos_i * s_o * s_w) * q0 + (-c_o * s_w - cos_i * s_o * c_w) * q1
    y = (s_o * c_w + cos_i * c_o * s_w) * q0 + (-s_o * s_w + cos_i * c_o * c_w) * q1
    z = (sin_i * s_w) * q0 + (sin_i * c_w) * q1
    return x, y, z


def apparent_radec_deg(a: float, e: float, inc_deg: float, node_deg: float,
                       peri_deg: float, M_deg: float, obs_icrf
                       ) -> tuple[float, float]:
    """ICRS RA/Dec as RADECeclXV computes them (object ecliptic, obs ICRF)."""
    obj_icrf = ecliptic_to_icrf(*ecliptic_xyz_from_elements(
        a, e, inc_deg, node_deg, peri_deg, M_deg))
    rel = [obj_icrf[i] - obs_icrf[i] for i in range(3)]
    delta = math.sqrt(sum(v * v for v in rel))
    ra = math.degrees(math.atan2(rel[1], rel[0])) % 360.0
    dec = math.degrees(math.asin(max(-1.0, min(1.0, rel[2] / delta))))
    return ra, dec


def sky_separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = (ra1 - ra2) * math.cos(math.radians(0.5 * (dec1 + dec2)))
    return math.hypot(dra, dec1 - dec2)


def angle_in_rate_cone(obj_deg: float, centre_deg: float, hwidth_deg: float
                       ) -> bool:
    """Whether a motion PA is inside the rate_cut direction cone.

    Detos1 uses atan2 ∈ [−180°, 180°]. A centre set to a field RA (e.g.
    209.4°, not a PA) compared without wrapping rejects pre-turnaround
    motion at −168.7° even when half-width is 180°.
    """
    dang = (centre_deg - obj_deg + 180.0) % 360.0 - 180.0
    return abs(dang) <= hwidth_deg


def mean_motion_deg_per_day(a_au: float) -> float:
    """n = 360° / P, P = a^{3/2} yr in days. Matches Detos1 with gmb≈1."""
    return 360.0 / (a_au ** 1.5 * 365.25)


def epoch_geometry(a: float, e: float, inc_deg: float, node_deg: float,
                   peri_deg: float, M_deg: float, jpl_path, element_jd: float,
                   obs_jd: float, field_ra: float | None = None,
                   field_dec: float | None = None,
                   survey: GridSurvey | None = None
                   ) -> tuple[float, float, float, float]:
    """Apparent (RA, Dec, sep_deg, rate_arcsec_hr) at obs_jd.

    Detos1 advances M from the element epoch to the pointing JD, then
    measures rate over the next two hours (GetSurvey's second ObsPos).
    """
    if field_ra is None or field_dec is None:
        surv = _require_survey(survey, "epoch_geometry")
        field_ra = surv.field_ra_deg if field_ra is None else field_ra
        field_dec = surv.field_dec_deg if field_dec is None else field_dec
    n = mean_motion_deg_per_day(a)
    m1 = M_deg + n * (obs_jd - element_jd)
    m2 = M_deg + n * (obs_jd + TWO_HOURS_DAY - element_jd)
    obs1 = parse_jpl_horizons_icrf(jpl_path, obs_jd)
    obs2 = parse_jpl_horizons_icrf(jpl_path, obs_jd + TWO_HOURS_DAY)
    ra1, dec1 = apparent_radec_deg(a, e, inc_deg, node_deg, peri_deg, m1, obs1)
    ra2, dec2 = apparent_radec_deg(a, e, inc_deg, node_deg, peri_deg, m2, obs2)
    dra = (ra1 - ra2) * math.cos(math.radians(dec1))
    ddec = dec2 - dec1
    rate = math.hypot(dra, ddec) / TWO_HOURS_DAY * 3600.0 / 24.0
    sep = sky_separation_deg(ra1, dec1, field_ra, field_dec)
    return ra1, dec1, sep, rate


def cell_index(value: float, step: float) -> float:
    return math.floor(value / step) * step


def cell_key(a: float, q: float, sin_ifree: float, hx: float) -> tuple:
    """(a, q, sin i_free, H) cell for well-constrained orbits (``aq_grid``)."""
    return (
        round(cell_index(a, A_STEP), 6),
        round(cell_index(q, Q_STEP), 6),
        round(cell_index(sin_ifree, SI_STEP), 6),
        round(cell_index(hx, H_STEP), 6),
    )


def rih_cell_key(r_au: float, i_deg: float, hx: float) -> tuple:
    """(r, ecliptic i, H) cell when (a, e) come from a model prior."""
    return (
        round(cell_index(r_au, R_STEP), 6),
        round(cell_index(i_deg, I_STEP), 6),
        round(cell_index(hx, H_STEP), 6),
    )


def bounds_from_key(key: tuple) -> dict:
    """Bounds for an ``aq_grid`` cell key."""
    a0, q0, si0, h0 = key
    return {
        "a": (a0, a0 + A_STEP),
        "q": (q0, q0 + Q_STEP),
        "sin_ifree": (max(0.0, si0), min(1.0, si0 + SI_STEP)),
        "Hx": (h0, h0 + H_STEP),
    }


def rih_bounds_from_key(key: tuple) -> dict:
    """Bounds for a ``model_ae`` (r, i, H) cell key."""
    r0, i0, h0 = key
    return {
        "r": (r0, r0 + R_STEP),
        "i": (i0, i0 + I_STEP),
        "Hx": (h0, h0 + H_STEP),
    }


MODEL_PATH_ENV = "SURVEY_DEBIAS_MODEL"


def default_orbit_model_path(root=None) -> Path:
    """Default orbit-model file or directory for ``model_ae`` debiasing.

    ``$SURVEY_DEBIAS_MODEL`` if set, else ``<root>/Models/OSSOS`` (``root``
    defaults to the current directory). The full OSSOS ModelUsed tables are
    not shipped with this package; a short header sample for tests lives
    under ``tests/data/OSSOS/``. Override with ``--model PATH``.
    """
    env = os.environ.get(MODEL_PATH_ENV)
    if env:
        return Path(env).expanduser()
    return Path(root or Path.cwd()) / "Models" / "OSSOS"


def _detect_orbit_model_format(path: Path) -> str:
    """Return ``ossos_modelused`` or ``l7`` from a commented column header."""
    with path.open() as fh:
        for line in fh:
            if not line.lstrip().startswith("#"):
                continue
            low = line.lower()
            if "comment" in low and "dist" in low:
                return "ossos_modelused"
            if "comp" in low and "dist" in low:
                return "l7"
    with path.open() as fh:
        for line in fh:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            n = len(line.split())
            if n >= 10:
                return "ossos_modelused"
            if n >= 9:
                return "l7"
            break
    raise ValueError(f"cannot detect orbit-model format for {path}")


def _normalize_model_comp(label: str) -> str:
    """Strip ModelUsed trailing underscores; keep a short component tag."""
    return str(label).strip().rstrip("_") or "unknown"


@dataclass(frozen=True)
class OrbitModelCatalog:
    """Keplerian lookup table used as an empirical p(a, e | r, i) prior.

    All dynamical components in the file are kept. Selecting on (r, i)
    mixes cold / hot / resonant / etc. in the model's own proportions —
    appropriate when a detection's component is unknown.

    Accepts OSSOS Models 1.0 ``ModelUsed.dat`` tables and legacy L7 files.
    """

    a: np.ndarray
    e: np.ndarray
    i_deg: np.ndarray
    dist: np.ndarray
    comp: np.ndarray

    def __len__(self) -> int:
        return int(self.a.size)

    @classmethod
    def _from_arrays(cls, a_list, e_list, i_list, d_list, c_list, path: Path
                     ) -> "OrbitModelCatalog":
        if not a_list:
            raise ValueError(f"no model rows in {path}")
        return cls(
            a=np.asarray(a_list, dtype=float),
            e=np.asarray(e_list, dtype=float),
            i_deg=np.asarray(i_list, dtype=float),
            dist=np.asarray(d_list, dtype=float),
            comp=np.asarray(c_list, dtype=object),
        )

    @classmethod
    def from_file(cls, path, comp_label: str | None = None) -> "OrbitModelCatalog":
        """Load one model file (OSSOS ModelUsed or L7).

        If ``comp_label`` is set (typical when loading a directory of
        component files), every row is tagged with that label. Otherwise
        the file's own ``comment`` / ``comp`` column is used.
        """
        path = Path(path)
        fmt = _detect_orbit_model_format(path)
        a_list: list[float] = []
        e_list: list[float] = []
        i_list: list[float] = []
        d_list: list[float] = []
        c_list: list[str] = []
        with path.open() as fh:
            for line in fh:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                if fmt == "ossos_modelused":
                    # a e i Omega omega M H epoch dist comment
                    if len(parts) < 10:
                        continue
                    a_list.append(float(parts[0]))
                    e_list.append(float(parts[1]))
                    i_list.append(float(parts[2]))
                    d_list.append(float(parts[8]))
                    label = comp_label or _normalize_model_comp(parts[9])
                    c_list.append(label)
                else:
                    # a e i node peri M H dist comp [j k]
                    if len(parts) < 9:
                        continue
                    a_list.append(float(parts[0]))
                    e_list.append(float(parts[1]))
                    i_list.append(float(parts[2]))
                    d_list.append(float(parts[7]))
                    label = comp_label or _normalize_model_comp(parts[8])
                    c_list.append(label)
        return cls._from_arrays(a_list, e_list, i_list, d_list, c_list, path)

    @classmethod
    def from_directory(cls, path) -> "OrbitModelCatalog":
        """Concatenate every model table in a directory.

        Component labels come from each filename stem before ``-ModelUsed``
        (e.g. ``Classical-ModelUsed.dat`` → ``Classical``).
        """
        path = Path(path)
        files = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in {".dat", ".txt", ""}
            and not p.name.startswith(".")
        )
        modelused = [p for p in files if "modelused" in p.name.lower()]
        if modelused:
            files = modelused
        if not files:
            raise FileNotFoundError(f"no orbit model files in {path}")
        catalogs = []
        for fp in files:
            stem = fp.stem
            for sep in ("-ModelUsed", "_ModelUsed", "-modelused", "ModelUsed"):
                if sep in stem:
                    stem = stem.split(sep)[0]
                    break
            catalogs.append(cls.from_file(fp, comp_label=stem or fp.stem))
        return cls.concatenate(catalogs)

    @classmethod
    def from_path(cls, path) -> "OrbitModelCatalog":
        """Load a model file or a directory of component model files."""
        path = Path(path)
        if path.is_dir():
            return cls.from_directory(path)
        if not path.is_file():
            raise FileNotFoundError(f"orbit model path not found: {path}")
        return cls.from_file(path)

    @classmethod
    def from_l7(cls, path) -> "OrbitModelCatalog":
        """Backward-compatible alias for :meth:`from_path`."""
        return cls.from_path(path)

    @classmethod
    def concatenate(cls, catalogs: list["OrbitModelCatalog"]) -> "OrbitModelCatalog":
        if not catalogs:
            raise ValueError("concatenate() requires at least one catalog")
        return cls(
            a=np.concatenate([c.a for c in catalogs]),
            e=np.concatenate([c.e for c in catalogs]),
            i_deg=np.concatenate([c.i_deg for c in catalogs]),
            dist=np.concatenate([c.dist for c in catalogs]),
            comp=np.concatenate([c.comp for c in catalogs]),
        )

    def select(self, r_lo: float, r_hi: float, i_lo: float, i_hi: float
               ) -> "OrbitModelCatalog":
        """Objects with discovery distance and ecliptic i in the window."""
        mask = (
            (self.dist >= r_lo) & (self.dist < r_hi)
            & (self.i_deg >= i_lo) & (self.i_deg < i_hi)
        )
        return OrbitModelCatalog(
            a=self.a[mask], e=self.e[mask], i_deg=self.i_deg[mask],
            dist=self.dist[mask], comp=self.comp[mask],
        )

    def select_expanding(self, r_lo: float, r_hi: float, i_lo: float, i_hi: float,
                         min_n: int = MODEL_AE_MIN_CANDIDATES,
                         max_expand: int = MODEL_AE_MAX_EXPAND
                         ) -> tuple["OrbitModelCatalog", float, float]:
        """Grow the (r, i) window symmetrically until ``min_n`` objects.

        Returns ``(subset, dr_half, di_half)`` actually used.
        """
        r_mid = 0.5 * (r_lo + r_hi)
        i_mid = 0.5 * (i_lo + i_hi)
        dr0 = max(0.5 * (r_hi - r_lo), 0.5 * R_STEP)
        di0 = max(0.5 * (i_hi - i_lo), 0.5 * I_STEP)
        for k in range(max_expand):
            dr = dr0 * (2.0 ** k)
            di = di0 * (2.0 ** k)
            sub = self.select(r_mid - dr, r_mid + dr, i_mid - di, i_mid + di)
            if len(sub) >= min_n:
                return sub, dr, di
        sub = self.select(
            r_mid - dr0 * (2.0 ** max_expand),
            r_mid + dr0 * (2.0 ** max_expand),
            i_mid - di0 * (2.0 ** max_expand),
            i_mid + di0 * (2.0 ** max_expand),
        )
        if len(sub) == 0:
            raise RuntimeError(
                f"no model objects near r∈[{r_lo}, {r_hi}), i∈[{i_lo}, {i_hi})"
            )
        return sub, dr0 * (2.0 ** (max_expand - 1)), di0 * (2.0 ** (max_expand - 1))

    def component_fractions(self) -> dict[str, float]:
        if len(self) == 0:
            return {}
        vals, counts = np.unique(self.comp, return_counts=True)
        n = float(len(self))
        return {str(v): float(c) / n for v, c in zip(vals, counts)}

    def reachable_at(self, r_au: float) -> np.ndarray:
        """Boolean mask: ellipse contains heliocentric distance ``r_au``."""
        q = self.a * (1.0 - self.e)
        q_ap = self.a * (1.0 + self.e)
        return (q <= r_au + 1e-8) & (r_au <= q_ap + 1e-8)

    def sample_ae(self, rng: np.random.Generator, r_au: float | None = None,
                  max_tries: int = 10000) -> tuple[float, float, str]:
        """Draw (a, e, comp), optionally requiring the orbit to reach ``r_au``."""
        if len(self) == 0:
            raise RuntimeError("empty OrbitModelCatalog")
        if r_au is None:
            idx = int(rng.integers(0, len(self)))
            return float(self.a[idx]), float(self.e[idx]), str(self.comp[idx])
        idxs = np.flatnonzero(self.reachable_at(r_au))
        if idxs.size == 0:
            raise RuntimeError(
                f"no model (a,e) reaches r={r_au:.3f} AU in this (r,i) window"
            )
        idx = int(rng.choice(idxs))
        return float(self.a[idx]), float(self.e[idx]), str(self.comp[idx])


class Detections(list):
    """Detection rows, the file columns they came from, and derived columns.

    Each row holds the canonical values the run uses. Original cell text is
    on ``_source``; names in ``_computed`` were derived for that row.
    ``input_columns`` is the file header. ``columns_used`` maps each program
    name that was read (including ``mag``) to the file column it came from.
    ``computed_columns`` is the derived quantities across the table, in
    results-file order (``bias`` is added after the run). ``notes`` describes
    those derivations.
    """

    def __init__(self, rows=(), *, input_columns, computed_columns,
                 column_map, columns_used, notes=()):
        super().__init__(rows)
        self.input_columns = list(input_columns)
        self.computed_columns = list(computed_columns)
        self.column_map = dict(column_map)
        self.columns_used = dict(columns_used)
        self.notes = list(notes)


def _table_lines(path: Path) -> list[str]:
    """CSV records, skipping blank lines and ``#`` comments before the header."""
    lines = path.read_text().splitlines()
    start = 0
    while start < len(lines):
        stripped = lines[start].strip()
        if stripped and not stripped.startswith("#"):
            break
        start += 1
    if start >= len(lines):
        raise ValueError(f"{path}: no header row")
    return lines[start:]


def _bind_columns(path, fieldnames: list[str], survey: GridSurvey) -> dict[str, str]:
    """Program column → file column for each ``detection_columns`` entry."""
    fields = set(fieldnames)
    missing = [
        f"{canon}={file_col}"
        for canon, file_col in survey.detection_columns.items()
        if file_col not in fields
    ]
    if missing:
        have = ", ".join(fieldnames) or "(none)"
        raise ValueError(
            f"{path}: detection_columns names column(s) not in the file: "
            f"{', '.join(missing)}. File columns: {have}."
        )
    return dict(survey.detection_columns)


def _missing_required(bound: dict[str, str], survey: GridSurvey) -> list[str]:
    missing = []
    if "mag" not in bound:
        missing.append("mag")
    if survey.bias_method == "model_ae":
        for name in ("d_bary", "i"):
            if name not in bound:
                missing.append(name)
    elif survey.bias_method == "aq_grid":
        if "d_bary" not in bound:
            missing.append("d_bary")
        if "a" not in bound:
            missing.append("a")
        if "e" not in bound and "q" not in bound:
            missing.append("e or q")
        if "i" not in bound and "ifree" not in bound:
            missing.append("i or ifree")
    else:
        raise ValueError(f"unknown bias_method {survey.bias_method!r}")
    return missing


def _require_columns(path, fieldnames, missing: list[str], survey: GridSurvey) -> None:
    if not missing:
        return
    have = ", ".join(fieldnames) or "(none)"
    if survey.bias_method == "model_ae":
        need = "mag, d_bary, and i"
    else:
        need = "mag, a, d_bary, e or q, and i or ifree"
    declared = ", ".join(
        f"{k}={v}" for k, v in survey.detection_columns.items()
    )
    raise ValueError(
        f"{path}: missing required column(s): {', '.join(missing)}. "
        f"{survey.bias_method} reads {need} from [detection_columns]. "
        f"Declared: {declared}. File columns: {have}."
    )


def _cell_text(source: dict, file_col: str | None):
    if file_col is None:
        return None
    value = source.get(file_col)
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def _parse_float(text: str, where: str, column: str) -> float:
    try:
        return float(text)
    except ValueError as ex:
        raise ValueError(
            f"{where}: column {column!r} value {text!r} is not a number"
        ) from ex


def _fill_orbital_elements(a, e, q, where: str):
    """Fill whichever of ``a``, ``e``, ``q`` is missing from the other two."""
    computed = []
    if a is None and e is not None and q is not None:
        if e >= 1.0:
            raise ValueError(f"{where}: cannot derive a from e={e} and q={q}")
        a = q / (1.0 - e)
        computed.append("a")
    if e is None and a is not None and q is not None:
        if a == 0.0:
            raise ValueError(f"{where}: cannot derive e from a=0")
        e = 1.0 - q / a
        computed.append("e")
    if q is None and a is not None and e is not None:
        q = a * (1.0 - e)
        computed.append("q")
    return a, e, q, computed


def _derivation_notes(rows: list[dict]) -> list[str]:
    computed = {name for row in rows for name in row["_computed"]}
    notes = []
    if "q" in computed:
        notes.append("q = a*(1-e)")
    if "e" in computed:
        notes.append("e = 1 - q/a")
    if "a" in computed:
        notes.append("a = q/(1-e)")
    if any(
        "ifree" in row["_computed"] and row.get("Omega") is None for row in rows
    ):
        notes.append(
            "ifree from ecliptic i and a with Omega = 0 "
            "(no Omega value on that row)"
        )
    if "name" in computed:
        notes.append("name is the 1-based row number (no name column)")
    return notes


def load_detections(path, survey: GridSurvey, colour_for=None) -> Detections:
    """Read a detections table and assign bias cells.

    ``colour_for(survey, block, comp)`` returns ``(filter, colour, group)``:
    the block's filter, ``filter - model_band`` for the object's spectral
    group, and that group's name (see :class:`survey_debias.pointings.Project`).
    Without it the colour is 0 and the filter blank.

    The reader keeps every column in the file. Columns listed in
    ``survey.detection_columns`` are the ones interpreted (program name =
    file column), including ``mag``. A column that is not listed is kept
    in the results and is not used for the cell.

    ``model_ae`` requires ``mag``, ``d_bary``, and ecliptic ``i`` in that
    table. ``a`` and ``e`` are read only when listed. ``aq_grid`` also
    requires ``a`` and either ``e`` or ``q``, and ``i`` or ``ifree``.

    Blank optional cells are absent. When two of ``a``, ``e``, and ``q``
    are present, the third is derived. ``ifree`` is taken from the file
    when present; otherwise it is computed from ecliptic ``i`` and ``a``
    (``Omega`` when that column has a value, otherwise 0). ``Hx`` is always
    computed from the magnitude and ``d_bary``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"detections file not found: {path}")
    reader = csv.DictReader(_table_lines(path))
    if not reader.fieldnames:
        raise ValueError(f"{path}: no header row")
    if any(h is None or not str(h).strip() for h in reader.fieldnames):
        raise ValueError(f"{path}: header has a blank column name")
    fieldnames = [str(h).strip() for h in reader.fieldnames]
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError(f"{path}: duplicate column name in the header")
    reader.fieldnames = fieldnames
    bound = _bind_columns(path, fieldnames, survey)
    _require_columns(path, fieldnames, _missing_required(bound, survey), survey)

    rows = []
    for number, raw in enumerate(reader, start=1):
        where = f"{path}: row {number}"
        source = {
            key: ("" if value is None else str(value).strip())
            for key, value in raw.items()
            if key is not None
        }
        parsed = {}
        for canon, file_col in bound.items():
            if canon == "mag":
                continue
            text = _cell_text(source, file_col)
            if text is None:
                parsed[canon] = None
            elif canon in ("name", "comp", "survey", "block"):
                parsed[canon] = text
            else:
                parsed[canon] = _parse_float(text, where, file_col)
        mag_file = bound["mag"]
        mag_text = _cell_text(source, mag_file)
        if mag_text is None:
            raise ValueError(f"{where}: blank magnitude in column {mag_file!r}")
        mag = _parse_float(mag_text, where, mag_file)
        computed: list[str] = []

        name = parsed.get("name")
        if not name:
            name = str(number)
            computed.append("name")
        for key in DETECTION_DEFAULTS:
            if parsed.get(key) is None and key in survey.detection_defaults:
                parsed[key] = survey.detection_defaults[key]
                if key != "comp":
                    computed.append(key)
        for key in ("survey", "block"):
            if not parsed.get(key):
                raise ValueError(
                    f"{where}: blank {key}; fill the column or set "
                    f"[detection_defaults] {key}"
                )
        filt, colour, group = "", 0.0, "default"
        if colour_for is not None:
            try:
                filt, colour, group = colour_for(
                    parsed["survey"], parsed["block"], parsed.get("comp"))
            except (OSError, ValueError) as ex:
                raise ValueError(f"{where}: {ex}") from None
        computed.extend(("filter", "colour_group", "colour"))
        i = parsed.get("i")
        d = parsed.get("d_bary")
        if d is None:
            raise ValueError(f"{where}: blank d_bary")
        if survey.bias_method == "model_ae" and i is None:
            raise ValueError(f"{where}: blank i")

        a, e, q, derived_orbit = _fill_orbital_elements(
            parsed.get("a"), parsed.get("e"), parsed.get("q"), where,
        )
        computed.extend(derived_orbit)
        hx = apparent_to_Hr(mag, d, colour=colour)
        computed.append("Hx")

        ifree = parsed.get("ifree")
        if ifree is None and i is not None and a is not None:
            node = parsed.get("Omega")
            ifree = compute_ifree(i, 0.0 if node is None else node, a)
            computed.append("ifree")
        sin_ifree = None
        if ifree is not None:
            sin_ifree = math.sin(math.radians(ifree))
            computed.append("sin_ifree")

        if survey.bias_method == "model_ae":
            cell = rih_cell_key(d, i, hx)
        elif a is None or q is None or ifree is None or sin_ifree is None:
            raise ValueError(
                f"{where}: aq_grid needs a, q (or e), d_bary, and i or ifree"
            )
        else:
            cell = cell_key(a, q, sin_ifree, hx)
        computed.append("cell")

        row = {
            "name": name,
            "survey": parsed["survey"],
            "block": parsed["block"],
            "filter": filt,
            "colour_group": group,
            "colour": colour,
            "mag": mag,
            "d_bary": d,
            "Hx": hx,
            "cell": cell,
            "_source": source,
            "_computed": computed,
        }
        for key, value in (
            ("a", a), ("e", e), ("i", i), ("q", q), ("ifree", ifree),
            ("sin_ifree", sin_ifree), ("Omega", parsed.get("Omega")),
            ("Omfree", parsed.get("Omfree")), ("omfree", parsed.get("omfree")),
            ("comp", parsed.get("comp")),
        ):
            if value is not None:
                row[key] = value
        rows.append(row)

    computed_names = {name for row in rows for name in row["_computed"]}
    computed_columns = [name for name in _DERIVED_ORDER if name in computed_names]
    computed_columns.extend(
        name for name in computed_names if name not in computed_columns
    )
    return Detections(
        rows,
        input_columns=fieldnames,
        computed_columns=computed_columns,
        column_map=survey.detection_columns,
        columns_used=dict(bound),
        notes=_derivation_notes(rows),
    )


def _computed_value(row: dict, name: str):
    """Value of a derived column, blank when this row did not derive it."""
    if name not in _BIAS_COLUMNS and name not in row.get("_computed", ()):
        return None
    return row.get(name)


_BIAS_COLUMNS = ("block_bias", "bias", "bias_se")
BIAS_MODES = ("union", "block")


def _format_computed(name: str, value) -> str:
    if value is None:
        return ""
    if name == "cell":
        return str(tuple(value))
    if name in {"name", "comp"} or isinstance(value, str):
        return str(value)
    number = float(value)
    if name == "Hx":
        return f"{number:.4f}"
    if name == "colour":
        return f"{number:.4f}"
    if name in _BIAS_COLUMNS:
        return f"{number:.7g}"
    return f"{number:.6g}"


def write_bias_results(out_path, detections: Detections, survey: GridSurvey,
                       bias_mode: str = "union", colour_note: str = "") -> None:
    """Write input columns, derived columns, and bias.

    A derived name that is already a column in the file is written as
    ``<name>_computed``. Comment lines record the detection columns, the
    colours, how each derived quantity was obtained, and ``bias_mode``:
    ``union`` (bias is the sum over every block of the project) or
    ``block`` (bias is the detection's own block).
    """
    if bias_mode not in BIAS_MODES:
        raise ValueError(f"bias_mode must be one of {BIAS_MODES}")
    computed = list(detections.computed_columns)
    for name in _BIAS_COLUMNS:
        if any(name in row for row in detections) and name not in computed:
            computed.append(name)
    computed_headers = [
        f"{name}_computed" if name in detections.input_columns else name
        for name in computed
    ]
    lines = [
        f"# survey: {survey.name}",
        f"# bias_method: {survey.bias_method}",
        f"# bias_mode: {bias_mode}",
        "# input columns: " + (", ".join(detections.input_columns) or "(none)"),
        "# detection columns: " + ", ".join(
            f"{k}={v}" for k, v in detections.columns_used.items()
        ),
    ]
    if survey.detection_defaults:
        lines.append("# detection defaults: " + ", ".join(
            f"{k}={v}" for k, v in survey.detection_defaults.items()))
    lines.append(f"# model_band {survey.model_band}; colour = filter - model_band"
                 + (f" ({colour_note})" if colour_note else ""))
    lines.append("# computed columns: " + (", ".join(computed_headers) or "(none)"))
    for note in detections.notes:
        lines.append(f"# derived: {note}")
    lines.append(
        "# Hx = mag - colour - 5log10(r Δ) + 2.5log10(Bowell Φ), "
        "r = Δ = d_bary, G=-0.12"
    )
    if bias_mode == "union":
        lines.append(
            "# bias = sum of block_bias over every block of the surveys in the "
            "project (P(detect) in any block); bias_se adds the blocks' MC errors"
        )
    out_lines = ["\n".join(lines)]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([*detections.input_columns, *computed_headers])
    for row in detections:
        source = row["_source"]
        values = [source.get(col, "") for col in detections.input_columns]
        values.extend(
            _format_computed(name, _computed_value(row, name)) for name in computed
        )
        writer.writerow(values)
    out_lines.append(buffer.getvalue().rstrip("\n"))
    Path(out_path).write_text("\n".join(out_lines) + "\n")


def can_write_detections_full(detections) -> bool:
    """True when every row has the elements the CFEPS detections file stores."""
    return bool(detections) and all(
        row.get(key) is not None for row in detections for key in _CFEPS_ELEMENTS
    )


# CFEPS.detections columns, then the six ac2c72 grid-debiasing fields.
DETECTIONS_FULL_COLUMNS = (
    "cl p j k sh object mag e_mag Filt Hsur dist e_dist Nobs time av_xres av_yres "
    "max_x max_y a e_a e e_e i e_i Omega e_Omega omega e_omega tperi e_tperi "
    "RAdeg DEdeg JD rate MPC ifree Omfree omfree Hx comp bias"
)


def write_detections_full(out_path, detections: list[dict], survey: GridSurvey,
                          header_lines: str | None = None) -> None:
    """Write an OSSOS detections file plus Laplace-free elements, Hx, comp, bias.

    Columns through ``MPC`` match ``CFEPS.detections`` (including ``e_e``).
    The six extra columns are:

    - ifree: free inclination (deg) to the Laplace plane (grid cell)
    - Omfree: free node (deg); 0 when the catalog has no Ω
    - omfree: free periapse (deg); 0 when unknown
    - Hx: H_r used for the (a, q, sin i_free, H) cell
    - comp: dynamical component (cold/hot)
    - bias: Horvitz–Thompson P(detect | cell)
    """
    if header_lines is None:
        if survey.bias_method == "model_ae":
            grid_lines = (
                f"# Bias method: model_ae — cells in (r, i, H); "
                f"(a, e) from OSSOS model p(a,e|r,i)\n"
                f"# Grid size:\n"
                f"# h_step:  {H_STEP}\n"
                f"# r_step:  {R_STEP}\n"
                f"# i_step:  {I_STEP}\n"
            )
        else:
            grid_lines = (
                f"# Bias method: aq_grid — cells in (a, q, sin i_free, H)\n"
                f"# Grid size:\n"
                f"# h_step:  {H_STEP}\n"
                f"# a_step:  {A_STEP}\n"
                f"# q_step:  {Q_STEP}\n"
                f"# si_step: {SI_STEP}\n"
            )
        header_lines = (
            f"# File: {survey.detections_full_name}\n"
            f"#\n"
            f"# Grid debiasing ac2c72; {survey.name}\n"
            f"# H_{survey.model_band} from {survey.mag_column} - colour "
            f"- 5log10(r Δ) + 2.5log10(Bowell Φ), r=Δ=d_bary, G=-0.12\n"
            f"# Extra after MPC: ifree Omfree omfree (Laplace-free elements; "
            f"Omfree=omfree=0 if unknown), Hx, comp, bias\n"
            f"#\n"
            f"{grid_lines}"
            f"#\n"
        )
    lines = [header_lines, DETECTIONS_FULL_COLUMNS]
    for d in detections:
        name = str(d["name"])
        ref_jd = survey.paper_reference_jd or d.get("field_jd", 0.0)
        nobs = int(d.get("n_epochs", 1))
        filt = d.get("filter") or survey.model_band
        omfree = float(d.get("Omfree") or 0.0)
        omegafree = float(d.get("omfree") or 0.0)
        comp = d.get("comp")
        if comp is None or str(comp).strip() == "":
            comp = "-"
        # After e: e_e, i, e_i, Omega, e_Omega, omega, e_omega, tperi, e_tperi
        lines.append(
            f"cla m -1 -1 S {name:7s} {d.get('mag', d['Hx']):.2f} 0.100 {filt} "
            f"{d['Hx']:.2f} {d['d_bary']:.3f} 0.100 "
            f"{nobs} 0.0000 0.083 0.073 0.311 0.343 {d['a']:11.6f} 0.1012 {d['e']:.6f} 0.001009 "
            f"{d['i']:6.3f} 0.100 0.000 0.100 0.000 0.100 0.000 0.100 "
            f"{d.get('field_ra', 0.0):.3f} {d.get('field_dec', 0.0):.3f} {ref_jd:.5f} 0.40 {name:7s} "
            f"{d['ifree']:6.3f} {omfree:6.3f} {omegafree:6.3f} "
            f"{d['Hx']:.2f} {comp} {d['bias']:.7f}"
        )
    Path(out_path).write_text("\n".join(lines) + "\n")


def sample_aq(rng: np.random.Generator, a_bounds: tuple, q_bounds: tuple,
              max_tries: int = 10000) -> tuple[float, float]:
    """Uniform draw in the (a, q) rectangle restricted to 0 < q < a."""
    a0, a1 = a_bounds
    q0, q1 = q_bounds
    for _ in range(max_tries):
        a = float(rng.uniform(a0, a1))
        q = float(rng.uniform(q0, q1))
        if 0.0 < q < a:
            return a, q
    raise RuntimeError(
        f"empty (a,q) cell a=[{a0}, {a1}) q=[{q0}, {q1}); no bound orbit with q < a"
    )


def icrs_los_unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    """ICRS unit vector at (RA, Dec). Detos1 FoV tests are in this frame."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return np.array([
        math.cos(dec) * math.cos(ra),
        math.cos(dec) * math.sin(ra),
        math.sin(dec),
    ])


def barycentric_on_icrs_los(obs_icrf, ra_deg: float, dec_deg: float, r_au: float
                            ) -> np.ndarray | None:
    """Far |R|=r intersection of the ICRS LOS, returned in J2000 ecliptic.

    Observatory vectors from the observer CSV are converted to ICRF; the FoV (RA, Dec)
    is ICRS. Orbit elements are ecliptic, so the intersection is rotated with
    icrf_to_ecliptic (the same trick as los_circular_elements).
    """
    los = icrs_los_unit(ra_deg, dec_deg)
    obs = np.asarray(obs_icrf, dtype=float)
    b = 2.0 * float(obs @ los)
    c = float(obs @ obs) - r_au * r_au
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    t = 0.5 * (-b + root)
    if t <= 0.0:
        t = 0.5 * (-b - root)
        if t <= 0.0:
            return None
    obj_icrf = obs + t * los
    return np.array(icrf_to_ecliptic(*obj_icrf))


def sample_orbital_radius(a: float, e: float, rng: np.random.Generator) -> float:
    """Draw barycentric r on the ellipse, r ∈ [q, Q] = [a(1-e), a(1+e)]."""
    if e < 1e-12:
        return a
    q = a * (1.0 - e)
    q_ap = a * (1.0 + e)
    if q_ap <= q:
        return a
    return float(rng.uniform(q, q_ap))


def true_anomaly_from_radius(a: float, e: float, r_au: float) -> float:
    """|f| in radians from the orbit equation. Caller chooses the sign of f."""
    if e < 1e-12:
        return 0.0
    cos_f = (a * (1.0 - e * e) / r_au - 1.0) / e
    return math.acos(max(-1.0, min(1.0, cos_f)))


def mean_anomaly_from_true(e: float, f_rad: float) -> float:
    """M = E − e sin E, with E from true anomaly f (radians)."""
    if e < 1e-12:
        return f_rad
    cos_f = math.cos(f_rad)
    sin_f = math.sin(f_rad)
    den = 1.0 + e * cos_f
    cos_E = (e + cos_f) / den
    sin_E = math.sqrt(max(0.0, 1.0 - e * e)) * sin_f / den
    ecc = math.atan2(sin_E, cos_E)
    return ecc - e * math.sin(ecc)


def argument_of_latitude(R, inc_deg: float, node_deg: float) -> float:
    """u = ω + f in radians from ecliptic position and (i, Ω)."""
    inc = math.radians(inc_deg)
    node = math.radians(node_deg)
    xhat = np.array([math.cos(node), math.sin(node), 0.0])
    yhat = np.array([
        -math.cos(inc) * math.sin(node),
        math.cos(inc) * math.cos(node),
        math.sin(inc),
    ])
    R = np.asarray(R, dtype=float)
    return math.atan2(float(R @ yhat), float(R @ xhat))


def _radius_on_ellipse(a: float, e: float, r_au: float) -> float | None:
    q = a * (1.0 - e)
    q_ap = a * (1.0 + e)
    if r_au < q - 1e-8 or r_au > q_ap + 1e-8:
        return None
    return min(q_ap, max(q, r_au))


def nodes_from_inclination(R, inc_deg: float) -> list[float]:
    """Ascending nodes Ω (deg) whose plane of inclination i contains R.

    n · R = 0 with n = (sin i sin Ω, −sin i cos Ω, cos i). Empty if |β| > i.
    """
    x, y, z = (float(c) for c in np.asarray(R, dtype=float))
    r = math.hypot(math.hypot(x, y), z)
    if r < 1e-18:
        return []
    inc = math.radians(inc_deg)
    si, ci = math.sin(inc), math.cos(inc)
    if abs(si) < 1e-12:
        return [0.0] if abs(z / r) < 1e-8 else []
    # x sin Ω − y cos Ω = −z cot i
    amp_a, amp_b, target = x, -y, -z * ci / si
    amp = math.hypot(amp_a, amp_b)
    if amp < 1e-18 or abs(target) > amp + 1e-10:
        return []
    psi = math.atan2(amp_b, amp_a)
    alpha = math.asin(max(-1.0, min(1.0, target / amp)))
    out = []
    seen = set()
    for ang in (alpha - psi, math.pi - alpha - psi):
        node = math.degrees(ang) % 360.0
        key = round(node, 8)
        if key in seen:
            continue
        seen.add(key)
        out.append(node)
    return out


def peri_m_from_position(a: float, e: float, inc_deg: float, node_deg: float,
                         R, f_sign: float = 1.0) -> tuple[float, float]:
    """(ω, M) in degrees from (a, e, i, Ω) and ecliptic position R.

    r = |R| fixes |true anomaly| f; u is the argument of latitude of R;
    ω = u − f; M follows from f.
    """
    r_au = float(np.linalg.norm(R))
    f_abs = true_anomaly_from_radius(a, e, r_au)
    f_rad = f_abs if f_sign >= 0.0 else -f_abs
    u_rad = argument_of_latitude(R, inc_deg, node_deg)
    peri = math.degrees(u_rad - f_rad) % 360.0
    mean_anom = math.degrees(mean_anomaly_from_true(e, f_rad)) % 360.0
    return peri, mean_anom


def keplerian_at_radec_r(a: float, e: float, inc_deg: float,
                         ra_deg: float, dec_deg: float, r_au: float,
                         obs_icrf, f_sign: float = 1.0, node_index: int = 0
                         ) -> tuple[float, float, float, float, float, float] | None:
    """Map (a, e, i) and ICRS (RA, Dec, r) to (a, e, i, Ω, ω, M).

    This is the reusable geometric step. H is not used (photometry only).
    RA/Dec are ICRS; elements are J2000 ecliptic. The LOS is intersected
    at |R|=r in ICRF and rotated with icrf_to_ecliptic.

    Returns None if r is off the ellipse, the ray misses the sphere, or
    |β| > i so no node exists. Two Ω solutions in general; node_index
    selects one. f_sign chooses inbound vs outbound true anomaly.
    """
    r_au = _radius_on_ellipse(a, e, r_au)
    if r_au is None:
        return None
    pos_ecl = barycentric_on_icrs_los(obs_icrf, ra_deg, dec_deg, r_au)
    if pos_ecl is None:
        return None
    nodes = nodes_from_inclination(pos_ecl, inc_deg)
    if not nodes:
        return None
    node = nodes[int(node_index) % len(nodes)]
    peri, mean_anom = peri_m_from_position(
        a, e, inc_deg, node, pos_ecl, f_sign
    )
    return a, e, inc_deg, node, peri, mean_anom


def poles_through_position_at_ifree(R, ifree_deg: float, a_au: float) -> list:
    """Orbit poles P with P·R = 0 and angle(P, Laplace pole) = i_free.

    Two solutions in general (the plane can tilt either way around R). Empty
    if |β_lp| > i_free, so the line of sight cannot sit on that free
    inclination.
    """
    lp = _orbit_pole(laplace_inclination(a_au), laplace_node(a_au))
    rhat = np.asarray(R, dtype=float)
    nrm = np.linalg.norm(rhat)
    if nrm < 1e-18:
        return []
    rhat = rhat / nrm
    ref = np.array([1.0, 0.0, 0.0]) if abs(rhat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(rhat, ref)
    n1 = np.linalg.norm(e1)
    if n1 < 1e-18:
        return []
    e1 = e1 / n1
    e2 = np.cross(rhat, e1)
    amp_a = float(e1 @ lp)
    amp_b = float(e2 @ lp)
    amp = math.hypot(amp_a, amp_b)
    target = math.cos(math.radians(ifree_deg))
    if amp < 1e-18 or abs(target) > amp + 1e-10:
        return []
    phi = math.atan2(amp_b, amp_a)
    dth = math.acos(max(-1.0, min(1.0, target / amp)))
    out = []
    seen = []
    for th in (phi + dth, phi - dth):
        pole = math.cos(th) * e1 + math.sin(th) * e2
        pole = pole / np.linalg.norm(pole)
        key = tuple(round(float(c), 10) for c in pole)
        if key in seen:
            continue
        seen.append(key)
        i_ecl = math.degrees(math.acos(max(-1.0, min(1.0, float(pole[2])))))
        si = math.sin(math.radians(i_ecl))
        if si < 1e-12:
            node_ecl = 0.0
        else:
            node_ecl = math.degrees(math.atan2(pole[0] / si, -pole[1] / si)) % 360.0
        out.append((max(i_ecl, 0.05), node_ecl))
    return out


def sample_mosaic_icrs(rng: np.random.Generator,
                       ra_deg: float | None = None,
                       dec_deg: float | None = None,
                       side_deg: float | None = None,
                       width_deg: float | None = None,
                       height_deg: float | None = None,
                       survey: GridSurvey | None = None
                       ) -> tuple[float, float]:
    """Uniform (RA, Dec) in the rectangle; width is on-sky, as in getsur.f95.

    The RA half-extent is ``width / 2 / cos(dec)`` at the field centre.
    """
    if side_deg is not None:
        width_deg = side_deg if width_deg is None else width_deg
        height_deg = side_deg if height_deg is None else height_deg
    if None in (ra_deg, dec_deg, width_deg, height_deg):
        surv = _require_survey(survey, "sample_mosaic_icrs")
        ra_deg = surv.field_ra_deg if ra_deg is None else ra_deg
        dec_deg = surv.field_dec_deg if dec_deg is None else dec_deg
        width_deg = surv.mosaic_width_deg if width_deg is None else width_deg
        height_deg = surv.mosaic_height_deg if height_deg is None else height_deg
    half_ra = 0.5 * width_deg / math.cos(math.radians(dec_deg))
    return (
        float(ra_deg + rng.uniform(-half_ra, half_ra)),
        float(dec_deg + rng.uniform(-0.5 * height_deg, 0.5 * height_deg)),
    )


def aimed_elements(a: float, e: float, ifree_deg: float,
                   ra_deg: float, dec_deg: float, r_au: float,
                   obs_icrf, f_sign: float = 1.0, pole_index: int = 0
                   ) -> tuple[float, float, float, float] | None:
    """Solve (i, Ω, ω, M) for a grid-cell i_free at ICRS (RA, Dec, r).

    i_free plus the ecliptic position fixes (i, Ω) (the orbit pole through
    R at i_free from the Laplace pole). Then peri_m_from_position gives
    (ω, M). Use keplerian_at_radec_r when ecliptic i is already known.
    """
    r_au = _radius_on_ellipse(a, e, r_au)
    if r_au is None:
        return None
    pos_ecl = barycentric_on_icrs_los(obs_icrf, ra_deg, dec_deg, r_au)
    if pos_ecl is None:
        return None
    poles = poles_through_position_at_ifree(pos_ecl, ifree_deg, a)
    if not poles:
        return None
    inc, node = poles[int(pole_index) % len(poles)]
    peri, mean_anom = peri_m_from_position(
        a, e, inc, node, pos_ecl, f_sign
    )
    return inc, node, peri, mean_anom


def sample_aimed_elements(a: float, e: float, ifree_deg: float, obs_icrf,
                          rng: np.random.Generator, max_tries: int = 40,
                          survey: GridSurvey | None = None,
                          r_au: float | None = None
                          ) -> tuple[float, float, float, float] | None:
    """FoV-aimed (i, Ω, ω, M) for one (a, e, i_free) draw.

    Samples an ICRS location in the mosaic and, unless ``r_au`` is given,
    a radius on [q, Q], then inverts. Discrete branches (±f, two poles)
    are chosen uniformly. Returns None if no inversion succeeds (cheap:
    i_free below the Laplace latitude).
    """
    for _ in range(max_tries):
        ra, dec = sample_mosaic_icrs(rng, survey=survey)
        r_draw = sample_orbital_radius(a, e, rng) if r_au is None else float(r_au)
        f_sign = 1.0 if rng.random() < 0.5 else -1.0
        pole_index = int(rng.integers(0, 2))
        el = aimed_elements(
            a, e, ifree_deg, ra, dec, r_draw, obs_icrf, f_sign, pole_index
        )
        if el is not None:
            return el
    return None


def sample_aimed_elements_at_i(a: float, e: float, inc_deg: float, obs_icrf,
                               rng: np.random.Generator, r_au: float,
                               max_tries: int = 40,
                               survey: GridSurvey | None = None
                               ) -> tuple[float, float, float, float] | None:
    """FoV-aimed (i, Ω, ω, M) pinning ecliptic inclination and distance.

    Used by ``model_ae`` debiasing: (a, e) come from the orbit model, while
    discovery (r, i) are the measured cell coordinates.
    """
    for _ in range(max_tries):
        ra, dec = sample_mosaic_icrs(rng, survey=survey)
        f_sign = 1.0 if rng.random() < 0.5 else -1.0
        node_index = int(rng.integers(0, 2))
        el = keplerian_at_radec_r(
            a, e, inc_deg, ra, dec, r_au, obs_icrf, f_sign, node_index
        )
        if el is not None:
            _a, _e, inc, node, peri, mean_anom = el
            return inc, node, peri, mean_anom
    return None


def aimed_detection_bias(n_aimed: int, geom_weight_sum: float) -> float:
    """Horvitz–Thompson P(detect | cell) from FoV-aimed draws.

    n_detected / n_aimed is P(detected | FoV). Each aimed orbit carries
    geometric_detection_prob(A, i, β) so the product is P(detect | cell),
    the same quantity isotropic (Ω, ω, M) sampling estimates ~1e5× slower.
    geom_weight_sum is Σ 1_detected P_geom over aimed draws.
    """
    if n_aimed <= 0:
        return 0.0
    return float(geom_weight_sum) / float(n_aimed)


CHECK_PLOT_KEYS = ("ra", "dec", "a", "e", "i", "Omega", "omega", "M")


def empty_check_samples() -> dict:
    return {key: [] for key in CHECK_PLOT_KEYS}


def record_check_sample(store: dict, ra: float, dec: float, a: float, e: float,
                        inc: float, node: float, peri: float, M: float) -> None:
    store["ra"].append(float(ra))
    store["dec"].append(float(dec))
    store["a"].append(float(a))
    store["e"].append(float(e))
    store["i"].append(float(inc))
    store["Omega"].append(float(node) % 360.0)
    store["omega"].append(float(peri) % 360.0)
    store["M"].append(float(M) % 360.0)


def as_check_arrays(samples: dict) -> dict:
    return {key: np.asarray(samples[key], dtype=float) for key in CHECK_PLOT_KEYS}


def stack_check_samples(parts: list) -> dict:
    """Concatenate per-cell check-sample dicts for a run-level plot."""
    out = {}
    for key in CHECK_PLOT_KEYS:
        chunks = [np.asarray(part[key], dtype=float) for part in parts]
        chunks = [c for c in chunks if c.size]
        out[key] = np.concatenate(chunks) if chunks else np.array([], dtype=float)
    return out


def check_plot_tag(label) -> str:
    """Filename stem for check plots: detection name, or cell key."""
    if isinstance(label, tuple):
        return "cell_" + "_".join(f"{float(v):.4g}" for v in label).replace(".", "p")
    text = str(label).strip() or "object"
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def write_bias_check_plots(out_dir, sampled: dict, detected: dict, tag: str,
                           field_ra: float | None = None,
                           field_dec: float | None = None,
                           side_deg: float | None = None,
                           width_deg: float | None = None,
                           height_deg: float | None = None,
                           survey: GridSurvey | None = None) -> list:
    """RA/Dec and a/e/i/Ω/ω/M check plots: sampled vs detected (flag≥4).

    Sampled = aimed orbits sent through Detos1. Detected = those with
    flag≥4 (all epochs for a multi-epoch survey). `tag` is the filename
    stem. The two RA/Dec clouds should fill the mosaic the same way if
    detection is not a spatial cut inside the field; the element
    histograms should match if detection is not a function of those
    elements.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    from pathlib import Path

    if side_deg is not None:
        width_deg = side_deg if width_deg is None else width_deg
        height_deg = side_deg if height_deg is None else height_deg
    if None in (field_ra, field_dec, width_deg, height_deg):
        surv = _require_survey(survey, "write_bias_check_plots")
        field_ra = surv.field_ra_deg if field_ra is None else field_ra
        field_dec = surv.field_dec_deg if field_dec is None else field_dec
        width_deg = surv.mosaic_width_deg if width_deg is None else width_deg
        height_deg = surv.mosaic_height_deg if height_deg is None else height_deg
    detected_title = survey.check_detected_title if survey else "detected flag≥4"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampled = as_check_arrays(sampled)
    detected = as_check_arrays(detected)
    n_s = int(sampled["ra"].size)
    n_d = int(detected["ra"].size)
    width_deg = width_deg / math.cos(math.radians(field_dec))
    half_w = 0.5 * width_deg
    half_h = 0.5 * height_deg
    pad_w = 0.4 * width_deg
    pad_h = 0.4 * height_deg
    ra_lim = (field_ra - half_w - pad_w, field_ra + half_w + pad_w)
    dec_lim = (field_dec - half_h - pad_h, field_dec + half_h + pad_h)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0),
                             gridspec_kw={"height_ratios": [1.35, 1.0]})
    scat_s, scat_d = axes[0]
    hist_ra, hist_dec = axes[1]
    for ax, data, n, color, title in (
            (scat_s, sampled, n_s, "0.35", f"sampled for Detos1 (n={n_s})"),
            (scat_d, detected, n_d, "C0", f"detected flag≥4 (n={n_d})"),
    ):
        if n:
            ax.scatter(data["ra"], data["dec"], s=6, alpha=0.35, c=color,
                       linewidths=0, rasterized=True)
        ax.add_patch(Rectangle(
            (field_ra - half_w, field_dec - half_h), width_deg, height_deg,
            fill=False, edgecolor="k", lw=1.0,
        ))
        ax.plot(field_ra, field_dec, "k+", ms=9, mew=1.2)
        ax.set_xlim(*ra_lim)
        ax.set_ylim(*dec_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("RA [deg, ICRS]")
        ax.set_ylabel("Dec [deg, ICRS]")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    _overlay_hist(hist_ra, sampled["ra"], detected["ra"], "RA [deg, ICRS]", n_s, n_d)
    _overlay_hist(hist_dec, sampled["dec"], detected["dec"], "Dec [deg, ICRS]", n_s, n_d)
    fig.suptitle(f"{tag}: epoch-1 RA/Dec  ({detected_title})",
                 fontsize=11)
    fig.tight_layout()
    radec_path = out_dir / f"check_{tag}_radec.png"
    fig.savefig(radec_path, dpi=140)
    plt.close(fig)

    elem_labels = (
        ("a", "a [au]", False),
        ("e", "e", False),
        ("i", "i [deg]", False),
        ("Omega", r"$\Omega$ [deg]", True),
        ("omega", r"$\omega$ [deg]", True),
        ("M", "M [deg]", True),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    for ax, (key, xlabel, circular) in zip(axes.ravel(), elem_labels):
        _overlay_hist(ax, sampled[key], detected[key], xlabel, n_s, n_d,
                      circular=circular)
    fig.suptitle(
        f"{tag}: elements sent to Detos1 vs detected (flag≥4). "
        "Densities should match if detection is independent of these elements.",
        fontsize=10,
    )
    fig.tight_layout()
    elem_path = out_dir / f"check_{tag}_elements.png"
    fig.savefig(elem_path, dpi=140)
    plt.close(fig)
    return [radec_path, elem_path]


def _overlay_hist(ax, sampled, detected, xlabel: str, n_s: int, n_d: int,
                  circular: bool = False) -> None:
    sampled = np.asarray(sampled, dtype=float)
    detected = np.asarray(detected, dtype=float)
    if circular:
        hist_range = (0.0, 360.0)
        bins = 36
    elif sampled.size:
        lo, hi = float(np.min(sampled)), float(np.max(sampled))
        if hi <= lo:
            hi = lo + 1e-6
        pad = 0.05 * (hi - lo)
        hist_range = (lo - pad, hi + pad)
        bins = min(40, max(12, int(np.sqrt(sampled.size))))
    else:
        hist_range = None
        bins = 20
    if sampled.size:
        ax.hist(sampled, bins=bins, range=hist_range, density=True,
                histtype="stepfilled", alpha=0.35, color="0.45",
                label=f"sampled ({n_s})")
    if detected.size:
        ax.hist(detected, bins=bins, range=hist_range, density=True,
                histtype="step", color="C0", lw=1.6,
                label=f"detected flag≥4 ({n_d})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=7, frameon=False)
    if circular:
        ax.set_xlim(0.0, 360.0)

"""Unit tests for grid-bias helpers. No Fortran required."""
from __future__ import annotations

import csv
import dataclasses
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JWST_CHAR = ROOT / "tests" / "data" / "Surveys" / "JWST"

try:
    from survey_debias import grid_bias
except ImportError:  # not installed: run from a source checkout
    sys.path.insert(0, str(ROOT / "src"))
    from survey_debias import grid_bias

# Test fixture with JWST Sample A geometry; the library itself defines no
# surveys (see examples/jwst_sample_a.py).
SURVEY = grid_bias.GridSurvey(
    name="JWST Sample A",
    detection_columns={
        "mag": "m_f150w2",
        "name": "name",
        "a": "a",
        "e": "e",
        "i": "i",
        "d_bary": "d_bary",
        "comp": "comp",
    },
    detection_defaults={"survey": "JWST_A", "block": "sampleA"},
    paper_reference_jd=2459974.5,
    detections_full_name="JWST-free-cla_m.detections-full",
    bias_method="model_ae",
)
# The geometry the helpers read, as a pointings.list tile provides it.
FIELD = grid_bias.FieldView(
    name="JWST_A/sampleA",
    field_ra_deg=209.3875,
    field_dec_deg=-10.865278,
    mosaic_width_deg=math.sqrt(0.05),
    mosaic_height_deg=math.sqrt(0.05),
    epoch_jd=(2459969.32118, 2459973.96785, 2459979.90854),
    observer_csv="JWST.csv",
    paper_reference_jd=2459974.5,
)
# m_r = m_F150W2 + 1, so F150W2 - r = -1.
F150W2_COLOUR = -1.0


class GridBiasHelpers(unittest.TestCase):
    def test_ifree_zero_on_laplace_plane(self):
        a = 44.0
        ip = grid_bias.laplace_inclination(a)
        om = grid_bias.laplace_node(a)
        self.assertAlmostEqual(grid_bias.compute_ifree(ip, om, a), 0.0, places=6)

    def test_ecliptic_from_ifree_roundtrip(self):
        rng = np.random.default_rng(0)
        a = 44.0
        for ifree in (0.5, 2.0, 5.0, 15.0, 30.0):
            for _ in range(20):
                i, node = grid_bias.ecliptic_from_ifree(ifree, a, rng)
                got = grid_bias.compute_ifree(i, node, a)
                self.assertAlmostEqual(got, ifree, places=5)

    def test_omega_zero_is_not_aligned_with_laplace_node(self):
        a = 43.7
        i = 1.85
        ifree_omega0 = grid_bias.compute_ifree(i, 0.0, a)
        ifree_aligned = grid_bias.compute_ifree(i, grid_bias.laplace_node(a), a)
        self.assertGreater(ifree_omega0, 2.0)
        self.assertLess(ifree_aligned, 0.5)

    def test_hr_inverts_appmag_without_constant_offset(self):
        m_f150w2 = 26.0
        d = 44.0
        h = grid_bias.apparent_to_Hr(m_f150w2, d, colour=F150W2_COLOUR)
        m_r = m_f150w2 - F150W2_COLOUR
        opposition_approx = m_r - 10.0 * math.log10(d)
        # Bowell Φ < 1 at the implied phase, so H is brighter than 10log10(d).
        self.assertLess(h, opposition_approx)
        self.assertGreater(opposition_approx - h, 0.1)
        self.assertLess(opposition_approx - h, 0.5)
        # The old +0.35 term moved H in the opposite direction.
        old = opposition_approx + 0.35
        self.assertGreater(old - h, 0.4)

    def test_sample_aq_near_circular_cell(self):
        rng = np.random.default_rng(1)
        # JPB13-like: a=42.9, e=0, cell a in [42.8, 43.0), q in [42.8, 43.0)
        a, q = grid_bias.sample_aq(rng, (42.8, 43.0), (42.8, 43.0))
        self.assertGreater(a, q)
        self.assertGreater(q, 0.0)
        self.assertGreaterEqual(a, 42.8)
        self.assertLess(a, 43.0)

    def test_sample_aq_rejects_q_greater_than_a_everywhere(self):
        rng = np.random.default_rng(2)
        with self.assertRaises(RuntimeError):
            grid_bias.sample_aq(rng, (40.0, 40.2), (41.0, 41.2), max_tries=50)

    def test_old_098_a_rule_empties_circular_cells(self):
        a0, q0, q1 = 42.8, 42.8, 43.0
        a = a0
        high = min(q1, a * 0.98)
        self.assertLess(high, q0)

    def test_mosaic_area_is_width_times_height(self):
        self.assertAlmostEqual(FIELD.mosaic_side_deg ** 2, 0.05, places=12)
        self.assertEqual(FIELD.n_epochs, 3)

    def test_cell_key_stable(self):
        key = grid_bias.cell_key(44.25, 42.55, 0.0452, 11.37)
        self.assertEqual(key[0], 44.2)
        self.assertEqual(key[1], 42.4)
        bounds = grid_bias.bounds_from_key(key)
        self.assertAlmostEqual(bounds["a"][0], 44.2, places=6)
        self.assertAlmostEqual(bounds["a"][1], 44.4, places=6)
        self.assertAlmostEqual(bounds["Hx"][1] - bounds["Hx"][0], grid_bias.H_STEP)

    def test_figure20_is_not_the_1e5_detection_rate(self):
        lon, lat = grid_bias.icrs_to_ecliptic(
            FIELD.field_ra_deg, FIELD.field_dec_deg
        )
        self.assertAlmostEqual(lon, 211.15, places=1)
        self.assertAlmostEqual(lat, 1.07, places=2)
        p7 = grid_bias.geometric_detection_prob(
            FIELD.mosaic_area_deg2, 7.0, lat
        )
        # ~1 per 10^5 draws is the on-sky geometry of 0.05 deg², not Fig. 20.
        self.assertGreater(p7, 5e-6)
        self.assertLess(p7, 2e-5)
        p_cold = grid_bias.geometric_detection_prob(
            FIELD.mosaic_area_deg2, 2.5, lat
        )
        self.assertGreater(p_cold, p7)
        self.assertEqual(
            grid_bias.geometric_detection_prob(
                FIELD.mosaic_area_deg2, 0.5, lat
            ),
            0.0,
        )

    def test_aimed_at_field_reaches_jwst_latitude(self):
        inc, node, peri, M = grid_bias.aimed_at_field(survey=FIELD)
        lon, lat = grid_bias.icrs_to_ecliptic(
            FIELD.field_ra_deg, FIELD.field_dec_deg
        )
        self.assertAlmostEqual(inc, abs(lat), places=5)
        arglat = peri + M
        self.assertAlmostEqual(math.sin(math.radians(arglat)), 1.0 if lat >= 0 else -1.0, places=6)
        lam = (node + arglat) % 360.0
        self.assertAlmostEqual(lam, lon % 360.0, places=4)

    def test_ecliptic_icrf_roundtrip(self):
        x, y, z = -0.55, 0.83, -0.002
        xe, ye, ze = grid_bias.icrf_to_ecliptic(*grid_bias.ecliptic_to_icrf(x, y, z))
        self.assertAlmostEqual(xe, x, places=12)
        self.assertAlmostEqual(ye, y, places=12)
        self.assertAlmostEqual(ze, z, places=12)
        _xi, _yi, zi = grid_bias.ecliptic_to_icrf(x, y, z)
        # JWST near the ecliptic; ICRF z is ~sin(ε) * y ≈ 0.33 AU.
        self.assertGreater(zi, 0.3)

    def test_jwst_csv_observer_is_ecliptic_and_converts(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        obs = grid_bias.parse_jpl_horizons_icrf(path, FIELD.epoch_jd[0])
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in obs)), 1.0, places=2)
        self.assertGreater(obs[2], 0.3)

    def test_los_plant_sits_on_field_line_of_sight(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        a, e, inc, node, peri, M = grid_bias.los_circular_elements(
            FIELD.field_ra_deg, FIELD.field_dec_deg, 44.0, path,
            FIELD.epoch_jd[0],
        )
        self.assertEqual(e, 0.0)
        self.assertAlmostEqual(a, 44.0, places=5)
        # Reconstruct ecliptic position from the circular-element recipe.
        lat = math.degrees(math.asin(math.sin(math.radians(inc))
                                     * math.sin(math.radians(peri + M))))
        lon = (node + peri + M) % 360.0
        x = a * math.cos(math.radians(lon)) * math.cos(math.radians(lat))
        y = a * math.sin(math.radians(lon)) * math.cos(math.radians(lat))
        z = a * math.sin(math.radians(lat))
        obj_icrf = grid_bias.ecliptic_to_icrf(x, y, z)
        obs = grid_bias.parse_jpl_horizons_icrf(path, FIELD.epoch_jd[0])
        los = (
            obj_icrf[0] - obs[0],
            obj_icrf[1] - obs[1],
            obj_icrf[2] - obs[2],
        )
        nrm = math.sqrt(sum(c * c for c in los))
        los = tuple(c / nrm for c in los)
        ra = math.radians(FIELD.field_ra_deg)
        dec = math.radians(FIELD.field_dec_deg)
        want = (
            math.cos(dec) * math.cos(ra),
            math.cos(dec) * math.sin(ra),
            math.sin(dec),
        )
        dot = sum(u * v for u, v in zip(los, want))
        sep_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        self.assertLess(sep_deg, 0.02)

    def test_plant_apparent_radec_is_mosaic_centre(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        jd = FIELD.epoch_jd[0]
        a, e, inc, node, peri, M = grid_bias.los_circular_elements(
            FIELD.field_ra_deg, FIELD.field_dec_deg, 44.0, path, jd
        )
        obs = grid_bias.parse_jpl_horizons_icrf(path, jd)
        ra, dec = grid_bias.apparent_radec_deg(a, e, inc, node, peri, M, obs)
        sep = grid_bias.sky_separation_deg(
            ra, dec, FIELD.field_ra_deg, FIELD.field_dec_deg
        )
        self.assertLess(sep * 60.0, 0.1)
        # Mixed frames (object ICRS, observatory left ecliptic) miss by ~26',
        # larger than the mosaic half-width ~6.7'.
        obs_ecl = grid_bias.icrf_to_ecliptic(*obs)
        ra_m, dec_m = grid_bias.apparent_radec_deg(
            a, e, inc, node, peri, M, obs_ecl
        )
        sep_m = grid_bias.sky_separation_deg(
            ra_m, dec_m, FIELD.field_ra_deg, FIELD.field_dec_deg
        )
        self.assertGreater(sep_m, 0.2)
        self.assertGreater(sep_m, FIELD.mosaic_side_deg / 2.0)

    def test_rate_cut_209_was_field_ra_not_motion_pa(self):
        # Debug log: epoch1 object PA −168.7°, .eff centre 209.4°, hwidth 180°.
        obj = -168.68213509828240
        self.assertGreater(abs(209.4 - obj), 180.0)
        self.assertTrue(grid_bias.angle_in_rate_cone(obj, 209.4, 180.0))
        self.assertTrue(grid_bias.angle_in_rate_cone(obj, 0.0, 180.0))
        # After turnaround the PA is near the old centre, so the unwrapped
        # test passed epochs 2–3 by accident.
        self.assertLess(abs(209.4 - 191.3), 180.0)

    def test_paper_reference_jd_is_not_an_observation(self):
        # Eduardo et al. 2026 §V: JD 2459974.5 is the orbit-fit origin,
        # "near the midpoint of the observation period", not epoch 2.
        self.assertEqual(SURVEY.paper_reference_jd, 2459974.5)
        self.assertNotIn(SURVEY.paper_reference_jd, FIELD.epoch_jd)
        # CADC visit midpoints: ~4.6 d then ~5.9 d, not 1 d and not 5+4 at 00:00.
        d12 = FIELD.epoch_jd[1] - FIELD.epoch_jd[0]
        d23 = FIELD.epoch_jd[2] - FIELD.epoch_jd[1]
        self.assertGreater(d12, 4.0)
        self.assertLess(d12, 5.5)
        self.assertGreater(d23, 5.0)
        self.assertLess(d23, 7.0)
        self.assertAlmostEqual(FIELD.epoch_jd[0], 2459969.32118, places=4)
        self.assertAlmostEqual(FIELD.epoch_jd[2], 2459979.90854, places=4)

    def test_keplerian_plant_stays_in_mosaic_at_cadc_epochs(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        element_jd = FIELD.epoch_jd[0]
        a, e, inc, node, peri, M = grid_bias.los_circular_elements(
            FIELD.field_ra_deg, FIELD.field_dec_deg, 44.0, path, element_jd
        )
        half = FIELD.mosaic_side_deg / 2.0
        for jd in FIELD.epoch_jd:
            ra, dec, sep, rate = grid_bias.epoch_geometry(
                a, e, inc, node, peri, M, path, element_jd, jd, survey=FIELD
            )
            self.assertLess(sep, half, msg=f"JD {jd} sep={sep * 60:.2f}'")
            self.assertGreaterEqual(rate, FIELD.rate_cut_min_arcsec_hr)
            self.assertLessEqual(rate, FIELD.rate_cut_max_arcsec_hr)

    def test_true_anomaly_inverts_orbit_equation(self):
        a, e, f = 44.0, 0.08, math.radians(35.0)
        r = a * (1.0 - e * e) / (1.0 + e * math.cos(f))
        got = grid_bias.true_anomaly_from_radius(a, e, r)
        self.assertAlmostEqual(got, f, places=10)
        M = grid_bias.mean_anomaly_from_true(e, f)
        x, y, z = grid_bias.ecliptic_xyz_from_elements(a, e, 0.0, 0.0, 0.0,
                                                       math.degrees(M))
        self.assertAlmostEqual(math.sqrt(x * x + y * y + z * z), r, places=8)

    def test_keplerian_at_radec_r_is_aei_to_full_elements(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        obs = grid_bias.parse_jpl_horizons_icrf(path, FIELD.epoch_jd[0])
        a, e, inc, r = 44.0, 0.05, 3.0, 43.5
        got = grid_bias.keplerian_at_radec_r(
            a, e, inc, FIELD.field_ra_deg, FIELD.field_dec_deg,
            r, obs, f_sign=1.0, node_index=0,
        )
        self.assertIsNotNone(got)
        a2, e2, inc2, node, peri, M = got
        self.assertEqual((a2, e2, inc2), (a, e, inc))
        ra, dec = grid_bias.apparent_radec_deg(a, e, inc, node, peri, M, obs)
        sep = grid_bias.sky_separation_deg(
            ra, dec, FIELD.field_ra_deg, FIELD.field_dec_deg
        )
        self.assertLess(sep * 60.0, 0.1)
        xyz = grid_bias.ecliptic_xyz_from_elements(a, e, inc, node, peri, M)
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in xyz)), r, places=5)

    def test_keplerian_at_radec_r_matches_circular_los_plant(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        jd = FIELD.epoch_jd[0]
        a, e, inc, node0, peri0, M0 = grid_bias.los_circular_elements(
            FIELD.field_ra_deg, FIELD.field_dec_deg, 44.0, path, jd
        )
        obs = grid_bias.parse_jpl_horizons_icrf(path, jd)
        got = grid_bias.keplerian_at_radec_r(
            a, e, inc, FIELD.field_ra_deg, FIELD.field_dec_deg,
            a, obs, f_sign=1.0, node_index=0,
        )
        self.assertIsNotNone(got)
        _a, _e, _i, node, peri, M = got
        ra, dec = grid_bias.apparent_radec_deg(a, e, inc, node, peri, M, obs)
        sep = grid_bias.sky_separation_deg(
            ra, dec, FIELD.field_ra_deg, FIELD.field_dec_deg
        )
        self.assertLess(sep * 60.0, 0.1)
        dang = abs((node - node0 + 180.0) % 360.0 - 180.0)
        self.assertLess(min(dang, abs(dang - 180.0)), 1.0)

    def test_keplerian_at_radec_r_rejects_i_below_latitude(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        obs = grid_bias.parse_jpl_horizons_icrf(path, FIELD.epoch_jd[0])
        got = grid_bias.keplerian_at_radec_r(
            44.0, 0.02, 0.2, FIELD.field_ra_deg, FIELD.field_dec_deg,
            44.0, obs,
        )
        self.assertIsNone(got)

    def test_aimed_elements_land_on_icrs_los(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        jd = FIELD.epoch_jd[0]
        obs = grid_bias.parse_jpl_horizons_icrf(path, jd)
        a, e, ifree, r = 44.0, 0.05, 3.0, 43.5
        el = grid_bias.aimed_elements(
            a, e, ifree, FIELD.field_ra_deg, FIELD.field_dec_deg,
            r, obs, f_sign=1.0, pole_index=0,
        )
        self.assertIsNotNone(el)
        inc, node, peri, M = el
        ra, dec = grid_bias.apparent_radec_deg(a, e, inc, node, peri, M, obs)
        sep = grid_bias.sky_separation_deg(
            ra, dec, FIELD.field_ra_deg, FIELD.field_dec_deg
        )
        self.assertLess(sep * 60.0, 0.1)
        xyz = grid_bias.ecliptic_xyz_from_elements(a, e, inc, node, peri, M)
        r_got = math.sqrt(sum(c * c for c in xyz))
        self.assertAlmostEqual(r_got, r, places=5)
        self.assertAlmostEqual(
            grid_bias.compute_ifree(inc, node, a), ifree, places=4
        )

    def test_aimed_elements_need_icrs_to_ecliptic_rotation(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        obs = grid_bias.parse_jpl_horizons_icrf(path, FIELD.epoch_jd[0])
        r_au = 44.0
        pos_ecl = grid_bias.barycentric_on_icrs_los(
            obs, FIELD.field_ra_deg, FIELD.field_dec_deg, r_au
        )
        self.assertIsNotNone(pos_ecl)
        los = grid_bias.icrs_los_unit(
            FIELD.field_ra_deg, FIELD.field_dec_deg
        )
        b = 2.0 * float(np.dot(obs, los))
        c = float(np.dot(obs, obs)) - r_au * r_au
        t = 0.5 * (-b + math.sqrt(b * b - 4.0 * c))
        pos_icrf = np.asarray(obs) + t * los
        # Field is near the ecliptic; ICRF z of a TNO at Dec≈−11° is ~−8 AU.
        self.assertLess(abs(pos_ecl[2]), 2.0)
        self.assertLess(pos_icrf[2], -5.0)

    def test_aimed_branches_share_position_not_periapse(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        obs = grid_bias.parse_jpl_horizons_icrf(path, FIELD.epoch_jd[0])
        a, e, ifree, r = 44.0, 0.07, 4.0, 44.5
        plus = grid_bias.aimed_elements(
            a, e, ifree, FIELD.field_ra_deg, FIELD.field_dec_deg,
            r, obs, f_sign=1.0, pole_index=0,
        )
        minus = grid_bias.aimed_elements(
            a, e, ifree, FIELD.field_ra_deg, FIELD.field_dec_deg,
            r, obs, f_sign=-1.0, pole_index=0,
        )
        self.assertIsNotNone(plus)
        self.assertIsNotNone(minus)
        self.assertAlmostEqual(plus[0], minus[0], places=6)
        self.assertAlmostEqual(plus[1], minus[1], places=6)
        self.assertGreater(abs((plus[2] - minus[2] + 180.0) % 360.0 - 180.0), 1.0)
        xyz_p = grid_bias.ecliptic_xyz_from_elements(a, e, *plus)
        xyz_m = grid_bias.ecliptic_xyz_from_elements(a, e, *minus)
        for i in range(3):
            self.assertAlmostEqual(xyz_p[i], xyz_m[i], places=6)

    def test_sample_aimed_elements_stays_in_mosaic(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        obs = grid_bias.parse_jpl_horizons_icrf(path, FIELD.epoch_jd[0])
        rng = np.random.default_rng(7)
        half = FIELD.mosaic_side_deg / 2.0
        hits = 0
        for _ in range(25):
            el = grid_bias.sample_aimed_elements(43.5, 0.04, 2.5, obs, rng, survey=FIELD)
            self.assertIsNotNone(el)
            inc, node, peri, M = el
            ra, dec = grid_bias.apparent_radec_deg(
                43.5, 0.04, inc, node, peri, M, obs
            )
            # Pointings.list is an on-sky square: RA half-width is half/cos(dec).
            half_ra = half / math.cos(math.radians(FIELD.field_dec_deg))
            self.assertLessEqual(abs(ra - FIELD.field_ra_deg), half_ra + 1e-3)
            self.assertLessEqual(abs(dec - FIELD.field_dec_deg), half + 1e-3)
            hits += 1
        self.assertEqual(hits, 25)

    def test_aimed_fails_when_ifree_below_field_latitude(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        obs = grid_bias.parse_jpl_horizons_icrf(path, FIELD.epoch_jd[0])
        el = grid_bias.aimed_elements(
            44.0, 0.02, 0.05, FIELD.field_ra_deg, FIELD.field_dec_deg,
            44.0, obs,
        )
        self.assertIsNone(el)

    def test_ht_aimed_bias_multiplies_geom(self):
        # 40% of aimed plants detected, each with P_geom = 1e-5.
        n_aimed = 1000
        geom_weight = 400 * 1e-5
        bias = grid_bias.aimed_detection_bias(n_aimed, geom_weight)
        self.assertAlmostEqual(bias, 4e-6, places=12)
        self.assertEqual(grid_bias.aimed_detection_bias(0, 0.0), 0.0)

    def test_aimed_pgeom_uses_object_latitude(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        obs = grid_bias.parse_jpl_horizons_icrf(path, FIELD.epoch_jd[0])
        rng = np.random.default_rng(3)
        _, beta_field = grid_bias.icrs_to_ecliptic(
            FIELD.field_ra_deg, FIELD.field_dec_deg
        )
        n_pos = 0
        n_field_zero = 0
        for _ in range(40):
            el = grid_bias.sample_aimed_elements(44.0, 0.03, 1.0, obs, rng, survey=FIELD)
            self.assertIsNotNone(el)
            pg = grid_bias.geometric_prob_for_aimed(44.0, 0.03, *el, survey=FIELD)
            self.assertGreater(pg, 0.0)
            n_pos += 1
            if grid_bias.geometric_detection_prob(
                    FIELD.mosaic_area_deg2, el[0], beta_field) == 0.0:
                n_field_zero += 1
        self.assertEqual(n_pos, 40)
        self.assertGreater(n_field_zero, 0)

    def test_circular_aimed_matches_los_plant_sky(self):
        path = JWST_CHAR / "epoch1" / "JWST.csv"
        if not path.is_file():
            self.skipTest(f"missing {path}")
        jd = FIELD.epoch_jd[0]
        a, e, inc0, node0, peri0, M0 = grid_bias.los_circular_elements(
            FIELD.field_ra_deg, FIELD.field_dec_deg, 44.0, path, jd
        )
        ifree = grid_bias.compute_ifree(inc0, node0, a)
        obs = grid_bias.parse_jpl_horizons_icrf(path, jd)
        el = grid_bias.aimed_elements(
            a, e, ifree, FIELD.field_ra_deg, FIELD.field_dec_deg,
            a, obs, f_sign=1.0, pole_index=0,
        )
        self.assertIsNotNone(el)
        ra, dec = grid_bias.apparent_radec_deg(a, e, *el, obs)
        sep = grid_bias.sky_separation_deg(
            ra, dec, FIELD.field_ra_deg, FIELD.field_dec_deg
        )
        self.assertLess(sep * 60.0, 0.1)

    def test_write_bias_check_plots(self):
        rng = np.random.default_rng(11)
        half = FIELD.mosaic_side_deg / 2.0
        n_s, n_d = 80, 30
        sampled = grid_bias.empty_check_samples()
        detected = grid_bias.empty_check_samples()
        for i in range(n_s):
            ra = FIELD.field_ra_deg + rng.uniform(-half, half)
            dec = FIELD.field_dec_deg + rng.uniform(-half, half)
            rec = (ra, dec, 43.8 + 0.01 * i, 0.04, 2.5, 10.0 * i, 20.0 * i, 5.0 * i)
            grid_bias.record_check_sample(sampled, *rec)
            if i < n_d:
                grid_bias.record_check_sample(detected, *rec)
        with tempfile.TemporaryDirectory() as tmp:
            paths = grid_bias.write_bias_check_plots(
                tmp, sampled, detected, "unit", survey=FIELD
            )
            self.assertEqual(len(paths), 2)
            names = {p.name for p in paths}
            self.assertEqual(names, {
                "check_unit_radec.png",
                "check_unit_elements.png",
            })
            for path in paths:
                self.assertGreater(path.stat().st_size, 1000)

    def test_stack_check_samples_concatenates_cells(self):
        a = grid_bias.empty_check_samples()
        b = grid_bias.empty_check_samples()
        grid_bias.record_check_sample(a, 209.4, -10.8, 44.0, 0.05, 3.0, 10, 20, 30)
        grid_bias.record_check_sample(b, 209.5, -10.9, 44.1, 0.06, 4.0, 40, 50, 60)
        stacked = grid_bias.stack_check_samples(
            [grid_bias.as_check_arrays(a), grid_bias.as_check_arrays(b)]
        )
        self.assertEqual(stacked["ra"].size, 2)
        self.assertAlmostEqual(stacked["a"][1], 44.1)

    def test_check_plot_tag_is_per_object(self):
        self.assertEqual(grid_bias.check_plot_tag("JPB04"), "JPB04")
        self.assertEqual(grid_bias.check_plot_tag("JPB 13"), "JPB_13")
        self.assertTrue(grid_bias.check_plot_tag((44.2, 42.4, 0.045, 11.3)).startswith("cell_"))

    def test_template_pointings_are_gone(self):
        self.assertFalse(hasattr(grid_bias, "setup_pointings"))
        self.assertFalse(hasattr(grid_bias, "render_pointings_text"))

    def test_detections_full_columns_match_header_and_cfeps(self):
        cfeps = ROOT / "examples" / "Surveys" / "CFEPS" / "CFEPS.detections"
        if not cfeps.is_file():
            self.skipTest(f"missing {cfeps}")
        cfeps_header = None
        for line in cfeps.read_text().splitlines():
            if line.startswith("#") and " object " in line:
                cfeps_header = line.lstrip("# ").split()
                break
        self.assertIsNotNone(cfeps_header)
        names = grid_bias.DETECTIONS_FULL_COLUMNS.split()
        self.assertEqual(names[:len(cfeps_header)], cfeps_header)
        self.assertEqual(names[len(cfeps_header):],
                         ["ifree", "Omfree", "omfree", "Hx", "comp", "bias"])
        row = {
            "name": "JPB04", "Hx": 8.12, "d_bary": 46.2, "a": 43.7,
            "e": 0.06, "i": 1.85, "ifree": 3.2, "comp": "cold", "bias": 1.2e-5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.detections-full"
            grid_bias.write_detections_full(path, [row], SURVEY)
            data = [ln for ln in path.read_text().splitlines()
                    if ln and not ln.startswith("#")]
            self.assertEqual(data[0].split(), names)
            tokens = data[1].split()
            self.assertEqual(len(tokens), len(names), msg=tokens)
            parsed = dict(zip(names, tokens))
            self.assertEqual(parsed["e_e"], "0.001009")
            self.assertEqual(parsed["ifree"], "3.200")
            self.assertEqual(parsed["Omfree"], "0.000")
            self.assertEqual(parsed["omfree"], "0.000")
            self.assertEqual(parsed["Hx"], "8.12")
            self.assertEqual(parsed["comp"], "cold")
            self.assertEqual(parsed["bias"], "0.0000120")


class ModelAePrior(unittest.TestCase):
    def test_library_defines_no_surveys(self):
        for name in ("JWST_SAMPLE_A", "N26_HELIOSTACK", "FIELD_RA_DEG", "EPOCH_JD"):
            self.assertFalse(hasattr(grid_bias, name), name)
        self.assertEqual(grid_bias.GridSurvey.bias_method, "aq_grid")

    def test_survey_dependent_helpers_require_survey(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            grid_bias.sample_mosaic_icrs(rng)
        ra, dec = grid_bias.sample_mosaic_icrs(
            rng, ra_deg=10.0, dec_deg=0.0, side_deg=0.1
        )
        self.assertLessEqual(abs(ra - 10.0), 0.05)
        self.assertLessEqual(abs(dec), 0.05)
        # A redder filter (positive colour) means a brighter model-band H.
        self.assertAlmostEqual(
            grid_bias.apparent_to_Hr(26.0, 44.0, colour=0.5),
            grid_bias.apparent_to_Hr(26.0, 44.0) - 0.5,
        )

    def test_mosaic_ra_extent_is_width_over_cos_dec(self):
        rng = np.random.default_rng(5)
        ras = [grid_bias.sample_mosaic_icrs(rng, ra_deg=100.0, dec_deg=60.0,
                                            width_deg=1.0, height_deg=0.1)[0]
               for _ in range(4000)]
        self.assertGreater(max(ras) - min(ras), 1.9)
        self.assertLessEqual(max(ras) - min(ras), 2.0)

    def test_rih_cell_from_measured_r_and_i(self):
        key = grid_bias.rih_cell_key(46.20, 1.85, 8.12)
        self.assertEqual(key, (46.0, 1.0, 8.1))
        bounds = grid_bias.rih_bounds_from_key(key)
        self.assertEqual(bounds["r"], (46.0, 47.0))
        self.assertEqual(bounds["i"], (1.0, 2.0))

    def test_ossos_modelused_directory_mixes_populations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = (
                "# a e i Omega omega M H epoch dist comment\n"
            )
            (root / "Classical-ModelUsed.dat").write_text(
                header
                + "44.0 0.04 2.0 10 20 30 8.0 2453157.5 43.8 coldl_\n"
                + "44.2 0.05 2.2 10 20 30 8.0 2453157.5 44.1 coldl_\n"
            )
            (root / "Scattering-ModelUsed.dat").write_text(
                header
                + "50.0 0.30 12.0 10 20 30 8.0 2453157.5 40.0 scatterin\n"
            )
            cat = grid_bias.OrbitModelCatalog.from_path(root)
            self.assertEqual(len(cat), 3)
            fracs = cat.component_fractions()
            self.assertAlmostEqual(fracs["Classical"], 2.0 / 3.0)
            self.assertAlmostEqual(fracs["Scattering"], 1.0 / 3.0)
            low = cat.select(43.0, 45.0, 1.0, 3.0)
            self.assertEqual(set(low.comp), {"Classical"})
            a, e, comp = low.sample_ae(np.random.default_rng(0), r_au=44.0)
            self.assertEqual(comp, "Classical")
            self.assertLessEqual(a * (1.0 - e), 44.0 + 1e-8)
            self.assertGreaterEqual(a * (1.0 + e), 44.0 - 1e-8)

    def test_l7_file_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.txt"
            path.write_text(
                "# a e i node peri M H dist comp j k\n"
                "44.0 0.04 2.0 10 20 30 8.0 43.8 c 0 0\n"
            )
            cat = grid_bias.OrbitModelCatalog.from_path(path)
            self.assertEqual(len(cat), 1)
            self.assertEqual(cat.comp[0], "c")
            self.assertAlmostEqual(float(cat.dist[0]), 43.8)

    def test_default_model_is_under_survey_root(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(grid_bias.MODEL_PATH_ENV, None)
            self.assertEqual(
                grid_bias.default_orbit_model_path(Path("/surveys/jwst")),
                Path("/surveys/jwst") / "Models" / "OSSOS",
            )

    def test_default_model_env_override(self):
        with mock.patch.dict(os.environ, {grid_bias.MODEL_PATH_ENV: "/models/L7.txt"}):
            self.assertEqual(
                grid_bias.default_orbit_model_path(Path("/surveys/jwst")),
                Path("/models/L7.txt"),
            )

    def test_committed_header_sample_loads_each_component(self):
        sample = ROOT / "tests" / "data" / "OSSOS"
        cat = grid_bias.OrbitModelCatalog.from_path(sample)
        counts = {
            name: int((cat.comp == name).sum())
            for name in (
                "Classical", "Detached", "Inner",
                "Plutinos", "Scattering", "Twotinos",
            )
        }
        self.assertEqual(counts, {name: 3 for name in counts})
        self.assertEqual(len(cat), 18)


def _write_csv(directory: Path, text: str) -> Path:
    path = directory / "det.csv"
    path.write_text(text)
    return path


def _read_results(path: Path) -> tuple[list[str], list[dict]]:
    comments = []
    body = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            comments.append(line)
        else:
            body.append(line)
    return comments, list(csv.DictReader(body))


class DetectionReader(unittest.TestCase):
    def test_model_ae_reads_distance_magnitude_and_inclination(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), (
                "# short-arc catalog\n"
                "snr,i,d_bary,m_f150w2\n"
                "12,1.85,46.20,26.0\n"
                "9,26.7,41.00,25.5\n"
            ))
            survey = dataclasses.replace(SURVEY, detection_columns={
                "mag": "m_f150w2", "i": "i", "d_bary": "d_bary",
            })
            rows = grid_bias.load_detections(path, survey)
        self.assertEqual(rows.input_columns, ["snr", "i", "d_bary", "m_f150w2"])
        self.assertEqual(rows.columns_used["mag"], "m_f150w2")
        self.assertEqual([row["name"] for row in rows], ["1", "2"])
        self.assertNotIn("a", rows[0])
        self.assertNotIn("e", rows[0])
        self.assertNotIn("q", rows[0])
        hx = grid_bias.apparent_to_Hr(26.0, 46.20)
        self.assertAlmostEqual(rows[0]["Hx"], hx)
        self.assertEqual(rows[0]["cell"], grid_bias.rih_cell_key(46.20, 1.85, hx))
        self.assertIn("Hx", rows.computed_columns)
        self.assertIn("cell", rows.computed_columns)
        self.assertIn("name", rows.computed_columns)
        self.assertNotIn("q", rows.computed_columns)
        self.assertFalse(grid_bias.can_write_detections_full(rows))
        self.assertTrue(any("row number" in note for note in rows.notes))

    def test_blank_orbital_elements_stay_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), (
                "name,a,e,i,d_bary,m_f150w2,snr\n"
                "J1,43.7,0.06,1.85,46.2,26.0,4\n"
                "J2,,,1.40,46.2,26.0,5\n"
            ))
            survey = dataclasses.replace(SURVEY, detection_columns={
                "mag": "m_f150w2", "name": "name", "a": "a", "e": "e",
                "i": "i", "d_bary": "d_bary",
            })
            rows = grid_bias.load_detections(path, survey)
            rows[0]["bias"] = 1.5e-5
            rows[1]["bias"] = 2.5e-5
            out = Path(tmp) / "bias_results.csv"
            grid_bias.write_bias_results(out, rows, survey)
            comments, table = _read_results(out)
        self.assertAlmostEqual(rows[0]["q"], 43.7 * (1.0 - 0.06))
        self.assertNotIn("q", rows[1])
        self.assertIn("q", rows.computed_columns)
        self.assertIn("ifree", rows[0])
        self.assertNotIn("ifree", rows[1])
        self.assertTrue(any("Omega = 0" in note for note in rows.notes))
        self.assertEqual(list(table[0]), [
            "name", "a", "e", "i", "d_bary", "m_f150w2", "snr",
            "survey", "block", "filter", "colour_group", "colour",
            "q", "ifree", "sin_ifree", "Hx", "cell", "bias",
        ])
        self.assertEqual(table[0]["survey"], "JWST_A")
        self.assertEqual(table[0]["block"], "sampleA")
        self.assertTrue(any("bias_mode: union" in line for line in comments))
        self.assertEqual(table[0]["name"], "J1")
        self.assertEqual(table[0]["snr"], "4")
        self.assertEqual(table[1]["a"], "")
        self.assertEqual(table[1]["q"], "")
        self.assertEqual(table[1]["ifree"], "")
        self.assertIn("J2", table[1]["name"])
        self.assertTrue(table[0]["Hx"])
        self.assertTrue(table[1]["cell"])
        self.assertTrue(any("q = a*(1-e)" in line for line in comments))
        self.assertIn("computed columns:", " ".join(comments))

    def test_detection_columns_rename_file_headers(self):
        survey = dataclasses.replace(SURVEY, detection_columns={
            "mag": "m_f150w2", "d_bary": "dist", "i": "incl", "name": "object",
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), (
                "object,incl,dist,m_f150w2,note\n"
                "JPB04,1.85,46.20,26.0,arc\n"
            ))
            rows = grid_bias.load_detections(path, survey)
        self.assertEqual(rows[0]["name"], "JPB04")
        self.assertAlmostEqual(rows[0]["i"], 1.85)
        self.assertAlmostEqual(rows[0]["d_bary"], 46.20)
        self.assertEqual(rows.columns_used["d_bary"], "dist")
        self.assertEqual(rows.columns_used["i"], "incl")
        self.assertEqual(rows[0]["_source"]["note"], "arc")
        self.assertNotIn("a", rows[0])

    def test_full_catalog_cell_matches_elements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), (
                "name,a,e,i,d_bary,comp,m_f150w2\n"
                "JPB04,43.7,0.06,1.85,46.20,cold,26.0\n"
            ))
            rows = grid_bias.load_detections(path, SURVEY)
        a, e, i, d = 43.7, 0.06, 1.85, 46.20
        hx = grid_bias.apparent_to_Hr(26.0, d)
        self.assertEqual(rows[0]["cell"], grid_bias.rih_cell_key(d, i, hx))
        self.assertAlmostEqual(rows[0]["q"], a * (1.0 - e))
        self.assertAlmostEqual(
            rows[0]["ifree"], grid_bias.compute_ifree(i, 0.0, a),
        )
        self.assertTrue(grid_bias.can_write_detections_full(rows))

    def test_aq_grid_derives_e_from_a_and_q(self):
        survey = dataclasses.replace(
            SURVEY, bias_method="aq_grid", detection_columns={
                "mag": "mag", "name": "name", "a": "a", "q": "q",
                "i": "i", "d_bary": "d_bary",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), (
                "name,a,q,i,d_bary,mag\n"
                "N1,44.0,42.0,3.0,43.5,24.0\n"
            ))
            rows = grid_bias.load_detections(path, survey)
        self.assertAlmostEqual(rows[0]["e"], 1.0 - 42.0 / 44.0)
        self.assertIn("e", rows.computed_columns)
        self.assertNotIn("q", rows.computed_columns)
        self.assertEqual(len(rows[0]["cell"]), 4)

    def test_ifree_column_is_kept_and_not_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), (
                "name,a,e,i,ifree,d_bary,m_f150w2\n"
                "J1,43.7,0.06,1.85,3.2,46.2,26.0\n"
            ))
            survey = dataclasses.replace(SURVEY, detection_columns={
                "mag": "m_f150w2", "name": "name", "a": "a", "e": "e",
                "i": "i", "ifree": "ifree", "d_bary": "d_bary",
            })
            rows = grid_bias.load_detections(path, survey)
        self.assertAlmostEqual(rows[0]["ifree"], 3.2)
        self.assertNotIn("ifree", rows.computed_columns)
        self.assertIn("sin_ifree", rows.computed_columns)

    def test_input_hx_is_kept_beside_computed_hx(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), (
                "i,d_bary,m_f150w2,Hx\n"
                "1.85,46.2,26.0,9.99\n"
            ))
            survey = dataclasses.replace(SURVEY, detection_columns={
                "mag": "m_f150w2", "i": "i", "d_bary": "d_bary",
            })
            rows = grid_bias.load_detections(path, survey)
            rows[0]["bias"] = 0.01
            out = Path(tmp) / "bias_results.csv"
            grid_bias.write_bias_results(out, rows, survey)
            _comments, table = _read_results(out)
        self.assertEqual(table[0]["Hx"], "9.99")
        self.assertNotEqual(table[0]["Hx_computed"], "9.99")
        self.assertTrue(table[0]["Hx_computed"])

    def test_missing_model_ae_column_names_the_file_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), "name,a,e,mag\nJ1,44,0.1,26\n")
            with self.assertRaisesRegex(ValueError, "not in the file") as caught:
                grid_bias.load_detections(path, SURVEY)
        message = str(caught.exception)
        self.assertIn("d_bary", message)
        self.assertIn("m_f150w2", message)
        self.assertIn("name, a, e, mag", message)

    def test_aq_grid_requires_an_orbit(self):
        survey = dataclasses.replace(
            SURVEY, bias_method="aq_grid", detection_columns={
                "mag": "m_f150w2", "i": "i", "d_bary": "d_bary",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), "i,d_bary,m_f150w2\n1.5,44,26\n")
            with self.assertRaisesRegex(ValueError, "missing required"):
                grid_bias.load_detections(path, survey)

    def test_detection_columns_reject_unknown_and_missing_targets(self):
        with self.assertRaisesRegex(ValueError, "detection_columns"):
            dataclasses.replace(
                SURVEY, detection_columns={"mag": "m", "distance": "dist"},
            )
        with self.assertRaisesRegex(ValueError, "both"):
            dataclasses.replace(SURVEY, detection_columns={
                "mag": "m", "a": "elem", "e": "elem",
            })
        with self.assertRaisesRegex(ValueError, "mag"):
            dataclasses.replace(SURVEY, detection_columns={"i": "i", "d_bary": "d"})
        survey = dataclasses.replace(SURVEY, detection_columns={
            "mag": "m_f150w2", "i": "i", "d_bary": "dist",
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(Path(tmp), "i,d_bary,m_f150w2\n1.5,44,26\n")
            with self.assertRaisesRegex(ValueError, "dist"):
                grid_bias.load_detections(path, survey)

    def test_two_program_columns_cannot_share_a_file_column(self):
        with self.assertRaisesRegex(ValueError, "both"):
            dataclasses.replace(SURVEY, detection_columns={
                "mag": "m_f150w2", "i": "m_f150w2", "d_bary": "d_bary",
            })


if __name__ == "__main__":
    unittest.main()

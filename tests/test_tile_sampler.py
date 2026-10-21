"""FoV-aimed plant weights when a survey is found at fewer epochs than it has."""
from __future__ import annotations

import tempfile
import unittest

from survey_debias.grid_bias import geometric_prob_for_aimed, keplerian_at_radec_r
from survey_debias.pointings import Block, Tile, observer_icrf


def _TileSampler(block):
    # Imported here: test_cli checks that a dry run never loads grid_bias_run.
    from survey_debias.grid_bias_run import _TileSampler as sampler

    return sampler(block)

JD = 2459140.5
RA, DEC = 353.0, -4.5      # DEEP B1, near opposition at JD


def tile(ra=RA, dec=DEC, jd=JD, side=1.0):
    return Tile(ra_deg=ra, dec_deg=dec, width_deg=side, height_deg=side,
                area_deg2=side * side, fill=1.0, jd=jd, observer="500",
                eff_name="B.eff", shape="rect", line=1)


def block(tiles_per_epoch, required):
    tmp = tempfile.mkdtemp()
    return Block(survey="S", name="B", char_dirs=[tmp] * len(tiles_per_epoch),
                 tiles=tiles_per_epoch, detections_required=required)


def on_axis_orbit(t, inc=10.0):
    """An i = 10 deg orbit at 44 au through the field centre at t.jd (P_geom > 0)."""
    obs = observer_icrf(".", t.observer, t.jd)
    return keplerian_at_radec_r(44.0, 0.05, inc, t.ra_deg, t.dec_deg, 44.0, obs, 1.0, 0)


class AimTilesTest(unittest.TestCase):
    def test_every_epoch_required_aims_at_epoch_one(self):
        b = block([[tile()], [tile(jd=JD + 30)], [tile(jd=JD + 60)]], required=3)
        self.assertEqual([ep for ep, _t in b.aim_tiles], [0])
        self.assertAlmostEqual(b.aim_area_deg2, 1.0)

    def test_k_of_n_aims_at_every_epoch(self):
        b = block([[tile()], [tile(jd=JD + 30)], [tile(jd=JD + 60)]], required=2)
        self.assertEqual([ep for ep, _t in b.aim_tiles], [0, 1, 2])
        self.assertAlmostEqual(b.aim_area_deg2, 3.0)


class WeightTest(unittest.TestCase):
    def test_single_aim_epoch_is_the_old_weight(self):
        t = tile()
        s = _TileSampler(block([[t], [tile(jd=JD + 1)]], required=2))
        el = on_axis_orbit(t)
        self.assertAlmostEqual(s.weight(t, *el),
                               geometric_prob_for_aimed(*el, area_deg2=1.0), places=15)

    def test_a_repeated_field_does_not_change_the_weight(self):
        # The same box at the same JD as two epochs: either can produce the
        # plant, so the weight is P_geom(2A) / 2 = P_geom(A).
        t0, t1 = tile(), tile()
        s = _TileSampler(block([[t0], [t1]], required=1))
        el = on_axis_orbit(t0)
        single = geometric_prob_for_aimed(*el, area_deg2=1.0)
        self.assertAlmostEqual(s.weight(t0, *el) / single, 1.0, places=6)
        self.assertAlmostEqual(s.weight(t1, *el) / single, 1.0, places=6)

    def test_a_field_the_orbit_misses_is_not_counted(self):
        t0, t1 = tile(), tile(ra=RA + 20.0)
        s = _TileSampler(block([[t0], [t1]], required=1))
        el = on_axis_orbit(t0)
        self.assertAlmostEqual(s.weight(t0, *el),
                               geometric_prob_for_aimed(*el, area_deg2=2.0), places=15)

    def test_a_later_field_that_tracks_the_orbit_is_counted(self):
        # The orbit's position 30 days later, as a second epoch's field.
        from survey_debias.grid_bias import apparent_radec_deg, mean_motion_deg_per_day

        t0 = tile()
        a, e, inc, node, peri, M = on_axis_orbit(t0)
        jd1 = JD + 30.0
        ra1, dec1 = apparent_radec_deg(a, e, inc, node, peri,
                                       M + mean_motion_deg_per_day(a) * 30.0,
                                       observer_icrf(".", "500", jd1))
        t1 = tile(ra=ra1, dec=dec1, jd=jd1)
        s = _TileSampler(block([[t0], [t1]], required=1))
        w = s.weight(t0, a, e, inc, node, peri, M)
        p0 = geometric_prob_for_aimed(a, e, inc, node, peri, M, area_deg2=2.0)
        p1 = geometric_prob_for_aimed(a, e, inc, node, peri,
                                      M + mean_motion_deg_per_day(a) * 30.0, area_deg2=2.0)
        self.assertAlmostEqual(w, 1.0 / (1.0 / p0 + 1.0 / p1), places=15)


if __name__ == "__main__":
    unittest.main()

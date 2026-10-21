"""Tests for free inclinations (Huang, Gladman & Volk 2022)."""
from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from survey_debias import ifree  # noqa: E402


def classical_rows():
    with ifree.TABLE_FILE.open() as fh:
        return [r for r in csv.DictReader(fh) if r["RESO"] == "0"]


class Grid(unittest.TestCase):
    def test_reproduces_the_double_average_table(self):
        rows = classical_rows()
        a, e, i, node, peri, table = (np.array([float(r[k]) for r in rows])
                                      for k in ("a", "e", "inc", "Omega", "omega", "Ifree"))
        got, _ = ifree.free_inclination(a, e, i, node, peri)
        self.assertGreater(np.mean(np.abs(got - table) < 0.1), 0.97)
        self.assertGreater(np.mean((got < 4.0) == (table < 4.0)), 0.995)

    def test_omega_symmetry(self):
        w = np.array([10.0, 170.0, 190.0, 350.0, 450.0])
        np.testing.assert_allclose(ifree.wrap_omega(w), [10.0, 10.0, 10.0, 10.0, 90.0])
        q1 = ifree.forcing_pole(44.0, 0.05, 2.0, 30.0)
        q2 = ifree.forcing_pole(44.0, 0.05, 2.0, 150.0)
        self.assertEqual(q1, q2)

    def test_pole_on_forcing_pole_is_free_of_inclination(self):
        qf, pf = ifree.forcing_pole(44.0, 0.05, 2.0, 30.0)
        inc, node = float(np.hypot(qf, pf)), float(np.degrees(np.arctan2(pf, qf)))
        self.assertAlmostEqual(float(ifree.free_inclination(44.0, 0.05, inc, node, 30.0)[0]), 0.0,
                               places=6)

    def test_forcing_pole_vector_matches_q_p(self):
        v = ifree.forcing_pole_vector(44.0, 0.05, 2.0, 30.0)
        qf, pf = ifree.forcing_pole(44.0, 0.05, 2.0, 30.0)
        self.assertAlmostEqual(float(np.degrees(np.arccos(v[2]))), float(np.hypot(qf, pf)), places=6)

    def test_in_grid(self):
        self.assertTrue(ifree.in_grid(44.0, 0.1, 3.0))
        self.assertFalse(ifree.in_grid(38.0, 0.1, 3.0))
        self.assertFalse(ifree.in_grid(44.0, 0.3, 3.0))


class Frames(unittest.TestCase):
    def test_state_round_trip(self):
        el = (44.0, 0.08, 3.0, 120.0, 250.0, 33.0)
        pos, vel = ifree._state(*el, ifree.K2)
        back = ifree._elements(pos, vel, ifree.K2)
        np.testing.assert_allclose(back, el, atol=1e-8)

    def test_barycentric_shift_is_small(self):
        a, e, i, node, peri, M = ifree.barycentric_elements(44.0, 0.08, 3.0, 120.0, 250.0, 33.0,
                                                            2461200.5)
        self.assertLess(abs(i - 3.0), 0.2)
        self.assertLess(abs(a - 44.0), 0.5)


class Table(unittest.TestCase):
    def test_lookup_by_number_provisional_and_ossos(self):
        self.assertAlmostEqual(ifree.huang_lookup("15760")["Ifree"], 2.7091)
        self.assertEqual(ifree.huang_lookup("1994ES2")["Name"], "(1994 ES2)")
        self.assertIsNotNone(ifree.huang_lookup("nothing", "15760"))
        self.assertIsNone(ifree.huang_lookup("nothing"))


if __name__ == "__main__":
    unittest.main()

"""Tests for synthetic catalogues."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from survey_debias import hfit, joint, synthetic  # noqa: E402
from survey_debias.survey_efficiency import ModelPopulation  # noqa: E402

import test_joint  # noqa: E402
from project_fixture import make_jwst_project, short_tempdir  # noqa: E402

try:
    import ossssim  # noqa: F401
    HAVE_OSSSSIM = True
except ImportError:
    HAVE_OSSSSIM = False


class Draws(unittest.TestCase):
    def test_h_follows_the_cumulative(self):
        theta = (0.5, 0.3, 8.0, 2.0)
        h, n_range = synthetic.draw_h(np.random.default_rng(1), 200000, theta, (4.0, 10.0))
        cum = hfit.cumulative(np.array([4.0, 7.0, 10.0]), *theta)
        self.assertAlmostEqual(n_range, cum[2] - cum[0], delta=1e-6 * n_range)
        frac = (cum[1] - cum[0]) / (cum[2] - cum[0])
        self.assertAlmostEqual(np.mean(h < 7.0), frac, delta=0.005)

    def test_expected_count_matches_the_joint_likelihood(self):
        terms, _ = test_joint.two_surveys()
        like = joint.JointLikelihood(terms, sigma_c=0.2)
        theta = (0.5, 0.3, 9.0, 1.0)
        for t in terms:
            ref = like.expected(t, theta[:3], theta[3], {"H": 0.0})[0]
            got = synthetic.expected_count(t.eps, theta, t.colour, 0.2, t.filter == "r")
            self.assertAlmostEqual(got, ref, delta=1e-9 * ref)
        h_o = synthetic.h_o_for_count(terms[0].eps, theta[:3], 50.0, 0.0, 0.2, True)
        self.assertAlmostEqual(
            synthetic.expected_count(terms[0].eps, (*theta[:3], h_o), 0.0, 0.2, True), 50.0,
            delta=1e-6)


@unittest.skipUnless(HAVE_OSSSSIM, "needs ossssim")
class Inject(unittest.TestCase):
    def test_catalogue_rows(self):
        model = ModelPopulation.from_file(ROOT / "tests" / "data" / "OSSOS" / "Classical-ModelUsed.dat")
        with short_tempdir() as tmp:
            make_jwst_project(Path(tmp))
            rows = synthetic.inject(Path(tmp) / "characterization", "jwst_sample_a", model,
                                    (0.6, 1.0, 0.0, -10.0), -1.0, 0.2, (5.0, 9.0), 200, seed=2,
                                    colour_file=Path(tmp) / "colour.toml", form="exponential")
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["block"], "jwst_sample_a")
            self.assertAlmostEqual(r["mag"] - (r["H_r"] + r["colour"]), 16.6, delta=1.5)


if __name__ == "__main__":
    unittest.main()

"""Tests for complete surveys (format = "complete", e.g. the MPC catalogue below an H limit)."""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from survey_debias import hfit, joint, joint_cli  # noqa: E402
from survey_debias.survey_efficiency import (  # noqa: E402
    H_GRID, CompleteSpec, EfficiencyTable, complete_efficiency, complete_overlap,
    step_weights)

import test_joint  # noqa: E402

ALPHA, H_O = 0.8, 2.0
F_CUT = 0.75


def h_grid():
    return np.arange(H_GRID[0], H_GRID[1] + 0.5 * H_GRID[2], H_GRID[2])


def n_below(h_lim, alpha=ALPHA, h_o=H_O):
    return 10.0 ** (alpha * (h_lim - h_o))


def complete_survey(h_lim, rng, filt="r", colour=0.0):
    """A complete catalogue: every object below h_lim, kept with probability F_CUT."""
    spec = CompleteSpec("MPC", filt, h_lim)
    eps = complete_efficiency(spec, F_CUT, np.full(8, F_CUT))
    n = rng.poisson(F_CUT * n_below(h_lim))
    # Inverse CDF of 10^(alpha (H - h_o)) below h_lim.
    h = h_lim + np.log10(rng.random(n)) / ALPHA
    like = joint.CompleteSample(np.array([f"c{k}" for k in range(n)]), h)
    return joint.SurveyTerm("MPC", filt, like, eps, colour, kind="complete", h_lim=h_lim)


class StepTable(unittest.TestCase):
    def test_step_integrates_exactly_to_the_limit(self):
        h = h_grid()
        for h_lim in (5.5, 5.513, 5.25):
            total = float(np.sum(step_weights(h, h_lim) * np.gradient(h)))
            self.assertAlmostEqual(total, h_lim - (h[0] - 0.5 * H_GRID[2]), places=9)

    def test_efficiency_round_trip(self):
        spec = CompleteSpec("MPC", "M", 5.5, 0.15)
        table = complete_efficiency(spec, F_CUT, np.full(4, F_CUT), {"cut": "i < 5 deg"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eps_cold.csv"
            table.save(path)
            back = EfficiencyTable.load(path)
        self.assertEqual(back.filter, "M")
        self.assertAlmostEqual(float(back.meta["f_cut"]), F_CUT)
        self.assertEqual(back.meta["complete"], "M < 5.5")
        self.assertEqual(back.meta["excluded_blocks"], "none")
        self.assertAlmostEqual(float(back.eps[back.h < 5.4].min()), F_CUT, places=5)
        self.assertEqual(float(back.eps[back.h > 5.52].max()), 0.0)

    def test_overlap_is_the_other_survey_below_the_limit(self):
        h = h_grid()
        other = EfficiencyTable("A", "g", h, np.full(len(h), 0.01), np.full((3, len(h)), 0.01),
                                {"cut": "i < 5 deg"})
        spec = CompleteSpec("MPC", "M", 5.5)
        sharp = complete_overlap(spec, other, 0.2, 0.8, 0.0)
        self.assertEqual(sharp.filter, "model")
        np.testing.assert_allclose(sharp.eps, 0.01 * step_weights(h, 5.3))
        smooth = complete_overlap(spec, other, 0.2, 0.8, 0.25)
        k = int(np.argmin(np.abs(h - 5.3)))
        self.assertAlmostEqual(float(smooth.eps[k]), 0.005, places=4)
        self.assertEqual(smooth.boot.shape, (3, len(h)))


class CompleteLikelihood(unittest.TestCase):
    def test_expected_count_and_conditional_are_analytic(self):
        term = complete_survey(5.0, np.random.default_rng(1))
        like = joint.JointLikelihood([term], sigma_c=0.0, form="exponential")
        shape = (ALPHA, 1.0, 0.0)
        lam = like.expected(term, shape, H_O, {})[0]
        self.assertAlmostEqual(lam, F_CUT * n_below(5.0), delta=2e-3 * lam)
        got = like.conditional(term, shape, H_O, 0.0)
        log_dn = hfit.log_differential_exponential(term.like.h, ALPHA, H_O)
        ref = float(np.sum(log_dn) - term.n * math.log(n_below(5.0)))
        self.assertAlmostEqual(got, ref, delta=2e-3 * abs(ref))

    def test_scatter_widens_the_lf_in_the_survey_filter(self):
        term = complete_survey(5.0, np.random.default_rng(1), filt="M", colour=0.2)
        term.sigma_h = 0.15
        like = joint.JointLikelihood([term], sigma_c=0.2, form="exponential")
        h = np.array([4.0, 5.0])
        wide = like.log_lambda_term(term, h, (ALPHA, 1.0, 0.0), H_O, 0.2)
        ref = like.log_lambda_f(h, (ALPHA, 1.0, 0.0), H_O, 0.2, "M", 0.15)
        np.testing.assert_allclose(wide, ref)
        narrow = like.log_lambda_f(h, (ALPHA, 1.0, 0.0), H_O, 0.2, "M")
        # Exponential LF: a Gaussian spread multiplies lambda by exp((alpha ln10 sigma)^2 / 2).
        boost = 0.5 * (ALPHA * math.log(10.0)) ** 2 * (0.25 ** 2 - 0.2 ** 2)
        np.testing.assert_allclose(wide - narrow, boost, rtol=1e-3)

    def test_fit_recovers_the_bright_end(self):
        term = complete_survey(6.0, np.random.default_rng(7))
        self.assertGreater(term.n, 200)
        like = joint.JointLikelihood([term], sigma_c=0.0, form="exponential")
        fit = joint.fit_joint(like)
        self.assertAlmostEqual(fit.theta[0], ALPHA, delta=0.15)
        n_fit = n_below(6.0, fit.theta[0], fit.theta[3])
        self.assertAlmostEqual(n_fit, term.n / F_CUT, delta=3 * math.sqrt(term.n) / F_CUT)


class HWindow(unittest.TestCase):
    def test_unit_window_changes_nothing(self):
        terms, _ = test_joint.two_surveys()
        cond = terms[0].like
        before = cond.loglike_exponential(0.6)
        cond.apply_h_window(lambda h: np.ones_like(np.asarray(h, dtype=float)))
        self.assertAlmostEqual(cond.loglike_exponential(0.6), before, places=10)

    def test_window_removes_the_bright_end(self):
        terms, _ = test_joint.two_surveys()
        cond = terms[0].like
        cut = float(np.median(cond.sample.h))
        den = cond.den.copy()
        cond.apply_h_window(lambda h: (np.asarray(h) > cut).astype(float))
        self.assertTrue(np.all(cond.den[:, cond.grid <= cut] == 0.0))
        np.testing.assert_allclose(cond.den[:, cond.grid > cut], den[:, cond.grid > cut])


class BuildTerms(unittest.TestCase):
    def test_complete_survey_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "colour.toml").write_text(
                '[default]\n"g-g" = 0.0\n"r-g" = -0.7\n"M-g" = -0.5\n'
                '[cold]\n"g-g" = 0.0\n"r-g" = -0.8\n"M-g" = -0.6\n')
            mpc = root / "characterization" / "MPC"
            mpc.mkdir(parents=True)
            (mpc / "MPC.toml").write_text(
                '[survey]\nsurveys = ["MPC"]\nformat = "complete"\nfilter = "M"\n'
                'h_lim = 5.5\nsigma_h = 0.15\n'
                'detections_relpath = "characterization/MPC/MPC.csv"\n'
                '[detection_columns]\nname = "desig"\nh = "h"\ni = "i"\n')
            (mpc / "MPC.csv").write_text(
                "# test\ndesig,h,i,comp\nA,4.6,3.9,cold\nB,5.2,1.7,cold\nC,5.6,2.0,cold\n"
                "D,5.0,9.0,\n")
            spec = CompleteSpec.load(mpc / "MPC.toml")
            complete_efficiency(spec, F_CUT, np.full(4, F_CUT)).save(mpc / "eps_cold.csv")
            cfg = {"joint": {"root": ".", "colour_group": "cold", "model_band": "r",
                             "priority": ["MPC"]},
                   "survey": [{"name": "MPC", "format": "complete",
                               "spec": "characterization/MPC/MPC.toml",
                               "select": {"comp": "cold"}}]}
            like, info = joint_cli.build_terms(cfg, root)
            term = like.surveys[0]
            self.assertEqual(term.kind, "complete")
            self.assertEqual(sorted(term.like.names), ["A", "B"])
            self.assertAlmostEqual(term.colour, 0.2)
            self.assertEqual(like.names[-1], "delta_M")
            self.assertEqual(info["surveys"]["MPC"]["h_lim"], 5.5)
            cfg["survey"][0]["h_lim"] = 5.0
            like, _ = joint_cli.build_terms(cfg, root)
            self.assertEqual(list(like.surveys[0].like.names), ["A"])
            self.assertEqual(float(like.surveys[0].eps.eps[like.surveys[0].eps.h > 5.02].max()), 0.0)
            self.assertTrue(np.isfinite(like.loglike([0.8, 1.0, 0.0, 2.0, 0.0])))


if __name__ == "__main__":
    unittest.main()

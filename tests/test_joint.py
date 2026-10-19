"""Tests for the joint (conditional + Poisson) H-distribution fit."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from survey_debias import hfit, joint  # noqa: E402
from survey_debias.grid_bias import apparent_to_Hr  # noqa: E402
from survey_debias.survey_efficiency import EfficiencyTable  # noqa: E402

TRUE_ALPHA, TRUE_HO = 0.6, 0.0
D_RANGE = (40.0, 46.0)
H_GRID = np.arange(0.0, 16.0, 0.02)


def selection(m50, lim):
    return hfit.SurveySelection([hfit.parse_efficiency_spec(f"single:0.95,{m50},0.2,{lim}")])


def mu_of(d):
    return 25.0 - apparent_to_Hr(25.0, d, colour=0.0)


def make_survey(name, filt, colour, sel, frac, rng, n_mu=400):
    """A survey whose detections are drawn from lambda_f * eps (sigma_c = 0)."""
    mus = np.array([mu_of(d) for d in np.linspace(*D_RANGE, n_mu)])
    s_of_h = np.array([np.mean(sel(h + mus, 3.0)) for h in H_GRID])
    eps = EfficiencyTable(name, filt, H_GRID, frac * s_of_h, np.zeros((0, len(H_GRID))), {})
    lam_f = np.exp(hfit.log_differential_exponential(H_GRID - colour, TRUE_ALPHA, TRUE_HO))
    expected = float(np.sum(lam_f * eps.eps * 0.02))
    n = rng.poisson(expected)
    h_out, d_out = [], []
    cdf = np.cumsum(lam_f * (s_of_h > 0)) / np.sum(lam_f * (s_of_h > 0))
    while len(h_out) < n:
        h = float(np.interp(rng.random(), cdf, H_GRID))
        d = float(rng.uniform(*D_RANGE))
        if rng.random() < sel(np.array([h + mu_of(d)]), 3.0)[0]:
            h_out.append(h)
            d_out.append(d)
    h, d = np.array(h_out), np.array(d_out)
    m = h + np.array([mu_of(x) for x in d])
    sample = hfit.HSample(np.array([f"{name}{k}" for k in range(n)]), h, m, np.ones(n),
                          hfit.opposition_rate(d), "H", name,
                          group=np.array([f"{name}/b"] * n, dtype=object), d=d)
    like = hfit.ConditionalLikelihood(sample, {f"{name}/b": sel}, float(h.min()) - 1.0,
                                      measurement="none")
    return joint.SurveyTerm(name, filt, like, eps, colour), expected


def two_surveys(seed=3):
    rng = np.random.default_rng(seed)
    a, la = make_survey("A", "r", 0.0, selection(24.0, 24.6), 1e-2, rng)
    b, lb = make_survey("B", "H", -1.0, selection(28.5, 29.2), 2e-6, rng)
    return [a, b], (la, lb)


class Lambda(unittest.TestCase):
    def test_colour_spread_is_a_convolution(self):
        terms, _ = two_surveys()
        like = joint.JointLikelihood(terms, sigma_c=0.2, form="exponential")
        h = np.array([10.0, 11.0])
        got = np.exp(like.log_lambda_f(h, (0.6, 1.0, 0.0), 0.0, -1.0, "H"))
        c = np.linspace(-1.0 - 1.6, -1.0 + 1.6, 4001)
        w = np.exp(-0.5 * ((c + 1.0) / 0.2) ** 2)
        w /= w.sum()
        ref = [np.sum(w * np.exp(hfit.log_differential_exponential(x - c, 0.6, 0.0))) for x in h]
        np.testing.assert_allclose(got, ref, rtol=1e-4)
        # The model band has neither colour nor spread.
        np.testing.assert_allclose(like.log_lambda_f(h, (0.6, 1.0, 0.0), 0.0, -1.0, "r"),
                                   hfit.log_differential_exponential(h, 0.6, 0.0))

    def test_expected_counts_and_h_o(self):
        terms, truth = two_surveys()
        like = joint.JointLikelihood(terms, sigma_c=0.0, form="exponential")
        shape = (TRUE_ALPHA, 1.0, 0.0)
        for term, lam in zip(terms, truth):
            self.assertAlmostEqual(like.expected(term, shape, TRUE_HO, {"H": 0.0})[0], lam,
                                   delta=1e-6 * lam)
        h_o = like.best_h_o(shape, {"H": 0.0})
        n = sum(t.n for t in terms)
        k = sum(like.expected(t, shape, h_o, {"H": 0.0})[0] for t in terms)
        self.assertAlmostEqual(k, n, delta=1e-6 * n)

    def test_overlap_reduces_the_expected_count(self):
        terms, _ = two_surveys()
        b = terms[1]
        table = EfficiencyTable("AxB", "model", H_GRID, 0.1 * b.eps.eps, np.zeros((0, len(H_GRID))), {})
        like = joint.JointLikelihood(terms, sigma_c=0.0, form="exponential")
        before = like.expected(b, (0.6, 1.0, 0.0), 0.0, {"H": 0.0})[0]
        b.overlaps.append(("A", table))
        after = like.expected(b, (0.6, 1.0, 0.0), 0.0, {"H": 0.0})[0]
        self.assertLess(after, before)


class Fit(unittest.TestCase):
    def test_recovers_slope_and_normalisation(self):
        terms, _ = two_surveys(seed=5)
        like = joint.JointLikelihood(terms, sigma_c=0.0, sigma_delta=0.05, form="exponential")
        fit = joint.fit_joint(like)
        alpha, h_o, delta = fit.theta[0], fit.theta[3], fit.theta[4]
        self.assertAlmostEqual(alpha, TRUE_ALPHA, delta=0.05)
        self.assertAlmostEqual(h_o, TRUE_HO, delta=0.15)
        self.assertLess(abs(delta), 0.15)
        self.assertTrue(math.isfinite(fit.loglike))
        self.assertGreaterEqual(fit.loglike, like.loglike([TRUE_ALPHA, 1.0, 0.0, TRUE_HO, 0.0]) - 1e-6)


if __name__ == "__main__":
    unittest.main()

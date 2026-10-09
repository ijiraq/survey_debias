"""Tests for the tapered H-distribution fit."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from survey_debias import hfit  # noqa: E402

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    import corner  # noqa: F401
    import emcee  # noqa: F401
    HAVE_FIT_EXTRAS = HAVE_SCIPY
except ImportError:
    HAVE_FIT_EXTRAS = False

EFF_TEXT = """\
# comment
track_frac= 1.0 29.92 -0.61
rates=  0.00 20.00
function= single
single_param=    0.95      25.00       0.30
mag_lim= 26.50
"""

TRUE = (0.6, 0.4, 7.0)
DIST_MOD = (14.0, 14.6)   # m - H range of the synthetic geometries
H_LO, H_HI = 4.0, 12.5


def synthetic(n_detect: int, seed: int):
    """Detections of a tapered population through EFF_TEXT's selection."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.eff"
        path.write_text(EFF_TEXT)
        selection = hfit.SurveySelection([hfit.read_efficiency_file(path)])
    grid = np.linspace(H_LO, H_HI, 20001)
    cum = hfit.cumulative(grid, *TRUE, 0.0)
    cum = (cum - cum[0]) / (cum[-1] - cum[0])
    h_all, m_all, n_all = [], [], 0
    while len(h_all) < n_detect:
        h = np.interp(rng.random(20000), cum, grid)
        m = h + rng.uniform(*DIST_MOD, size=h.size)
        seen = rng.random(h.size) < selection(m, 1.0)
        n_all += h.size
        h_all.extend(h[seen])
        m_all.extend(m[seen])
    h = np.array(h_all[:n_detect])
    m = np.array(m_all[:n_detect])
    bias = selection(m, 1.0) * 1e-3
    sample = hfit.HSample(
        np.array([str(i) for i in range(n_detect)]), h, m, bias,
        np.full(n_detect, np.nan), "Hx", "synthetic",
    )
    return sample, selection


class FunctionalForm(unittest.TestCase):
    def test_differential_is_derivative_of_cumulative(self):
        h = np.linspace(3.0, 11.0, 2001)
        args = (0.4, 0.25, 8.1, -2.6)
        numeric = np.gradient(hfit.cumulative(h, *args), h)
        analytic = np.exp(hfit.log_differential(h, *args))
        np.testing.assert_allclose(numeric[5:-5], analytic[5:-5], rtol=1e-3)

    def test_h_o_reproduces_count(self):
        theta = (0.5, 0.3, 7.5)
        h_o = hfit.h_o_for(theta, 5.0, 9.0, 1.0e5)
        n = (hfit.cumulative(9.0, *theta, h_o) - hfit.cumulative(5.0, *theta, h_o))
        self.assertAlmostEqual(float(n) / 1.0e5, 1.0, places=10)

    def test_h_o_vectorised(self):
        thetas = np.array([[0.5, 0.3, 7.5], [0.7, 0.5, 6.0]])
        h_o = hfit.h_o_for(thetas.T, 5.0, 9.0, np.array([1e5, 2e5]))
        for t, ho, c in zip(thetas, h_o, (1e5, 2e5)):
            n = hfit.cumulative(9.0, *t, ho) - hfit.cumulative(5.0, *t, ho)
            self.assertAlmostEqual(float(n) / c, 1.0, places=8)


class Efficiency(unittest.TestCase):
    def test_reads_single_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.eff"
            path.write_text(EFF_TEXT)
            epoch = hfit.read_efficiency_file(path)
        self.assertEqual(len(epoch.blocks), 1)
        block = epoch.blocks[0]
        self.assertEqual(block.function, "single")
        self.assertEqual(block.params, [0.95, 25.0, 0.3])
        self.assertEqual(block.mag_lim, 26.5)
        self.assertEqual(epoch.track, (1.0, 29.92, -0.61))
        self.assertFalse(epoch.rate_dependent)

    def test_selection_floor_and_mag_lim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.eff"
            path.write_text(EFF_TEXT)
            epoch = hfit.read_efficiency_file(path)
        s = epoch(np.array([20.0, 25.0, 26.4, 26.6]), 1.0)
        self.assertAlmostEqual(s[0], 0.95, places=6)
        self.assertAlmostEqual(s[1], 0.475, places=6)
        self.assertEqual(s[2], 0.0)   # eta < 0.01
        self.assertEqual(s[3], 0.0)   # fainter than mag_lim

    def test_rate_blocks_and_lookup(self):
        text = (
            "rates= 0 2\nfunction= lookup\n"
            "lookup_param= 24.0 0.9\nlookup_param= 25.0 0.5\nlookup_param= 26.0 0.1\n"
            "rates= 2 10\nfunction= double\ndouble_param= 0.8 25.0 0.3 0.2\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.eff"
            path.write_text(text)
            epoch = hfit.read_efficiency_file(path)
        self.assertTrue(epoch.rate_dependent)
        s = epoch(np.array([23.0, 24.5, 26.5, 24.0, 25.0]),
                  np.array([1.0, 1.0, 1.0, 5.0, 5.0]))
        np.testing.assert_allclose(s[:3], [0.9, 0.7, 0.0])
        expect = 0.2 * (1 - np.tanh(-1 / 0.3)) * (1 - np.tanh(-1 / 0.2))
        self.assertAlmostEqual(s[3], expect, places=6)
        # No mag_lim: OSSSSim counts a detection as characterised only if eta >= 0.4.
        self.assertEqual(s[4], 0.0)

    def test_spec_string(self):
        epoch = hfit.parse_efficiency_spec("single:0.9,25,0.3,26")
        self.assertEqual(epoch.blocks[0].mag_lim, 26.0)
        with self.assertRaises(ValueError):
            hfit.parse_efficiency_spec("single:0.9,25")

    def test_multi_epoch_product(self):
        e = hfit.parse_efficiency_spec("single:0.9,25,0.3")
        both = hfit.SurveySelection([e, e])
        m = np.array([24.0, 25.0])
        np.testing.assert_allclose(both(m, 1.0), e(m, 1.0) ** 2)


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class Recovery(unittest.TestCase):
    def test_mle_recovers_known_parameters(self):
        sample, selection = synthetic(3000, seed=1)
        like = hfit.ConditionalLikelihood(sample, selection, h_min=H_LO, measurement="none")
        priors = hfit.default_priors(sample.h, H_LO)
        mle = hfit.fit_mle(like, priors)
        truth = like.loglike(*TRUE)
        self.assertGreaterEqual(mle.loglike, truth - 1e-6)
        # Truth inside the 99.9% likelihood-ratio region (chi2, 3 dof).
        self.assertLess(2.0 * (mle.loglike - truth), 16.27)
        self.assertAlmostEqual(mle.theta[0], TRUE[0], delta=0.08)
        # The exponential is the tapered form's beta -> 0 limit.
        self.assertGreaterEqual(mle.loglike, mle.loglike_exp - 1e-6)

    def test_pit_is_uniform_at_truth(self):
        sample, selection = synthetic(1500, seed=2)
        like = hfit.ConditionalLikelihood(sample, selection, h_min=H_LO, measurement="none")
        _d, p = hfit.ks_uniform(like.pit(*TRUE))
        self.assertGreater(p, 0.01)

    def test_bias_does_not_change_shape_likelihood(self):
        sample, selection = synthetic(300, seed=3)
        like = hfit.ConditionalLikelihood(sample, selection, h_min=H_LO, measurement="none")
        scaled = hfit.HSample(sample.names, sample.h, sample.m,
                              sample.bias * 7.0, sample.rate, "Hx", "x")
        like2 = hfit.ConditionalLikelihood(scaled, selection, h_min=H_LO, measurement="none")
        self.assertAlmostEqual(like.loglike(*TRUE), like2.loglike(*TRUE))

    def test_zero_selection_at_detection_is_an_error(self):
        sample, selection = synthetic(50, seed=4)
        sample.m[0] = 28.0
        with self.assertRaisesRegex(ValueError, "selection is zero"):
            hfit.ConditionalLikelihood(sample, selection, h_min=H_LO, measurement="none")


SURVEY_TOML = """\
[survey]
name = "Synthetic"
detections_relpath = "objects.csv"
bias_method = "model_ae"

[detection_columns]
d_bary = "dist"
i = "incl"
mag = "mag"
name = "name"

[detection_defaults]
survey = "T"
block = "t"
"""


def write_characterization(root: Path, blocks=(("t", EFF_TEXT),), survey="T") -> None:
    d = root / "characterization" / survey
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, (name, text) in enumerate(blocks):
        lines.append(f"0.2 0.2 {209.0 + k} -10.0 2459969.3 1.0 500 {name}.eff\n")
        (d / f"{name}.eff").write_text(text + "filter= r\n")
    (d / "pointings.list").write_text("".join(lines))


def grouped_sample(seed: int, n_each: int = 1200):
    """Two blocks with different depths, as one HSample with groups."""
    deep = EFF_TEXT
    shallow = EFF_TEXT.replace("25.00", "24.00").replace("26.50", "25.50")
    parts, sels = [], {}
    for k, (name, text) in enumerate((("T/deep", deep), ("T/shallow", shallow))):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.eff"
            path.write_text(text)
            sel = hfit.SurveySelection([hfit.read_efficiency_file(path)])
        sels[name] = sel
        rng = np.random.default_rng(seed + k)
        grid = np.linspace(H_LO, H_HI, 20001)
        cum = hfit.cumulative(grid, *TRUE, 0.0)
        cum = (cum - cum[0]) / (cum[-1] - cum[0])
        h_all, m_all = [], []
        while len(h_all) < n_each:
            h = np.interp(rng.random(20000), cum, grid)
            m = h + rng.uniform(*DIST_MOD, size=h.size)
            seen = rng.random(h.size) < sel(m, 1.0)
            h_all.extend(h[seen])
            m_all.extend(m[seen])
        parts.append((name, np.array(h_all[:n_each]), np.array(m_all[:n_each])))
    names = np.concatenate([[f"{g}{i}" for i in range(len(h))] for g, h, _ in parts])
    h = np.concatenate([p[1] for p in parts])
    m = np.concatenate([p[2] for p in parts])
    group = np.array([p[0] for p in parts for _ in range(len(p[1]))], dtype=object)
    bias = np.concatenate([sels[g](mm, 1.0) for g, _, mm in parts]) * 1e-3
    sample = hfit.HSample(names, h, m, bias, np.full(len(h), np.nan), "Hx", "synthetic",
                          group=group)
    return sample, sels


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class GroupedSelection(unittest.TestCase):
    def test_each_block_uses_its_own_efficiency(self):
        sample, sels = grouped_sample(seed=11)
        like = hfit.ConditionalLikelihood(sample, sels, h_min=H_LO, measurement="none")
        priors = hfit.default_priors(sample.h, H_LO)
        mle = hfit.fit_mle(like, priors)
        self.assertLess(2.0 * (mle.loglike - like.loglike(*TRUE)), 16.27)
        self.assertAlmostEqual(mle.theta[0], TRUE[0], delta=0.08)
        _d, p = hfit.ks_uniform(like.pit(*TRUE))
        self.assertGreater(p, 0.01)
        # One shared (deep) selection mis-models the shallow block.
        wrong = hfit.ConditionalLikelihood(sample, sels["T/deep"], h_min=H_LO, measurement="none")
        self.assertLess(wrong.loglike(*TRUE), like.loglike(*TRUE) - 10.0)
        diag = {d["group"]: d for d in hfit.group_diagnostics(like, TRUE)}
        self.assertEqual(diag["T/shallow"]["n"], 1200)
        self.assertGreater(diag["T/shallow"]["ks_p_value"], 0.001)

    def test_missing_group_selection_is_an_error(self):
        sample, sels = grouped_sample(seed=12, n_each=50)
        with self.assertRaisesRegex(ValueError, "no selection for T/shallow"):
            hfit.ConditionalLikelihood(sample, {"T/deep": sels["T/deep"]}, h_min=H_LO, measurement="none")

    def test_take_matches_subset(self):
        sample, sels = grouped_sample(seed=13, n_each=200)
        like = hfit.ConditionalLikelihood(sample, sels, h_min=H_LO, measurement="none")
        idx = np.arange(0, like.n, 3)
        direct = hfit.ConditionalLikelihood(like.sample.subset(idx), sels, h_min=H_LO, measurement="none")
        taken = like.take(idx)
        self.assertAlmostEqual(taken.loglike(*TRUE), direct.loglike(*TRUE), places=6)


def _ht_sample(bias, group, key, bias_se=None, mode="union"):
    n = len(bias)
    return hfit.HSample(
        np.array([str(i) for i in range(n)]), np.full(n, 6.0), np.full(n, 22.0),
        np.asarray(bias, float), np.full(n, np.nan), "Hx", "t",
        group=np.array(group, dtype=object),
        bias_key=hfit._object_array(key),
        bias_se=None if bias_se is None else np.asarray(bias_se, float),
        bias_mode=mode,
    )


class HorvitzThompson(unittest.TestCase):
    def test_union_count_and_variance(self):
        bias = [0.1, 0.1, 0.2, 0.5]
        s = _ht_sample(bias, ["S/a", "S/b", "S/a", "S/b"],
                       [("d", "c1"), ("d", "c1"), ("d", "c2"), ("d", "c3")],
                       bias_se=[0.01, 0.01, 0.04, 0.0])
        ht = hfit.ht_count(s, 5.0, 7.0)
        w = 1.0 / np.array(bias)
        self.assertAlmostEqual(ht.count, w.sum())
        self.assertAlmostEqual(ht.sigma_sampling ** 2, np.sum(w * (w - 1)))
        # Detections sharing an estimate add coherently: (10 + 10) * 0.1, 5 * 0.2.
        self.assertAlmostEqual(ht.sigma_bias ** 2, (20 * 0.1) ** 2 + (5 * 0.2) ** 2)
        self.assertEqual(ht.mode, "union")
        self.assertEqual(ht.n_used, 4)

    def test_rel_err_fallback_fills_unknown_errors(self):
        s = _ht_sample([0.1, 0.1], ["S/a"] * 2, [("d", "c")] * 2)
        self.assertEqual(hfit.ht_count(s, 5.0, 7.0).sigma_bias, 0.0)
        ht = hfit.ht_count(s, 5.0, 7.0, rel_err_fallback=0.1)
        self.assertAlmostEqual(ht.sigma_bias, 2.0)

    def test_block_mode_combines_blocks_by_inverse_variance(self):
        # Block a: 4 at bias 0.5 -> count 8, var 8. Block b: 2 at 0.25 -> 8, var 24.
        s = _ht_sample([0.5] * 4 + [0.25] * 2, ["S/a"] * 4 + ["S/b"] * 2,
                       [("S", "a", "d", "c")] * 4 + [("S", "b", "d", "c")] * 2,
                       bias_se=[0.0] * 6, mode="block")
        ht = hfit.ht_count(s, 5.0, 7.0)
        self.assertEqual(ht.mode, "block")
        self.assertAlmostEqual(ht.per_block["S/a"]["count"], 8.0)
        self.assertAlmostEqual(ht.per_block["S/b"]["count"], 8.0)
        self.assertAlmostEqual(ht.per_block["S/a"]["sigma"] ** 2, 8.0)
        self.assertAlmostEqual(ht.per_block["S/b"]["sigma"] ** 2, 24.0)
        f = np.array([1 / 8, 1 / 24]) / (1 / 8 + 1 / 24)
        self.assertAlmostEqual(ht.count, 8.0)
        self.assertAlmostEqual(ht.sigma ** 2, f[0] ** 2 * 8 + f[1] ** 2 * 24)
        self.assertLess(ht.sigma ** 2, 8.0)

    def test_h_range_and_override(self):
        s = _ht_sample([0.5, 0.5], ["S/a"] * 2, [("d", "c")] * 2)
        s.h[1] = 9.0
        self.assertEqual(hfit.ht_count(s, 5.0, 7.0).n_used, 1)
        self.assertAlmostEqual(hfit.ht_count(s, 5.0, 10.0, bias=[0.25, 0.5]).count, 6.0)


class BiasTables(unittest.TestCase):
    def test_from_results_shares_keys(self):
        s = _ht_sample([0.1, 0.1, 0.2], ["S/a"] * 3,
                       [("d", "c1"), ("d", "c1"), ("d", "c2")],
                       bias_se=[0.01, 0.01, np.nan])
        table = hfit.bias_table(s, rel_err_fallback=0.3)
        self.assertEqual(len(table.keys), 2)
        np.testing.assert_allclose(table.rel_err, [0.1, 0.3])
        np.testing.assert_allclose(table.biases(), s.bias)
        rng = np.random.default_rng(0)
        f = np.array([table.perturbation(rng) for _ in range(20000)])
        np.testing.assert_allclose(f.mean(axis=0), 1.0, atol=0.01)
        np.testing.assert_allclose(f.std(axis=0), [0.1, 0.3], rtol=0.05)
        self.assertIn("results file", table.note)

    def test_missing_error_is_reported(self):
        s = _ht_sample([0.1], ["S/a"], [("d", "c1")])
        self.assertIn("no MC error", hfit.bias_table(s).note)

    def test_union_from_bias_files(self):
        from survey_debias.bias_files import BiasEstimate, save_bias_file
        from survey_debias.pointings import Project

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_characterization(root, (("a", EFF_TEXT), ("b", EFF_TEXT)))
            cell = (42.0, 2.0, 7.25)
            save_bias_file(root / "characterization/T/bias_grid_rih.csv", {
                ("a", "default", cell): BiasEstimate(0.1, 100, 25, 0.01),
                ("b", "default", cell): BiasEstimate(0.3, 100, 25, 0.06),
            }, "model_ae")
            s = _ht_sample([0.4], ["T/a"], [("default", "(42.0, 2.0, 7.25)")])
            table = hfit.bias_table(s, Project(root, use_ossssim=False), "model_ae")
            self.assertEqual(len(table.keys), 2)
            np.testing.assert_allclose(table.matrix, [[1.0, 1.0]])
            np.testing.assert_allclose(table.biases(), [0.4])
            np.testing.assert_allclose(sorted(table.rel_err), [0.1, 0.2])
            self.assertIn("2 block-cell bias estimates", table.note)

            own = _ht_sample([0.3], ["T/b"], [("T", "b", "default", "(42.0, 2.0, 7.25)")],
                             mode="block")
            table = hfit.bias_table(own, Project(root, use_ossssim=False), "model_ae")
            self.assertEqual(table.keys, [("T", "b", "default", cell)])


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class Bootstrap(unittest.TestCase):
    def test_count_spread_matches_ht_sigma(self):
        sample, selection = synthetic(400, seed=21)
        n = len(sample)
        sample = hfit.HSample(sample.names, sample.h, sample.m, sample.bias, sample.rate,
                              "Hx", "x", bias_se=np.full(n, np.nan))
        like = hfit.ConditionalLikelihood(sample, selection, h_min=H_LO, measurement="none")
        priors = hfit.default_priors(sample.h, H_LO)
        mle = hfit.fit_mle(like, priors)
        h_norm = float(like.h_half_k.min())
        table = hfit.bias_table(sample, rel_err_fallback=0.0)
        ht = hfit.ht_count(like.sample, H_LO, h_norm)
        rng = np.random.default_rng(4)
        boot = hfit.bootstrap(like, sample, table, priors, mle.theta, 60, rng, H_LO, h_norm)
        self.assertEqual(boot.n_failed, 0)
        self.assertEqual(boot.samples.shape, (60, 4))
        ratio = boot.count.std(ddof=1) / ht.sigma
        self.assertGreater(ratio, 0.6)
        self.assertLess(ratio, 1.5)
        self.assertAlmostEqual(np.median(boot.count) / ht.count, 1.0, delta=0.2)
        # Shape scatter is of the order of the MCMC width, not zero.
        self.assertGreater(boot.samples[:, 0].std(), 0.005)

    def test_narrow_bootstrap_is_flagged(self):
        rng = np.random.default_rng(0)
        post = rng.normal([0.5, 1.0, 7.0, -4.0], [0.02, 0.6, 2.5, 0.8], size=(4000, 4))
        boot = rng.normal([0.5, 1.0, 7.0, -4.0], [0.02, 0.6, 0.3, 0.8], size=(200, 4))
        notes = hfit.bootstrap_width_warnings(boot, post)
        self.assertEqual(len(notes), 1)
        self.assertIn("h_b", notes[0])

    def test_bias_errors_widen_the_count(self):
        sample, selection = synthetic(200, seed=22)
        like = hfit.ConditionalLikelihood(sample, selection, h_min=H_LO, measurement="none")
        priors = hfit.default_priors(sample.h, H_LO)
        mle = hfit.fit_mle(like, priors)
        h_norm = float(like.h_half_k.min())
        spreads = []
        for rel in (0.0, 0.5):
            table = hfit.bias_table(sample, rel_err_fallback=rel)
            boot = hfit.bootstrap(like, sample, table, priors, mle.theta, 40,
                                  np.random.default_rng(9), H_LO, h_norm)
            spreads.append(np.std(np.log(boot.count)))
        self.assertGreater(spreads[1], spreads[0])


@unittest.skipUnless(HAVE_FIT_EXTRAS, "scipy, emcee, or corner not installed")
class Command(unittest.TestCase):
    def test_end_to_end(self):
        from survey_debias import hfit_cli

        sample, _selection = synthetic(400, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_characterization(root)
            (root / "s.toml").write_text(SURVEY_TOML)
            lines = ["# detection columns: test", "# bias_mode: union",
                     "name,mag,dist,incl,survey,block,colour_group,cell,Hx,bias,bias_se"]
            for k in range(len(sample)):
                cell = f"\"(42.0, 2.0, {0.5 * int(sample.h[k] / 0.5) + 0.25:.2f})\""
                lines.append(f"{k},{sample.m[k]:.4f},42.0,1.0,T,t,default,{cell},"
                             f"{sample.h[k]:.4f},{sample.bias[k]:.6g},"
                             f"{0.05 * sample.bias[k]:.6g}")
            (root / "bias_results.csv").write_text("\n".join(lines) + "\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = hfit_cli.main([
                    "--survey", str(root / "s.toml"), "--root", str(root),
                    "--walkers", "12", "--steps", "300", "--burn", "100",
                    "--ppc-draws", "20", "--bootstrap", "8",
                ])
            self.assertEqual(code, 0, out.getvalue())
            out_dir = root / "hfit"
            for name in ("hfit_summary.txt", "hfit_summary.json",
                         "hfit_samples.csv", "hfit_bootstrap.csv", "hfit_corner.png",
                         "hfit_detected.png", "hfit_debiased.png"):
                self.assertTrue((out_dir / name).is_file(), name)
            summary = json.loads((out_dir / "hfit_summary.json").read_text())
            self.assertEqual(summary["n_detections"], 400)
            self.assertIn("posterior", summary)
            self.assertEqual(summary["bootstrap"]["replicates"], 8)
            self.assertEqual([g["group"] for g in summary["groups"]], ["T/t"])
            self.assertGreater(summary["normalisation"]["ht_sigma_bias"], 0.0)
            header = (out_dir / "hfit_samples.csv").read_text().splitlines()[0]
            self.assertEqual(header, "alpha,beta,h_b,h_o,loglike")
            header = (out_dir / "hfit_bootstrap.csv").read_text().splitlines()[0]
            self.assertEqual(header, "alpha,beta,h_b,h_o,ht_count")
            text = (out_dir / "hfit_summary.txt").read_text()
            self.assertIn("bootstrap", text)

    def test_missing_results_reports_error(self):
        from survey_debias import hfit_cli

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "s.toml").write_text(SURVEY_TOML)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = hfit_cli.main(["--survey", str(root / "s.toml"),
                                      "--root", str(root), "--no-mcmc"])
            self.assertEqual(code, 2)
            self.assertIn("run survey_debias first", err.getvalue())


if __name__ == "__main__":
    unittest.main()

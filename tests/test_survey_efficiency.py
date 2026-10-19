"""Tests for survey population efficiencies eps(H)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from survey_debias import survey_efficiency as se  # noqa: E402
from survey_debias.pointings import load_survey_blocks  # noqa: E402

from project_fixture import make_jwst_project, make_survey, short_tempdir  # noqa: E402

MODEL = ROOT / "tests" / "data" / "OSSOS" / "Classical-ModelUsed.dat"

try:
    import ossssim  # noqa: F401
    HAVE_OSSSSIM = True
except ImportError:
    HAVE_OSSSSIM = False


def fake_plants(survey, block, n, rng):
    in_field = rng.random(n) < 0.5
    mu = np.where(in_field, rng.uniform(16.0, 17.0, n), np.nan)
    return se.PlantTable(
        survey, np.full(n, block), np.full(n, "coldl"), rng.uniform(42, 47, n),
        rng.uniform(0, 0.1, n), rng.uniform(0, 8, n), rng.uniform(38, 50, n),
        np.zeros(n), rng.uniform(0, 1e-5, n), in_field, mu, np.where(in_field, 3.0, np.nan),
    )


class UnitCharacterization(unittest.TestCase):
    def test_unit_eff_keeps_geometry_and_drops_selection(self):
        with short_tempdir() as tmp:
            base = make_survey(Path(tmp), "S", ["0.2 0.2 210.0 -10.0 2459969.3 0.9 500 b.eff\n"],
                               {"b": "r"})
            text = se.unit_efficiency_text(base / "b.eff")
        self.assertIn("rate_cut= 0.03 8.66 0.0 180.0", text)
        self.assertIn("filter= r", text)
        self.assertIn("function= lookup", text)
        self.assertIn("mag_lim= 40.0", text)
        self.assertNotIn("single_param", text)

    def test_copy_has_the_same_pointings(self):
        with short_tempdir() as tmp:
            make_jwst_project(Path(tmp) / "p")
            unit = se.make_unit_characterization(Path(tmp) / "p" / "characterization",
                                                 "jwst_sample_a", Path(tmp) / "u")
            real = load_survey_blocks(Path(tmp) / "p" / "characterization", "jwst_sample_a", False)
            copy = load_survey_blocks(unit, "jwst_sample_a", False)
        self.assertEqual(list(real), list(copy))
        self.assertEqual(real["jwst_sample_a"].n_epochs, copy["jwst_sample_a"].n_epochs)
        self.assertEqual(copy["jwst_sample_a"].detections_required, 3)


class Model(unittest.TestCase):
    def test_components(self):
        cold = se.ModelPopulation.from_file(MODEL)
        self.assertEqual(set(cold.comp), {"coldl"})
        both = se.ModelPopulation.from_file(MODEL, ("coldl", "hot"))
        self.assertEqual(len(both), len(cold) + 1)
        with self.assertRaises(ValueError):
            se.ModelPopulation.from_file(MODEL, ("kernel",))


class Efficiency(unittest.TestCase):
    def test_eps_is_the_weighted_selection(self):
        rng = np.random.default_rng(1)
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", ["0.2 0.2 210.0 -10.0 2459969.3 0.9 500 b.eff\n"],
                        {"b": "r"})
            blocks = load_survey_blocks(Path(tmp) / "characterization", "S", False)
        plants = fake_plants("S", "b", 400, rng)
        table = se.survey_efficiency(plants, blocks, h_grid=(10.0, 14.0, 0.5), n_boot=8)
        sel = blocks["b"].selection
        k = plants.in_field
        for j, h in enumerate(table.h):
            s = np.array([sel.measured_selection(np.array([h + m]), 3.0)[0] for m in plants.mu[k]])
            self.assertAlmostEqual(table.eps[j], float(np.sum(plants.w[k] * s)) / 400, delta=1e-9)
        self.assertEqual(table.boot.shape, (8, len(table.h)))
        self.assertEqual(table.filter, "r")

    def test_cut_and_round_trip(self):
        rng = np.random.default_rng(2)
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", ["0.2 0.2 210.0 -10.0 2459969.3 0.9 500 b.eff\n"],
                        {"b": "r"})
            blocks = load_survey_blocks(Path(tmp) / "characterization", "S", False)
            plants = fake_plants("S", "b", 300, rng)
            full = se.survey_efficiency(plants, blocks, h_grid=(8.0, 14.0, 0.25), n_boot=4)
            cut = se.survey_efficiency(plants, blocks, se.SampleCut(i_max=4.0),
                                       h_grid=(8.0, 14.0, 0.25), n_boot=4)
            self.assertTrue(np.all(cut.eps <= full.eps + 1e-15))
            path = Path(tmp) / "eps.csv"
            cut.save(path)
            back = se.EfficiencyTable.load(path)
            plants.save(Path(tmp) / "p.npz")
            again = se.PlantTable.load(Path(tmp) / "p.npz")
        np.testing.assert_allclose(back.eps, cut.eps, rtol=1e-5)
        self.assertEqual(back.meta["cut"], "i < 4 deg")
        np.testing.assert_array_equal(again.block, plants.block)


@unittest.skipUnless(HAVE_OSSSSIM, "needs ossssim")
class GeometryPass(unittest.TestCase):
    def test_plants_land_in_the_three_epoch_field(self):
        model = se.ModelPopulation.from_file(MODEL)
        with short_tempdir() as tmp:
            make_jwst_project(Path(tmp))
            char = Path(tmp) / "characterization"
            plants = se.geometry_pass(char, "jwst_sample_a", model, 300, seed=3)
            blocks = load_survey_blocks(char, "jwst_sample_a", False)
            area = se.three_epoch_area_arcmin2(plants, blocks)
        box = blocks["jwst_sample_a"].aim_area_deg2 * 3600.0
        self.assertGreater(plants.in_field.sum(), 0)
        self.assertGreater(area, 0.3 * box)
        self.assertLessEqual(area, box)
        mu = plants.mu[plants.in_field]
        self.assertTrue(np.all((mu > 15.5) & (mu < 17.5)))


if __name__ == "__main__":
    unittest.main()

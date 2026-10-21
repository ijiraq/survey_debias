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
    angles = rng.uniform(0, 360, (3, n))
    return se.PlantTable(
        survey, np.full(n, block), np.full(n, "coldl"), rng.uniform(42, 47, n),
        rng.uniform(0, 0.1, n), rng.uniform(0, 8, n), rng.uniform(38, 50, n),
        *angles, np.full(n, 2459969.3), rng.uniform(0, 1e-5, n), in_field, mu,
        np.where(in_field, 3.0, np.nan),
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


class Overlaps(unittest.TestCase):
    def _blocks(self, tmp, survey, ra, jd):
        make_survey(Path(tmp), survey, [f"0.5 0.5 {ra} 0.0 {jd} 0.9 500 b.eff\n"], {"b": "r"})
        return load_survey_blocks(Path(tmp) / "characterization", survey, False)

    def test_reachable_pairs_use_cold_drift(self):
        with short_tempdir() as tmp:
            a = self._blocks(tmp, "A", 100.0, 2457000.5)
            near = self._blocks(tmp, "B", 110.0, 2457000.5 + 5 * 365.25)
            far = self._blocks(tmp, "C", 160.0, 2457000.5 + 5 * 365.25)
        self.assertEqual(se.reachable_pairs(a, near), [("b", "b")])
        self.assertEqual(se.reachable_pairs(a, far), [])

    def test_overlap_is_the_product_of_both_selections(self):
        rng = np.random.default_rng(4)
        with short_tempdir() as tmp:
            a = self._blocks(tmp, "A", 100.0, 2457000.5)
            b = self._blocks(tmp, "B", 100.0, 2457000.5)
        plants = fake_plants("A", "b", 200, rng)
        idx = np.flatnonzero(plants.in_field)
        in_b = rng.random(idx.size) < 0.5
        ov = se.OverlapTable("A", "B", idx, in_b, np.where(in_b, "b", ""),
                             np.where(in_b, plants.mu[idx] + 0.1, np.nan),
                             np.where(in_b, 3.0, np.nan))
        table = se.overlap_efficiency(plants, a, ov, b, 0.0, 0.0, h_grid=(8.0, 9.0, 1.0),
                                      n_boot=0)
        sel = a["b"].selection
        j = idx[in_b]
        for k, h in enumerate(table.h):
            s = [sel.measured_selection(np.array([h + plants.mu[x]]), 3.0)[0]
                 * sel.measured_selection(np.array([h + plants.mu[x] + 0.1]), 3.0)[0] for x in j]
            self.assertAlmostEqual(table.eps[k], float(np.sum(plants.w[j] * s)) / 200, delta=1e-12)
        single = se.survey_efficiency(plants, a, h_grid=(8.0, 9.0, 1.0), n_boot=0)
        self.assertTrue(np.all(table.eps <= single.eps))


class FreeInclinationAiming(unittest.TestCase):
    def test_poles_through_position_at_free_inclination(self):
        rng = np.random.default_rng(4)
        centre = np.array([0.03, -0.001, 1.0])
        centre /= np.linalg.norm(centre)
        for _ in range(20):
            r = np.array([*rng.normal(size=2), 0.02])
            r /= np.linalg.norm(r)
            for pole in se.poles_at_angle(r, centre, 2.5):
                self.assertAlmostEqual(float(pole @ r), 0.0, places=10)
                self.assertAlmostEqual(float(np.degrees(np.arccos(pole @ centre))), 2.5, places=8)
        self.assertEqual(se.poles_at_angle(np.array([0.0, 0.0, 1.0]), centre, 2.5), [])

    def test_ifree_cut_needs_ifree_plants(self):
        rng = np.random.default_rng(5)
        plants = fake_plants("A", "b", 50, rng)
        with self.assertRaises(ValueError):
            se.SampleCut(ifree_max=4.0).mask(plants)
        plants.aim = "ifree"
        plants.ifree = np.linspace(0.0, 8.0, 50)
        m = se.SampleCut(ifree_max=4.0).mask(plants)
        self.assertEqual(int(m.sum()), int(np.sum(plants.ifree < 4.0)))
        self.assertEqual(se.SampleCut(ifree_max=4.0).describe(), "i_free < 4 deg")

    def test_plant_table_keeps_ifree_and_aim(self):
        rng = np.random.default_rng(6)
        plants = fake_plants("A", "b", 20, rng)
        plants.aim, plants.ifree = "ifree", rng.uniform(0, 5, 20)
        with short_tempdir() as tmp:
            path = Path(tmp) / "plants.npz"
            plants.save(path)
            back = se.PlantTable.load(path)
        self.assertEqual(back.aim, "ifree")
        np.testing.assert_allclose(back.ifree, plants.ifree)


@unittest.skipUnless(HAVE_OSSSSIM, "needs ossssim")
class GeometryPassFree(unittest.TestCase):
    def test_plants_keep_the_model_free_inclination(self):
        model = se.ModelPopulation.from_file(MODEL)
        with short_tempdir() as tmp:
            make_jwst_project(Path(tmp))
            char = Path(tmp) / "characterization"
            plants = se.geometry_pass(char, "jwst_sample_a", model, 300, seed=3, aim="ifree")
        self.assertEqual(plants.aim, "ifree")
        ok = plants.w > 0
        self.assertGreater(ok.sum(), 50)
        self.assertGreater(plants.in_field.sum(), 0)
        self.assertTrue(np.all(np.isfinite(plants.ifree[ok])))
        # Aimed orbits sit near their model row's free inclination (the forcing pole
        # moves a little with the aimed i and omega).
        self.assertLess(float(np.median(plants.ifree[ok])), 6.0)


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

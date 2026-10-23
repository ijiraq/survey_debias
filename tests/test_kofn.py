"""Surveys that find an object at K of their N epochs (DEEP: 4 of 29 nights)."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy.stats import binom

from project_fixture import make_survey, short_tempdir
from survey_debias.efficiency import KofNSelection, SurveySelection, at_least
from survey_debias.pointings import Project

LINE = "0.2 0.2 10.0 0.0 2459000.5 {fill} 500 b1.eff\n"


def kofn_block(tmp, epochs=5, required=2, fill=0.8):
    make_survey(Path(tmp), "S", [LINE.format(fill=fill)], {"b1": "r"}, epochs=epochs,
                detections_required=required)
    return Project(tmp, use_ossssim=False).block("S", "b1")


class AtLeastTest(unittest.TestCase):
    def test_binomial_tail(self):
        p = np.linspace(0.0, 1.0, 11)
        for n, k in ((5, 4), (8, 4), (4, 4), (29, 4), (3, 1)):
            np.testing.assert_allclose(at_least([p] * n, k), binom.sf(k - 1, n, p), atol=1e-12)

    def test_unequal_probabilities(self):
        a, b, c = np.array(0.5), np.array(0.2), np.array(0.9)
        self.assertAlmostEqual(float(at_least([a, b, c], 2)),
                               0.5 * 0.2 + 0.5 * 0.9 + 0.2 * 0.9 - 2 * 0.5 * 0.2 * 0.9)

    def test_fewer_epochs_than_required(self):
        self.assertEqual(float(at_least([np.array(0.9)] * 3, 4)), 0.0)


class KofNSelectionTest(unittest.TestCase):
    def test_k_equal_n_is_the_and_product(self):
        with short_tempdir() as tmp:
            block = kofn_block(tmp, epochs=3, required=2)
            epochs = block.selection.epochs
        m = np.linspace(26.0, 31.0, 51)
        np.testing.assert_allclose(KofNSelection(epochs, 3)(m, 1.0),
                                   SurveySelection(epochs)(m, 1.0), atol=1e-14)

    def test_block_selection_uses_its_epochs_and_fill(self):
        with short_tempdir() as tmp:
            block = kofn_block(tmp, epochs=5, required=2, fill=0.8)
            self.assertTrue(block.k_of_n)
            sel = block.kofn_selection([0, 2, 4], with_fill=True)
            bare = block.kofn_selection([0, 2, 4], with_fill=False)
            one = block.selection.epochs[0]
        m = np.linspace(26.0, 31.0, 51)
        p = one(m, 1.0)
        np.testing.assert_allclose(sel(m, 1.0), binom.sf(1, 3, 0.8 * p), atol=1e-12)
        np.testing.assert_allclose(bare(m, 1.0), binom.sf(1, 3, p), atol=1e-12)
        self.assertEqual(len(sel.epochs), 3)

    def test_every_epoch_required_is_not_k_of_n(self):
        with short_tempdir() as tmp:
            self.assertFalse(kofn_block(tmp, epochs=3, required=3).k_of_n)


class EfficiencyTest(unittest.TestCase):
    def _plants(self, masks, in_field):
        from survey_debias.survey_efficiency import PlantTable

        n = len(masks)
        f = np.full(n, 1.0)
        return PlantTable("S", np.full(n, "b1", dtype=object), np.full(n, "coldl", dtype=object),
                          a=f * 44, e=f * 0.05, i=f * 2.0, r=f * 44, node=f, peri=f, M=f,
                          epoch=f * 2459000.5, w=f * 1e-4, in_field=np.array(in_field),
                          mu=f * 16.0, rate=f * 3.0, epochs=np.array(masks, dtype=np.int64))

    def test_eps_uses_each_plants_epochs(self):
        from survey_debias.survey_efficiency import survey_efficiency

        with short_tempdir() as tmp:
            block = kofn_block(tmp, epochs=5, required=2)
            plants = self._plants([0b00011, 0b11111, 0b00001], [True, True, False])
            table = survey_efficiency(plants, {"b1": block}, n_boot=0)
            h = table.h
            expect = np.zeros(len(h))
            for mask in (0b00011, 0b11111):
                epochs = [e for e in range(5) if mask >> e & 1]
                sel = block.kofn_selection(epochs, with_fill=False)
                expect += 1e-4 * sel.measured_selection(h + 16.0, 3.0) / 3
        np.testing.assert_allclose(table.eps, expect, rtol=1e-3, atol=1e-12)

    def test_plants_without_epoch_record_are_refused(self):
        from survey_debias.survey_efficiency import survey_efficiency

        with short_tempdir() as tmp:
            block = kofn_block(tmp, epochs=5, required=2)
            plants = self._plants([-1, -1], [True, True])
            with self.assertRaises(ValueError):
                survey_efficiency(plants, {"b1": block}, n_boot=0)


class JointGroupsTest(unittest.TestCase):
    def test_detections_get_their_epochs_selection(self):
        from survey_debias.joint_cli import _selection_groups, group_block

        with short_tempdir() as tmp:
            block = kofn_block(tmp, epochs=5, required=2)
            rows = [{"name": "A", "block": "b1", "epochs": "1;3;5"},
                    {"name": "B", "block": "b1", "epochs": "5;3;1"},
                    {"name": "C", "block": "b1", "epochs": "2;4"}]
            labels, selections = _selection_groups("S", rows, {"b1": block})
        self.assertEqual(labels, ["S/b1|1;3;5", "S/b1|1;3;5", "S/b1|2;4"])
        self.assertEqual(set(selections), {"S/b1|1;3;5", "S/b1|2;4"})
        self.assertEqual(len(selections["S/b1|1;3;5"].epochs), 3)
        self.assertEqual(group_block(labels[0]), "b1")

    def test_k_of_n_detection_needs_epochs(self):
        from survey_debias.joint_cli import _selection_groups

        with short_tempdir() as tmp:
            block = kofn_block(tmp, epochs=5, required=2)
            for bad in ("", "3", "1;9"):
                with self.subTest(epochs=bad), self.assertRaises(ValueError):
                    _selection_groups("S", [{"name": "A", "block": "b1", "epochs": bad}],
                                      {"b1": block})


class CutFractionTest(unittest.TestCase):
    """The cumulative plot divides by the model fraction inside the cut, not a plant fraction."""

    MODEL = Path(__file__).resolve().parent / "data" / "OSSOS" / "Classical-ModelUsed.dat"

    def test_recorded_f_cut_is_used(self):
        from survey_debias.joint_cli import _CutFractions

        self.assertEqual(_CutFractions().get({"f_cut": "0.71440"}, Path(".")), 0.7144)

    def test_computed_from_the_model_rows(self):
        from survey_debias.joint_cli import _CutFractions
        from survey_debias.survey_efficiency import (ModelPopulation, SampleCut,
                                                     model_cut_fraction)

        meta = {"model": str(self.MODEL), "components": "coldl,coldh,kernel",
                "cut": "i_free < 4 deg"}
        model = ModelPopulation.from_file(self.MODEL, ("coldl", "coldh", "kernel"))
        expect, _ = model_cut_fraction(model, SampleCut(None, None, None, 4.0))
        self.assertAlmostEqual(_CutFractions().get(meta, Path(".")), expect)
        self.assertGreater(expect, 0.0)

    def test_missing_model_gives_none(self):
        from survey_debias.joint_cli import _CutFractions

        self.assertIsNone(_CutFractions().get({"model": "no/such/model.dat"}, Path("/tmp")))


if __name__ == "__main__":
    unittest.main()

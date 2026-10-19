"""Survey/block characterization, colours, attribution, and bias files."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from project_fixture import COLOUR_TOML, make_survey, short_tempdir, write_eff  # noqa: E402
from survey_debias import bias_files, grid_bias, pointings  # noqa: E402
from survey_debias.pointings import Project, parse_pointings  # noqa: E402

try:
    from ossssim.survey import SurveyCharacterization  # noqa: F401
    HAVE_OSSSSIM = True
except ImportError:
    HAVE_OSSSSIM = False


class PointingsParsing(unittest.TestCase):
    def _parse(self, text):
        with short_tempdir() as tmp:
            path = Path(tmp) / "pointings.list"
            path.write_text(text)
            return parse_pointings(path)

    def test_rect_with_and_without_keyword(self):
        tiles = self._parse(
            "# header\n"
            "0.2 0.1 210.0 -10.0 2459969.5 0.9 JWST.csv a.eff\n"
            "rect 0.1111111 0.1666667 14:07:53.33 -11:21:38 2452672.8585 1.0 HST.csv b.eff\n"
        )
        self.assertEqual(len(tiles), 2)
        a, b = tiles
        self.assertEqual((a.shape, a.eff_name, a.observer, a.line), ("rect", "a.eff", "JWST.csv", 2))
        self.assertAlmostEqual(a.area_deg2, 0.02)
        self.assertAlmostEqual(a.fill, 0.9)
        self.assertAlmostEqual(b.ra_deg, 15.0 * (14 + 7 / 60 + 53.33 / 3600), places=6)
        self.assertAlmostEqual(b.dec_deg, -(11 + 21 / 60 + 38 / 3600), places=6)
        self.assertAlmostEqual(b.jd, 2452672.8585)

    def test_ears_area_counts_each_ear_once(self):
        (t,) = self._parse("ears 150.0 2.0 2459000.5 1.0 568 m.eff\n")
        rad2 = (180.0 / math.pi) ** 2
        body = 4 * pointings._EARS_W * pointings._EARS_H * rad2
        ears = 4 * pointings._EARS_EW * pointings._EARS_EH * rad2
        self.assertEqual(t.shape, "ears")
        self.assertAlmostEqual(t.area_deg2, body + ears, places=9)
        self.assertGreater(t.box_area_deg2, t.area_deg2)

    def test_poly_reads_vertex_lines(self):
        (t,) = self._parse(
            "poly 4 30.0 0.0 2459000.5 1.0 500 p.eff\n"
            "-0.1 -0.05\n0.1 -0.05\n0.1 0.05\n-0.1 0.05\n"
        )
        self.assertEqual(t.shape, "poly")
        self.assertAlmostEqual(t.area_deg2, 0.02)
        self.assertAlmostEqual(t.width_deg, 0.2)
        self.assertAlmostEqual(t.height_deg, 0.1)
        self.assertAlmostEqual(t.ra_deg, 30.0)

    def test_bad_lines_name_the_file_and_line(self):
        with self.assertRaisesRegex(ValueError, r"pointings.list:2"):
            self._parse("# c\n0.2 0.2 10.0\n")
        with self.assertRaisesRegex(ValueError, "no pointings"):
            self._parse("# only a comment\n")


class Blocks(unittest.TestCase):
    def test_tiles_group_by_eff_name(self):
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", [
                "0.2 0.2 10.0 0.0 2459000.5 1.0 500 b1.eff\n",
                "0.1 0.1 10.3 0.0 2459000.5 1.0 500 b1.eff\n",
                "0.2 0.2 50.0 5.0 2459001.5 1.0 500 b2.eff\n",
            ], {"b1": "r", "b2": "g"})
            blocks = Project(tmp, use_ossssim=False).blocks("S")
        self.assertEqual(sorted(blocks), ["b1", "b2"])
        b1 = blocks["b1"]
        self.assertEqual(len(b1.tiles[0]), 2)
        self.assertAlmostEqual(b1.area_deg2, 0.05)
        self.assertEqual(b1.filter, "r")
        self.assertEqual(blocks["b2"].filter, "g")
        self.assertEqual(b1.n_epochs, 1)

    def test_epoch_directories_and_mismatch(self):
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", ["0.2 0.2 10.0 0.0 2459000.5 1.0 500 b1.eff\n"],
                        {"b1": "r"}, epochs=2)
            block = Project(tmp, use_ossssim=False).block("S", "b1")
            self.assertEqual(block.n_epochs, 2)
            self.assertEqual(len(block.selection.epochs), 2)
            (Path(tmp) / "characterization/S/epoch2/pointings.list").write_text(
                "0.2 0.2 10.0 0.0 2459000.5 1.0 500 other.eff\n")
            write_eff(Path(tmp) / "characterization/S/epoch2/other.eff", "r")
            with self.assertRaisesRegex(ValueError, "differ from epoch1"):
                Project(tmp, use_ossssim=False).blocks("S")

    def test_filter_must_match_across_epochs_and_exist(self):
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", ["0.2 0.2 10.0 0.0 2459000.5 1.0 500 b1.eff\n"],
                        {"b1": "r"}, epochs=2)
            write_eff(Path(tmp) / "characterization/S/epoch2/b1.eff", "g")
            with self.assertRaisesRegex(ValueError, "same 'filter='"):
                Project(tmp, use_ossssim=False).blocks("S")

    def test_unknown_block_and_detos_name_limits(self):
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", [
                "0.2 0.2 10.0 0.0 2459000.5 1.0 500 blockname_1.eff\n",
                "0.2 0.2 20.0 0.0 2459000.5 1.0 500 blockname_2.eff\n",
            ], {"blockname_1": "r", "blockname_2": "r"})
            with mock.patch.object(pointings, "detos_reports_keys", return_value=False):
                with self.assertRaisesRegex(ValueError, "first 10"):
                    Project(tmp, use_ossssim=False).blocks("S")
            with mock.patch.object(pointings, "detos_reports_keys", return_value=True):
                self.assertEqual(len(Project(tmp, use_ossssim=False).blocks("S")), 2)
        with short_tempdir() as tmp:
            long = "b" * 33
            make_survey(Path(tmp), "S", [f"0.2 0.2 10.0 0.0 2459000.5 1.0 500 {long}.eff\n"],
                        {long: "r"})
            with mock.patch.object(pointings, "detos_reports_keys", return_value=True):
                with self.assertRaisesRegex(ValueError, "longer than 32"):
                    Project(tmp, use_ossssim=False).blocks("S")
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", ["0.2 0.2 10.0 0.0 2459000.5 1.0 500 b1.eff\n"],
                        {"b1": "r"})
            with self.assertRaisesRegex(ValueError, "no line tagged nope.eff"):
                Project(tmp, use_ossssim=False).block("S", "nope")

    def test_block_view_feeds_geometry_helpers(self):
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", [
                "0.2 0.1 100.0 60.0 2459000.5 1.0 500 b1.eff\n",
            ], {"b1": "r"})
            block = Project(tmp, use_ossssim=False).block("S", "b1")
        view = block.field_view()
        self.assertIsInstance(view, grid_bias.FieldView)
        self.assertAlmostEqual(view.mosaic_width_deg, 0.2, places=9)
        self.assertAlmostEqual(view.mosaic_height_deg, 0.1, places=9)
        self.assertEqual(view.epoch_jd, (2459000.5,))
        self.assertEqual((view.rate_cut_min_arcsec_hr, view.rate_cut_max_arcsec_hr),
                         (0.03, 8.66))
        rng = np.random.default_rng(1)
        pts = np.array([grid_bias.sample_mosaic_icrs(rng, survey=view) for _ in range(2000)])
        half_ra = 0.1 / math.cos(math.radians(60.0))
        self.assertLessEqual(np.abs(pts[:, 0] - 100.0).max(), half_ra + 1e-9)
        self.assertGreater(np.abs(pts[:, 0] - 100.0).max(), 0.9 * half_ra)
        self.assertLessEqual(np.abs(pts[:, 1] - 60.0).max(), 0.05 + 1e-9)

    @unittest.skipUnless(HAVE_OSSSSIM, "ossssim not installed")
    def test_ossssim_area_cross_check(self):
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", [
                "0.2 0.2 10.0 0.0 2459000.5 1.0 500 r1.eff\n",
                "ears 150.0 2.0 2459000.5 1.0 500 e1.eff\n",
            ], {"r1": "r", "e1": "r"})
            ours = Project(tmp, use_ossssim=False).blocks("S")
            theirs = Project(tmp, use_ossssim=True).blocks("S")
        for name in ("r1", "e1"):
            self.assertAlmostEqual(ours[name].area_deg2, theirs[name].area_deg2,
                                   delta=0.01 * ours[name].area_deg2)

    @unittest.skipUnless(HAVE_OSSSSIM and pointings.detos_loads_survey_roots(),
                         "needs ossssim with multi-epoch survey roots")
    def test_ossssim_reads_the_same_survey_root(self):
        from ossssim import SurveyCharacterization

        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", ["0.2 0.2 10.0 0.0 2459000.5 1.0 500 b1.eff\n"],
                        {"b1": "r"}, epochs=3)
            block = Project(tmp).block("S", "b1")
            loaded = SurveyCharacterization.from_directory(str(block.survey_root))
        self.assertEqual(loaded.n_epochs, block.n_epochs)
        self.assertEqual(loaded.detections_required, block.detections_required)
        self.assertEqual({p.key for p in loaded}, {block.root_key})

    def test_long_paths_are_refused_before_fortran(self):
        tile = parse_pointings.__globals__["Tile"](
            ra_deg=0, dec_deg=0, width_deg=1, height_deg=1, area_deg2=1, fill=1,
            jd=0, observer="500", eff_name="b.eff", shape="rect", line=1)
        with self.assertRaisesRegex(ValueError, "longer than 100"):
            pointings.check_fortran_paths(Path("/" + "x" * 120), [tile], limit=100)
        pointings.check_fortran_paths(Path("/" + "x" * 120), [tile], limit=2048)
        with self.assertRaisesRegex(ValueError, "longer than 2048"):
            pointings.check_fortran_paths(Path("/" + "x" * 2100), [tile], limit=2048)


class Colours(unittest.TestCase):
    def _project(self, tmp, model_band="r", text=COLOUR_TOML):
        (Path(tmp) / "colour.toml").write_text(text)
        return Project(tmp, use_ossssim=False, model_band=model_band)

    def test_rebased_to_model_band_by_group(self):
        with short_tempdir() as tmp:
            colours = self._project(tmp).colours
        self.assertAlmostEqual(colours.colour("H", "default"), -1.0)
        self.assertAlmostEqual(colours.colour("F", "hot"), 0.3)
        self.assertAlmostEqual(colours.colour("F", "cold_classical"), 0.4)
        self.assertAlmostEqual(colours.colour("r", "cold"), 0.0)
        self.assertEqual(colours.group("cold_classical"), "cold")
        self.assertEqual(colours.group("resonant"), "default")
        self.assertEqual(colours.group(None), "default")

    def test_colour_for_uses_block_filter(self):
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", ["0.2 0.2 10.0 0.0 2459000.5 1.0 500 b1.eff\n"],
                        {"b1": "H"})
            project = self._project(tmp)
            filt, colour, group = project.colour_for("S", "b1", "cold")
        self.assertEqual((filt, group), ("H", "cold"))
        self.assertAlmostEqual(colour, -1.0)

    def test_missing_filter_and_band_are_errors(self):
        with short_tempdir() as tmp:
            colours = self._project(tmp).colours
            with self.assertRaisesRegex(ValueError, "no colour for filter 'z'"):
                colours.colour("z", "default")
            with self.assertRaisesRegex(ValueError, "filter\\(s\\) z"):
                colours.validate({"H", "z"})
        with short_tempdir() as tmp:
            with self.assertRaisesRegex(ValueError, "model_band 'i'"):
                self._project(tmp, model_band="i").colours.colour("H")

    def test_bad_colour_files(self):
        for text, msg in (
            ('[cold]\n"r-g" = 0.1\n', r"\[default\]"),
            ('[default]\n"rg" = 0.1\n', "band ratio"),
            ('[default]\n"F150W2-g" = 0.1\n', "one character"),
            ('[default]\n"r-g" = "x"\n', "not a number"),
        ):
            with self.subTest(text=text), short_tempdir() as tmp:
                with self.assertRaisesRegex(ValueError, msg):
                    self._project(tmp, text=text).colours

    def test_missing_file_uses_ossssim_colours(self):
        with short_tempdir() as tmp:
            colours = Project(tmp, use_ossssim=False).colours
        self.assertIn("not found", colours.note)
        self.assertAlmostEqual(colours.colour("r"), 0.0)


class SurveyConf(unittest.TestCase):
    LINE = ["0.2 0.2 10.0 0.0 2459000.5 1.0 500 b1.eff\n"]

    def test_read_survey_conf(self):
        with short_tempdir() as tmp:
            base = Path(tmp)
            self.assertIsNone(pointings.read_survey_conf(base))
            (base / "survey.conf").write_text("# c\n\nother = 1\ndetections_required = 2  # AND\n")
            self.assertEqual(pointings.read_survey_conf(base), 2)
            (base / "survey.conf").write_text("detections_required = two\n")
            with self.assertRaisesRegex(ValueError, "survey.conf:1"):
                pointings.read_survey_conf(base)

    def test_multi_epoch_block_carries_requirement_and_root(self):
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", self.LINE, {"b1": "r"}, epochs=3,
                        detections_required=2)
            block = Project(tmp, use_ossssim=False).block("S", "b1")
        self.assertEqual(block.n_epochs, 3)
        self.assertEqual(block.detections_required, 2)
        self.assertEqual(block.survey_root.name, "S")
        self.assertEqual(block.root_key, "S/b1")

    def test_single_epoch_defaults_to_one(self):
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", self.LINE, {"b1": "r"})
            block = Project(tmp, use_ossssim=False).block("S", "b1")
            self.assertEqual(block.detections_required, 1)
            self.assertEqual(block.root_key, "S/b1")
            (Path(tmp) / "characterization/S/survey.conf").write_text("detections_required = 2\n")
            with self.assertRaisesRegex(ValueError, "single epoch"):
                Project(tmp, use_ossssim=False).blocks("S")

    def test_multi_epoch_needs_valid_survey_conf(self):
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", self.LINE, {"b1": "r"}, epochs=2)
            conf = Path(tmp) / "characterization/S/survey.conf"
            conf.unlink()
            with self.assertRaisesRegex(ValueError, "detections_required = 2"):
                Project(tmp, use_ossssim=False).blocks("S")
            for bad in (0, 3):
                conf.write_text(f"detections_required = {bad}\n")
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "between 1"):
                    Project(tmp, use_ossssim=False).blocks("S")

    def test_stray_epoch_like_directory_is_refused(self):
        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", self.LINE, {"b1": "r"}, epochs=2)
            extra = Path(tmp) / "characterization/S/visit3"
            extra.mkdir()
            (extra / "pointings.list").write_text(self.LINE[0])
            with self.assertRaisesRegex(ValueError, "visit3"):
                Project(tmp, use_ossssim=False).blocks("S")


class _FakeOSSSSim:
    """Records the characterization directory of every simulate() call."""

    def __init__(self, characterization_directory, seed):
        self.characterization_directory = characterization_directory
        self.dirs = []

    def simulate(self, row, colors, model_band, debug=False):
        self.dirs.append(Path(self.characterization_directory).name)
        return dict(flag=4, Survey="S/b1", RA=0.1, DEC=0.0)


class SimulatorModes(unittest.TestCase):
    def _sim(self, tmp, root_mode):
        from survey_debias import grid_bias_run

        make_survey(Path(tmp), "S", ["0.2 0.2 10.0 0.0 2459000.5 1.0 500 b1.eff\n"],
                    {"b1": "r"}, epochs=3)
        blocks = Project(tmp, use_ossssim=False).blocks("S")
        with mock.patch.object(grid_bias_run, "OSSSSim", _FakeOSSSSim), \
                mock.patch.object(grid_bias_run, "detos_loads_survey_roots",
                                  return_value=root_mode):
            sim = grid_bias_run.GridBiasSimulator("S", blocks, None, "r")
        return sim, blocks["b1"]

    def test_root_mode_calls_detos1_once_on_the_survey_root(self):
        from survey_debias.grid_bias_run import attributed_to

        with short_tempdir() as tmp:
            sim, block = self._sim(tmp, root_mode=True)
            self.assertTrue(sim.root_mode)
            self.assertEqual(sim.sim.dirs, ["S"])  # priming
            rows = sim.epoch_rows(44, 0.0, 5, 0, 0, 0, 8)
        self.assertEqual(len(rows), 1)
        self.assertEqual(sim.sim.dirs, ["S", "S"])
        self.assertTrue(attributed_to(rows, block))

    def test_older_ossssim_loops_over_epoch_directories(self):
        with short_tempdir() as tmp:
            sim, _ = self._sim(tmp, root_mode=False)
            self.assertFalse(sim.root_mode)
            rows = sim.epoch_rows(44, 0.0, 5, 0, 0, 0, 8)
        self.assertEqual(len(rows), 3)
        self.assertEqual(sim.sim.dirs[3:], ["epoch1", "epoch2", "epoch3"])


class _StubSimulator:
    """Detos1 stand-in: every call is flag 4, attributed per ``survey_for``."""

    def __init__(self, survey_for, n_epochs=1):
        self.survey_for = survey_for
        self.n_epochs = n_epochs
        self.calls = 0
        self.comps = []

    def epoch_rows(self, a, e, inc, node, peri, M, H, epoch_jd=None, comp="default",
                   debug=False):
        self.calls += 1
        self.comps.append(comp)
        name = self.survey_for(self.calls)
        return [dict(flag=4, Survey=name, RA=0.1, DEC=0.0) for _ in range(self.n_epochs)]


class Attribution(unittest.TestCase):
    def _block(self, tmp, name="b1", epochs=0):
        make_survey(Path(tmp), "S", [f"0.2 0.2 10.0 0.0 2459000.5 1.0 500 {name}.eff\n"],
                    {name: "r"}, epochs=epochs)
        return Project(tmp, use_ossssim=False).block("S", name)

    def test_attributed_to_needs_every_epoch_and_truncated_name(self):
        from survey_debias.grid_bias_run import attributed_to

        with short_tempdir() as tmp:
            block = self._block(tmp, "averylongblockname", epochs=2)
        self.assertEqual(block.detos_name, "averylongb")
        good = dict(flag=4, Survey=b"averylongb")
        self.assertTrue(attributed_to([good, dict(good, Survey="averylongb ")], block))
        self.assertFalse(attributed_to([good, dict(good, flag=3)], block))
        self.assertFalse(attributed_to([good, dict(good, Survey="other")], block))

    def test_attributed_to_survey_block_keys(self):
        from survey_debias.grid_bias_run import attributed_to

        with short_tempdir() as tmp:
            single = self._block(tmp, "averylongblockname")
        self.assertEqual(single.detos_keys(), ["S/averylongblockname"])
        good = dict(flag=4, Survey="S/averylongblockname")
        self.assertTrue(attributed_to([good], single))
        self.assertFalse(attributed_to([dict(good, Survey="S/other")], single))
        self.assertFalse(attributed_to([dict(good, Survey="T/averylongblockname")], single))
        with short_tempdir() as tmp:
            multi = self._block(tmp, "b1", epochs=2)
        self.assertEqual(multi.detos_keys(), ["epoch1/b1", "epoch2/b1"])
        rows = [dict(flag=4, Survey=b"epoch1/b1"), dict(flag=4, Survey="epoch2/b1 ")]
        self.assertTrue(attributed_to(rows, multi))
        self.assertFalse(attributed_to([rows[0], dict(rows[1], flag=2)], multi))

    def test_attributed_to_root_mode_single_row(self):
        from survey_debias.grid_bias_run import attributed_to

        with short_tempdir() as tmp:
            multi = self._block(tmp, "b1", epochs=3)
        self.assertEqual(multi.root_key, "S/b1")
        self.assertTrue(attributed_to([dict(flag=4, Survey=b"S/b1 ")], multi))
        self.assertFalse(attributed_to([dict(flag=3, Survey="S/b1")], multi))
        self.assertFalse(attributed_to([dict(flag=4, Survey="epoch1/b1")], multi))

    def test_attributed_to_per_epoch_k_of_n(self):
        from survey_debias.grid_bias_run import attributed_to

        with short_tempdir() as tmp:
            make_survey(Path(tmp), "S", ["0.2 0.2 10.0 0.0 2459000.5 1.0 500 b1.eff\n"],
                        {"b1": "r"}, epochs=3, detections_required=2)
            block = Project(tmp, use_ossssim=False).block("S", "b1")
        good = [dict(flag=4, Survey=f"epoch{i}/b1") for i in (1, 2, 3)]
        self.assertTrue(attributed_to(good, block))
        self.assertTrue(attributed_to([good[0], dict(flag=0, Survey=""), good[2]], block))
        self.assertFalse(attributed_to([good[0], dict(flag=0, Survey=""),
                                        dict(good[2], flag=2)], block))

    def test_cell_bias_counts_only_this_blocks_detections(self):
        from survey_debias.grid_bias_run import compute_cell_bias

        with short_tempdir() as tmp:
            block = self._block(tmp)
            sim = _StubSimulator(lambda k: "S/b1" if k % 2 == 0 else "S/elsewhere")
            bounds = {"a": (42.0, 44.0), "q": (38.0, 40.0),
                      "sin_ifree": (0.0, 0.1), "Hx": (7.0, 7.5)}
            est = compute_cell_bias(sim, block, bounds, seed=3, target=40,
                                    group="cold", max_aimed=200)
        self.assertEqual(est.n_detected, 40)
        self.assertEqual(est.n_drawn, 80)
        self.assertEqual(set(sim.comps), {"cold"})
        self.assertGreater(est.bias, 0.0)
        self.assertGreater(est.bias_se, 0.0)
        self.assertLess(est.bias_se, est.bias)

    def test_block_that_never_detects(self):
        from survey_debias.grid_bias_run import compute_cell_bias

        bounds = {"a": (42.0, 44.0), "q": (38.0, 40.0),
                  "sin_ifree": (0.0, 0.1), "Hx": (7.0, 7.5)}
        with short_tempdir() as tmp:
            block = self._block(tmp)
            sim = _StubSimulator(lambda k: "elsewhere")
            est = compute_cell_bias(sim, block, bounds, seed=3, target=10,
                                    must_detect=False, max_aimed=30)
            self.assertEqual((est.bias, est.n_detected, est.n_drawn), (0.0, 0, 30))
            with self.assertRaisesRegex(RuntimeError, "no simulated twin"):
                compute_cell_bias(sim, block, bounds, seed=3, target=10,
                                  must_detect=True, max_aimed=30)


class BiasFiles(unittest.TestCase):
    def test_round_trip(self):
        entries = {
            ("b1", "default", (38.0, 10.0, 7.25)): bias_files.BiasEstimate(1.5e-3, 4000, 100, 1.2e-4),
            ("b2", "cold", (38.0, 10.0, 7.25)): bias_files.BiasEstimate(2.0e-4, 2000, 0, 0.0),
            ("b3", "default", (40.0, 2.0, 7.75)): bias_files.BiasEstimate(3.0e-3, 100),
        }
        with short_tempdir() as tmp:
            path = Path(tmp) / "characterization" / "S" / bias_files.bias_file_name("model_ae")
            bias_files.save_bias_file(path, entries, "model_ae")
            self.assertEqual(path.name, "bias_grid_rih.csv")
            self.assertEqual(path.read_text().splitlines()[0],
                             "block,colour_group,r_bin,i_bin,h_bin,bias,n_drawn,n_detected,bias_se")
            back = bias_files.load_bias_file(path, "model_ae")
            project = bias_files.load_project_bias(Path(tmp) / "characterization", ["S"], "model_ae")
        self.assertEqual(set(back), set(entries))
        for key, est in entries.items():
            got = back[key]
            self.assertAlmostEqual(got.bias, est.bias)
            self.assertEqual(got.n_drawn, est.n_drawn)
            self.assertEqual(got.n_detected, est.n_detected)
            if est.bias_se is None:
                self.assertIsNone(got.bias_se)
            else:
                self.assertAlmostEqual(got.bias_se, est.bias_se)
        self.assertIn(("S", "b1", "default", (38.0, 10.0, 7.25)), project)

    def test_relative_error_fallbacks(self):
        est = bias_files.BiasEstimate
        self.assertAlmostEqual(est(2e-3, 10, 4, 5e-4).relative_error(), 0.25)
        self.assertAlmostEqual(est(2e-3, 10, 25).relative_error(), 0.2)
        self.assertEqual(est(2e-3, 10).relative_error(0.05), 0.05)
        self.assertIsNone(est(2e-3, 10).relative_error())

    def test_mc_bias_is_mean_and_standard_error(self):
        x = np.array([0.0, 0.0, 0.2, 0.4, 0.0, 0.1])
        bias, se = bias_files.mc_bias(len(x), x.sum(), (x * x).sum())
        self.assertAlmostEqual(bias, x.mean())
        self.assertAlmostEqual(se, x.std() / math.sqrt(len(x)))

    def test_union_sums_blocks_and_adds_errors_in_quadrature(self):
        from survey_debias.grid_bias_run import _union

        class B:
            def __init__(self, name):
                self.survey, self.name = "S", name

        cell = (38.0, 10.0, 7.25)
        entries = {
            ("S", "b1", "default", cell): bias_files.BiasEstimate(1e-3, 10, 5, 3e-4),
            ("S", "b2", "default", cell): bias_files.BiasEstimate(2e-3, 10, 5, 4e-4),
        }
        bias, se = _union(entries, [B("b1"), B("b2")], "default", cell)
        self.assertAlmostEqual(bias, 3e-3)
        self.assertAlmostEqual(se, 5e-4)
        entries[("S", "b2", "default", cell)].bias_se = None
        self.assertIsNone(_union(entries, [B("b1"), B("b2")], "default", cell)[1])

    def test_legacy_cache_is_cell_keyed(self):
        with short_tempdir() as tmp:
            path = Path(tmp) / "bias_grid.csv"
            path.write_text("a_bin,q_bin,si_bin,h_bin,bias,n_drawn\n44,40,0.1,7.5,0.002,500\n")
            legacy = bias_files.load_legacy_cache(path, "aq_grid")
            self.assertEqual(legacy[(44.0, 40.0, 0.1, 7.5)].bias, 0.002)
            path.write_text("block,a_bin,q_bin,si_bin,h_bin,bias,n_drawn\n")
            self.assertEqual(bias_files.load_legacy_cache(path, "aq_grid"), {})


if __name__ == "__main__":
    unittest.main()

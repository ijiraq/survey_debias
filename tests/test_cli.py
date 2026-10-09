"""Tests for survey loading and the survey_debias command. No Fortran required."""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

try:
    import survey_debias
except ImportError:  # not installed: run from a source checkout
    sys.path.insert(0, str(ROOT / "src"))
    import survey_debias
from survey_debias import cli, survey_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_fixture import make_jwst_project, make_survey, short_tempdir  # noqa: E402

DETECTIONS_HEADER = "name,a,e,i,d_bary,comp,m_f150w2\n"

MINIMAL_TOML = (
    '[survey]\n'
    'name = "Test"\n'
    'detections_relpath = "objects.csv"\n'
    'bias_method = "model_ae"\n'
    "\n"
    "[detection_columns]\n"
    'd_bary = "sun_dist"\n'
    'i = "incl"\n'
    'mag = "mag"\n'
    'name = "name"\n'
    "\n"
    "[detection_defaults]\n"
    'survey = "test"\n'
    'block = "test"\n'
)


class SurveyLoading(unittest.TestCase):
    def test_toml_and_python_examples_match(self):
        for toml_name, py_spec in (
            ("jwst_sample_a.toml", "jwst_sample_a.py:JWST_SAMPLE_A"),
            ("hst_n26.toml", "hst_n26.py:N26_HELIOSTACK"),
        ):
            from_toml = survey_debias.load_survey(str(EXAMPLES / toml_name))
            from_py = survey_debias.load_survey(str(EXAMPLES / py_spec))
            self.assertEqual(from_toml, from_py, toml_name)

    def test_python_module_with_single_survey_needs_no_name(self):
        survey = survey_debias.load_survey(str(EXAMPLES / "jwst_sample_a.py"))
        self.assertEqual(survey.bias_method, "model_ae")
        self.assertEqual(survey.model_band, "r")
        self.assertEqual(survey.detection_defaults["survey"], "JWST_A")

    def test_mag_column_belongs_in_detection_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.toml"
            path.write_text(MINIMAL_TOML.replace(
                'bias_method = "model_ae"\n',
                'bias_method = "model_ae"\nmag_column = "mag"\n',
            ))
            with self.assertRaisesRegex(ValueError, r"\[detection_columns\]"):
                survey_debias.load_survey(str(path))

    def test_old_geometry_fields_point_to_characterization(self):
        for field, value in (
            ("mag_color_offset", "1.0"),
            ("field_ra_deg", "209.3875"),
            ("eff_file", '"test.eff"'),
            ("epoch_jd", "[2459969.3]"),
            ("observer_csv", '"JWST.csv"'),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "old.toml"
                path.write_text(MINIMAL_TOML.replace(
                    'name = "Test"\n', f'name = "Test"\n{field} = {value}\n'
                ))
                with self.assertRaisesRegex(ValueError, "no longer belong") as cm:
                    survey_debias.load_survey(str(path))
                text = str(cm.exception)
                self.assertIn(field, text)
                self.assertTrue("pointings.list" in text and "colour.toml" in text)

    def test_survey_and_block_need_a_column_or_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.toml"
            path.write_text(MINIMAL_TOML.split("[detection_defaults]")[0])
            with self.assertRaisesRegex(ValueError, "survey"):
                survey_debias.load_survey(str(path))
            path.write_text(MINIMAL_TOML.replace('block = "test"\n', ""))
            with self.assertRaisesRegex(ValueError, "block"):
                survey_debias.load_survey(str(path))

    def test_renamed_distance_and_inclination_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            survey_path = root / "test.toml"
            survey_path.write_text(MINIMAL_TOML)
            (root / "objects.csv").write_text(
                "name,mag,sun_dist,obs_dist,incl,H\n"
                "A,26.0,46.2,45.0,1.85,8.1\n"
            )
            survey = survey_debias.load_survey(str(survey_path))
            from survey_debias.grid_bias import load_detections
            rows = load_detections(root / "objects.csv", survey)
        self.assertEqual(survey.detection_columns["d_bary"], "sun_dist")
        self.assertEqual(survey.detection_columns["i"], "incl")
        self.assertNotIn("mag_column", survey_config._FIELDS)
        self.assertAlmostEqual(rows[0]["d_bary"], 46.2)
        self.assertAlmostEqual(rows[0]["i"], 1.85)
        self.assertAlmostEqual(rows[0]["mag"], 26.0)
        self.assertEqual(rows[0]["name"], "A")
        self.assertEqual(rows[0]["survey"], "test")
        self.assertEqual(rows[0]["block"], "test")
        self.assertEqual(rows[0]["_source"]["H"], "8.1")
        self.assertEqual(rows[0]["_source"]["obs_dist"], "45.0")

    def test_unknown_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.toml"
            text = (EXAMPLES / "hst_n26.toml").read_text()
            text = text.replace(
                'bias_method = "aq_grid"',
                'bias_method = "aq_grid"\nmosaic_area = 0.02',
            )
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "unknown GridSurvey field"):
                survey_debias.load_survey(str(path))

    def test_missing_field_and_bad_method_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing required"):
            survey_config.survey_from_mapping({"name": "x"})
        base = survey_config.load_survey_toml(EXAMPLES / "hst_n26.toml")
        values = {f: getattr(base, f) for f in survey_config._FIELDS}
        with self.assertRaisesRegex(ValueError, "bias_method"):
            survey_config.survey_from_mapping({**values, "bias_method": "box"})
        with self.assertRaisesRegex(ValueError, "model_band"):
            survey_config.survey_from_mapping({**values, "model_band": "F150W2"})


class CommandLine(unittest.TestCase):
    def _run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_lists_model_ae_cells_without_simulator(self):
        with short_tempdir() as tmp:
            root = make_jwst_project(Path(tmp))
            (root / "data").mkdir()
            (root / "data" / "jwst_sampleA.csv").write_text(
                DETECTIONS_HEADER
                + "JPB04,43.7,0.06,1.85,46.20,cold,26.0\n"
                + "JPB07,45.7,0.05,1.40,46.20,cold,26.0\n"
                + "JPB19,41.6,0.01,26.7,41.00,hot,25.5\n"
            )
            code, out, err = self._run(
                "--survey", str(EXAMPLES / "jwst_sample_a.toml"),
                "--root", str(root), "--dry-run",
            )
        self.assertEqual(code, 0, err)
        self.assertIn("3 objects, 2 cells", out)
        self.assertIn("[cold]", out)
        self.assertIn("JPB04, JPB07", out)
        self.assertIn("JWST_A/sampleA: filter W, 3 epoch(s), 1 tile(s)", out)
        self.assertIn("JWST_A/sampleA: 3 detection(s)", out)
        self.assertIn("orbit model:", out)
        self.assertIn("MISSING", out)  # no orbit model under the temp root
        self.assertNotIn("MISSING or invalid", out)
        self.assertNotIn("survey_debias.grid_bias_run", sys.modules)

    def test_dry_run_reports_missing_characterization(self):
        with short_tempdir() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data" / "jwst_sampleA.csv").write_text(
                DETECTIONS_HEADER + "JPB04,43.7,0.06,1.85,46.20,cold,26.0\n"
            )
            code, out, err = self._run(
                "--survey", str(EXAMPLES / "jwst_sample_a.toml"),
                "--root", str(root), "--dry-run",
            )
        self.assertEqual(code, 0, err)
        self.assertIn("characterization:", out)
        self.assertIn("MISSING or invalid", out)
        self.assertIn("JWST_A", out)

    def test_dry_run_per_block_counts_blocks(self):
        with short_tempdir() as tmp:
            root = Path(tmp)
            make_survey(root, "S1", [
                "0.2 0.2 10.0 0.0 2459000.5 1.0 500 b1.eff\n",
                "0.2 0.2 30.0 0.0 2459000.5 1.0 500 b2.eff\n",
            ], {"b1": "r", "b2": "r"})
            survey = root / "survey.toml"
            survey.write_text(
                '[survey]\nname = "Multi"\n'
                'detections_relpath = "objects.csv"\nbias_method = "model_ae"\n\n'
                '[detection_columns]\nname = "name"\nmag = "mag"\n'
                'd_bary = "d"\ni = "i"\nsurvey = "survey"\nblock = "block"\n'
            )
            (root / "objects.csv").write_text(
                "name,mag,d,i,survey,block\n"
                "A,24.0,42.0,2.0,S1,b1\n"
                "B,24.0,42.0,2.0,S1,b2\n"
            )
            code, out, err = self._run(
                "--survey", str(survey), "--root", str(root), "--dry-run",
            )
            self.assertEqual(code, 0, err)
            self.assertIn("1 cells, 2 block-cell bias estimates", out)
            self.assertIn("bias=union over blocks", out)
            code, out, err = self._run(
                "--survey", str(survey), "--root", str(root), "--dry-run",
                "--per-block-bias",
            )
        self.assertEqual(code, 0, err)
        self.assertIn("bias=own block", out)
        self.assertIn("S1/b1: 1 detection(s)", out)
        self.assertIn("S1/b2: 1 detection(s)", out)

    def test_bad_survey_exits_with_error(self):
        code, _, err = self._run("--survey", "/no/such/survey.toml", "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("survey_debias:", err)

    def test_dry_run_model_ae_without_orbital_elements(self):
        with short_tempdir() as tmp:
            root = make_jwst_project(Path(tmp))
            survey = root / "survey.toml"
            survey.write_text(_with_detection_columns(
                'mag = "m_f150w2"\nd_bary = "d_bary"\ni = "i"\n'
            ))
            (root / "data").mkdir()
            (root / "data" / "jwst_sampleA.csv").write_text(
                "i,d_bary,m_f150w2\n"
                "1.85,46.20,26.0\n"
                "1.40,46.20,26.0\n"
            )
            code, out, err = self._run(
                "--survey", str(survey),
                "--root", str(root), "--dry-run",
            )
        self.assertEqual(code, 0, err)
        self.assertIn("2 objects, 1 cells", out)
        self.assertIn("file columns: i, d_bary, m_f150w2", out)
        self.assertIn("derived columns:", out)
        self.assertIn("Hx", out)
        self.assertIn("colour", out)
        self.assertIn("bias_results.csv", out)

    def test_dry_run_applies_detection_columns(self):
        with short_tempdir() as tmp:
            root = make_jwst_project(Path(tmp))
            survey = root / "survey.toml"
            survey.write_text(_with_detection_columns(
                'mag = "m_f150w2"\n'
                'name = "object"\n'
                'i = "incl"\n'
                'd_bary = "dist"\n'
            ))
            (root / "data").mkdir()
            (root / "data" / "jwst_sampleA.csv").write_text(
                "object,incl,dist,m_f150w2\n"
                "JPB04,1.85,46.20,26.0\n"
            )
            code, out, err = self._run(
                "--survey", str(survey), "--root", str(root), "--dry-run",
            )
        self.assertEqual(code, 0, err)
        self.assertIn("JPB04", out)
        self.assertIn("d_bary=dist", out)
        self.assertIn("i=incl", out)

    def test_missing_detection_column_exits_with_error(self):
        with short_tempdir() as tmp:
            root = make_jwst_project(Path(tmp))
            (root / "data").mkdir()
            (root / "data" / "jwst_sampleA.csv").write_text("name,mag\nJ1,26\n")
            code, _, err = self._run(
                "--survey", str(EXAMPLES / "jwst_sample_a.toml"),
                "--root", str(root), "--dry-run",
            )
        self.assertEqual(code, 2)
        self.assertIn("not in the file", err)
        self.assertIn("survey_debias:", err)


def _with_detection_columns(body: str) -> str:
    text = (EXAMPLES / "jwst_sample_a.toml").read_text()
    head = text.split("\n[detection_columns]", 1)[0].rstrip()
    return (head + "\n\n[detection_columns]\n" + body.strip() + "\n\n"
            '[detection_defaults]\nsurvey = "JWST_A"\nblock = "sampleA"\n')


if __name__ == "__main__":
    unittest.main()

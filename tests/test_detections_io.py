"""Tests for reading OSSOS/CFEPS .detections files."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from survey_debias import detections_io  # noqa: E402

EFF = """\
rates= 0.0 20.0
function= single
single_param= 0.9 24.0 0.3
mag_lim= 24.5
filter= {filt}
"""

HEADER = ("# cl p  j   k sh     object  mag e_mag Filt Hsur dist  e_dist    Nobs   time  "
          "av_xres av_yres max_x  max_y       a          e_a        e        e_e      i  "
          "    e_i      Omega e_Omega    omega   e_omega      tperi    e_tperi   RAdeg   "
          "DEdeg      JD        rate   MPC\n")


def line(cl, p, sh, name, i, ra, dec, jd):
    return (f"{cl} {p} -1 -1 {sh} {name} 23.5 0.1 g 7.0 42.0 0.01 20 2.0 0.1 0.1 0.2 0.2 "
            f"44.0 0.01 0.05 0.001 {i} 0.01 10.0 0.1 20.0 0.1 53000.0 10.0 "
            f"{ra} {dec} {jd} 3.0 X\n")


def write_survey(root: Path) -> Path:
    d = root / "characterization" / "S"
    d.mkdir(parents=True)
    (d / "pointings.list").write_text(
        "1.0 1.0 10.0 0.0 2453000.5 0.9 500 L3f-smooth.eff\n"
        "1.0 1.0 40.0 5.0 2453100.5 0.9 500 Q9a.eff\n"
        "1.0 1.0 41.5 5.0 2453101.5 0.9 500 Q9b.eff\n"
    )
    for name in ("L3f-smooth", "Q9a", "Q9b"):
        (d / f"{name}.eff").write_text(EFF.format(filt="g"))
    (d / "S.detections").write_text(
        "# test release\n" + HEADER
        + line("cla", "m", "S", "L3f01", 2.0, 10.1, 0.1, 2453000.6)
        + line("cla", "m", "I", "l3f02nt", 7.0, 10.2, 0.2, 2453000.6)
        + line("res", "-", "S", "Q9x01", 1.0, 41.4, 5.1, 2453101.6)
        + line("cla", "o", "S", "Q9x02", 1.0, 40.2, 4.9, 2453100.6)
    )
    return root / "characterization"


class Reader(unittest.TestCase):
    def test_columns_and_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = detections_io.read_detections_file(write_survey(Path(tmp)) / "S" / "S.detections")
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["object"], "L3f01")
        self.assertEqual(rows[0]["Filt"], "g")
        self.assertAlmostEqual(rows[0]["dist"], 42.0)

    def test_blocks_from_names_and_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = detections_io.convert(write_survey(Path(tmp)), "S", detections_io.ColdCut())
        self.assertEqual([r["block"] for r in rows], ["L3f-smooth", "L3f-smooth", "Q9b", "Q9a"])
        self.assertEqual([r["comp"] for r in rows], ["cold", "other", "other", "other"])
        self.assertTrue(all(r["filter"] == "g" for r in rows))

    def test_explicit_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocks = detections_io.load_survey_blocks(write_survey(Path(tmp)), "S", False)
        self.assertEqual(detections_io.assign_block("zz01", blocks, {"zz": "Q9a"}), "Q9a")
        with self.assertRaises(ValueError):
            detections_io.assign_block("zz01", blocks)

    def test_cut(self):
        cut = detections_io.ColdCut(parts=("m", "o"), secure_only=True)
        self.assertTrue(cut({"cl": "cla", "p": "o", "sh": "S", "i": 1.0}))
        self.assertFalse(cut({"cl": "cla", "p": "m", "sh": "I", "i": 1.0}))
        self.assertFalse(cut({"cl": "cla", "p": "m", "sh": "S", "i": 5.0}))

    def test_free_inclination_cut(self):
        cut = detections_io.ColdCut(i_max=None, ifree_max=4.0)
        row = {"cl": "cla", "p": "m", "sh": "S", "object": "nothing", "MPC": "",
               "a": 44.0, "e": 0.05, "i": 4.6, "Omega": 90.0, "omega": 20.0}
        ifree, source = detections_io.release_ifree(dict(row))
        self.assertEqual(source, "huang2022_grid")
        self.assertLess(ifree, 4.0)            # i = 4.6 near the forcing node is cold
        self.assertTrue(cut(dict(row)))
        self.assertFalse(cut({**row, "Omega": 270.0}))
        self.assertEqual(cut.describe(), "cl in ('cla',), p in ('m',), i_free < 4 deg")

    def test_free_inclination_from_huang_table(self):
        ifree, source = detections_io.release_ifree({"object": "x", "MPC": "15760",
                                                     "a": 44.0, "e": 0.0, "i": 9.0,
                                                     "Omega": 0.0, "omega": 0.0})
        self.assertEqual(source, "huang2022_table")
        self.assertAlmostEqual(ifree, 2.7091)

    def test_duplicates_are_kept_in_the_priority_survey(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Duplicates"
            path.write_text("# OSSOS  CFEPS\no3e20PD   L4h05PD\no3l13PD L3s05\n")
            groups = detections_io.read_duplicates(path)
        names = {"OSSOS_v11": ["o3e20PD", "o5x"], "CFEPS_g": ["L4h05PD", "L3s05"]}
        drop = detections_io.drop_duplicates(names, groups, ["OSSOS_v11", "CFEPS_g"])
        self.assertEqual(drop, {"OSSOS_v11": set(), "CFEPS_g": {"L4h05PD"}})
        drop = detections_io.drop_duplicates(names, groups, ["CFEPS_g", "OSSOS_v11"])
        self.assertEqual(drop, {"OSSOS_v11": {"o3e20PD"}, "CFEPS_g": set()})

    def test_command_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_survey(Path(tmp))
            out = Path(tmp) / "out.csv"
            detections_io.main(["--root", tmp, "--survey", "S", "--out", str(out)])
            lines = [x for x in out.read_text().splitlines() if not x.startswith("#")]
            rows = list(csv.DictReader(lines))
        self.assertEqual(rows[0]["survey"], "S")
        self.assertEqual(sum(r["comp"] == "cold" for r in rows), 1)


if __name__ == "__main__":
    unittest.main()

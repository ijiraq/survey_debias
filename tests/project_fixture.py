"""Small characterization trees for tests (no Fortran)."""
from __future__ import annotations

import tempfile
from pathlib import Path


def short_tempdir() -> tempfile.TemporaryDirectory:
    """A temp directory short enough for OSSSSim's 100-character paths."""
    base = "/tmp" if Path("/tmp").is_dir() else None
    return tempfile.TemporaryDirectory(dir=base)


EFF_TEMPLATE = """\
# test efficiency
rate_cut= 0.03 8.66 0.0 180.0
track_frac= 1.0 29.92 -0.61
filter= {filter}
rates= 0.00 20.00
function= single
single_param= 0.95 {m0} 0.20
mag_lim= {mag_lim}
"""

COLOUR_TOML = """\
[default]
"g-g" = 0.0
"r-g" = -0.7
"H-g" = -1.7
"t-g" = -0.4

[cold]
"g-g" = 0.0
"r-g" = -0.9
"H-g" = -1.9
"t-g" = -0.5
"""

JWST_LINE = "0.2236068 0.2236068 209.3875 -10.865278 {jd} 1.0 500 {block}.eff\n"


def write_eff(path: Path, filter: str = "H", m0: float = 29.0, mag_lim: float = 30.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EFF_TEMPLATE.format(filter=filter, m0=m0, mag_lim=mag_lim))
    return path


def write_survey_conf(base: Path, detections_required: int) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    path = base / "survey.conf"
    path.write_text(f"# test survey\ndetections_required = {detections_required}\n")
    return path


def make_survey(root: Path, survey: str, lines: list[str], effs: dict,
                epochs: int = 0, detections_required: int | None = None) -> Path:
    """``characterization/{survey}`` with ``lines`` and ``{block: filter}`` .eff files.

    ``epochs > 0`` writes the same layout into ``epoch1..epochN`` and a
    ``survey.conf`` requiring ``detections_required`` epochs (default all).
    """
    base = Path(root) / "characterization" / survey
    dirs = [base] if epochs == 0 else [base / f"epoch{i}" for i in range(1, epochs + 1)]
    if epochs:
        write_survey_conf(base, detections_required or epochs)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / "pointings.list").write_text("# test pointings\n" + "".join(lines))
        for block, spec in effs.items():
            filt, m0 = spec if isinstance(spec, tuple) else (spec, 29.0)
            write_eff(d / f"{block}.eff", filt, m0)
    return base


def make_jwst_project(root: Path, colour: bool = True) -> Path:
    """The examples/jwst_sample_a layout: jwst_sample_a/epoch{1,2,3}/, block jwst_sample_a."""
    root = Path(root)
    write_survey_conf(root / "characterization" / "jwst_sample_a", 3)
    for i, jd in enumerate((2459969.32118, 2459973.96785, 2459979.90854), start=1):
        d = root / "characterization" / "jwst_sample_a" / f"epoch{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "pointings.list").write_text(JWST_LINE.format(jd=jd, block="jwst_sample_a"))
        write_eff(d / "jwst_sample_a.eff", "H")
    if colour:
        (root / "colour.toml").write_text(COLOUR_TOML)
    return root

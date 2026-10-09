"""``survey_debias`` command: Horvitz–Thompson grid debiasing of a survey.

The project is supplied with ``--survey`` (a TOML file or a Python
``GridSurvey``; see :mod:`survey_debias.survey_config`); geometry comes from
``characterization/{survey}/pointings.list``. The simulator is imported only
for a real run, so ``--dry-run`` lists surveys, blocks, tiles, and cells
without starting OSSSSim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .grid_bias import TARGET_DETECTIONS, default_orbit_model_path, load_detections
from .survey_config import load_survey


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="survey_debias",
        description="Grid-cell Horvitz–Thompson debiasing of a pencil-beam TNO "
                    "survey with the OSSOS Survey Simulator (follows Kavelaars "
                    "et al. 2022).",
    )
    parser.add_argument(
        "--survey", required=True,
        help="Survey definition: path/to/survey.toml, path/to/survey.py:NAME, "
             "or package.module:NAME.",
    )
    parser.add_argument(
        "--root", default=".",
        help="Survey directory with the detections, characterization, and "
             "models; debiasing outputs are written here (default: current "
             "directory).",
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_DETECTIONS,
        help=f"Detections required per cell (default: {TARGET_DETECTIONS}).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--model", default=None,
        help="Orbit model file or directory for p(a,e|r,i) (model_ae only). "
             "Accepts OSSOS Models or CFEPS L7 catalog file formats. Default: "
             "$SURVEY_DEBIAS_MODEL, else <root>/Models/OSSOS.",
    )
    parser.add_argument(
        "--per-block-bias", action="store_true",
        help="Give each detection its own block's bias instead of the union "
             "over every block of the project.",
    )
    parser.add_argument(
        "--check-plots-dir", default=None,
        help="Directory for sampled-vs-detected check plots "
             "(default: <root>/check_plots, one {survey}/{block}/ per block).",
    )
    parser.add_argument(
        "--no-check-plots", action="store_true",
        help="Skip writing check plots.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load the survey and detections, list the bias cells, and exit "
             "without running the simulator.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _describe_project(project, names) -> tuple[list, list[str]]:
    """Print surveys, blocks, and tiles; return (blocks, problems)."""
    blocks, problems = [], []
    for s in names:
        try:
            found = project.blocks(s)
        except (OSError, ValueError) as ex:
            problems.append(str(ex))
            print(f"  {s}: MISSING or invalid ({ex})")
            continue
        for b in found.values():
            blocks.append(b)
            print(f"  {s}/{b.name}: filter {b.filter}, {b.n_epochs} epoch(s), "
                  f"{len(b.tiles[0])} tile(s), {b.area_deg2:.4f} deg² "
                  f"(sampled {b.aim_area_deg2:.4f} deg²)")
            for t in b.tiles[0]:
                obs = Path(b.char_dirs[0]) / t.observer
                kind = "file" if obs.is_file() else "MPC code"
                print(f"    line {t.line} {t.shape} {t.width_deg:.4f}x{t.height_deg:.4f} deg "
                      f"at {t.ra_deg:.5f},{t.dec_deg:.5f} JD {t.jd:.5f} "
                      f"observer {t.observer} ({kind})")
    return blocks, problems


def dry_run(survey, root: Path, model_path: Path | None,
            per_block_bias: bool = False) -> int:
    from .pointings import Project

    project = Project.for_survey(root, survey)
    names = project.survey_names(survey.surveys)
    print(f"project: {survey.name} (bias_method={survey.bias_method}, "
          f"model_band={survey.model_band}, "
          f"bias={'own block' if per_block_bias else 'union over blocks'})")
    print(f"characterization: {project.char_root} "
          f"({'found' if project.char_root.is_dir() else 'MISSING'}); "
          f"surveys: {', '.join(names) or '(none found)'}")
    blocks, problems = _describe_project(project, names)
    try:
        print(project.colours.note)
        project.colours.validate({b.filter for b in blocks})
    except (OSError, ValueError) as ex:
        problems.append(str(ex))
        print(f"colours: {ex}")
    detections_path = root / survey.detections_relpath

    def colour_for(s, b, comp):
        try:
            return project.colour_for(s, b, comp)
        except (OSError, ValueError) as ex:
            if str(ex) not in problems:
                problems.append(str(ex))
            return "", 0.0, "default"

    detections = load_detections(detections_path, survey, colour_for=colour_for)
    cells = sorted({(d["colour_group"], d["cell"]) for d in detections})
    n_est = (len({(d["survey"], d["block"], d["colour_group"], d["cell"]) for d in detections})
             if per_block_bias else len(cells) * len(blocks))
    print(f"detections: {detections_path} ({len(detections)} objects, {len(cells)} cells, "
          f"{n_est} block-cell bias estimates)")
    counts = {}
    for d in detections:
        counts[(d["survey"], d["block"])] = counts.get((d["survey"], d["block"]), 0) + 1
    for (s, b), n in sorted(counts.items()):
        print(f"  {s}/{b}: {n} detection(s)")
    print(f"file columns: {', '.join(detections.input_columns)}")
    print("detection columns: " + ", ".join(
        f"{name}={column}" for name, column in detections.columns_used.items()
    ))
    print(
        "derived columns: "
        + ", ".join([*detections.computed_columns, "block_bias", "bias", "bias_se"])
    )
    for note in detections.notes:
        print(f"  derived: {note}")
    print(f"results: {root / survey.results_name}")
    for group, key in cells:
        members = [str(d["name"]) for d in detections
                   if d["cell"] == key and d["colour_group"] == group]
        print(f"  [{group}] {key}: {', '.join(members)}")
    if survey.bias_method == "model_ae":
        model = Path(model_path or default_orbit_model_path(root))
        print(f"orbit model: {model} ({'found' if model.exists() else 'MISSING'})")
    for problem in problems:
        print(f"MISSING or invalid: {problem}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        survey = load_survey(args.survey)
    except (OSError, ValueError) as ex:
        print(f"survey_debias: {ex}", file=sys.stderr)
        return 2
    root = Path(args.root)
    model_path = Path(args.model) if args.model else None
    if model_path is not None and survey.bias_method != "model_ae":
        print("survey_debias: --model is ignored for bias_method="
              f"{survey.bias_method}", file=sys.stderr)
    try:
        if args.dry_run:
            return dry_run(survey, root, model_path, args.per_block_bias)
        from .grid_bias_run import run_grid_bias

        run_grid_bias(
            survey, root, target=args.target, seed=args.seed,
            check_plots_dir=Path(args.check_plots_dir) if args.check_plots_dir else None,
            no_check_plots=args.no_check_plots,
            model_path=model_path,
            per_block_bias=args.per_block_bias,
        )
    except (OSError, ValueError) as ex:
        print(f"survey_debias: {ex}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

# survey_debias

Horvitz–Thompson debiasing of pencil-beam trans-Neptunian object (TNO)
surveys using the OSSOS Survey Simulator (`ossssim`).

Each detection receives a weight \(1/P\), where \(P\) is the probability that
an object like it would have been detected by the survey. Summing the weights
estimates the intrinsic population represented by the detected sample.

## Requirements

- Python ≥ 3.10
- `numpy`, `astropy`, `matplotlib` (check plots only)
- `ossssim`, the OSSOS SurveySimulator Python package with its compiled
  Fortran library
- For `model_ae` surveys: the OSSOS Models 1.0 ModelUsed tables (see
  [OSSOS model prior](#ossos-model-prior-model_ae-only))

## How the bias is computed

Each detection is assigned to a cell. For each cell the runner plants
synthetic objects whose orbits pass through the survey field, runs them
through `ossssim` at every survey epoch, and keeps going until `--target`
detections accumulate (default 5000). The cell's bias is

    bias = P(detected | in field) × P(orbit crosses field)

The first factor comes from the simulator (efficiency, rate cuts, epochs).
The second is the geometric chance that an orbit with that inclination passes
through the mosaic at the field's ecliptic latitude.

Two cell definitions are supported, selected by `GridSurvey.bias_method`:

| Method     | Use when                          | Cell                         | Synthetic (a, e) drawn from        |
|------------|-----------------------------------|------------------------------|------------------------------------|
| `aq_grid`  | the orbit is well determined      | a, q, sin i_free, H          | uniform within the cell            |
| `model_ae` | short arc; only r, i, H trusted   | r (1 au), i (1°), H (0.1)    | OSSOS model objects near (r, i)    |

`model_ae` keeps every dynamical population in the model (Classical,
Scattering, resonant, …) in its model proportions, because a short-arc
detection's population is unknown. The resulting bias therefore depends on
the orbit model used as the prior.

## Surveys

A survey is a `GridSurvey` in `grid_bias.py`: field centre, mosaic size,
epoch Julian dates, magnitude column and its offset to OSSOS \(r\), observer
position file, efficiency file, rate cuts, epoch directory layout, and
`bias_method`.

Defined now:

- `JWST_SAMPLE_A`: JWST GO 1568 (Eduardo et al. 2026); `model_ae`; detection
  requires flag ≥ 4 at all three epochs.
- `N26_HELIOSTACK`: HST GO-9433 reanalysis (Napier et al. 2026); `aq_grid`;
  single 15-day heliostack.

## Survey directory layout

The runner reads and writes inside a survey root passed as `--root`:

    <root>/
      data/<detections>.csv           # GridSurvey.detections_relpath
      characterization/
        pointings.template
        <observer_csv>                # ICRF observer positions (e.g. JPL Horizons)
        <eff_file>                    # detection efficiency
        epoch1/ epoch2/ ...           # epoch_layout="subdir" only

These files are survey-specific and are not part of this repository.

The detections CSV needs the columns `name, a, e, i, d_bary, comp` plus the
survey's magnitude column (`m_f150w2` for JWST, `m_stmag` for N26). An
`ifree` column is used when present. For `model_ae`, the catalog `a` and `e`
are only carried into the output file; they do not define the cell.

`pointings.template` is a Python format string using `{epoch} {jd} {ra} {dec}
{side} {width} {height} {fill} {observer_csv} {eff_file}`. It is rendered to
`pointings.list` at run time, so that generated file does not need to be
versioned.

## OSSOS model prior (`model_ae` only)

The full tables are not stored in this repository. Download them into
`Models/OSSOS/` (gitignored):

    mkdir -p Models/OSSOS
    for c in Classical Detached Inner Plutinos Scattering Twotinos; do
      curl -fsSL -o Models/OSSOS/$c-ModelUsed.dat \
        https://raw.githubusercontent.com/OSSOS/OSSOS_Models/$c/$c/ModelUsed-check-8.66.dat
    done

These are the OSSOS 1.0 nominal populations calibrated to the OSSOS++ sample
(Bannister et al. 2018, ApJS 236, 18). Each row is tagged with its population
from the filename.

To use another model, pass `--model PATH` (a single file or a directory).
Legacy L7-style files also load. `tests/data/OSSOS/` contains a short excerpt
for unit tests only; it is not a usable prior.

## Running

There is no installed command yet. Write a small wrapper for each survey:

```python
# run_jwst.py
from pathlib import Path

from grid_bias import JWST_SAMPLE_A
from grid_bias_run import main

if __name__ == "__main__":
    main(survey=JWST_SAMPLE_A, default_root=Path("/path/to/jwst_survey"))
```

Then run it with this repository on `PYTHONPATH`:

    PYTHONPATH=/path/to/survey_debias python run_jwst.py --root /path/to/jwst_survey

Options:

- `--root`: survey root directory
- `--target`: detections required per cell (default 5000)
- `--seed`: random seed (default 42)
- `--check-plots-dir`: where to write check plots (default `<root>/check_plots`)
- `--no-check-plots`: skip check plots
- `--model`: orbit model file or directory (`model_ae` only)

## Outputs

Written into `<root>`:

- `bias_grid.csv` (`aq_grid`) or `bias_grid_rih.csv` (`model_ae`): one bias
  per cell. This doubles as a cache; re-running skips finished cells. Delete
  it to recompute.
- `<detections_full_name>`: OSSOS/CFEPS-format detections file. Columns
  through `MPC` match `CFEPS.detections`; six columns follow:
  - `ifree`, `Omfree`, `omfree`: free elements relative to the Laplace plane
    (`Omfree` and `omfree` are 0 when unknown)
  - `Hx`: absolute magnitude \(H_r\) used for the cell
  - `comp`: component from the detections CSV
  - `bias`: \(P\) for the detection's cell
- `check_plots/`: sampled versus detected RA/Dec and orbital-element
  distributions, one set per detection plus one combined (`all`).

The run ends by printing `sum 1/bias`, the debiased count of objects like
those detected. It covers only cells that contain detections and is not an
extrapolation to unsampled parts of (r, i, H) or (a, q, i, H).

## Adding a survey

1. Define a new `GridSurvey` in `grid_bias.py` and choose `bias_method`.
2. Prepare the survey directory described above.
3. Call `main(survey=YOUR_SURVEY)` from a wrapper.

## Tests

    python -m unittest test_grid_bias -v

Tests that need survey files not in this repository are skipped.

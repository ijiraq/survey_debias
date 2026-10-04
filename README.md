# survey_debias

Horvitz–Thompson debiasing of pencil-beam trans-Neptunian object (TNO)
surveys using the OSSOS Survey Simulator (`ossssim`).

Each detection receives a weight $1/P$, where $P$ is the probability that an
object like it would have been detected by the survey. Summing the weights
estimates the total population the survey is capable of detecting; see
[Population estimates](#population-estimates-from-the-weights) for why, and
for the parts of the population a survey can never see.

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

## Population estimates from the weights

### Why $\hat N = \sum 1/P$ estimates the total population

Suppose the true population has $N$ objects and object $k$ is detected with
probability $P_k$. Summing $1/P_k$ over the objects that were detected gives

$$
\mathbb{E}\Big[\sum_{k\,\in\,\text{detected}} \frac{1}{P_k}\Big]
= \sum_{k\,\in\,\text{all}} P_k \cdot \frac{1}{P_k} = N .
$$

This is the Horvitz–Thompson estimator. Two consequences that are easy to get
wrong:

- $P$ is only needed for the objects actually detected. Parts of phase space
  with no detections are not ignored: on average they are represented by the
  detections, each of which stands in for $1/P_k$ similar objects.
- The same weights can be used inside a fit. For a luminosity function,
  maximising the weighted log-likelihood $\sum_k w_k \log f(H_k \mid \theta)$
  with $w_k = 1/P_k$ gives a consistent estimate of the shape, and
  $\sum_k w_k$ over $H < H_{\lim}$ gives its normalisation: the number of
  objects brighter than $H_{\lim}$.

### What the estimate cannot include

The derivation requires $P_k > 0$ for every object in the population being
estimated. Objects the survey could never detect have $P = 0$; they never
appear in the sum, and no reweighting brings them back. For a pencil-beam
survey these blind regions include:

- **Inclination below the field's ecliptic latitude** ($i < |\beta|$): such
  orbits never reach the field, so the geometric probability is zero.
- **Too faint** (apparent magnitude beyond the efficiency limit): at a given
  distance this sets an $H$ limit, and that limit is brighter for more
  distant objects.
- **Sky motion outside the rate cuts**: in practice, objects too distant or
  too nearby for the search rates.

Orbits that merely happen to be elsewhere during the survey are not a blind
region: any orbit with $i > |\beta|$ has some chance of crossing the field,
and that chance is already in the geometric factor of $P$.

So define the estimated population to match what the survey can see (for
example, "$H_r < H_{\lim}$ and $i > |\beta|$ at the distances sampled"), or
extrapolate into the blind regions with an explicit model and say so.

### Other cautions

- **Variance.** With few detections, a single object with small $P$ can
  dominate $\hat N$ and the fitted luminosity function. The estimator is
  unbiased on average but can be very noisy. Bootstrap over detections, or
  use a variance estimate that accounts for the weights; a plain weighted
  likelihood understates the uncertainty because it treats $\sum w_k$ as the
  number of independent detections.
- **Model dependence (`model_ae`).** $P$ is averaged over the orbit model's
  $p(a, e \mid r, i)$, so $\hat N$ and the luminosity-function normalisation
  assume that model's mix of orbits at each $(r, i)$.
- **Cell averaging.** $P$ is the average over a cell. This is accurate only
  if $P$ varies little across the cell, which is why the $H$ step is small
  (0.1 mag).

## Surveys

A survey is a `GridSurvey` (from `grid_bias.py`): field centre, mosaic size,
epoch Julian dates, magnitude column and its offset to OSSOS $r$, observer
position file, efficiency file, rate cuts, epoch directory layout, and
`bias_method`.

This library defines no surveys. Each survey is defined in the project that
holds its characterization files and passed to the runner as `survey=`.
`examples/` has one script per survey to seed those projects:

- `examples/jwst_sample_a.py`: JWST GO 1568 Sample A (Eduardo et al. 2026);
  `model_ae`; detection requires flag ≥ 4 at all three epochs. Seed for
  `jwst-tno-followup`.
- `examples/hst_n26.py`: HST GO-9433 reanalysis (Napier et al. 2026);
  `aq_grid`; single 15-day heliostack. Seed for `hst-tno-followup`.

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

There is no installed command yet. Each survey project keeps a small script
that defines its `GridSurvey` and calls `main`. Copy one of the examples as a
starting point:

```python
# run_debias.py (in the survey project)
from pathlib import Path

from grid_bias import GridSurvey
from grid_bias_run import main

MY_SURVEY = GridSurvey(
    name="My survey",
    field_ra_deg=..., field_dec_deg=...,
    mosaic_width_deg=..., mosaic_height_deg=...,
    epoch_jd=(...,),
    mag_color_offset=..., mag_column="...",
    observer_csv="...", eff_file="...",
    bias_method="aq_grid",  # or "model_ae"
)

if __name__ == "__main__":
    main(survey=MY_SURVEY, default_root=Path.cwd())
```

Then run it with this repository on `PYTHONPATH`:

    PYTHONPATH=/path/to/survey_debias python run_debias.py --root /path/to/survey

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
  - `Hx`: absolute magnitude $H_r$ used for the cell
  - `comp`: component from the detections CSV
  - `bias`: $P$ for the detection's cell
- `check_plots/`: sampled versus detected RA/Dec and orbital-element
  distributions, one set per detection plus one combined (`all`).

The run ends by printing `sum 1/bias`, the Horvitz–Thompson estimate of the
population the survey can detect. It excludes regions where the survey has
zero sensitivity; see
[What the estimate cannot include](#what-the-estimate-cannot-include).

## Adding a survey

1. In the survey's own project, copy an example from `examples/`.
2. Edit the `GridSurvey` values and choose `bias_method`.
3. Prepare the survey directory described above and run the script.

Library helpers that depend on survey geometry take `survey=` (or explicit
field values) and raise `ValueError` if neither is given.

## Tests

    python -m unittest test_grid_bias -v

Tests that need survey files not in this repository are skipped.

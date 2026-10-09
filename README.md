# survey_debias

Horvitz–Thompson debiasing of pencil-beam trans-Neptunian object (TNO)
surveys using the OSSOS Survey Simulator (`ossssim`).

Each detection receives a weight $1/P$, where $P$ is the probability that an
object like it would have been detected by the survey. Summing the weights
estimates the total population the survey is capable of detecting; see
[Population estimates](#population-estimates-from-the-weights) for why, and
for the parts of the population a survey can never see.

## Installation

Requirements:

- Python ≥ 3.11
- `numpy`, `astropy`, `matplotlib`
- `ossssim` (distribution `OSSSurveySimulator`), the OSSOS Survey Simulator
  Python package with its compiled Fortran library. Install it first, from
  the SurveySimulator repository.
- For `model_ae` surveys: the OSSOS Models 1.0 ModelUsed tables (see
  [OSSOS model prior](#ossos-model-prior-model_ae-only))

Then install this package, which provides the `survey_debias` command:

    pip install -e /path/to/survey_debias

The H-distribution fit (`survey_debias_hfit`) also needs `scipy`, `emcee`,
and `corner`:

    pip install -e '/path/to/survey_debias[fit]'

Package layout:

- `survey_debias.grid_bias`: geometry, cells, orbit-model prior, and
  output helpers (no simulator import)
- `survey_debias.pointings`: reads `characterization/{survey}/pointings.list`
  into surveys, blocks, and tiles
- `survey_debias.efficiency`: OSSOS `.eff` files and the selection they define
- `survey_debias.colours`: `colour.toml` and per-detection colours
- `survey_debias.bias_files`: the per-survey bias tables
- `survey_debias.grid_bias_run`: the simulator-backed bias runner
- `survey_debias.survey_config`: loads a `GridSurvey` from TOML or Python
- `survey_debias.cli`: the `survey_debias` command
- `survey_debias.hfit`, `hfit_plots`, `hfit_cli`: the H-distribution fit and
  the `survey_debias_hfit` command

## How the bias is computed

A project combines one or more **surveys**, each a directory
`characterization/{survey}/` with its own `pointings.list`. Every line of
`pointings.list` names an efficiency file `{block}.eff`; the lines that name
the same file are the **tiles** of one **block**. Each detection names its
survey and block (columns or defaults, see [Surveys](#surveys)).

Each detection is assigned to a cell. For each block and cell the runner
plants synthetic objects whose orbits pass through one of the block's tiles
(chosen with probability proportional to tile area), runs them through
`ossssim` at every epoch of that survey, and keeps going until `--target`
detections accumulate (default 5000). A plant counts for the block only if
OSSSSim's `detos1` reports flag ≥ 4 at every epoch **and** attributes the
detection to that block's `.eff` file each time. The block's bias for the
cell is the mean, over aimed plants, of

    x = 1[detected in block] × P(orbit crosses the block's footprint)

with Monte Carlo standard error $\sqrt{(\overline{x^2} - \bar x^2)/n}$. The
first factor comes from the simulator (efficiency, rate cuts, epochs, the
colour into the block's filter); the second is the geometric chance that an
orbit with that inclination passes through the block's total area at the
field's ecliptic latitude.

The bias given to a detection is, by default, the **union** over every block
in the project: $P = \sum_\text{blocks} P_\text{block}$, the probability that
an object in that cell is detected anywhere in the project. That is the
Horvitz–Thompson inclusion probability for the combined sample.
`--per-block-bias` instead gives each detection its own block's bias, which
treats every block as a separate HT estimate of the same population (the
fit then combines blocks by inverse variance; see
[Normalisation](#normalisation-and-its-uncertainty)). Use the union when the
blocks are one survey programme; use per-block biases to compare blocks or
when blocks see different populations.

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
  unbiased on average but can be very noisy. `survey_debias_hfit` reports
  the HT variance (sampling plus Monte Carlo bias error) and can bootstrap
  over detections (`--bootstrap`); a plain weighted likelihood understates
  the uncertainty because it treats $\sum w_k$ as the number of independent
  detections.
- **Model dependence (`model_ae`).** $P$ is averaged over the orbit model's
  $p(a, e \mid r, i)$, so $\hat N$ and the luminosity-function normalisation
  assume that model's mix of orbits at each $(r, i)$.
- **Cell averaging.** $P$ is the average over a cell. This is accurate only
  if $P$ varies little across the cell, which is why the $H$ step is small
  (0.1 mag).

## Surveys

The `--survey` file describes a **project**: how to read the detections and
how to compute the bias. It is a `GridSurvey` (`survey_debias.GridSurvey`)
given as TOML or Python. Geometry (pointings, epochs, observer, efficiency)
is **not** in this file; it is read from each survey's characterization
directory, and colours from `colour.toml`.

This package defines no surveys. Each project is defined where its
characterization files live and passed to `survey_debias --survey`.
`examples/` has both forms for each project, to seed those projects:

- `examples/jwst_sample_a.toml` / `.py`: JWST GO 1568 Sample A (Eduardo et
  al. 2026); `model_ae`; survey `JWST_A`, block `sampleA`, three epochs.
  Seed for `jwst-tno-followup`.
- `examples/hst_n26.toml` / `.py`: HST GO-9433 reanalysis (Napier et al.
  2026); `aq_grid`; survey and block `N26`, a single 15-day heliostack.
  Seed for `hst-tno-followup`.
- `examples/pointings.list`: every `pointings.list` line format.
- `examples/colour.toml`: colours for the default and cold groups, including
  JWST F150W2 (`W`) and HST F606W STMAG (`F`).

A TOML file has up to three tables:

- `[survey]`: `name`, `bias_method` (`aq_grid` or `model_ae`),
  `detections_relpath`, and optionally `model_band` (the band the H
  distribution and orbit model are in; default `r`), `colour_file` (default
  `colour.toml`), `surveys` (survey directories to use; default every
  directory under `characterization/` with a `pointings.list`),
  `paper_reference_jd`, the rate cut used for plots
  (`rate_cut_min_arcsec_hr`, `rate_cut_max_arcsec_hr`), and output names
  (`results_name`, `detections_full_name`, `check_detected_title`).
- `[detection_columns]`: maps each program column that is read onto the
  column name in the detections table (below).
- `[detection_defaults]`: values for `survey`, `block`, or `comp` when the
  table has no such column. `survey` and `block` must come from a column or
  a default.

```toml
[survey]
name = "JWST Sample A"
model_band = "r"
detections_relpath = "data/jwst_sampleA.csv"
bias_method = "model_ae"

[detection_columns]
mag = "m_f150w2"
d_bary = "d_bary"
i = "i"
comp = "comp"

[detection_defaults]
survey = "JWST_A"
block = "sampleA"
```

Unknown or missing required fields are errors. The geometry fields of
earlier versions (`field_ra_deg`, `field_dec_deg`, `mosaic_width_deg`,
`mosaic_height_deg`, `epoch_jd`, `observer_csv`, `eff_file`, `epoch_layout`,
`fill_factor`, `mag_color_offset`, ...) are rejected with a message that
points to `pointings.list` or `colour.toml`.

## Survey directory layout

The runner reads and writes inside a project root passed as `--root`:

    <root>/
      data/<detections>.csv              # detections_relpath
      colour.toml                        # colour_file
      Models/OSSOS/                      # default orbit model (model_ae only)
      characterization/
        {survey}/
          pointings.list                 # one line per tile
          {block}.eff                    # one per block, with a filter= line
          <observer>.csv                 # JPL Horizons vectors, if not an MPC code
          bias_grid[_rih].csv            # written: this survey's bias table
        {survey}/                        # or, for a multi-epoch survey:
          epoch1/pointings.list  epoch1/{block}.eff
          epoch2/pointings.list  epoch2/{block}.eff
          ...

These files are survey-specific and are not part of this repository.

### `pointings.list`

The formats are those of OSSSSim `getsur.f95`; every line ends with
`JD fill observer {block}.eff`:

    [rect] width height ra dec jd fill observer eff   # on-sky sizes in degrees
    ears ra dec jd fill observer eff                   # MegaCam footprint
    poly n ra dec jd fill observer eff                 # then n lines "dx dy" [deg]

RA is decimal degrees or `hh:mm:ss`, Dec decimal degrees or `dd:mm:ss`.
`observer` is a JPL Horizons vector CSV in the same directory or an MPC
code. Lines that name the same `{block}.eff` are tiles of one block: plants
are aimed at a tile chosen in proportion to its area, at that tile's JD,
and the geometric probability uses the block's total area. Footprint areas
are cross-checked against OSSSSim's own reader.

A multi-epoch survey has one `epoch{i}/` directory per epoch, each with its
own `pointings.list` (that epoch's JDs) and `{block}.eff` files, and the same
block names in every epoch. A detection must be found, in the same block, at
every epoch.

OSSSSim's `detos1` reports which block detected an object as the key
`{directory}/{block}`, where `{directory}` is the survey directory (or
`epoch{i}`); each part may be at most 32 characters, and characterization
paths at most 2048. Older OSSSSim builds report only the first 10
characters of `{block}.eff` and open paths of at most 100 characters; with
those, blocks of one survey must differ within the first 10 characters.
The runner detects which build is installed and checks these limits before
calling the simulator.

### `.eff` files and filters

Each `{block}.eff` is an OSSOS efficiency file and must contain a
`filter=` line with a one-character filter name (`A`–`z`, as OSSSSim
indexes filters). Different blocks of a survey may use different filters;
every epoch of one block must use the same filter. `rate_cut=` (if present)
sets the rate range used for check plots.

### `colour.toml`

Colours are a property of the objects, not of a survey, so they live in one
file per project in the `ossssim.color.PhotSpec` layout: one table per
spectral group, each mapping band ratios `"X-g"` to colours in magnitudes.

```toml
[default]
"g-g" = 0.0
"r-g" = -0.7
"W-g" = -1.7    # JWST F150W2: m_r = m_F150W2 + 1

[cold]
"g-g" = 0.0
"r-g" = -0.85
"W-g" = -1.85
```

A detection's spectral group is the first group whose name appears in its
`comp` value (`cold_classical` uses `[cold]`), otherwise `[default]`. Its
colour is `filter − model_band` for that group, so

$$H_\text{model} = m_\text{filter} - \text{colour} - 5\log_{10}(r\Delta) + \text{phase}$$

and the simulator is run with the same group and colours, so the bias is
computed for the same filter. Every block filter and `model_band` must have
a colour in every group. Without `colour.toml`, ossssim's built-in
`PhotSpec.COLORS` are used.

### Detections table

The detections table is a CSV. Blank lines and `#` comments before the header
are ignored.

`[detection_columns]` names every column that is read. The key is the
program name and the value is the header in the detections table. `mag`
is the apparent magnitude in the block's filter (`m_f150w2` for JWST,
`m_stmag` for N26). `model_ae` also reads `d_bary` and ecliptic `i`. `a`
and `e` are read only when they are listed; they do not place the cell.
`aq_grid` also reads `a` and either `e` or `q`, and `i` or `ifree`. Other
program names are `name`, `survey`, `block`, `comp`, `Omega`, `Omfree`,
and `omfree`.

Every column in the file is kept in the results. A column that is not
listed in `[detection_columns]` is not used for the cell. A blank cell in
a listed column is left unused. When two of `a`, `e`, and `q` are present,
the third is derived. `ifree` is taken from the file when that column is
listed; otherwise it is computed from `i`, `a`, and `Omega`, with Ω = 0
when that row has no node.

Cells are kept separately per spectral group, because the colour changes
the simulated magnitudes.

## OSSOS model prior (`model_ae` only)

The full tables are not shipped with this package. Download them once, for
example into a shared directory:

    mkdir -p Models/OSSOS
    for c in Classical Detached Inner Plutinos Scattering Twotinos; do
      curl -fsSL -o Models/OSSOS/$c-ModelUsed.dat \
        https://raw.githubusercontent.com/OSSOS/OSSOS_Models/$c/$c/ModelUsed-check-8.66.dat
    done

These are the OSSOS 1.0 nominal populations calibrated to the OSSOS++ sample
(Bannister et al. 2018, ApJS 236, 18). Each row is tagged with its population
from the filename.

The model is found, in order, from `--model PATH` (a single file or a
directory), the `SURVEY_DEBIAS_MODEL` environment variable, or
`<root>/Models/OSSOS`. Legacy L7-style files also load. `tests/data/OSSOS/` contains a short excerpt
for unit tests only; it is not a usable prior.

## Running

    survey_debias --survey jwst_sample_a.toml --root /path/to/project

`--survey` accepts a TOML file, `path/to/survey.py:NAME`, or
`package.module:NAME` (`:NAME` may be omitted if the module defines exactly
one `GridSurvey`).

Check the setup first with `--dry-run`. Without running the simulator it
lists every survey, block (filter, epochs, tiles, area), and tile; the
colours; the detections per block; the cells and their objects; and
anything missing or invalid (characterization, `.eff` filters, colours,
orbit model):

    survey_debias --survey jwst_sample_a.toml --root /path/to/project --dry-run

Options:

- `--root`: project directory (default: current directory)
- `--target`: detections required per block and cell (default 5000)
- `--seed`: random seed (default 42)
- `--model`: orbit model file or directory (`model_ae` only)
- `--per-block-bias`: give each detection its own block's bias instead of
  the union over every block
- `--check-plots-dir`: where to write check plots (default `<root>/check_plots`)
- `--no-check-plots`: skip check plots
- `--dry-run`: validate inputs and list cells; no simulation

Before the first cell of each block the runner plants a bright object on a
tile's line of sight and stops if OSSSSim does not detect it in that block,
which catches characterization and observer errors early. In union mode
every block is simulated for every cell, including blocks with no
detections in that cell. Each estimate stops after at most
`20 × --target` aimed plants and keeps what it has, with its MC error; a
block with no detections in the cell may record zero, but a cell with a
real detection in the block and no simulated detection is an error.

A project defined in Python can also be run as a script; the examples call
`survey_debias.grid_bias_run.main(survey=...)`, which takes the same options
apart from `--survey`.

## Outputs

- `characterization/{survey}/bias_grid.csv` (`aq_grid`) or
  `bias_grid_rih.csv` (`model_ae`): one row per block, colour group, and
  cell, with `bias`, `n_drawn` (aimed plants), `n_detected`, and `bias_se`
  (Monte Carlo standard error). Each survey's file is a cache: re-running
  skips finished rows, and a missing row is computed. Delete a file (or
  rows) to recompute. A root-level `bias_grid[_rih].csv` from earlier
  versions is imported into a single-block project, without MC errors.
- `<root>/bias_results.csv` (`GridSurvey.results_name`): the columns that
  were in the detections table, then the derived columns: `survey`,
  `block`, `filter`, `colour_group`, `colour`, `Hx`, `cell` (and `q`, `a`,
  `e`, `ifree`, `sin_ifree`, `name` when not already in the file),
  `block_bias` (the detection's own block), `bias` (union or own block), and
  `bias_se`. Comment lines at the top list the file columns, the column map,
  the colours, `bias_mode: union` or `block`, and how each derived value was
  obtained. A derived name that is already an input column is written as
  `<name>_computed`.
- `<root>/<detections_full_name>`: OSSOS/CFEPS-format detections file,
  written when every detection has `a`, `e`, and `i`. Columns through `MPC`
  match `CFEPS.detections`; six columns follow:
  - `ifree`, `Omfree`, `omfree`: free elements relative to the Laplace plane
    (`Omfree` and `omfree` are 0 when unknown)
  - `Hx`: absolute magnitude in `model_band` used for the cell
  - `comp`: component from the detections table (`-` when the table has none)
  - `bias`: $P$ for the detection's cell
- `check_plots/{survey}/{block}/`: sampled versus detected RA/Dec and
  orbital-element distributions, one set per detection plus one combined
  (`all`).

The run ends by printing `sum 1/bias`, the Horvitz–Thompson estimate of the
population the project can detect. It excludes regions where the survey has
zero sensitivity; see
[What the estimate cannot include](#what-the-estimate-cannot-include).

## Fitting the H distribution

After `survey_debias` has written `bias_results.csv`, `survey_debias_hfit`
fits the exponentially tapered form (Kavelaars et al. 2021; Petit et al.
2023, [ApJL 946, L4](https://iopscience.iop.org/article/10.3847/2041-8213/acc525),
Eq. B4):

$$N(<H) = 10^{\alpha (H - H_o)} \exp\left[-10^{-\beta (H - H_B)}\right]$$

    survey_debias_hfit --survey survey.toml --root /path/to/project [--bootstrap 200]

### Likelihood

The shape $(\alpha, \beta, H_B)$ comes from the conditional likelihood of
each detection's H given its geometry and that it was detected. With
$\lambda(H) = dN/dH$, apparent magnitude $m_k$ and fitted $H_k$, detection
$k$ contributes

$$f_k(H_k) = \frac{\lambda(H_k)\, S(m_k)}{\int \lambda(H)\, S(m_k + H - H_k)\, dH}$$

where $S(m)$ is the magnitude selection of the detection's own block, read
from `characterization/{survey}/[epoch{i}/]{block}.eff`: $\eta(m)$
(single, double, linear, square, or lookup; zero below 0.01) × tracking
fraction × the characterisation limit (`mag_lim`, or $\eta \ge 0.4$ without
one), multiplied over epochs, as in OSSSSim. $m_k$ is in the block's filter
and $H_k$ in `model_band`, so $m_k - H_k$ carries the detection's colour.
Detections from blocks of different depth or filter each use their own
$S$. This assumes the H distribution does not depend on orbit (the
population factorises as $\lambda(H)\,g(\text{orbit})$), needs no orbit
model, and gives ordinary rather than weight-inflated uncertainties. $H_o$
and the bias cancel from it. Photometric scatter is not modelled.

### Normalisation and its uncertainty

$H_o$ is set from the Horvitz–Thompson count $\hat N = \sum_k w_k$,
$w_k = 1/\text{bias}_k$, of detections with
$H_\text{min} \le H \le h_\text{norm}$. `h_norm` defaults to the faintest H
at which every detection's geometry still has half its peak selection, so
the count is not depleted by objects too faint to see. Its variance has two
parts:

- **sampling**: $\sum_k w_k (w_k - 1)$, the Poisson scatter of which
  objects were detected;
- **bias Monte Carlo error**: detections that share a bias estimate (same
  colour group and cell, and in block mode the same block) move together,
  so each estimate $j$ adds $(\sum_{k \in j} w_k)^2\,(\sigma_j / P_j)^2$,
  with $\sigma_j$ the `bias_se` written by `survey_debias`.

With per-block biases (`bias_mode: block`) each block gives its own HT count
of the whole population; these are combined with inverse-variance weights,
and the summary lists each block's count. The MCMC posterior of $H_o$ draws
the count from a lognormal with this mean and variance.

Bias files written before `bias_se` existed have no MC error. The summary
says so; `--bias-rel-err` supplies a relative error for those estimates
(and for results without `bias_se`). It does not override recorded errors.

### Bootstrap

`--bootstrap N` repeats the fit on `N` resampled data sets. Each replicate:

1. resamples detections with replacement **within each block**, drawing a
   Poisson($n_\text{block}$) number of detections, so the HT count keeps the
   Poisson scatter that its $\sum w(w-1)$ variance describes;
2. multiplies every block-cell bias estimate by an independent lognormal
   factor with mean 1 and that estimate's relative MC error (the union bias
   of a detection is the sum of its perturbed block estimates);
3. refits the maximum likelihood, starting from the full-sample MLE, and
   sets $H_o$ from the replicate's HT count.

The spread of the replicates is a frequentist check on the MCMC intervals,
and its $H_o$ spread includes the bias MC error. **Caveat:** each refit is warm-started from the full-sample MLE
and is a local optimisation, so when the likelihood is broad or
multi-modal (for example $\beta$ and $H_B$ unconstrained by the data)
replicates stay near the starting mode and the bootstrap understates the
uncertainty. Compare with the MCMC posterior and trust the wider of the
two; the summary flags any parameter whose bootstrap 68% interval is less
than half the posterior's. Replicates whose fit fails are dropped and counted in the summary.

The parameters are strongly degenerate (Petit et al. 2023 make the same
point): if the detections do not reach well past $H_B$, or the taper sits
brighter than the brightest detection, $\beta$ and $H_B$ are set by their
priors. The summary flags this.

### Options

- `--h-column`: H column in the results (default `Hx`, or `Hx_computed`).
- `--h-min`, `--h-max`: fit range (defaults: 1 mag brighter than the
  brightest detection; no faint limit).
- `--h-norm`: normalisation limit for $H_o$.
- `--eff-file` (repeat per epoch) or `--efficiency single:A,m0,w[,mag_lim]`:
  replace every block's efficiency files (for tests and comparisons).
- `--rate`: on-sky rate in "/hr, used when the efficiency depends on rate
  (default: opposition rate from `d_bary`).
- `--bootstrap N`: bootstrap replicates (default 0, none).
- `--bias-rel-err X`: relative MC error for bias estimates that have none.
- `--prior NAME=LO,HI`: uniform prior for `alpha` (default 0.05–2),
  `beta` (0.02–2), or `h_b` (`h_min` − 5 to the faintest detection + 5).
- `--walkers`, `--steps`, `--burn`, `--thin`, `--seed`: emcee settings
  (defaults 32, 6000, 1000, 5, 42). `--no-mcmc` stops after the
  maximum-likelihood fit.
- `--out-dir`: default `<root>/hfit`.

### Fit outputs

- `hfit_summary.txt`, `hfit_summary.json`: maximum-likelihood parameters
  and $\ln L$; posterior medians with 68% intervals; bootstrap medians and
  intervals; comparison with a single exponential
  $N(<H) = 10^{\alpha(H-H_o)}$ ($\Delta\ln L$, ΔAIC, ΔBIC, and an
  approximate likelihood-ratio p-value); goodness of fit, overall and per
  block (detections, PIT KS p-value, efficiency files); the normalisation
  with its sampling and bias-MC parts (and per-block counts in block mode);
  and warnings (short chains, posteriors at a prior edge, unconstrained
  taper, missing MC errors, failed bootstrap replicates).
- `hfit_samples.csv`: posterior samples of `alpha, beta, h_b, h_o` and
  $\ln L$.
- `hfit_bootstrap.csv` (with `--bootstrap`): `alpha, beta, h_b, h_o,
  ht_count` per replicate.
- `hfit_corner.png`: corner plot of the posterior, with the
  maximum-likelihood point in red and, with `--bootstrap`, the bootstrap
  replicates overlaid.
- `hfit_detected.png`: histogram of detected H against the model's
  prediction for those detections ($\sum_k f_k(H)$), with the
  single-exponential prediction for comparison; the histogram of
  $F_k(H_k)$ (the probability integral transform, flat when the model is
  right); and its empirical CDF with a 95% KS band, with one curve per
  survey when there are several.
- `hfit_debiased.png`: HT cumulative and differential H distributions
  against the model, with the best-fit parameters inset. H fainter than
  `h_norm` is shaded, because the HT estimate is incomplete there.

Goodness of fit is reported two ways. The KS p-value of the PIT values at
the maximum likelihood is conservative because the parameters were fitted
to the same data. The posterior predictive p-value is the fraction of
posterior draws for which simulated detections, at the observed
geometries, fit worse than the real ones. Values near 0 mean the form does
not describe the data.

## Adding a survey

1. In the project, copy an example from `examples/` (TOML or Python) and
   set `name`, `bias_method`, `model_band`, the detections path, and
   `[detection_columns]`.
2. For each survey, create `characterization/{survey}/` (or
   `{survey}/epoch{i}/`) with `pointings.list`, one `{block}.eff` per block
   (with a `filter=` line), and the observer files. Add a `survey` and
   `block` column to the detections, or set them in `[detection_defaults]`.
3. Add every block filter to `colour.toml` for every spectral group.
4. Check with `survey_debias --dry-run`, then run without it.

## Tests

    python -m pytest tests

The tests do not need the compiled simulator for most checks (the bias
loop is tested with a stub simulator); a few cross-check footprint areas
against `ossssim` when it is installed. Tests that need survey files not
in this repository are skipped.

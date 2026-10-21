# Joint cold-classical H-distribution fit across surveys

Several surveys at different ecliptic latitudes and longitudes, in different
filters, are fitted together with an extended (Poisson) likelihood instead of
per-detection biases and a Horvitz-Thompson (HT) count.

## Method

For survey `s` in filter `f`, `eps_s(H_f)` is the fraction of the whole cold
population with absolute magnitude `H_f` (in `f`) that `s` detects and
characterises. With the cold LF `lambda(H_r)` (tapered form, `H_o` free) and
each filter's mean colour `cbar_f + delta_f` with an intrinsic spread `sigma_c`:

    lambda_f(H) = int lambda(H - c) N(c; cbar_f + delta_f, sigma_c) dc   (no colour in r)
    Lambda_s    = int lambda_f(H) eps_s(H) dH   - overlaps with higher-priority surveys
    ln L = sum_s [ sum_k ln f_k + N_s ln Lambda_s - Lambda_s ] + sum_f ln N(delta_f; 0, sigma_delta)

`f_k` is each detection's conditional density given its own distance and its
block's selection (`hfit.ConditionalLikelihood`); the Poisson terms tie the
surveys' counts, and so their normalisations, together. Colours are a shift and
a convolution inside the fit, so nothing is re-simulated for a colour change.
Defaults: cold `g - r = 0.9`, `r - F150W2 = 1.0` (`colour.toml` `[cold]`),
`sigma_delta = 0.05`, `sigma_c = 0.2`.

`eps_s` is computed once per survey (`survey_debias_eff`). OSSSSim decides only
geometry, on a copy of the characterization with unit efficiency: cold model
rows (OSSOS 1.0 `Classical-ModelUsed.dat`: `coldl`, `coldh`, `kernel`) are
aimed into each block and weighted by the geometric probability of the orbit
being in the block's box; a plant is in the field when Detos1 returns flag 4.
Each plant keeps `mu = m - H`, and

    eps_b(H) = mean_j[ w_j 1(in field)_j S_bar_b(H + mu_j, rate_j) ]

with `S_bar_b` the block's real `.eff` selection (efficiency on the true
magnitude, tracking and characterisation over the photometric noise,
multiplied over epochs). The sample cut (e.g. `i < 5 deg`) is applied to the
plants as to the detections; `eps` stays a fraction of the whole cold
population.

## Cold sample by free inclination (i_free < 4 deg)

`eps_tag = "coldfree"` (`SurveyCharacterizations/joint/coldfree_all*.toml`)
selects the cold classicals by free inclination, I_free < 4 deg (Huang,
Gladman & Volk 2022, ApJS 259, 54), instead of ecliptic i < 5 deg. A
non-resonant orbit keeps its inclination to the local forcing pole while its
ecliptic i cycles; the 4 deg cut in I_free separates the cold and hot
components much more cleanly (their Figs 1 and 6).

- **I_free** (`survey_debias.ifree`): Huang et al. Table 1 (double-averaged
  Hamiltonian, barycentric elements; bundled in `data/huang2022_ifree.csv`)
  where the object is listed, by OSSOS++ name, number or provisional
  designation; otherwise their forcing-pole grid (Table 2,
  `data/huang2022_qp_forced.npz`, nearest grid point as in their
  `_read_grid.py`) on barycentric elements (`barycentric_elements` converts
  heliocentric ones). On their 1332 classicals the grid matches Table 1 to
  < 0.1 deg for 98% and on the same side of 4 deg for 99.85%. 540 of the 549
  OSSOS++ main-belt classicals are in Table 1 (the release elements agree
  with Huang's barycentric ones to 0.001 deg).
- **Detections**: release surveys `cold = { classes = ["cla"], parts = ["m"],
  ifree_max = 4.0 }` (`detections_io.ColdCut(ifree_max=...)`, CSV columns
  `ifree`, `ifree_source`). jwst_sample_a and N26 keep their papers' cold
  classifications (`select = { comp = "cold" }`): their short arcs give no
  usable node.
- **Model**: the OSSOS 1.0 cold components are drawn about a forced plane
  (the mean pole of the cold rows is at i = 1.77 deg, node 88 deg), so
  planting with the ecliptic i and a random node does not keep their
  geometry. `survey_debias_eff --ifree-max 4` plants with `aim = "ifree"`
  (`plant_block_free`): each plant keeps its model row's I_free about the
  row's forcing pole; of the two orbit poles through the planted position at
  that angle one is drawn, and the weight is the geometric probability at
  the position's latitude above the forced plane,
  `A / (360 x 180 x sqrt(sin^2 I_free - sin^2 beta_f))` (a Monte Carlo of
  random poles about the forcing pole matches it to its 4% noise). The plant's
  own Huang I_free (from its aimed elements) carries the cut. 71.4% of the
  cold model rows have I_free < 4 deg (75.8% have i < 5 deg).
- `build_terms` checks that a release survey's i / i_free cut matches the
  `cut` its eps table was computed with.

The geometry matters most for fields off the forced plane: relative to the
i < 5 deg tables, the bright-end eps changes by 1.13 (OSSOS), 1.11
(CFEPS_g), 1.09 (presurvey), 0.83 (CFEPS_r), 1.41 (JWST), 1.59 (N26) and
0.19 (MA). At the shape of the earlier all-survey fit, the counts become
OSSOS 259 / 263 expected, CFEPS_g 47 / 42 (was 53 / 38.5), MA 8 / 4.3 (was
10 / 25), JWST 12 / 15, N26 5 / 6.5: the MA deficit and the CFEPS_g excess
of the i < 5 deg fits go away.

Fits (2000 steps; `SurveyCharacterizations/joint/joint_coldfree_all*`,
`joint_mpc_compare_coldfree/`):

| i_free < 4 | alpha | beta | H_B | N(H_r<5) | N(<5.5) | N(<6) | alpha within 0.5 lnL |
|---|---|---|---|---|---|---|---|
| all surveys | 0.18 ± 0.03 | 0.18 +0.03 -0.02 | 10.7 +1.0 -0.8 | 2.2 +2.2 -1.2 | 17 +10 -6 | 94 +29 -24 | 0.005-0.278 |
| + MPC (5 objects, H_r < 5.42) | 0.23 ± 0.04 | 0.22 ± 0.03 | 9.6 ± 0.7 | 1.0 +0.7 -0.4 | 11 +4 -3 | 77 +16 -13 | 0.129-0.278 |

Counts at the MLE with the MPC: OSSOS 259 / 270, CFEPS_r 4 / 3.4, CFEPS_g
47 / 39, presurvey 4 / 4.0, MA 8 / 4.4, JWST 12 / 11.6, N26 5 / 5.2, MPC 5 /
6.1. Without the MPC the fit predicts 9.1 +5.5 -3.7 MPC objects, 5 are found
(P(N <= 5) = 0.22). As with i < 5 deg, the MPC removes the small-alpha side
of the profile and halves the bright-end uncertainty; the single exponential
is rejected by Delta AIC = 120 (330 with the MPC).

## Complete surveys (the MPC)

A catalogue that has found every object below an H limit everywhere on the
sky is a survey too: `format = "complete"` in the joint configuration, with
its spec in a TOML (`SurveyCharacterizations/characterization/MPC/MPC.toml`:
filter, `h_lim`, `sigma_h`, detections, duplicates). Its efficiency needs no
plants:

    eps(H_f) = f_cut 1(H_f < h_lim)

with `f_cut` the fraction of the model's cold population inside the sample
cut (0.758 for i < 5 deg; `survey_debias_eff --complete SPEC --model M --i-max 5`
writes it, with the cell holding `h_lim` weighted so the sum integrates
exactly to the limit). Its conditional term is

    f_k = lambda_f(H_k) / int^{h_lim} lambda_f dH

with `lambda_f` spread by `sigma^2 = sigma_c^2 + sigma_h^2` (the catalogue's
H scatter on top of the colour spread; `sigma_h` alone when the catalogue is
in the model band). The MPC survey is in r: its `MPC.csv` H is H_r measured
from the MPC photometry (`characterization/MPC/mpc_photometry.py`), so it
has no colour offset to fit. A catalogue in another filter gets its
`delta_f` like any other survey. The overlaps and windows below use
`sigma_h`, plus `sigma_c` when the other survey's filter differs.

A complete survey must be first in `priority`: it owns every object below
its limit. The other surveys then

- drop their detections of its objects (its `duplicates_relpath` is added to
  `duplicates`, which may now be a list of files);
- subtract `eps_s(H_r + c_s) Phi((h_lim - H_r - c_M) / sigma)` from their
  expected counts (computed in `build_terms` from their own `eps_cold.csv`, so
  `h_lim`, `sigma_h` and `sigma_c` overrides stay consistent);
- multiply the selection in their conditional terms by the window
  `1 - P(H_M < h_lim | H)` (`ConditionalLikelihood.apply_h_window`), so the
  shape term does not expect bright objects they no longer keep.

`h_lim` and `sigma_h` can be overridden per configuration; the eps table is
re-stepped at the new limit from its stored `f_cut`.

**Synthetic check** (`validate_synthetic.py --mpc`, `synthetic/recovery_mpc.*`):
the five OSSOS + JWST realisations, each with an MPC-like census of the same
truth (5-8 objects with M < 5.5, like the 7 real ones) and fitted with and
without it. log10(N(<H) fit / truth), mean and rms over the realisations:

| | N(<5) | N(<5.5) | N(<6) | N(<7) |
|---|---|---|---|---|
| without MPC | -0.60, 0.73 | -0.32, 0.40 | -0.15, 0.22 | -0.02, 0.08 |
| with MPC | -0.02, 0.15 | +0.01, 0.12 | +0.01, 0.10 | +0.01, 0.07 |

Without it the bright end is extrapolated from H_r > 6 and comes out low by a
factor of 4 at H_r = 5; with it the bias is gone and the scatter is 5 times
smaller.

**Data, first version** (`SurveyCharacterizations/joint/cold_all_mpc.toml`
against the same configuration without the MPC, both with 2000 steps,
`mpc_compare.py`, `mpc_sensitivity.py`). These fits used the MPC's own H
(filter M, M - r = +0.20, limit 5.5, 7 objects); the sample is now in r
(H_r < 5.42, 9 objects with measured H_r) and has not been refitted:

| | alpha | beta | H_B | N(H_r<5) | N(<5.5) | N(<6) | alpha within 0.5 lnL |
|---|---|---|---|---|---|---|---|
| without MPC | 0.29 ± 0.05 | 0.22 +0.05 -0.04 | 9.1 +1.1 -0.9 | 2.3 +2.5 -1.5 | 19 +11 -9 | 109 ± 33 | 0.030-0.328 |
| with MPC | 0.27 ± 0.05 | 0.21 +0.05 -0.03 | 9.4 ± 0.9 | 2.0 +1.4 -0.8 | 17 +6 -4.5 | 102 ± 18 | 0.129-0.328 |

The fit without the MPC predicts 10.3 +5.9 -4.6 cold MPC objects below the
limit, against 7 found (posterior-predictive P(N <= 7) = 0.33): the census
agrees with the pencil-beam surveys and halves the bright-end uncertainty. It
also removes the small-alpha side of the profile (alpha < 0.13), where H_o
ran away, and the single exponential falls further behind (Delta AIC 97 to
258). At the MLE the largest systematics are the MPC colour (M - r = 0.1 or
0.3 for 0.2: N(<5) = 1.4 or 3.3 for 2.2) and its H scatter (sigma_h = 0 or
0.3: 2.7 or 1.2); the limit (5.0, 5.25) and counting the one-opposition
2025 ME398 move it by less.

## Running it

```bash
M=../KuiperBeltModels/OSSOS/Classical-ModelUsed.dat
survey_debias_eff --root ../SurveyCharacterizations --survey OSSOS_v11 --model $M \
    --i-max 5 --exclude-block 2015BS
survey_debias_eff --root ../SurveyCharacterizations --survey jwst_sample_a --model $M \
    --i-max 5.0001 --r-min 38 --r-max 55 --overlap-with OSSOS_v11
cd ../SurveyCharacterizations/joint
survey_debias_joint --config cold_jwst_ossos.toml
```

The configurations in `SurveyCharacterizations/joint/` list the `eps` commands
for every survey. `survey_debias_eff --check-detections` runs each block's own
release orbits through its characterization; blocks that recover few of their
own objects should be left out (`exclude_blocks` and `--exclude-block`).

## Validation

- **Geometry.** JWST Sample A's three-epoch cold area from the plants is
  117 arcmin² (the characterization README: 115 ± 2).
- **Against the per-detection bias.** In (r, i, H) cells the plant estimate
  matches the old `model_ae` bias within 1-2 sigma (old values up to 10% higher
  at the efficiency fall-off, where it drops unreachable aimed draws).
- **Synthetic catalogues** (`validate_synthetic.py`, `synthetic/recovery.*`):
  five realisations of a known tapered LF drawn through the real JWST and OSSOS
  characterizations with OSSSSim. `N(<H)` is recovered over H_r 6-13.5; `alpha`
  averages 0.31 for a true 0.25 (the truth is inside the Delta lnL = 0.5 range in
  4 of 5); fitting with `sigma_c` = 0.1 or 0.3 instead of 0.2 moves it by < 0.003.
- **HT cross-check** (`ht_crosscheck.py`): sum 1/bias with each detection's
  union bias from the plant tables estimates the cold objects with i < 5 deg,
  0.76 of the model's cold population. Synthetic: OSSOS 0.83, JWST 0.86 of the
  true N; data: OSSOS 0.73, JWST 1.08 of the fit's N.

## Problems found on the way

- **Geometric probability was pi/2 too large.** `geometric_detection_prob`
  integrated to pi/2 over the sky; fixed (exact form
  `A / (360 * 180 * sqrt(sin^2 i - sin^2 beta))`). Per-detection biases from
  before the fix are 57% too high.
- **`apparent_to_Hr` is ~0.2 mag off near opposition.** It assumes r = Delta =
  d, a phase angle of ~1/d rad. The joint fit uses the release `Hsur` (which
  matches OSSSSim to 0.001 mag) or `geometric_h` from the pointing and observer;
  the per-detection pipeline still uses `apparent_to_Hr`.
- **Blocks that reject their own detections.** OSSOS 2015BS (1/30), CFEPS
  L3s-smooth (0/6) and L3f-smooth (0/3), HiLat HL6r and HL7c: the objects move
  at an angle outside the `.eff` `rate_cut`. 2015BS and L3s are both near RA 0h.
  They are excluded until the characterizations are checked.
- **Overlaps.** 24% of the cold objects JWST could find in its field were in
  OSSOS 2015AP's field in 2015; CFEPS_g shares 5% of its field objects with
  OSSOS, and 4 cold detections (listed in `Duplicates`). Each object is counted
  in the first survey of `priority`, and the others' expectations are reduced
  by the overlap eps.

## Results

| Surveys | alpha | beta | H_B | delta | counts (observed / expected at MLE) |
|---|---|---|---|---|---|
| JWST A + OSSOS | 0.21 +0.05 -0.04 | 0.18 +0.05 -0.03 | 10.6 ± 1.2 | delta_H = 0.00 ± 0.05 | 12/11.2, 266/266.8 |
| + CFEPS r, g | 0.30 +0.05 -0.06 | 0.23 +0.07 -0.05 | 8.8 +1.2 -0.9 | delta_H = 0.00 ± 0.05, delta_g = -0.07 ± 0.04 | OSSOS 266/280, CFEPS_r 4/5.1, CFEPS_g 53/39.3, JWST 12/11.2 |
| all (+ presurvey, MA, N26) | 0.26 +0.07 -0.05 | 0.20 +0.07 -0.04 | 9.6 +1.5 -1.3 | delta_g = -0.07 ± 0.05; delta_H, delta_R, delta_t ~ 0 | OSSOS 266/268.5, CFEPS_r 4/4.9, CFEPS_g 53/38.5, presurvey 6/3.6, MA 11/25.4, JWST 12/11.2, N26 4/4.2 |

Posterior medians and 68% intervals; `alpha` is the faint-end slope, the taper
removes the bright end below `H_B`. A single exponential is rejected in both
(Delta AIC = 60 and 90). For JWST + OSSOS, `sigma_c` from 0.1 to 0.3 changes
lnL by 0.02, and JWST's fill factor 1.0 instead of 0.91 (E26's normalisation)
lowers N(<10) by 6% and alpha to 0.14. With CFEPS, CFEPS_g sees about a third
more cold objects than the OSSOS-normalised model predicts; the fit takes part
of that up as a bluer cold g - r (0.83).

With every survey the counts show where the surveys disagree: MA finds 11 cold
objects where 25 are expected (Poisson p ~ 0.001) and CFEPS_g 53 for 38.5,
while OSSOS, JWST and N26 agree. HiLat is not in the fit: its fields are at
ecliptic latitudes >= 12 deg, out of reach of i < 5 deg orbits. With this cut
the surveys do not constrain the cold inclination width, which stays fixed at
the OSSOS 1.0 model's; a wider sample cut (or HiLat's hot objects) would be
needed to fit it.

**The lower side of alpha in these fits is an artefact.** The fits above kept
H_o inside (-30, 40). Because N(<H) = 10^(alpha (H - H_o)), the H_o that
matches the counts goes as -log10(N/K)/alpha and leaves that box below
alpha ~ 0.13, so the profile and the posteriors are cut there (the JWST + OSSOS
fill-factor run ended at H_o = -30). The code now leaves H_o unbounded and
allows alpha down to 0.001 with a profile grid from 0.005; refit to see the
small-alpha side. The quoted lower errors on alpha, and the JWST + OSSOS MLE
(0.165, on a flat plateau at the cut), will change.

`survey_debias_joint --replot` redraws the plots from `joint_summary.json` and
`joint_samples.csv` without fitting; the cumulative plot shows each survey's
sum of 1/bias (union bias over its blocks from the plant tables, divided by the
model fraction of cold objects inside its cut) over the fitted N(<H).

The configurations and outputs are in `SurveyCharacterizations/joint/`
(`cold_jwst_ossos.toml`, `cold_jwst_ossos_cfeps.toml`, `cold_all.toml`,
`cold_all_mpc.toml`; `joint_<name>/joint_summary.{txt,json}`, samples and
plots; `mpc_compare.py` and `mpc_sensitivity.py` for the MPC comparison).
`mpc_compare.py --hfit-samples DIR/hfit_samples.csv` gives the same MPC
posterior-predictive count for a single-survey `survey_debias_hfit` run
(its H_o comes from the HT count).

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
(`cold_jwst_ossos.toml`, `cold_jwst_ossos_cfeps.toml`, `cold_all.toml`;
`joint_<name>/joint_summary.{txt,json}`, samples and plots).

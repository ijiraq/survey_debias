# LF shape-fit note

LaTeX note describing the conditional luminosity-function shape fit used in
`survey_debias`: the no-measurement-error case, and the versions that fold in
photometric scatter (OSSSSim `magran` / `.eff` `mag_error`) and optional
distance uncertainty. Its results section fits the dynamically cold objects
of JWST Sample A.

## Build

```bash
# regenerate figures and the Sample A cold-object fit table
python docs/lf_shape_fit/make_figures.py

# compile (needs a basic TeX Live with hyperref, booktabs, …)
pdflatex -output-directory=docs/lf_shape_fit docs/lf_shape_fit/lf_shape_fit.tex
pdflatex -output-directory=docs/lf_shape_fit docs/lf_shape_fit/lf_shape_fit.tex
```

`make_figures.py` reads the Sample A detections and the three epoch `.eff`
files from `../SurveyCharacterizations/characterization/jwst_sample_a/`
(next to this repository), keeps the objects with `comp = cold`, and fits
them with `survey_debias.hfit` from this checkout's `src/`.

## Outputs

| path | contents |
|------|----------|
| `lf_shape_fit.pdf` | the note |
| `figures/*.pdf` | figures embedded in the note |
| `results/jwst_shape_mle.csv` | tapered and single-exponential fits to the cold objects |
| `results/jwst_shape_notes.txt` | the same fits, with profile-likelihood ranges in alpha |
| `results/synthetic_recovery.txt` | synthetic cold-like recovery, with and without the scatter model |

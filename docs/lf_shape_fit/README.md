# LF shape-fit note

LaTeX note describing the conditional luminosity-function shape fit used in
`survey_debias`: the no-measurement-error case, and the versions that fold in
photometric scatter (OSSSSim `magran` / `.eff` `mag_error`) and optional
distance uncertainty.

## Build

```bash
# regenerate figures and the JWST Sample A demo MLE table
python docs/lf_shape_fit/make_figures.py

# compile (needs a basic TeX Live with hyperref, booktabs, …)
pdflatex -output-directory=docs/lf_shape_fit docs/lf_shape_fit/lf_shape_fit.tex
pdflatex -output-directory=docs/lf_shape_fit docs/lf_shape_fit/lf_shape_fit.tex
```

`make_figures.py` looks for `../jwst-tno-followup` next to this repository, or
falls back to `/tmp/jwst-tno-followup`.

## Outputs

| path | contents |
|------|----------|
| `lf_shape_fit.pdf` | the note |
| `figures/*.pdf` | figures embedded in the note |
| `results/jwst_shape_mle.csv` | demo MLE table for Sample A |
| `results/synthetic_recovery.txt` | synthetic no-error vs model recovery |

When the production `hfit` CLI reports `--measurement` comparisons, point the
results section at `hfit_summary.json` and keep this note as the narrative.

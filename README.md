# House Price Prediction with Linear Regression

End-to-end Ames Housing analysis: EDA → feature selection → cleaning/encoding →
correlation → 80/20 split → Linear Regression (MSE/RMSE/R²) → actual-vs-predicted
→ residuals → coefficient analysis → Ridge/Lasso bonus comparison.

## Files

| File | What it is |
|---|---|
| `house_price_linear_regression.ipynb` | The executed notebook (27 code + 21 markdown cells), all outputs included |
| `house_price_linear_regression.html` | Rendered read-only view of the same notebook |
| `figures/` | All 7 charts (target distribution, correlations, actual-vs-predicted, residuals, coefficients, model comparison) |
| `data/ames_housing_missing.csv` | Local dataset copy used as the offline fallback |
| `build_notebook.py` | Script that generates the `.ipynb` (edit this, rebuild, re-execute) |
| `exec_nb.py` | Helper that executes the notebook with nbclient |

## Results (held-out 20% test set, n = 586)

| Metric | Linear (OLS) | Ridge (α = 10) | Lasso (α = 100) |
|---|---|---|---|
| R² | 0.8901 | **0.8913** | 0.8894 |
| RMSE | $29,687 | **$29,527** | $29,775 |
| MAE | $16,979 | $16,681 | $16,892 |
| 5-fold CV RMSE | $29,142 | $28,492 | $28,637 |

Key drivers: `Neighborhood_Green_Hills` (+$123k), the `Overall_Qual`
thermometer ladder (~+$105k Poor→Excellent), `Gr_Liv_Area` (~+$29k per SD).

## Reproduce

```bash
# 1. (optional) edit the notebook source
python build_notebook.py            # regenerates house_price_linear_regression.ipynb
# 2. execute end-to-end (needs network for OpenML on first run; falls back to
#    data/ames_housing_missing.csv offline, then to California Housing)
python exec_nb.py house_price_linear_regression.ipynb
# 3. rendered view
python -m nbconvert --to html --no-input house_price_linear_regression.ipynb \
    --output house_price_linear_regression.html
```

Environment used: Python 3.13.3, pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0,
matplotlib 3.11.2, seaborn 0.13.2.

## Dataset & self-sourcing

- Primary source (no login): OpenML `ames_housing` v1
  (`sklearn.datasets.fetch_openml("ames_housing")`, 2,930 × 81) — same data as
  Kaggle's *House Prices: Advanced Regression Techniques* competition.
- Offline fallback: `data/ames_housing_missing.csv` (same columns, NA = "feature
  absent" pattern as the Kaggle file).
- To use the actual Kaggle `train.csv`: place it at `data/` and point `LOCAL_CSV`
  in the notebook's §2 cell at it — column names match, no other change needed.
- References: scikit-learn docs for `LinearRegression`, `Ridge`, `Lasso`,
  `mean_squared_error`, `r2_score`; Kaggle "House Prices: Advanced Regression
  Techniques"; *Hands-On Machine Learning* (Géron) California Housing example.

## Feature-selection summary

Kept (30 inputs → 118 encoded columns): area (`Gr_Liv_Area`, `Total_Bsmt_SF`,
`First_Flr_SF`, `Lot_Area`), location (`Neighborhood`), quality
(`Overall_Qual/Cond` 10-pt, `Exter/Kitchen/Bsmt/Heating_QC` 5-pt, thermometer
encoded), `Age`, rooms/baths (`Total_Bath` engineered), garage, lot/outdoor,
`Sale_Condition` context. Excluded: leaking `Price_per_SF`, collinear
`Year_Built`/`Year_Remod_Add`/`Longitude`/`Latitude`, timing columns,
near-empty `Pool_QC`/`Alley`/`Fence`/`Misc_Feature`, redundant finish details.

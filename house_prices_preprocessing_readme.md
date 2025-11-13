# House Prices — Data Cleaning & Preprocessing

**Project:** House Prices — Synthetic dataset preprocessing and baseline model

**Purpose**
This repository contains a small project that demonstrates a full Data Cleaning & Preprocessing pipeline (in the style of the Titanic project) applied to a House Prices dataset (synthetic). The goal is to teach the common steps you’ll need for preparing tabular data for machine learning: missing-value handling, encoding, scaling, outlier detection/removal, simple feature engineering, and training a baseline model.

---

## Contents

```
README.md
house_prices.csv                 # synthetic dataset (1,500 rows)
house_prices_cleaned.csv         # cleaned dataset produced by the script
house_prices_pipeline.py         # main script that runs cleaning + modeling
rf_regressor_baseline.joblib      # trained RandomForestRegressor model
```

> Note: file names above match the defaults used in `house_prices_pipeline.py`. Change the paths in the script if you move files.

---

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Install dependencies with pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## Quick Project Overview

1. **Load data** from `house_prices.csv`.
2. **Inspect** the dataset for missing values and data types.
3. **Impute** missing values:
   - numeric columns → median
   - categorical columns → most frequent (mode)
4. **Encode** categorical features with one-hot encoding (drop_first=True).
5. **Scale** numeric features with `StandardScaler`.
6. **Split** data into train/test sets (default `test_size=0.2`).
7. **Train** a baseline `RandomForestRegressor`.
8. **Evaluate** with RMSE, MAE and R².
9. **Save** cleaned dataset and trained model to disk.

---

## Usage

1. Place `house_prices.csv` in the same folder as the script (or update the `RAW_PATH` variable in the script).
2. Run the main pipeline script:

```bash
python house_prices_pipeline.py
```

Output files (defaults):
- `house_prices_cleaned.csv` — cleaned dataframe (numeric features scaled and one-hot encoded categoricals).
- `rf_regressor_baseline.joblib` — trained RandomForestRegressor model.

If you want a different CSV name or location, edit the `RAW_PATH`, `CLEANED_OUT`, and `MODEL_OUT` variables near the top of the script.

---

## Files and Key Functions

- `house_prices_pipeline.py` contains a linear pipeline with the following sections:
  - **Loading** & basic inspection
  - **Basic cleaning** (drop ID columns, check target exists)
  - **Imputation** for numeric and categorical features
  - **Encoding** using `pd.get_dummies()`
  - **Scaling** with `StandardScaler`
  - **Train/test split**
  - **Train model** using `RandomForestRegressor`
  - **Evaluation** and plotting (Actual vs Predicted)
  - **Saving outputs**

You can modularize this script into functions (e.g., `load_data`, `impute`, `encode`, `scale`, `train`) if you plan to expand it.

---

## Notes & Tips

- **ID column:** The script drops common ID columns like `Id` automatically. If your dataset uses a different id name, either keep it or add it to the drop list.

- **Domain-specific imputation:** Some columns are better handled with domain logic. For example:
  - `MasVnrArea` when `MasVnrType` is missing: set area to 0 and/or add a `MasVnr_missing` flag.
  - `GarageYrBlt` could be filled with `YearBuilt` if missing.

- **Encoding high-cardinality columns:** If one-hot encoding creates a very wide matrix (many neighborhoods), consider frequency-encoding or target-encoding instead.

- **Outliers:** The script does not remove outliers by default. Use IQR or z-score-based methods to detect and drop or cap extreme values if desired.

- **Modeling improvements:** Try `LightGBM`/`XGBoost`, feature interactions, hyperparameter tuning (GridSearchCV / RandomizedSearchCV), or cross-validation for better performance.

---

## Example: Recommended next steps (for a coursework/project report)

1. Create an EDA (Exploratory Data Analysis) notebook with:
   - Histograms for numeric features
   - Boxplots to visualize outliers
   - Correlation heatmap against `SalePrice`

2. Document your imputation choices: why median? why mode? show before/after plots.

3. Try two pipelines: (A) one-hot encoding + scaling, (B) frequency/target encoding + scaling. Compare performance.

4. Use 5-fold cross-validation when tuning hyperparameters and report mean ± std of RMSE.

5. Add explainability: show feature importances and SHAP values for the final model.

---

## License

This project and the synthetic dataset are provided for educational use. Feel free to reuse and modify the code.

---

## Contact

If you want a notebook version, extended EDA, or automated pipeline (sklearn `Pipeline` + `ColumnTransformer`), reply and I will provide it.
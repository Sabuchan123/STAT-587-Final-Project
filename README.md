# STAT-587 Final Project

This repository contains a set of tools and models for forecasting the direction of the S&P 500 using historical stock data.

## 📁 `Project/Models` Overview
The `Project/Models` directory contains the core modeling and evaluation code used in this project.

### 🧩 Key Components
- **Data preparation & feature engineering**
  - `H_prep.py` — imports raw data, cleans it, generates technical features, and produces classification targets.
  - `H_reduce.py` — auxiliary reduction/feature-selection utilities (e.g., step-wise regression using walk-forward validation).

- **Evaluation & metrics**
  - `H_eval.py` — contains evaluation helpers (rolling/forward-window backtesting, metric reporting, plotting utilities).
  - `H_helpers.py` — general helpers like logging, parameter formatting, and filesystem helpers.

- **Exploratory analysis & plots**
  - `EDA.py` — plotting utilities for inspecting index behavior, return distributions, correlations, and sector-level metrics.

- **Modeling scripts**
  - `logistic_regression.py` — logistic regression pipeline (with built-in cross-validation and backtesting).
  - `random_forest.py` — random forest classification pipeline (with optional LASSO feature selection and backtesting).
  - `SVM.py` — support vector machine classification pipeline (with grid search and backtesting).

- **Results**
  - `results/` — stores result CSVs and generated figures.

- **Notebook**
  - `guide.ipynb` — an interactive walkthrough that demonstrates how to load data, run models, and visualize results.

## 🚀 How to Run
> These scripts are designed to be used as modules (imported and called), rather than executed as standalone scripts.

A common workflow is to run `guide.ipynb`, which demonstrates how to:
1. Import the data via `H_prep.import_data()`
2. Clean the data with `H_prep.clean_data()`
3. Train and evaluate models (e.g., `run_logistic_regression`, `run_random_forest_classification`, `run_SVM_model`)

### Example (Python REPL / script)
```python
from Project.Models import H_prep
from Project.Models.logistic_regression import run_logistic_regression

DATA = H_prep.import_data(testing=True)
run_logistic_regression(DATA, FIND_OPTIMAL=False, DISPLAY_GRAPHS=True)
```

## 🧰 Dependencies
This project uses the `Project/env` virtual environment (Python 3.12). Key dependencies include:
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `pyarrow` (for reading parquet files)

If you need to install dependencies, activate the virtualenv and install via `pip install -r requirements.txt` (or install packages individually).

## 📂 Data
Input data is stored under `Project/Data/`, including:
- `raw_data_8_years.parquet` (main dataset)
- `stock_lookup_table.csv` (ticker → sector mapping)

## 📝 Notes
- The modeling code assumes a time-series forecasting setting (train/test split is sequential, not random).
- Most model scripts include optional hyperparameter tuning via `FIND_OPTIMAL=True`.
- Results are appended to `Project/Models/results/results.csv` when `EXPORT=True`.

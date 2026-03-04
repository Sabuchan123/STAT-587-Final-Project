import os
import joblib
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from helper_functions import get_cwd
from model_evaluation import rolling_window_backtest, get_final_metrics

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def to_binary_class(y):
    return (y >= 0).astype(int)

if __name__ == "__main__":
    # ------- Load raw data (no feature engineering) -------
    # Set testing=True for the 2-year dataset; False for the full 8-year dataset.
    TESTING = True

    cwd = get_cwd("STAT-587-Final-Project")
    parquet_file = "raw_data_2_years.parquet" if TESTING else "raw_data_8_years.parquet"

    print("------- Loading Raw Data")
    table = pq.read_table(cwd / "PyScripts" / "Data" / parquet_file)
    DATA = table.to_pandas()
    print(f"Loaded: {DATA.shape[0]} rows, {DATA.shape[1]} columns.")

    idx = pd.IndexSlice

    # ------- Clean: drop holiday rows and tickers with missing data -------
    STOCKS = DATA.loc[:, idx[:, 'Stocks', :]].dropna(how="all", axis=0)
    STOCKS = STOCKS.dropna(how="any", axis=1)
    to_drop = STOCKS.index[STOCKS.isna().all(axis=1)]
    STOCKS = STOCKS.drop(index=to_drop)

    # ------- Extract only Close, Open, High, Low, Volume for Stocks -------
    RAW_METRICS = ['Close', 'Open', 'High', 'Low', 'Volume']
    X = STOCKS.loc[:, idx[RAW_METRICS, :, :]].copy()

    # Flatten multi-level columns to single strings: "Close_AAPL", "Volume_MSFT", etc.
    X.columns = [f"{metric}_{ticker}" for metric, _, ticker in X.columns]
    print(f"Feature matrix shape: {X.shape[0]} rows, {X.shape[1]} columns.")

    # ------- Build target: next-day SPX return (binary: up=1, down=0) -------
    spx_return = (
        (DATA.loc[:, idx['Close', 'Index', '^SPX']] - DATA.loc[:, idx['Open', 'Index', '^SPX']])
        / DATA.loc[:, idx['Open', 'Index', '^SPX']]
    ).shift(-1).rename("Target")

    # Align X and target, drop any remaining NAs
    combined = X.join(spx_return).dropna()
    X = combined.drop(columns=["Target"])
    y_regression = combined["Target"]

    y_classification = to_binary_class(y_regression)
    print(f"Final shape — X: {X.shape}, y: {y_classification.shape}")

    # ------- Train/test split (80/20) -------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=1
    )

    tscv = TimeSeriesSplit(n_splits=3)

    # ------- Baseline: plain Logistic Regression (no regularization tuning) -------
    print("\n========== BASELINE: Plain Logistic Regression ==========")
    pipeline_base = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegression(C=1.0, solver='saga', random_state=1,
                                          max_iter=500, tol=1e-2))
    ])
    pipeline_base.fit(X_train, y_train)
    print("\n--- Rolling Window Backtest ---")
    rolling_window_backtest(pipeline_base, X, y_classification, verbose=1)
    print("\n--- Model Report ---")
    get_final_metrics(pipeline_base, X_train, y_train, X_test, y_test)

    # ------- Ridge (L2): LogisticRegressionCV — stores per-fold scores for bias-variance plot -------
    print("\n========== RIDGE (L2) Logistic Regression CV ==========")
    _ridge_cache = os.path.join(CACHE_DIR, 'base_ridge_cv.pkl')
    if os.path.exists(_ridge_cache):
        print("Loading Ridge CV pipeline from cache...")
        pipeline_ridge = joblib.load(_ridge_cache)
    else:
        pipeline_ridge = Pipeline([
            ('scaler',     StandardScaler()),
            ('classifier', LogisticRegressionCV(
                Cs=20, cv=tscv, penalty='l2', solver='saga',
                random_state=1, n_jobs=-1, max_iter=500, tol=1e-2
            ))
        ])
        pipeline_ridge.fit(X_train, y_train)
        joblib.dump(pipeline_ridge, _ridge_cache)

    ridge_cv = pipeline_ridge.named_steps['classifier']
    print(f"Best C (Ridge): {ridge_cv.C_[0]:.6f}")
    print(f"Cs: {ridge_cv.Cs_}")
    print("\n--- Model Report ---")
    get_final_metrics(pipeline_ridge, X_train, y_train, X_test, y_test)

    # ------- LASSO (L1): LogisticRegressionCV — stores per-fold scores for bias-variance plot -------
    print("\n========== LASSO (L1) Logistic Regression CV ==========")
    _lasso_cache = os.path.join(CACHE_DIR, 'base_lasso_cv.pkl')
    if os.path.exists(_lasso_cache):
        print("Loading LASSO CV pipeline from cache...")
        pipeline_lasso = joblib.load(_lasso_cache)
    else:
        pipeline_lasso = Pipeline([
            ('scaler',     StandardScaler()),
            ('classifier', LogisticRegressionCV(
                Cs=np.logspace(-6, 4, 20), cv=tscv, penalty='l1', solver='saga',
                random_state=1, n_jobs=-1, max_iter=500, tol=1e-2
            ))
        ])
        pipeline_lasso.fit(X_train, y_train)
        joblib.dump(pipeline_lasso, _lasso_cache)

    lasso_cv = pipeline_lasso.named_steps['classifier']
    print(f"Best C (LASSO): {lasso_cv.C_[0]:.6f}")
    print(f"Cs: {lasso_cv.Cs_}")
    print("\n--- Model Report ---")
    get_final_metrics(pipeline_lasso, X_train, y_train, X_test, y_test)

    # ------- Bias-Variance Tradeoff Plot (Ridge + LASSO) -------
    print("\n========== Generating Bias-Variance Tradeoff Plot ==========")

    def _bv_curves(cv_clf, pipeline, X_tr, y_tr, tscv_splitter, penalty, solver):
        """Extract CV test error from stored scores_ and compute train error per fold."""
        cs    = cv_clf.Cs_
        raw   = np.array(list(cv_clf.scores_.values())[0])
        if raw.ndim == 3:
            raw = raw[:, :, 0]                          # (n_folds, n_Cs)
        cv_err_mean = 1 - raw.mean(axis=0)
        cv_err_std  = raw.std(axis=0)

        scaler = pipeline.named_steps['scaler']
        train_scores = np.zeros_like(raw)
        for fold_idx, (tr, _) in enumerate(tscv_splitter.split(X_tr, y_tr)):
            X_fold = scaler.fit_transform(
                X_tr.iloc[tr] if hasattr(X_tr, 'iloc') else X_tr[tr])
            y_fold = y_tr.iloc[tr] if hasattr(y_tr, 'iloc') else y_tr[tr]
            for c_idx, c_val in enumerate(cs):
                clf = LogisticRegression(
                    C=c_val, penalty=penalty, solver=solver,
                    random_state=1, max_iter=500, tol=1e-2)
                clf.fit(X_fold, y_fold)
                train_scores[fold_idx, c_idx] = clf.score(X_fold, y_fold)
        tr_err_mean = 1 - train_scores.mean(axis=0)
        tr_err_std  = train_scores.std(axis=0)
        return cs, tr_err_mean, tr_err_std, cv_err_mean, cv_err_std

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Bias-Variance Tradeoff — Raw OHLCV Features\n(Train vs CV Test Error per Regularization Strength)',
                 fontsize=13, fontweight='bold')

    for ax, (label, cv_clf, pipe, pen, solv, color) in zip(axes, [
        ('Ridge (L2)', ridge_cv, pipeline_ridge, 'l2', 'saga', 'darkorange'),
        ('LASSO (L1)', lasso_cv, pipeline_lasso, 'l1', 'saga', 'seagreen'),
    ]):
        cs, tr_mean, tr_std, cv_mean, cv_std = _bv_curves(
            cv_clf, pipe, X_train, y_train, tscv, pen, solv)
        ax.semilogx(cs, tr_mean, marker='o', color='steelblue', linewidth=2, label='Train error')
        ax.fill_between(cs, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color='steelblue')
        ax.semilogx(cs, cv_mean, marker='s', color=color,      linewidth=2, label='CV Test error')
        ax.fill_between(cs, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color=color)
        ax.axvline(cv_clf.C_[0], color='red', linestyle='--',
                   label=f'Best C = {cv_clf.C_[0]:.4f}')
        ax.set_title(f'{label} — Bias-Variance Tradeoff')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Prediction Error')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'base_logistic_bias_variance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path)}")
    plt.close()

    # ------- Train vs Test Error Plot (direct, no CV averaging) -------
    print("\n========== Generating Train vs Test Error Plot (Direct Split) ==========")

    C_grid = np.logspace(-10, 2, 30)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Train vs Test Error — Raw OHLCV Features\n(Direct Train/Test Split, No CV)',
                  fontsize=13, fontweight='bold')

    for ax, (label, pen, solv, color) in zip(axes2, [
        ('Ridge (L2)', 'l2', 'saga', 'darkorange'),
        ('LASSO (L1)', 'l1', 'saga', 'seagreen'),
    ]):
        train_errors, test_errors = [], []
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)
        for c_val in C_grid:
            clf = LogisticRegression(
                C=c_val, penalty=pen, solver=solv,
                random_state=1, max_iter=500, tol=1e-2)
            clf.fit(X_tr_sc, y_train)
            train_errors.append(1 - clf.score(X_tr_sc, y_train))
            test_errors.append(1 - clf.score(X_te_sc, y_test))

        ax.semilogx(C_grid, train_errors, marker='o', color='steelblue',
                    linewidth=2, label='Train error')
        ax.semilogx(C_grid, test_errors, marker='s', color=color,
                    linewidth=2, label='Test error')
        ax.set_title(f'{label} — Train vs Test Error')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Prediction Error')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(output_dir, 'base_logistic_train_test.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path2)}")
    plt.close()

    # ===================================================================
    # PCA PRE-PROCESSING  (n_components=0.9  →  90% explained variance)
    # ===================================================================
    print("\n========== Applying PCA (90% variance) ==========")
    pca = PCA(n_components=0.9)
    scaler_pca = StandardScaler()
    X_train_sc  = scaler_pca.fit_transform(X_train)
    X_test_sc   = scaler_pca.transform(X_test)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca  = pca.transform(X_test_sc)
    print(f"PCA reduced {X_train.shape[1]} features → {X_train_pca.shape[1]} components (90% variance)")

    # ------- Ridge CV after PCA — cached -------
    print("\n========== RIDGE (L2) + PCA — LogisticRegressionCV ==========")
    _ridge_pca_cache = os.path.join(CACHE_DIR, 'base_ridge_pca_cv.pkl')
    if os.path.exists(_ridge_pca_cache):
        print("Loading Ridge+PCA CV from cache...")
        clf_ridge_pca = joblib.load(_ridge_pca_cache)
    else:
        clf_ridge_pca = LogisticRegressionCV(
            Cs=20, cv=tscv, penalty='l2', solver='saga',
            random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
        clf_ridge_pca.fit(X_train_pca, y_train)
        joblib.dump(clf_ridge_pca, _ridge_pca_cache)
    print(f"Best C (Ridge+PCA): {clf_ridge_pca.C_[0]:.6f}")
    print(f"Train accuracy: {clf_ridge_pca.score(X_train_pca, y_train):.4f}")
    print(f"Test  accuracy: {clf_ridge_pca.score(X_test_pca,  y_test):.4f}")

    # ------- LASSO CV after PCA — cached -------
    print("\n========== LASSO (L1) + PCA — LogisticRegressionCV ==========")
    _lasso_pca_cache = os.path.join(CACHE_DIR, 'base_lasso_pca_cv.pkl')
    if os.path.exists(_lasso_pca_cache):
        print("Loading LASSO+PCA CV from cache...")
        clf_lasso_pca = joblib.load(_lasso_pca_cache)
    else:
        clf_lasso_pca = LogisticRegressionCV(
            Cs=np.logspace(-6, 4, 20), cv=tscv, penalty='l1', solver='saga',
            random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
        clf_lasso_pca.fit(X_train_pca, y_train)
        joblib.dump(clf_lasso_pca, _lasso_pca_cache)
    print(f"Best C (LASSO+PCA): {clf_lasso_pca.C_[0]:.6f}")
    print(f"Train accuracy: {clf_lasso_pca.score(X_train_pca, y_train):.4f}")
    print(f"Test  accuracy: {clf_lasso_pca.score(X_test_pca,  y_test):.4f}")

    # ------- Bias-Variance Tradeoff Plot (PCA + Ridge/LASSO) -------
    print("\n========== Generating PCA Bias-Variance Tradeoff Plot ==========")

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('Bias-Variance Tradeoff — PCA Features (90% Variance)\n'
                  '(Train vs CV Test Error per Regularization Strength)',
                  fontsize=13, fontweight='bold')

    for ax, (label, cv_clf, pen, solv, color) in zip(axes3, [
        ('Ridge (L2)', clf_ridge_pca, 'l2', 'saga', 'darkorange'),
        ('LASSO (L1)', clf_lasso_pca, 'l1', 'saga', 'seagreen'),
    ]):
        cs   = cv_clf.Cs_
        raw  = np.array(list(cv_clf.scores_.values())[0])
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        cv_err_mean = 1 - raw.mean(axis=0)
        cv_err_std  = raw.std(axis=0)

        train_scores = np.zeros_like(raw)
        for fold_idx, (tr, _) in enumerate(tscv.split(X_train_pca, y_train)):
            X_fold = X_train_pca[tr]
            y_fold = y_train.iloc[tr] if hasattr(y_train, 'iloc') else y_train[tr]
            for c_idx, c_val in enumerate(cs):
                clf = LogisticRegression(
                    C=c_val, penalty=pen, solver=solv,
                    random_state=1, max_iter=500, tol=1e-2)
                clf.fit(X_fold, y_fold)
                train_scores[fold_idx, c_idx] = clf.score(X_fold, y_fold)
        tr_err_mean = 1 - train_scores.mean(axis=0)
        tr_err_std  = train_scores.std(axis=0)

        ax.semilogx(cs, tr_err_mean, marker='o', color='steelblue', linewidth=2, label='Train error')
        ax.fill_between(cs, tr_err_mean - tr_err_std, tr_err_mean + tr_err_std, alpha=0.15, color='steelblue')
        ax.semilogx(cs, cv_err_mean, marker='s', color=color, linewidth=2, label='CV Test error')
        ax.fill_between(cs, cv_err_mean - cv_err_std, cv_err_mean + cv_err_std, alpha=0.15, color=color)
        ax.axvline(cv_clf.C_[0], color='red', linestyle='--',
                   label=f'Best C = {cv_clf.C_[0]:.4f}')
        ax.set_title(f'{label} — Bias-Variance Tradeoff (PCA)')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Prediction Error')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path3 = os.path.join(output_dir, 'base_logistic_pca_bias_variance.png')
    plt.savefig(out_path3, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path3)}")
    plt.close()

    # ------- Train vs Test Error Plot after PCA (direct split) -------
    print("\n========== Generating PCA Train vs Test Error Plot (Direct Split) ==========")

    C_grid_pca = np.logspace(-5, 2, 30)

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    fig4.suptitle('Train vs Test Error — PCA Features (90% Variance)\n'
                  '(Direct Train/Test Split, No CV)',
                  fontsize=13, fontweight='bold')

    for ax, (label, pen, solv, color) in zip(axes4, [
        ('Ridge (L2)', 'l2', 'saga', 'darkorange'),
        ('LASSO (L1)', 'l1', 'saga', 'seagreen'),
    ]):
        train_errors, test_errors = [], []
        for c_val in C_grid_pca:
            clf = LogisticRegression(
                C=c_val, penalty=pen, solver=solv,
                random_state=1, max_iter=500, tol=1e-2)
            clf.fit(X_train_pca, y_train)
            train_errors.append(1 - clf.score(X_train_pca, y_train))
            test_errors.append(1 - clf.score(X_test_pca, y_test))

        ax.semilogx(C_grid_pca, train_errors, marker='o', color='steelblue',
                    linewidth=2, label='Train error')
        ax.semilogx(C_grid_pca, test_errors, marker='s', color=color,
                    linewidth=2, label='Test error')
        ax.set_title(f'{label} — Train vs Test Error (PCA)')
        ax.set_xlabel('C  (Inverse Regularization Strength)\n'
                      '← High Regularization, Simpler Model      '
                      'Low Regularization, More Complex →')
        ax.set_ylabel('Prediction Error')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path4 = os.path.join(output_dir, 'base_logistic_pca_train_test.png')
    plt.savefig(out_path4, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.abspath(out_path4)}")
    plt.close()

    # ===================================================================
    # MODEL COMPARISON TABLE
    # ===================================================================
    print("\n========== Model Comparison Table ==========")
    from sklearn.model_selection import cross_validate as _cv
    from sklearn.metrics import precision_score, recall_score, f1_score

    def _metrics(name, model, X_tr, y_tr, X_te, y_te, tscv_splitter):
        cv_res = _cv(model, X_tr, y_tr, cv=tscv_splitter,
                     return_train_score=True, n_jobs=-1)
        preds = model.predict(X_te)
        return {
            'Model':              name,
            'Avg Train Acc':      round(cv_res['train_score'].mean(), 4),
            'Std Train Acc':      round(cv_res['train_score'].std(),  4),
            'Avg CV Test Acc':    round(cv_res['test_score'].mean(),  4),
            'Std CV Test Acc':    round(cv_res['test_score'].std(),   4),
            'Hold-out Test Acc':  round(model.score(X_te, y_te),      4),
            'Precision':          round(precision_score(y_te, preds, zero_division=0), 4),
            'Recall':             round(recall_score(y_te, preds,    zero_division=0), 4),
            'F1':                 round(f1_score(y_te, preds,         zero_division=0), 4),
        }

    rows = [
        _metrics('Baseline (C=1)',    pipeline_base,  X_train, y_train, X_test, y_test, tscv),
        _metrics('Ridge CV (raw)',    pipeline_ridge, X_train, y_train, X_test, y_test, tscv),
        _metrics('LASSO CV (raw)',    pipeline_lasso, X_train, y_train, X_test, y_test, tscv),
        _metrics('Ridge CV (PCA)',    clf_ridge_pca,  X_train_pca, y_train, X_test_pca, y_test, tscv),
        _metrics('LASSO CV (PCA)',    clf_lasso_pca,  X_train_pca, y_train, X_test_pca, y_test, tscv),
    ]

    comparison_df = pd.DataFrame(rows).set_index('Model')
    print(comparison_df.to_string())

    # Save as CSV (raw-only, will be extended below)
    csv_path = os.path.join(output_dir, 'base_logistic_comparison.csv')
    comparison_df.to_csv(csv_path)
    print(f"\nComparison table (raw) saved to: {os.path.abspath(csv_path)}")

    # ===================================================================
    # DAY-OF-WEEK EXTENSION
    # Add one-hot encoded day-of-week (Mon–Fri) to raw OHLCV features
    # then re-run the same 5 models
    # ===================================================================
    print("\n========== Adding Day-of-Week Features ==========")

    dow_dummies = pd.get_dummies(X.index.dayofweek, prefix='DOW').astype(float)
    dow_dummies.index = X.index
    # Keep only Mon–Thu (drop Fri to avoid dummy variable trap)
    dow_dummies = dow_dummies.iloc[:, :-1]
    print(f"Day-of-week columns added: {list(dow_dummies.columns)}")

    X_dow = pd.concat([X, dow_dummies], axis=1)
    X_train_dow, X_test_dow, _, _ = train_test_split(
        X_dow, y_classification, test_size=0.2, random_state=1
    )
    print(f"Feature matrix with DOW: {X_dow.shape[1]} columns")

    # --- Baseline + DOW ---
    print("\n========== BASELINE + DOW ==========")
    pipeline_base_dow = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegression(C=1.0, solver='saga', random_state=1,
                                          max_iter=500, tol=1e-2))
    ])
    pipeline_base_dow.fit(X_train_dow, y_train)

    # --- Ridge CV + DOW — cached ---
    print("\n========== RIDGE CV + DOW ==========")
    _ridge_dow_cache = os.path.join(CACHE_DIR, 'base_ridge_dow_cv.pkl')
    if os.path.exists(_ridge_dow_cache):
        print("Loading Ridge+DOW CV from cache...")
        pipeline_ridge_dow = joblib.load(_ridge_dow_cache)
    else:
        pipeline_ridge_dow = Pipeline([
            ('scaler',     StandardScaler()),
            ('classifier', LogisticRegressionCV(
                Cs=20, cv=tscv, penalty='l2', solver='saga',
                random_state=1, n_jobs=-1, max_iter=500, tol=1e-2
            ))
        ])
        pipeline_ridge_dow.fit(X_train_dow, y_train)
        joblib.dump(pipeline_ridge_dow, _ridge_dow_cache)
    print(f"Best C (Ridge+DOW): {pipeline_ridge_dow.named_steps['classifier'].C_[0]:.6f}")

    # --- LASSO CV + DOW — cached ---
    print("\n========== LASSO CV + DOW ==========")
    _lasso_dow_cache = os.path.join(CACHE_DIR, 'base_lasso_dow_cv.pkl')
    if os.path.exists(_lasso_dow_cache):
        print("Loading LASSO+DOW CV from cache...")
        pipeline_lasso_dow = joblib.load(_lasso_dow_cache)
    else:
        pipeline_lasso_dow = Pipeline([
            ('scaler',     StandardScaler()),
            ('classifier', LogisticRegressionCV(
                Cs=np.logspace(-6, 4, 20), cv=tscv, penalty='l1', solver='saga',
                random_state=1, n_jobs=-1, max_iter=500, tol=1e-2
            ))
        ])
        pipeline_lasso_dow.fit(X_train_dow, y_train)
        joblib.dump(pipeline_lasso_dow, _lasso_dow_cache)
    print(f"Best C (LASSO+DOW): {pipeline_lasso_dow.named_steps['classifier'].C_[0]:.6f}")

    # --- PCA + DOW ---
    print("\n========== Applying PCA on DOW features (90% variance) ==========")
    pca_dow = PCA(n_components=0.9)
    scaler_pca_dow = StandardScaler()
    X_train_dow_sc  = scaler_pca_dow.fit_transform(X_train_dow)
    X_test_dow_sc   = scaler_pca_dow.transform(X_test_dow)
    X_train_dow_pca = pca_dow.fit_transform(X_train_dow_sc)
    X_test_dow_pca  = pca_dow.transform(X_test_dow_sc)
    print(f"PCA reduced {X_train_dow.shape[1]} features → {X_train_dow_pca.shape[1]} components")

    # --- Ridge CV + PCA + DOW — cached ---
    print("\n========== RIDGE CV + PCA + DOW ==========")
    _ridge_pca_dow_cache = os.path.join(CACHE_DIR, 'base_ridge_pca_dow_cv.pkl')
    if os.path.exists(_ridge_pca_dow_cache):
        print("Loading Ridge+PCA+DOW CV from cache...")
        clf_ridge_pca_dow = joblib.load(_ridge_pca_dow_cache)
    else:
        clf_ridge_pca_dow = LogisticRegressionCV(
            Cs=20, cv=tscv, penalty='l2', solver='saga',
            random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
        clf_ridge_pca_dow.fit(X_train_dow_pca, y_train)
        joblib.dump(clf_ridge_pca_dow, _ridge_pca_dow_cache)
    print(f"Best C (Ridge+PCA+DOW): {clf_ridge_pca_dow.C_[0]:.6f}")

    # --- LASSO CV + PCA + DOW — cached ---
    print("\n========== LASSO CV + PCA + DOW ==========")
    _lasso_pca_dow_cache = os.path.join(CACHE_DIR, 'base_lasso_pca_dow_cv.pkl')
    if os.path.exists(_lasso_pca_dow_cache):
        print("Loading LASSO+PCA+DOW CV from cache...")
        clf_lasso_pca_dow = joblib.load(_lasso_pca_dow_cache)
    else:
        clf_lasso_pca_dow = LogisticRegressionCV(
            Cs=np.logspace(-6, 4, 20), cv=tscv, penalty='l1', solver='saga',
            random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
        clf_lasso_pca_dow.fit(X_train_dow_pca, y_train)
        joblib.dump(clf_lasso_pca_dow, _lasso_pca_dow_cache)
    print(f"Best C (LASSO+PCA+DOW): {clf_lasso_pca_dow.C_[0]:.6f}")

    # --- DOW rows using same _metrics helper ---
    rows_dow = [
        _metrics('Baseline+DOW',       pipeline_base_dow,  X_train_dow, y_train, X_test_dow, y_test, tscv),
        _metrics('Ridge CV+DOW',       pipeline_ridge_dow, X_train_dow, y_train, X_test_dow, y_test, tscv),
        _metrics('LASSO CV+DOW',       pipeline_lasso_dow, X_train_dow, y_train, X_test_dow, y_test, tscv),
        _metrics('Ridge CV+PCA+DOW',   clf_ridge_pca_dow,  X_train_dow_pca, y_train, X_test_dow_pca, y_test, tscv),
        _metrics('LASSO CV+PCA+DOW',   clf_lasso_pca_dow,  X_train_dow_pca, y_train, X_test_dow_pca, y_test, tscv),
    ]
    dow_df = pd.DataFrame(rows_dow).set_index('Model')
    print("\nDay-of-Week Models:")
    print(dow_df.to_string())

    # ===================================================================
    # COMBINED COMPARISON TABLE (raw OHLCV vs raw OHLCV + DOW)
    # ===================================================================
    combined_df = pd.concat([comparison_df, dow_df])
    combined_df.index.name = 'Model'

    print("\n===== Combined Comparison Table (raw + DOW) =====")
    print(combined_df.to_string())

    # ===================================================================
    # LAG FEATURES  (lag1 … lag7 of raw OHLCV, no DOW)
    # ===================================================================
    print("\n========== Adding Lag1–Lag7 of Raw OHLCV Features ==========")

    lag_parts = [X]
    for lag in range(1, 8):
        lag_parts.append(X.shift(lag).add_suffix(f'_lag{lag}'))
    X_lag = pd.concat(lag_parts, axis=1)

    # Re-align with target (dropna removes first 7 rows that have NaN lags)
    combined_lag = X_lag.join(spx_return).dropna()
    X_lag = combined_lag.drop(columns=["Target"])
    y_lag = to_binary_class(combined_lag["Target"])
    print(f"Feature matrix with lags: {X_lag.shape[1]} columns, {X_lag.shape[0]} rows")

    X_train_lag, X_test_lag, y_train_lag, y_test_lag = train_test_split(
        X_lag, y_lag, test_size=0.2, random_state=1
    )

    # --- Baseline + Lags ---
    print("\n========== BASELINE + Lags ==========")
    pipeline_base_lag = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', LogisticRegression(C=1.0, solver='saga', random_state=1,
                                          max_iter=500, tol=1e-2))
    ])
    pipeline_base_lag.fit(X_train_lag, y_train_lag)

    # --- Ridge CV + Lags — cached ---
    print("\n========== RIDGE CV + Lags ==========")
    _ridge_lag_cache = os.path.join(CACHE_DIR, 'base_ridge_lag_cv.pkl')
    if os.path.exists(_ridge_lag_cache):
        print("Loading Ridge+Lag CV from cache...")
        pipeline_ridge_lag = joblib.load(_ridge_lag_cache)
    else:
        pipeline_ridge_lag = Pipeline([
            ('scaler',     StandardScaler()),
            ('classifier', LogisticRegressionCV(
                Cs=20, cv=tscv, penalty='l2', solver='saga',
                random_state=1, n_jobs=-1, max_iter=500, tol=1e-2
            ))
        ])
        pipeline_ridge_lag.fit(X_train_lag, y_train_lag)
        joblib.dump(pipeline_ridge_lag, _ridge_lag_cache)
    print(f"Best C (Ridge+Lag): {pipeline_ridge_lag.named_steps['classifier'].C_[0]:.6f}")

    # --- LASSO CV + Lags — cached ---
    print("\n========== LASSO CV + Lags ==========")
    _lasso_lag_cache = os.path.join(CACHE_DIR, 'base_lasso_lag_cv.pkl')
    if os.path.exists(_lasso_lag_cache):
        print("Loading LASSO+Lag CV from cache...")
        pipeline_lasso_lag = joblib.load(_lasso_lag_cache)
    else:
        pipeline_lasso_lag = Pipeline([
            ('scaler',     StandardScaler()),
            ('classifier', LogisticRegressionCV(
                Cs=np.logspace(-6, 4, 20), cv=tscv, penalty='l1', solver='saga',
                random_state=1, n_jobs=-1, max_iter=500, tol=1e-2
            ))
        ])
        pipeline_lasso_lag.fit(X_train_lag, y_train_lag)
        joblib.dump(pipeline_lasso_lag, _lasso_lag_cache)
    print(f"Best C (LASSO+Lag): {pipeline_lasso_lag.named_steps['classifier'].C_[0]:.6f}")

    # --- PCA + Lags ---
    print("\n========== Applying PCA on Lag features (90% variance) ==========")
    pca_lag = PCA(n_components=0.9)
    scaler_pca_lag = StandardScaler()
    X_train_lag_sc  = scaler_pca_lag.fit_transform(X_train_lag)
    X_test_lag_sc   = scaler_pca_lag.transform(X_test_lag)
    X_train_lag_pca = pca_lag.fit_transform(X_train_lag_sc)
    X_test_lag_pca  = pca_lag.transform(X_test_lag_sc)
    print(f"PCA reduced {X_train_lag.shape[1]} features → {X_train_lag_pca.shape[1]} components")

    # --- Ridge CV + PCA + Lags — cached ---
    print("\n========== RIDGE CV + PCA + Lags ==========")
    _ridge_pca_lag_cache = os.path.join(CACHE_DIR, 'base_ridge_pca_lag_cv.pkl')
    if os.path.exists(_ridge_pca_lag_cache):
        print("Loading Ridge+PCA+Lag CV from cache...")
        clf_ridge_pca_lag = joblib.load(_ridge_pca_lag_cache)
    else:
        clf_ridge_pca_lag = LogisticRegressionCV(
            Cs=20, cv=tscv, penalty='l2', solver='saga',
            random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
        clf_ridge_pca_lag.fit(X_train_lag_pca, y_train_lag)
        joblib.dump(clf_ridge_pca_lag, _ridge_pca_lag_cache)
    print(f"Best C (Ridge+PCA+Lag): {clf_ridge_pca_lag.C_[0]:.6f}")

    # --- LASSO CV + PCA + Lags — cached ---
    print("\n========== LASSO CV + PCA + Lags ==========")
    _lasso_pca_lag_cache = os.path.join(CACHE_DIR, 'base_lasso_pca_lag_cv.pkl')
    if os.path.exists(_lasso_pca_lag_cache):
        print("Loading LASSO+PCA+Lag CV from cache...")
        clf_lasso_pca_lag = joblib.load(_lasso_pca_lag_cache)
    else:
        clf_lasso_pca_lag = LogisticRegressionCV(
            Cs=np.logspace(-6, 4, 20), cv=tscv, penalty='l1', solver='saga',
            random_state=1, n_jobs=-1, max_iter=500, tol=1e-2)
        clf_lasso_pca_lag.fit(X_train_lag_pca, y_train_lag)
        joblib.dump(clf_lasso_pca_lag, _lasso_pca_lag_cache)
    print(f"Best C (LASSO+PCA+Lag): {clf_lasso_pca_lag.C_[0]:.6f}")

    rows_lag = [
        _metrics('Baseline+Lags',      pipeline_base_lag,  X_train_lag, y_train_lag, X_test_lag, y_test_lag, tscv),
        _metrics('Ridge CV+Lags',      pipeline_ridge_lag, X_train_lag, y_train_lag, X_test_lag, y_test_lag, tscv),
        _metrics('LASSO CV+Lags',      pipeline_lasso_lag, X_train_lag, y_train_lag, X_test_lag, y_test_lag, tscv),
        _metrics('Ridge CV+PCA+Lags',  clf_ridge_pca_lag,  X_train_lag_pca, y_train_lag, X_test_lag_pca, y_test_lag, tscv),
        _metrics('LASSO CV+PCA+Lags',  clf_lasso_pca_lag,  X_train_lag_pca, y_train_lag, X_test_lag_pca, y_test_lag, tscv),
    ]
    lag_df = pd.DataFrame(rows_lag).set_index('Model')
    print("\nLag Models:")
    print(lag_df.to_string())

    # ===================================================================
    # FINAL COMBINED TABLE: raw | raw+DOW | raw+Lags
    # ===================================================================
    full_df = pd.concat([comparison_df, dow_df, lag_df])
    full_df.index.name = 'Model'

    print("\n===== Full Comparison Table =====")
    print(full_df.to_string())

    # Save CSV
    combined_csv = os.path.join(output_dir, 'base_logistic_comparison.csv')
    full_df.to_csv(combined_csv)
    print(f"\nFull comparison table saved to: {os.path.abspath(combined_csv)}")

    # Build LaTeX manually with \midrule separating the three groups
    tex_path = os.path.join(output_dir, 'base_logistic_comparison.tex')
    col_fmt  = 'l' + 'r' * len(full_df.columns)
    col_header = ' & '.join(['Model'] + list(full_df.columns)) + r' \\'
    lasso_note = (
        r'$^\dagger$ Degenerate classifier: optimal $C = 10^{-6}$ shrinks all '
        r'coefficients to zero; model predicts majority class for every observation '
        r'(Recall = 1.0, Precision $\approx$ base rate).'
    )

    def _row(name, vals):
        dagger = r'$^\dagger$' if 'LASSO' in name else ''
        return name + dagger + ' & ' + ' & '.join(f'{v:.4f}' for v in vals) + r' \\'

    with open(tex_path, 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Logistic Regression Model Comparison: '
                r'Raw OHLCV vs Raw OHLCV + Day-of-Week vs Raw OHLCV + Lag1--7}' + '\n')
        f.write(r'\label{tab:base_logistic_comparison}' + '\n')
        f.write(r'\begin{tabular}{' + col_fmt + '}\n')
        f.write(r'\toprule' + '\n')
        f.write(col_header + '\n')
        f.write(r'\midrule' + '\n')
        # Group 1: raw
        for name, row in comparison_df.iterrows():
            f.write(_row(name, row.values) + '\n')
        f.write(r'\midrule' + '\n')
        # Group 2: DOW
        for name, row in dow_df.iterrows():
            f.write(_row(name, row.values) + '\n')
        f.write(r'\midrule' + '\n')
        # Group 3: Lags
        for name, row in lag_df.iterrows():
            f.write(_row(name, row.values) + '\n')
        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\par\smallskip' + '\n')
        f.write(r'\footnotesize ' + lasso_note + '\n')
        f.write(r'\end{table}' + '\n')

    print(f"LaTeX table saved to:           {os.path.abspath(tex_path)}")

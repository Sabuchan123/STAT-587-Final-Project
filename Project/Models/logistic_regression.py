from typing import Any, cast
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

from H_prep import clean_data, data_clean_param_selection, import_data
from H_eval import get_final_metrics, RollingWindowBacktest, utility_score, display_bias_variance_tradeoff
from H_helpers import log_result, append_params_to_dict, get_cwd

'''No need for hyperparameter tuning for Logistic Regression via GridSearchCV since LogisticRegressionCV performs internal CV to select the best C value. We will just use the default 10 values of C that LogisticRegressionCV tests.'''

VERBOSE=0

cwd=get_cwd("STAT-587-Final-Project")

if __name__=="__main__":
    WINDOW_SIZE=200
    HORIZON=40
    EXPORT=True
    TEST_SIZE=0.2
    tscv=TimeSeriesSplit(n_splits=5)
    custom_Cs=[0.05, 0.1, 1.0, 10.0]
    DATA=import_data()

    FIND_OPTIMAL=False
    W=4 # Greater w emphasizes more accuracy, lesser w emphasizes more robustness.

    parameters_={  # These are optimal as of 3/12/2026 11:00 PM w=4
        "lag_period": 2,
        "lookback_period": 18,
        "sector": True,
        "corr_threshold": 0.9,
        "corr_level": 2
    }

    if (FIND_OPTIMAL):
        # ------- Selection of Optimal lag_period and lookback_period Parameters -------
        base_Log_Reg_model=LogisticRegression(C=1.0, l1_ratio=1, solver='saga', class_weight='balanced', random_state=1, max_iter=1000, tol=1e-3, verbose=VERBOSE)
        base_Log_Reg_model_pipeline=Pipeline([('scaler', StandardScaler()), ('classifier', base_Log_Reg_model)])
        
        print("------- Finding Optimal lag_period Value")
        param_grid={
            'lag_period': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
            'sector': [True],
            'corr_level': [2]
        }

        for_display, best_parameters, best_score=data_clean_param_selection(*DATA, clone(base_Log_Reg_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, eff_support=True, w=W, **param_grid)
        display_bias_variance_tradeoff(for_display, key="lag_period", label='log_reg')
        best_lag=best_parameters['lag_period']
        print(f"Best Utility Score (lag_period): {best_score}")
        print(f"Best lag_period: {best_lag}")

        print("------- Finding Optimal lookback_period Value")
        param_grid={
            'lookback_period': [0, 7, 9, 11, 14, 16, 18, 21, 23, 25, 28],
            'lag_period': [best_lag],
            'sector': [True],
            'corr_level': [2]
        }
        
        for_display, best_parameters, best_score=data_clean_param_selection(*DATA, clone(base_Log_Reg_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, w=W, **param_grid)
        display_bias_variance_tradeoff(for_display, key="lookback_period", label='log_reg')
        best_lookback=best_parameters['lookback_period']
        print(f"Best Utility Score (lookback_period): {best_score}")
        print(f"Best lookback_period: {best_lookback}")

        # ------- Selection of Optimal data_clean() Parameters -------
        print("------- Finding Optimal data_clean() Parameters")
        param_grid={
            'lag_period': [best_lag],
            'lookback_period': [best_lookback],
            'sector': [True],
            'corr_level': [2],
            'corr_threshold': [0.8, 0.9, 0.95]
        }

        _, parameters_, best_score=data_clean_param_selection(*DATA, clone(base_Log_Reg_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, w=W, **param_grid)
        print(f"Best Utility Score {best_score}")
        print(f"Optimal parameter {parameters_}")


    download_params = {f"clean_data__{k}=": v for k, v in parameters_.items()}

    X, y_regression=cast(Any, clean_data(*DATA, **parameters_))
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_classification=to_binary_class(y_regression)
    X_train, X_test, y_train, y_test=train_test_split(X, y_classification, test_size=TEST_SIZE, random_state=1, shuffle=False)

    # ------- LASSO(Internal) APPLICATION -------
    Log_Reg_R=LogisticRegressionCV(Cs=custom_Cs, cv=tscv, l1_ratios=[1], solver='saga', class_weight='balanced', random_state=1, n_jobs=-1, max_iter=500, tol=1e-2, verbose=VERBOSE)

    Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Log_Reg_R)])

    Log_Reg_model_pipeline_R.fit(X_train, y_train)

    classifier = Log_Reg_model_pipeline_R.named_steps['classifier']
    tested_Cs = classifier.Cs_
    fold_scores = classifier.scores_[1]
    if fold_scores.ndim == 3:
        fold_scores = fold_scores[:, :, 0]
    mean_cv_scores = np.mean(fold_scores, axis=0) 
    results_df = pd.DataFrame({
        'C': tested_Cs,
        'score': mean_cv_scores
    })
    display_bias_variance_tradeoff(results_df, key='C', label='Logistic_Regression_L1', baseline=False)

    best_c = Log_Reg_model_pipeline_R.named_steps['classifier'].C_[0]
    Opt_Log_Reg_R=LogisticRegression(C=best_c, l1_ratio=1, solver='saga', random_state=1, max_iter=500, tol=1e-2)

    Opt_Log_Reg_model_pipeline_R=Pipeline([('scaler', StandardScaler()), ('classifier', Opt_Log_Reg_R)])

    rwb_obj=RollingWindowBacktest(clone(Opt_Log_Reg_model_pipeline_R), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results(label="LR_LASSO", model="LR")

    optimized_Log_Reg_R_=clone(Opt_Log_Reg_model_pipeline_R)
    optimized_Log_Reg_R_.fit(X_train, y_train)

    results=get_final_metrics(optimized_Log_Reg_R_, X_train, y_train, X_test, y_test, n_splits=10, label="LASSO(int.) Log. Reg.")
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results.update({'w': W})
        results=append_params_to_dict(results, optimized_Log_Reg_R_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'Project' / 'Models' / 'results', "results.csv")
    input("Press Enter to Finish...")
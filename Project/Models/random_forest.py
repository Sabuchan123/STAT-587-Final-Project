#!/usr/bin/env python3
from typing import Any, cast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from H_reduce import step_wise_reg_wfv
from H_prep import clean_data, data_clean_param_selection, import_data
from H_eval import RollingWindowBacktest, get_final_metrics, utility_score, display_bias_variance_tradeoff
from H_helpers import log_result, get_cwd, append_params_to_dict

VERBOSE=0
WINDOW_SIZE=220
HORIZON=40
EXPORT=True

cwd=get_cwd("STAT-587-Final-Project")

if __name__=="__main__":
    WINDOW_SIZE=200
    HORIZON=40
    EXPORT=True
    TEST_SIZE=0.2
    DATA=import_data()

    FIND_OPTIMAL=False
    W=4 # Greater w emphasizes more accuracy, lesser w emphasizes more robustness.

    parameters_={ # These are optimal as of 3/8/2026 4:00 PM w=4
        "lag_period": [1, 2, 3, 4],
        "lookback_period": 28,
        "sector": True,
        "corr_threshold": 0.95,
        "corr_level": 2
    }

    if (FIND_OPTIMAL):
        # ------- Selection of Optimal lag_period and lookback_period Parameters -------
        base_RF_model=RandomForestClassifier(max_depth=10, n_estimators=250, random_state=1, n_jobs=-1, class_weight='balanced')
        base_RF_model_pipeline=Pipeline([('scaler', StandardScaler()), ('classifier', base_RF_model)])
        
        print("------- Finding Optimal lag_period Value")
        param_grid={
            'lag_period': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]],
            'sector': [True],
            'corr_level': [2]
        }

        for_display, best_parameters, best_score=data_clean_param_selection(*DATA, clone(base_RF_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, eff_support=True, w=W, **param_grid)
        display_bias_variance_tradeoff(for_display, key="lag_period", label='RF')
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
        
        for_display, best_parameters, best_score=data_clean_param_selection(*DATA, clone(base_RF_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, w=W, **param_grid)
        display_bias_variance_tradeoff(for_display, key="lookback_period", label='RF')
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

        _, parameters_, best_score=data_clean_param_selection(*DATA, clone(base_RF_model_pipeline), TEST_SIZE, WINDOW_SIZE, HORIZON, w=W, **param_grid)
        print(f"Best Utility Score {best_score}")
        print(f"Optimal parameter {parameters_}")
        

    download_params = {f"clean_data__{k}=": v for k, v in parameters_.items()}

    X, y_regression=cast(Any, clean_data(*DATA, **parameters_))
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_classification=to_binary_class(y_regression)
    X_train, X_test, y_train, y_test=train_test_split(X, y_classification, test_size=TEST_SIZE, random_state=1, shuffle=False)

    tscv=TimeSeriesSplit(n_splits=5) # CHANGEABLE (OPTIONAL)

    # ------- LASSO APPLICATION -------
    print("\n\n------- LASSO RF Model -------")
    lasso_selector=SelectFromModel(LogisticRegression(l1_ratio=1, solver='saga', random_state=1, class_weight='balanced', max_iter=500, tol=5e-2), threshold='mean')
    RFClassifier_red_lasso=RandomForestClassifier(random_state=1, n_jobs=-1, class_weight='balanced')

    RF_pipeline_lasso=Pipeline([('scaler', StandardScaler()), 
                              ('feature_selector', lasso_selector),
                              ('classifier', RFClassifier_red_lasso)])

    param_grid={
        'feature_selector__estimator__C': [0.001, 0.01, 0.1, 1], 
        'classifier__max_depth': [2, 3, 5, 10],              
        'classifier__n_estimators': [500]
    }
    grid_search_LASSO=GridSearchCV(RF_pipeline_lasso, param_grid, cv=tscv, n_jobs=-1, return_train_score=True, verbose=VERBOSE)
    grid_search_LASSO.fit(X_train, y_train)

    rwb_obj=RollingWindowBacktest(clone(grid_search_LASSO.best_estimator_), X, y_classification, X_train, WINDOW_SIZE, HORIZON)
    rwb_obj.rolling_window_backtest(verbose=1)
    rwb_obj.display_wfv_results(label="RF_LASSO", model="RF")

    optimized_LASSO_=clone(grid_search_LASSO.best_estimator_)
    optimized_LASSO_.fit(X_train, y_train)

    results=get_final_metrics(optimized_LASSO_, X_train, y_train, X_test, y_test, label="LASSO RF")
    util_score=utility_score(results, rwb_obj)
    print(f"Utility Score {util_score:.4}")
    if (EXPORT):
        results.update({'utility_score': round(util_score, 3)})
        results.update({'w': W})
        results=append_params_to_dict(results, optimized_LASSO_)
        results.update(rwb_obj.results[2])
        results.update(download_params)
        log_result(results, cwd / 'Project' / 'Models' / 'results', "results.csv")
        
    input("Press Enter to Finish...")

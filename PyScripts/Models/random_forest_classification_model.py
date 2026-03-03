#!/usr/bin/env python3
from typing import Any, cast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from dimension_reduction import step_wise_reg_wfv
# from model_evaluation import classification_accuracy, display_feat_importances, classification_wfv_eval
from data_preprocessing_and_cleaning import clean_data
from sklearn.pipeline import Pipeline

if __name__=="__main__":
    X, y_regression=cast(Any, clean_data())
    X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2, random_state=1)
    def to_binary_class(y):
        return (y>=0).astype(int)
    y_train=to_binary_class(y_train)
    y_test=to_binary_class(y_test)
    tscv=TimeSeriesSplit(n_splits=3)
    
    # ------- BASE APPLICATION -------
    # RFClassifier_base=RandomForestClassifier(random_state=1, n_jobs=-1)

    # RF_pipeline_base=Pipeline([('scaler', StandardScaler()), 
    #                            ('classifier', RFClassifier_base)])

    # param_grid={
    #     'classifier__max_depth': [2, 3, 5],
    #     'classifier__n_estimators': [250, 500]
    # }
    # grid_search_base=GridSearchCV(RF_pipeline_base, param_grid, cv=tscv, n_jobs=-1, return_train_score=True, verbose=3)
    # grid_search_base.fit(X_train, y_train)

    # display_feat_importances(RF_pipeline_base, X_train)

    # acc, avg_dir=classification_accuracy(RF_pipeline_base.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # classification_wfv_eval(RF_pipeline_base, X_train, y_train)

    input("Press Enter to continue...")
    # ------- PCA APPLICATION -------
    # RFClassifier_PCA=RandomForestClassifier(random_state=1, n_jobs=-1)

    # RF_pipeline_PCA=Pipeline([('scaler', StandardScaler()),
    #                           ('reducer', PCA()),
    #                           ('classifier', RFClassifier_PCA)])
    
    # param_grid={
    #     'reducer__n_components': [0.8, 0.95],
    #     'classifier__max_depth': [2, 3, 5],
    #     'classifier__n_estimators': [250, 500]
    # }
    # grid_search_PCA=GridSearchCV(RF_pipeline_PCA, param_grid, cv=tscv, n_jobs=-1, return_train_score=True, verbose=3)
    # grid_search_PCA.fit(X_train, y_train)

    # display_feat_importances(RFClassifier_red_PCA, X_train)

    # acc, avg_dir=classification_accuracy(RFClassifier_red_PCA.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # classification_wfv_eval(RFClassifier_red_PCA, X_train, y_train)

    input("Press Enter to continue...")
    # ------- LASSO APPLICATION -------
    # lasso_selector=SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=1), threshold='mean')
    # RFClassifier_red_lasso=RandomForestClassifier(random_state=1, n_jobs=-1)

    # RF_pipeline_lasso=Pipeline([('scaler', StandardScaler()), 
    #                           ('feature_selector', lasso_selector),
    #                           ('classifier', RFClassifier_red_lasso)])

    # param_grid={
    #     'feature_selector__estimator__C': [0.001, 0.01, 0.1], 
    #     'classifier__max_depth': [2, 3, 5],              
    #     'classifier__n_estimators': [500]
    # }
    # grid_search_LASSO=GridSearchCV(RF_pipeline_lasso, param_grid, cv=tscv, n_jobs=-1, return_train_score=True, verbose=3)
    # grid_search_LASSO.fit(X_train, y_train)

    # display_feat_importances(RFClassifier_red_LASSO, X_train)

    # acc, avg_dir=classification_accuracy(RFClassifier_red_LASSO.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # classification_wfv_eval(RFClassifier_red_LASSO, X_train, y_train)

    input("Press Enter to continue...")
    # ------- RIDGE APPLICATION -------
    # ridge_selector=SelectFromModel(LogisticRegression(penalty='l2', solver='liblinear', random_state=1), threshold='mean')
    # RFClassifier_red_ridge=RandomForestClassifier(random_state=1, n_jobs=-1)

    # RF_pipeline_ridge=Pipeline([('scaler', StandardScaler()), 
    #                           ('feature_selector', ridge_selector),
    #                           ('classifier', RFClassifier_red_ridge)])

    
    # param_grid={
    #     'feature_selector__estimator__C': [0.001, 0.01, 0.1],
    #     'classifier__max_depth': [2, 3, 5],              
    #     'classifier__n_estimators': [500]
    # }
    # grid_search_ridge=GridSearchCV(RF_pipeline_ridge, param_grid, cv=tscv, n_jobs=-1, return_train_score=True, verbose=3)
    # grid_search_ridge.fit(X_train, y_train)

    # display_feat_importances(RFClassifier_red_RIDGE, X_train)

    # acc, avg_dir=classification_accuracy(RFClassifier_red_RIDGE.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # classification_wfv_eval(RFClassifier_red_RIDGE, X_train, y_train)

    input("Press Enter to continue...")
    # ------- STEP-WISE REGRESSION APPLICATION -------
    lasso_selector=SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=1), threshold='mean')
    RFClassifier_red_lasso=RandomForestClassifier(random_state=1, n_jobs=-1)

    RF_pipeline_lasso=Pipeline([('scaler', StandardScaler()), 
                              ('feature_selector', lasso_selector),
                              ('classifier', RFClassifier_red_lasso)])

    param_grid={
        'feature_selector__estimator__C': [0.001, 0.01, 0.1], 
        'classifier__max_depth': [2, 3, 5],              
        'classifier__n_estimators': [500]
    }
    grid_search_LASSO=GridSearchCV(RF_pipeline_lasso, param_grid, cv=tscv, n_jobs=-1, return_train_score=True, verbose=3)
    grid_search_LASSO.fit(X_train, y_train) 

    best_params_from_grid = grid_search_LASSO.best_params_

    RF_params = {k.replace('classifier__', ''): v 
             for k, v in best_params_from_grid.items() 
             if k.startswith('classifier__')}

    lasso_support = grid_search_LASSO.best_estimator_.named_steps['feature_selector'].get_support()

    lasso_coefficient_names = X_train.columns[lasso_support].tolist()

    X_train_red=X_train[lasso_coefficient_names]
    X_test_red=X_test[lasso_coefficient_names]

    RFClassifier_red_sw_wfv=RandomForestClassifier(**RF_params, random_state=1, n_jobs=1)

    X_train_final, X_test_final=step_wise_reg_wfv(RFClassifier_red_sw_wfv, X_train_red, y_train, X_test_red) 

    RFClassifier_red_sw_wfv.fit(X_train_final, X_test_final)

    # display_feat_importances(RFClassifier_red_sw_wfv, X_train)

    # acc, avg_dir=classification_accuracy(RFClassifier_red_sw_wfv.predict(X_test), y_test)
    # print("Accuracy (Test):", acc)
    # print("Average Direction (Test):", avg_dir)

    # classification_wfv_eval(RFClassifier_red_sw_wfv, X_train, y_train)

    input("Press Enter to Finish...")
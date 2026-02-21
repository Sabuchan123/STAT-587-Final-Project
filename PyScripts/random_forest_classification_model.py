#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, KFold, cross_validate
from dimension_reduction import apply_PCA
from model_evaluation import classification_accuracy, classification_cv_eval, display_feat_importances

from data_preprocessing_and_cleaning import clean_data, pull_features

X, y_regression=clean_data()

X_train, X_test, y_train, y_test=train_test_split(X, y_regression, test_size=0.2)
y_train=(y_train>=0).astype(int).to_numpy()
y_test=(y_test>=0).astype(int).to_numpy()
X_train, X_test=apply_PCA(X_train, X_test, n_comp=0.9)

RFRegressor_red=RandomForestClassifier(random_state=1, n_jobs=-1)
RFRegressor_red.fit(X_train, y_train)

display_feat_importances(RFRegressor_red, X_train)

acc, avg_dir=classification_accuracy(RFRegressor_red.predict(X_test), y_test)
print(acc)
print(avg_dir)
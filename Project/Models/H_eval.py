import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
import datetime

from H_helpers import safe_div

# Good note, standard deviation of any accuracies is 0.5 achieved by having a perfectly split accuracy set of all correct and all correct instances.

def classification_accuracy(probabilities, actuals, upper_cutoff =0.5, lower_cutoff =0.5) -> tuple[float, float, int]:
    probabilities = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
    mask=(probabilities > upper_cutoff) | (probabilities < lower_cutoff)
    skipped_days=np.sum(~mask)

    actual_active=actuals[mask]
    predictions_active=(probabilities[mask] > upper_cutoff).astype(int)
    avg_pred_direction=(np.sum(predictions_active) + skipped_days * 0.5) / (len(probabilities))
    accuracy=np.mean(predictions_active==actual_active)
    return accuracy, avg_pred_direction, skipped_days / len(probabilities)

def display_feat_importances_tree(model, X: pd.DataFrame, n: int =50) -> pd.DataFrame:
    model_feature_df=pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    model_feature_df.head(n=n).plot(kind='barh', x="Feature", y="Importance")
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
    return model_feature_df

def display_coef_importances_regression(model, X: pd.DataFrame, n: int =50) -> pd.DataFrame:
    importances=np.abs(model.coef_[0]) 
    model_coef_df=pd.DataFrame({
        'Feature': X.columns,
        'Coef': importances
    }).sort_values(by='Importance', ascending=False)
    model_coef_df.head(n=n).plot(kind='barh', x="Feature", y="Coefficient")
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
    return model_coef_df

def get_final_metrics(model_obj, X_train, y_train, X_test, y_test, n_splits: int =5, label: str | None =None) -> dict:
    tscv=TimeSeriesSplit(n_splits=n_splits)
    
    threshold_pairs=[(0.6, 0.4), (0.65, 0.35), (0.7, 0.3), (0.75, 0.25)]

    def conviction_accuracy(y_true, y_probas, upper_cutoff: float =0.65, lower_cutoff: float =0.35):
        probs=y_probas[:, 1] if y_probas.ndim == 2 else y_probas
        mask=(y_probas > upper_cutoff) | (y_probas < lower_cutoff)
        if not np.any(mask): return 0.0
        actual_active=y_true[mask]
        predictions_active=(y_probas[mask] > upper_cutoff).astype(int)
        return np.mean(predictions_active==actual_active)

    def conviction_up_precision(y_true, y_probas, upper_cutoff: float =0.65):
        probs=y_probas[:, 1] if y_probas.ndim == 2 else y_probas
        mask_up=(probs > upper_cutoff)
        if not np.any(mask_up): return 0.0
        return np.mean(y_true[mask_up] == 1)

    def conviction_up_recall(y_true, y_probas, upper_cutoff: float =0.65):
        probs=y_probas[:, 1] if y_probas.ndim == 2 else y_probas
        mask_up=(probs > upper_cutoff)
        actual_ups=np.sum(y_true == 1)
        if (actual_ups == 0): return 1.0
        return np.sum((mask_up) & (y_true == 1)) / actual_ups

    def conviction_down_precision(y_true, y_probas, lower_cutoff: float =0.35):
        probs=y_probas[:, 1] if y_probas.ndim == 2 else y_probas
        mask_down=(probs < lower_cutoff)
        if not np.any(mask_down): return 0.0
        return np.mean(y_true[mask_down] == 0)

    def conviction_down_recall(y_true, y_probas, lower_cutoff: float =0.35):
        probs=y_probas[:, 1] if y_probas.ndim == 2 else y_probas
        mask_down=(probs < lower_cutoff)
        actual_down=np.sum(y_true == 0)
        if (actual_down == 0): return 1.0
        return np.sum((mask_down) & (y_true == 0)) / actual_down
    
    def conviction_participation(y_true, y_probas, upper_cutoff=0.65, lower_cutoff=0.35):
        probs=y_probas[:, 1] if y_probas.ndim == 2 else y_probas
        mask=(probs > upper_cutoff) | (probs < lower_cutoff)
        return np.mean(mask)

    best_score=-1
    best_thresholds=(0.5, 0.5)
    best_cv_results=None

    for upper, lower in threshold_pairs:
        t_conviction_accuracy=make_scorer(conviction_accuracy, response_method="predict_proba", upper_cutoff=upper, lower_cutoff=lower)
        t_conviction_up_precision=make_scorer(conviction_up_precision, response_method="predict_proba", upper_cutoff=upper)
        t_conviction_up_recall=make_scorer(conviction_up_recall, response_method="predict_proba", upper_cutoff=upper)
        t_conviction_down_precision=make_scorer(conviction_down_precision, response_method="predict_proba", lower_cutoff=lower)
        t_conviction_down_recall=make_scorer(conviction_down_recall, response_method="predict_proba", lower_cutoff=lower)
        t_participation=make_scorer(conviction_participation, response_method="predict_proba", upper_cutoff=upper, lower_cutoff=lower)

        scoring_metrics={ 
            'conv_acc': t_conviction_accuracy,
            'up_prec': t_conviction_up_precision,
            'up_rec': t_conviction_up_recall,
            'down_prec': t_conviction_down_precision,
            'down_rec': t_conviction_down_recall,
            'part_rate': t_participation
        }

        cv_results=cross_validate(model_obj, X_train, y_train, cv=tscv, scoring=scoring_metrics, return_train_score=True, n_jobs=-1)

        test_part_rate=np.mean(cv_results['test_part_rate'])
        
        mean_cv_test=np.mean(cv_results['test_conv_acc'])
        std_cv_test=np.std(cv_results['test_conv_acc'])

        if mean_cv_test == 0:
            score = 0.0
        else:
            score = (mean_cv_test / (std_cv_test + 1e-6)) * test_part_rate

        score=(mean_cv_test / (std_cv_test + 1e-6)) * test_part_rate # Signal to Noise Ratio times participation rate
        if (score > best_score):
            best_score=score
            best_thresholds=(upper, lower)
            best_cv_results=cv_results

    current_part_rate=np.mean(best_cv_results['test_part_rate'])
        
    upper_cutoff, lower_cutoff=best_thresholds
    print(f"Best cutoff rates: ({lower_cutoff}, {upper_cutoff})")
    
    mean_train=np.mean(best_cv_results['train_conv_acc'])
    std_train=np.std(best_cv_results['train_conv_acc'])
    mean_cv_test=np.mean(best_cv_results['test_conv_acc'])
    std_cv_test=np.std(best_cv_results['test_conv_acc'])

    mean_test_up_recall=np.mean(best_cv_results['test_up_rec'])
    mean_test_up_precision=np.mean(best_cv_results['test_up_prec'])
    mean_test_down_recall=np.mean(best_cv_results['test_down_rec'])
    mean_test_down_precision=np.mean(best_cv_results['test_down_prec'])

    probs=model_obj.predict_proba(X_test)[:, 1]

    mask=(probs > upper_cutoff) | (probs < lower_cutoff)
    y_test_active=y_test[mask]
    probs_active=probs[mask]

    if not np.any(mask) and (upper_cutoff >= 0.5 and lower_cutoff <= 0.5):
        while not np.any(mask) and (upper_cutoff >= 0.55 and lower_cutoff <= 0.):
            upper_cutoff-=0.05
            lower_cutoff+=0.05
            mask=(probs > upper_cutoff) | (probs < lower_cutoff)
            y_test_active=y_test[mask]
            probs_active=probs[mask]
    elif not np.any(mask):
        final_score=0.0
        model_name=model_obj.__class__.__name__
        if (model_name=="Pipeline"):
            model_name=model_obj.named_steps['classifier'].__class__.__name__
        return { 
            "model_name": model_name,
            "label": label,
            "time_ran": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cv_train_avg_accuracy": -999,
            "cv_train_std_accuracy": -999,
            "cv_validation_avg_accuracy": -999,
            "cv_validation_std_accuracy": -999,
            "cv_validation_up_precision": -999,
            "cv_validation_up_recall": -999,
            "cv_validation_down_precision": -999,
            "cv_validation_down_recall": -999,
            "test_split_accuracy": -999,
            "test_split_participation_rate": -999,
            "test_split_thresholds": (-999, -999),
            "test_split_positive_prediction_rate": -999,
            "test_split_true_up_rate": -999,
            "test_split_true_down": -999,
            "test_split_true_up": -999,
            "test_split_negative_down": -999,
            "test_split_negative_up": -999,
            "test_split_up_precision": -999,
            "test_split_up_recall": -999,
            "test_split_down_precision": -999,
            "test_split_down_recall": -999,
            "train_split_rows": -999,
            "train_split_cols": -999,
            "test_split_rows": -999,
            "test_split_cols": -999
        }
    
    preds_active=(probs_active > upper_cutoff).astype(int)
    final_score=np.mean(preds_active == y_test_active)
    participation_rate=len(y_test_active) / len(y_test)
    conf_mat=confusion_matrix(y_test_active, preds_active, labels=[0, 1])

    up_precision=round(safe_div(conf_mat[1][1], conf_mat[1][1] + conf_mat[0][1]), 3)
    down_precision=round(safe_div(conf_mat[0][0], conf_mat[0][0] + conf_mat[1][0]), 3)
    true_up_rate=round(np.mean(y_test), 3)
    true_down_rate=1 - true_up_rate 

    print(f"--- Model Report ---")
    print(f"Avg CV Train Accuracy:                {mean_train:.4f} (±{std_train:.4f})")
    print(f"Avg CV Validation Accuracy:           {mean_cv_test:.4f} (±{std_cv_test:.4f})") # This test score is generated by splitting the training set into time series splits.
    print(f"Avg CV Validation Up Precision:       {mean_test_up_precision:.4f}")
    print(f"Avg CV Validation Up Recall:          {mean_test_up_recall:.4f}")
    print(f"Avg CV Validation Down Precision:     {mean_test_down_precision:.4f}")
    print(f"Avg CV Validation Down Recall:        {mean_test_down_recall:.4f}")
    print(f"Avg CV Validation Participation Rate: {current_part_rate:.4f}")
    print(f"CV Upper Cutoff Threshold: {upper_cutoff}")
    print(f"CV Lower Cutoff Threshold: {lower_cutoff}")
    print(f"Final Hold-out (Test) Score (Accuracy): {final_score:.4f}")
    print(f"Positive Prediction Rate (Test):        {np.mean(preds_active):.4f}")
    print(f"True Up Rate (Test):   {true_up_rate:.4f}")
    print(f"Up Precision (Test):   {up_precision:.4f}") # How correct is it when it guesses up
    print(f"Up Edge Rate (Test):   {up_precision - true_up_rate:.4f}") # How much more correct is it than guessing the average up rate
    print(f"True Down Rate (Test): {true_down_rate:.4f}") 
    print(f"Down Precision (Test): {down_precision:.4f}") # How correct is it when it guesses down
    print(f"Down Edge Rate (Test): {down_precision - true_down_rate:.4f}") # How much more correct is it than guessing the average down rate (1 - up rate)
    print(pd.DataFrame(conf_mat, index=["Actual Down", "Actual Up"], columns=["Predicted Down", "Predicted Up"]))  

    model_name=model_obj.__class__.__name__
    if (model_name=="Pipeline"):
        model_name=model_obj.named_steps['classifier'].__class__.__name__
    return {
        "model_name": model_name,
        "label": label,
        "time_ran": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cv_train_avg_accuracy": round(mean_train, 3),
        "cv_train_std_accuracy": round(std_train, 3),
        "cv_validation_avg_accuracy": round(mean_cv_test, 3),
        "cv_validation_std_accuracy": round(std_cv_test, 3),
        "cv_validation_up_precision": round(mean_test_up_precision, 3),
        "cv_validation_up_recall": round(mean_test_up_recall, 3),
        "cv_validation_down_precision": round(mean_test_down_precision, 3),
        "cv_validation_down_recall": round(mean_test_down_recall, 3),
        "test_split_accuracy": round(final_score, 3),
        "test_split_participation_rate": round(participation_rate, 3),
        "test_split_thresholds": best_thresholds,
        "test_split_positive_prediction_rate": round(np.mean(preds_active), 3),
        "test_split_true_up_rate": true_up_rate,
        "test_split_true_down": conf_mat[0][0],
        "test_split_true_up": conf_mat[1][1],
        "test_split_negative_down": conf_mat[1][0],
        "test_split_negative_up": conf_mat[0][1],
        "test_split_up_precision": up_precision,
        "test_split_up_recall": round(safe_div(conf_mat[1][1], conf_mat[1][1] + conf_mat[1][0]), 3),
        "test_split_down_precision": down_precision,
        "test_split_down_recall": round(safe_div(conf_mat[0][0], conf_mat[0][0] + conf_mat[0][1]), 3),
        "train_split_rows": X_train.shape[0],
        "train_split_cols": X_train.shape[1],
        "test_split_rows": X_test.shape[0],
        "test_split_cols": X_test.shape[1]
    }

class RollingWindowBacktest:
    def __init__(self, model: BaseEstimator, X: pd.DataFrame | None =None, y: pd.DataFrame | None =None, X_train: pd.DataFrame =None, window_size: int | None =None, horizon: int | None=None, upper_cutoff: float =0.65, lower_cutoff: float =0.35):
        self.model=model
        self.X=X
        self.y=y
        self.X_train=X_train
        self.window_size=window_size
        self.horizon=horizon
        self.results=None
        self.upper_cutoff=upper_cutoff
        self.lower_cutoff=lower_cutoff
    
    def rolling_window_backtest(self, verbose: int =1):
        train_accuracy=[]
        train_avg_direction=[]
        train_avg_skipped_days=[]
        test_accuracy=[]
        test_avg_direction=[]
        test_avg_skipped_days=[]


        if (self.window_size == None):
            self.window_size=min(self.X.shape[1] * 2, self.X.shape[0] // 3)
            if (self.window_size == (self.X.shape[0] // 3)):
                print("!!!WARNING!!! Overfitting will most likely occur.")
        if (self.horizon == None):
            self.horizon=self.window_size // 4
        elif (self.horizon > self.window_size): raise ValueError("horizon must be less than window_size")

        n=len(self.X)
        n_train=len(self.X_train)

        total_iterations=(n - self.horizon - self.window_size) // self.horizon + 1
        if (verbose != 0): print(f"Rolling Window Backtest over {total_iterations} iterations.")
        current_step=0
        for i in range(self.window_size, n - self.horizon, self.horizon):
            current_step += 1
            X_train_roll=self.X.iloc[i-self.window_size : i]
            y_train_roll=self.y.iloc[i-self.window_size : i]
            
            X_test_roll=self.X.iloc[i : i+self.horizon]
            y_test_roll=self.y.iloc[i : i+self.horizon]
            
            self.model.fit(X_train_roll, y_train_roll)
            probs=self.model.predict_proba(X_test_roll)
            
            acc, avg, skipped_days=classification_accuracy(probs, y_test_roll, self.upper_cutoff, self.lower_cutoff)
            if (skipped_days >= 0.99):
                acc=0.5
                avg=0.5
            if (i > n_train):
                test_accuracy.append(acc)
                test_avg_direction.append(avg)
                test_avg_skipped_days.append(skipped_days)
            else:
                train_accuracy.append(acc)
                train_avg_direction.append(avg)
                train_avg_skipped_days.append(skipped_days)
            if (verbose>0 and ((current_step%10) == 0)): print(f"{current_step * 100 / total_iterations:.2f}% complete. Current iteration: {current_step}, True iteration: {i + 1 - self.window_size}")
            
        train_avg_accuracy=np.mean(train_accuracy)
        train_std_accuracy=np.std(train_accuracy)
        test_avg_accuracy=np.mean(test_accuracy)
        test_std_accuracy=np.std(test_accuracy)
        if (verbose > 0):
            print(f"Average Rolling Accuracy (train) (rwb): {train_avg_accuracy:.4f} (±{train_std_accuracy:.4f})")
            print(f"Average Rolling Accuracy (test)  (rwb): {test_avg_accuracy:.4f} (±{test_std_accuracy:.4f})")
        self.results=[train_accuracy + test_accuracy, train_avg_direction + test_avg_direction, {
            "mwfv_train_avg_accuracy": round(train_avg_accuracy, 3), # Modified Walk Forward Validation is mwfv
            "mwfv_train_std_accuracy": round(train_std_accuracy, 3),
            "mwfv_test_avg_accuracy": round(test_avg_accuracy, 3),
            "mwfv_test_std_accuracy": round(test_std_accuracy, 3)
        }, train_avg_skipped_days + test_avg_skipped_days]
    
    def display_wfv_results(self, extra_metrics: bool =True, label: str | None =None) -> None:
        plt.figure(figsize=(12, 6))
        n_train=len(self.X_train)
        n_total=len(self.X)
        start_of_each_test=list(range(self.window_size, n_total - self.horizon, self.horizon))
        
        plt.plot(start_of_each_test, self.results[0], marker='o', linestyle='-', label='Segment Accuracy')
        plt.plot(start_of_each_test, self.results[1], color='gray',marker='o', linestyle='-', alpha=0.4, label='Prediction Direction')
        plt.plot(start_of_each_test, self.results[3], color='pink',marker='o', linestyle='-', alpha=0.4, label='Skipped Rate')
        plt.plot(start_of_each_test, [0.5 for _ in range(len(start_of_each_test))], linestyle="--", label="Base Line")
        if (extra_metrics):
            in_X_train=[x for x in start_of_each_test if x < n_train]
            in_X_test=[x for x in start_of_each_test if x >= n_train]
            if (in_X_train):
                m_train=self.results[2]["mwfv_train_avg_accuracy"]
                s_train=self.results[2]["mwfv_train_std_accuracy"]
                plt.plot(in_X_train + [n_train], [m_train] * (len(in_X_train) + 1), color="#8EFF32", alpha=0.8, linestyle="--", label="Train Mean")
                plt.fill_between(in_X_train + [n_train], m_train - s_train, m_train + s_train, color="#8EFF32", alpha=0.15)
            if (in_X_test):
                m_test=self.results[2]["mwfv_test_avg_accuracy"]
                s_test=self.results[2]["mwfv_test_std_accuracy"]
                plt.plot([n_train] + in_X_test, [m_test] * (len(in_X_test) + 1), color="#2C8FFF", alpha=0.8, linestyle="--", label="Test Mean")
                plt.fill_between([n_train] + in_X_test, m_test - s_test, m_test + s_test, color="#65ADFF", alpha=0.15)
        
        plt.axvspan(self.window_size, n_train, color='lightblue', alpha=0.3, label='In-Sample Rolling')
        plt.axvspan(n_train, n_total, color='#FFFACD', alpha=0.5, label='Out-of-Sample Test')
        plt.axvline(x=n_train, color='r', linestyle='--', label='Train/Test Split Boundary')

        plt.title("Rolling Window Backtest Accuracy")
        plt.xlabel("Sample Index (Start of Test Horizon)")
        plt.ylabel("Accuracy Rate")
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.text(n_train * 0.5, 0.05, 'In-Sample Rolling', horizontalalignment='center', color='gray')
        plt.text(((n_total + n_train) * 0.5), 0.05, 'Out-of-Sample Rolling', horizontalalignment='center', color='gray')
        
        plt.savefig
        plt.savefig(f'../{label}_rwv_display.png', dpi=600, bbox_inches="tight")
        plt.show()
        plt.close('all')

    def set(self, X: pd.DataFrame, y: pd.DataFrame, X_train: pd.DataFrame):
        self.X=X
        self.y=y
        self.X_train=X_train

def utility_score(results: dict, rwb: dict, w: float =4.0):
    val_cv_CoV=results['cv_validation_std_accuracy'] / results['cv_validation_avg_accuracy']
    test_rwb_CoV=rwb.results[2]['mwfv_test_std_accuracy'] / rwb.results[2]['mwfv_test_avg_accuracy']
    score=results['test_split_participation_rate'] * (results['test_split_accuracy'] + results['test_split_up_recall'] + results['test_split_down_recall'] - 1.5) - (1 / w) * (val_cv_CoV + 3 * test_rwb_CoV)
    return score

def display_bias_variance_tradeoff(results: pd.DataFrame, key: str, label: str | None =None):
    results_plot=results[[key, 'score']].copy()
    sort_key=results_plot[key].map(lambda x: max(x) if isinstance(x, list) else x)
    results_plot=results_plot.iloc[sort_key.argsort()]
    results_plot[key]=results_plot[key].map(str)

    plt.figure(figsize=(10, 5))
    plt.plot(results_plot[key], results_plot['score'], marker='o', label="Score")
    plt.plot(results_plot[key], [0.0 for _ in range (len(results_plot['score']))], linestyle="--", alpha=0.4, label="Baseline")
    plt.xlabel(f"{key} Items")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.savefig(f'../{key}_search_values_{label}.png', dpi=600, bbox_inches="tight")
    plt.show()

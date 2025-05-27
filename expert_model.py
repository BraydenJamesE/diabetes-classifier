"""
Brayden Edwards
AI541 - Final Project

"""
import numpy as np
import pandas as pd
import random
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from visualization import plot_model_calibration, plot_probs
from data import get_data, get_class_specific_example
from base_model import DiabetesModelBase # Parent model
from explanation import get_lime_explanation
from config import (
    RANDOM_STATE, 
    EXPERT_MODEL_FEATURES,
    TEST_RATIO, 
    VAL_RATIO,
    EXPERT_CATEGORICAL_FEATURES
) # Global variables

class ExpertDiabetesModel(DiabetesModelBase):
    def __init__(self, model):
        super().__init__(EXPERT_MODEL_FEATURES, model)

def print_evals(model, X, y) -> None:
    evaluations = model.get_evaluations(X, y)
    print(evaluations.get("classification_report"))
    print(evaluations.get("confusion_matrix"))
    print(f"AUC/ROC: {evaluations.get('roc_auc')}")
    print(f"F1: {evaluations.get('f1')}")
    print(f"balanced_acc: {evaluations.get('balanced_acc')}")
    print(f"accuracy: {evaluations.get('accuracy')}")

def rand_split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=TEST_RATIO, 
        stratify=y, 
        random_state=RANDOM_STATE
    ) # Pulling out the hold-out set. 

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=VAL_RATIO,
        stratify=y_train_val, 
        random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def smotenc_split_then_split_data(X, y):
    categorical_features_idx = [
        i for i, feat in enumerate(EXPERT_MODEL_FEATURES) 
        if feat in EXPERT_CATEGORICAL_FEATURES
    ]    
    sm = SMOTENC(
        categorical_features=categorical_features_idx, 
        sampling_strategy='auto', 
        random_state=RANDOM_STATE
    )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=TEST_RATIO, 
        stratify=y, 
        random_state=RANDOM_STATE
    ) # Pulling out the hold-out set. 

    X_resampled, y_resampled = sm.fit_resample(X_train_val, y_train_val)

    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, 
        test_size=VAL_RATIO,
        stratify=y_resampled, 
        random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def undersample_then_split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=TEST_RATIO, 
        stratify=y, 
        random_state=RANDOM_STATE
    ) # Pulling out the hold-out set. 

    rus = RandomUnderSampler(sampling_strategy='auto', random_state=RANDOM_STATE)
    X_resampled, y_resampled = rus.fit_resample(X_train_val, y_train_val)

    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, 
        test_size=VAL_RATIO,
        stratify=y_resampled, 
        random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_train_calibrate_threshold(model, X_train, X_val, y_train, y_val) -> None:
    model.train(X_train, y_train)
    model.calibrate(X_val, y_val)
    threshold, _ = model.find_threshold_for_f1(y_val, model.get_probabilities(X_val))
    model.set_threshold(threshold)

def class_imbalance_test():
    auc_key = "AUC/ROC"
    f1_key = "F1 Score"
    num_samples_key = "Num. of Train Samples"
    
    data_output_rf = {
        "Method": ["Do Nothing", "Undersample", "SMOTENC", "Cost Sen. Learning"],
        f1_key: [],
        auc_key: [],
        num_samples_key: []
    }
    data_output_xgb = {
        "Method": ["Do Nothing", "Undersample", "SMOTENC", "Cost Sen. Learning"],
        f1_key: [],
        auc_key: [],
        num_samples_key: []
    }
    def add_evals(rf_model: ExpertDiabetesModel, xgb_model: ExpertDiabetesModel) -> None:
        evaluations = rf_model.get_evaluations(X_test, y_test)
        data_output_rf.get(f1_key).append(evaluations.get("f1"))
        data_output_rf.get(auc_key).append(evaluations.get("roc_auc"))
        data_output_rf.get(num_samples_key).append(y_train.size)

        evaluations = xgb_model.get_evaluations(X_test, y_test)
        data_output_xgb.get(f1_key).append(evaluations.get("f1"))
        data_output_xgb.get(auc_key).append(evaluations.get("roc_auc"))
        data_output_xgb.get(num_samples_key).append(y_train.size)

    subset_ratio = 1
    X, y = get_data(EXPERT_MODEL_FEATURES, subset_ratio=subset_ratio)

    rf_params_no_class_weight = {'max_depth': 10, 'n_estimators': 100}
    xgb_params_no_class_weight = {'colsample_bytree': 0.6, 'gamma': 1, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 1.0}

    # DO NOTHING
    rf_model = ExpertDiabetesModel(RandomForestClassifier(**rf_params_no_class_weight, random_state=RANDOM_STATE))
    xgb_model = ExpertDiabetesModel(XGBClassifier(**xgb_params_no_class_weight, random_state=RANDOM_STATE))
    
    X_train, X_val, X_test, y_train, y_val, y_test = rand_split_data(X, y)
    apply_train_calibrate_threshold(rf_model, X_train, X_val, y_train, y_val)
    apply_train_calibrate_threshold(xgb_model, X_train, X_val, y_train, y_val)

    add_evals(rf_model, xgb_model)

    # UNDERSAMPLE
    rf_model = ExpertDiabetesModel(RandomForestClassifier(**rf_params_no_class_weight, random_state=RANDOM_STATE))
    xgb_model = ExpertDiabetesModel(XGBClassifier(**xgb_params_no_class_weight, random_state=RANDOM_STATE))
    
    X_train, X_val, X_test, y_train, y_val, y_test = undersample_then_split_data(X, y)
    apply_train_calibrate_threshold(rf_model, X_train, X_val, y_train, y_val)
    apply_train_calibrate_threshold(xgb_model, X_train, X_val, y_train, y_val)

    add_evals(rf_model, xgb_model)
  
    # SMOTE 
    rf_model = ExpertDiabetesModel(RandomForestClassifier(**rf_params_no_class_weight, random_state=RANDOM_STATE))
    xgb_model = ExpertDiabetesModel(XGBClassifier(**xgb_params_no_class_weight, random_state=RANDOM_STATE))
    
    X_train, X_val, X_test, y_train, y_val, y_test = smotenc_split_then_split_data(X, y)
    apply_train_calibrate_threshold(rf_model, X_train, X_val, y_train, y_val)
    apply_train_calibrate_threshold(xgb_model, X_train, X_val, y_train, y_val)

    add_evals(rf_model, xgb_model)

    # COST SENSITIVE LEARNING
    rf_params = {'class_weight': {0: 1, 1: 4}, 'max_depth': 10, 'n_estimators': 100}
    xgb_params = {'colsample_bytree': 0.6, 'gamma': 1, 'max_depth': 3, 'min_child_weight': 1, 'scale_pos_weight': 4, 'subsample': 1.0}

    rf_model = ExpertDiabetesModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE))
    xgb_model = ExpertDiabetesModel(XGBClassifier(**xgb_params, random_state=RANDOM_STATE))
      
    X_train, X_val, X_test, y_train, y_val, y_test = rand_split_data(X, y)
    apply_train_calibrate_threshold(rf_model, X_train, X_val, y_train, y_val)
    apply_train_calibrate_threshold(xgb_model, X_train, X_val, y_train, y_val)

    add_evals(rf_model, xgb_model)

    # Print the results
    rf_df = pd.DataFrame(data=data_output_rf)
    xgb_df = pd.DataFrame(data=data_output_xgb)

    print("Random Forest Results:")
    print(tabulate(rf_df, headers="keys", tablefmt='pretty'))
    print("\nXGBoost Results:")
    print(tabulate(xgb_df, headers="keys", tablefmt='pretty'))


def main():
    subset_ratio = 1 

    X, y = get_data(EXPERT_MODEL_FEATURES, subset_ratio=subset_ratio)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, 
        test_size=TEST_RATIO, 
        stratify=y, 
        random_state=RANDOM_STATE
    ) # Pulling out the hold-out set. 

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=VAL_RATIO,
        stratify=y_train_val, 
        random_state=RANDOM_STATE
    )
    
    rf_params = {'class_weight': {0: 1, 1: 4}, 'max_depth': 10, 'n_estimators': 100}
    xgb_params = {'colsample_bytree': 0.6, 'gamma': 1, 'max_depth': 3, 'min_child_weight': 1, 'scale_pos_weight': 4, 'subsample': 1.0}

    rf_model = ExpertDiabetesModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE))
    xgb_model = ExpertDiabetesModel(XGBClassifier(**xgb_params, random_state=RANDOM_STATE))

    rf_model.train(X_train, y_train)
    xgb_model.train(X_train, y_train)

    rf_model.calibrate(X_val, y_val)
    xgb_model.calibrate(X_val, y_val)

    # threshold, recall = rf_model.find_threshold_for_recall(y_val, rf_model.get_probabilities(X_val))
    # print(f"Threshold: {threshold}, Recall: {recall}")
    
    threshold, f1 = rf_model.find_threshold_for_f1(y_val, rf_model.get_probabilities(X_val))
    print(f"Threshold rf_model: {threshold}, F1 Score: {f1}")
    
    rf_model.set_threshold(threshold)

    threshold, f1 = xgb_model.find_threshold_for_f1(y_val, xgb_model.get_probabilities(X_val))
    print(f"Threshold xgb_model: {threshold}, F1 Score: {f1}")
    
    xgb_model.set_threshold(threshold)

    # plot_model_calibration(X_test, y_test, rf_model)
    # plot_model_calibration(X_test, y_test, xgb_model)

    prods_rf = rf_model.get_probabilities(X_test)
    prods_xgb = xgb_model.get_probabilities(X_test)

    # rand_example = random.randint(0, 1000000) 
    # class_of_interest = 1
    # x_example, y_true_example, prod_example = get_class_specific_example(X_test, y_test, prods_rf, class_of_interest, rand_example)
    # print("Random Forest")
    # print(f"True: \t{y_true_example}")
    # print(f"Predicted True: \t{prod_example}\n")
    
    # x_example, y_true_example, prod_example = get_class_specific_example(X_test, y_test, prods_xgb, class_of_interest, rand_example)
    # print("XGBoost")
    # print(f"True: \t{y_true_example}")
    # print(f"Predicted True: \t{prod_example}")

    # plot_probs(y_test, prods_rf, "RF probs")
    # plot_probs(y_test, prods_xgb, "XGB probs")

    print_evals(rf_model, X_test, y_test)
    print_evals(xgb_model, X_test, y_test)

    # get_lime_explanation(x_example, y_example, X_train, EXPERT_MODEL_FEATURES, EXPERT_CATEGORICAL_FEATURES, rf_model)

if __name__  == "__main__":
    class_imbalance_test()
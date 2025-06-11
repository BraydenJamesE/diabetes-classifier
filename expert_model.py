"""
**Author:** Brayden Edwards  
**Course:** CS 541 – Machine Learning Challenges in the Real World
**School:** Oregon State University
**Professor:** Dr. Kiri Wagstaff  
**Term:** Spring 2025  
"""
import pandas as pd
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    f1_score, 
    recall_score, 
    balanced_accuracy_score,
)

from data import get_data
from base_model import DiabetesModelBase # Parent model
from config import (
    RANDOM_STATE, 
    EXPERT_MODEL_FEATURES,
    TEST_RATIO, 
    VAL_RATIO,
    EXPERT_CATEGORICAL_FEATURES, 
    ROUND_DEC
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

def apply_train_calibrate_threshold(model: DiabetesModelBase, X_train, X_val, y_train, y_val) -> None:
    model.train(X_train, y_train)
    model.calibrate(X_val, y_val)
    threshold, _ = model.find_threshold_for_f1(y_val, model.get_probabilities(X_val))
    model.set_threshold(threshold)

def class_imbalance_test():
    auc_key = "AUC/ROC"
    f1_key = "F1 Score"
    recall_key = "Recall"
    precision_key = "Precision"
    brier_key = "Brier Score"
    bal_acc_key = "Balanced Accuracy"
    num_samples_key = "Num. of Train Samples"
    
    data_output_rf = {
        "Method": ["Do Nothing", "Undersample", "SMOTENC", "Cost Sen. Learning"],
        f1_key: [],
        recall_key: [], 
        precision_key: [],
        bal_acc_key: [],
        num_samples_key: []
    }
    data_output_xgb = {
        "Method": ["Do Nothing", "Undersample", "SMOTENC", "Cost Sen. Learning"],
        f1_key: [],
        recall_key: [], 
        precision_key: [],
        bal_acc_key: [],
        num_samples_key: []
    }
    data_output_dummy = {
        "Method": ["Do Nothing", "Undersample", "SMOTENC", "Cost Sen. Learning"],
        f1_key: [],
        recall_key: [],
        precision_key: [],
        bal_acc_key: [],
    }
    
    # Helper function
    def add_evals(df: pd.DataFrame, model: ExpertDiabetesModel, X_test: pd.DataFrame, y_test: pd.Series) -> None: 
        probs = model.get_probabilities(X_test)
        evaluations = model.get_evaluations(X_test, y_test)
        df.get(f1_key).append(evaluations.get("f1"))
        df.get(recall_key).append(evaluations.get("recall"))
        df.get(precision_key).append(evaluations.get("precision"))
        df.get(bal_acc_key).append(evaluations.get("balanced_acc"))
        df.get(num_samples_key).append(y_train.size)

    def add_dummy_evals(
        preds: pd.Series, 
        probs: pd.Series, 
        y_test: pd.Series, 
        is_null: bool = False
    ) -> None:
        if is_null:
            data_output_dummy.get(f1_key).append("--")
            data_output_dummy.get(auc_key).append("--")
            data_output_dummy.get(recall_key).append("--")
            data_output_dummy.get(precision_key).append("--")
            data_output_dummy.get(bal_acc_key).append("--")
            data_output_dummy.get(brier_key).append("--")
        else:
            data_output_dummy.get(f1_key).append(round(f1_score(y_test, preds), ROUND_DEC))
            data_output_dummy.get(recall_key).append(round(recall_score(y_test, preds), ROUND_DEC))
            data_output_dummy.get(precision_key).append(0) 
            data_output_dummy.get(bal_acc_key).append(round(balanced_accuracy_score(y_test, preds), ROUND_DEC))

    subset_ratio = 1
    X, y = get_data(EXPERT_MODEL_FEATURES, subset_ratio=subset_ratio)

    rf_params_no_class_weight = {'max_depth': 10, 'n_estimators': 150}
    xgb_params_no_class_weight = {'colsample_bytree': 0.6, 'gamma': 2, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 1.0}

    # DO NOTHING
    rf_model = ExpertDiabetesModel(RandomForestClassifier(**rf_params_no_class_weight, random_state=RANDOM_STATE))
    xgb_model = ExpertDiabetesModel(XGBClassifier(**xgb_params_no_class_weight, random_state=RANDOM_STATE))

    X_train, X_val, X_test, y_train, y_val, y_test = rand_split_data(X, y)
    rf_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val, threshold_opt_type='f1')
    xgb_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val, threshold_opt_type='f1')
    add_evals(data_output_rf, rf_model, X_test, y_test)
    add_evals(data_output_xgb, xgb_model, X_test, y_test)

    dummy_model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy_model.fit(X_train, y_train)
    dummy_preds = dummy_model.predict(X_test)
    dummy_probs = dummy_model.predict_proba(X_test)[:, 1]
    add_dummy_evals(dummy_preds, dummy_probs, y_test)

    # UNDERSAMPLE
    rf_model = ExpertDiabetesModel(RandomForestClassifier(**rf_params_no_class_weight, random_state=RANDOM_STATE))
    xgb_model = ExpertDiabetesModel(XGBClassifier(**xgb_params_no_class_weight, random_state=RANDOM_STATE))
    
    X_train, X_val, X_test, y_train, y_val, y_test = undersample_then_split_data(X, y)
    rf_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val, threshold_opt_type='f1')
    xgb_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val, threshold_opt_type='f1')
    add_evals(data_output_rf, rf_model, X_test, y_test)
    add_evals(data_output_xgb, xgb_model, X_test, y_test)

    dummy_model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy_model.fit(X_train, y_train)
    dummy_preds = dummy_model.predict(X_test)
    dummy_probs = dummy_model.predict_proba(X_test)[:, 1]
    add_dummy_evals(dummy_preds, dummy_probs, y_test)
  
    # SMOTE 
    rf_model = ExpertDiabetesModel(RandomForestClassifier(**rf_params_no_class_weight, random_state=RANDOM_STATE))
    xgb_model = ExpertDiabetesModel(XGBClassifier(**xgb_params_no_class_weight, random_state=RANDOM_STATE))
    
    X_train, X_val, X_test, y_train, y_val, y_test = smotenc_split_then_split_data(X, y)
    rf_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val, threshold_opt_type='f1')
    xgb_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val, threshold_opt_type='f1')
    add_evals(data_output_rf, rf_model, X_test, y_test)
    add_evals(data_output_xgb, xgb_model, X_test, y_test)

    dummy_model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy_model.fit(X_train, y_train)
    dummy_preds = dummy_model.predict(X_test)
    dummy_probs = dummy_model.predict_proba(X_test)[:, 1]
    add_dummy_evals(dummy_preds, dummy_probs, y_test)

    # COST SENSITIVE LEARNING
    rf_params = {'class_weight': {0: 1, 1: 4}, 'max_depth': 10, 'n_estimators': 150}
    xgb_params = {'colsample_bytree': 0.6, 'gamma': 2, 'max_depth': 5, 'min_child_weight': 1, 'scale_pos_weight': 4, 'subsample': 1.0}

    rf_model = ExpertDiabetesModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE))
    xgb_model = ExpertDiabetesModel(XGBClassifier(**xgb_params, random_state=RANDOM_STATE))
      
    X_train, X_val, X_test, y_train, y_val, y_test = rand_split_data(X, y)
    rf_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val, threshold_opt_type='f1')
    xgb_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val, threshold_opt_type='f1')
    add_evals(data_output_rf, rf_model, X_test, y_test)
    add_evals(data_output_xgb, xgb_model, X_test, y_test)
    
    dummy_model = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    dummy_model.fit(X_train, y_train)
    dummy_preds = dummy_model.predict(X_test)
    dummy_probs = dummy_model.predict_proba(X_test)[:, 1]
    add_dummy_evals(dummy_preds, dummy_probs, y_test)

    # Print the results
    rf_df = pd.DataFrame(data=data_output_rf)
    xgb_df = pd.DataFrame(data=data_output_xgb)
    dummy_df = pd.DataFrame(data=data_output_dummy)

    print()
    print(f"-"*6, "Expert Model Output", "-"*6)
    print("\nRandom Forest Results:")
    print(tabulate(rf_df, headers="keys", tablefmt='pretty'))
    print("\nXGBoost Results:")
    print(tabulate(xgb_df, headers="keys", tablefmt='pretty'))
    print("\nMajority Classifier Results:")
    print(tabulate(dummy_df, headers="keys", tablefmt='pretty'))
    print()
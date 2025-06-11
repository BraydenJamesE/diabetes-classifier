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


from data import get_data
from visualization import plot_model_calibration
from base_model import DiabetesModelBase # Parent model
from config import (
    RANDOM_STATE, 
    EXPERT_MODEL_FEATURES,
    TEST_RATIO, 
    VAL_RATIO,
    ROUND_DEC
) # Global variables

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

def representation_test():
    all_features = EXPERT_MODEL_FEATURES
    
    X_all_features, y = get_data(all_features)
    health_survey_features = ["MentHlth", "PhysHlth",] # "GenHlth" -> Findings found that this feature should be included.
    calculation_features = ["Income", "Education"] # "Age" -> Findings found that this feature should be included.

    rf_params = {'class_weight': {0: 1, 1: 4}, 'max_depth': 10, 'n_estimators': 150}
    xgb_params = {'colsample_bytree': 0.6, 'gamma': 2, 'max_depth': 5, 'min_child_weight': 1, 'scale_pos_weight': 4, 'subsample': 1.0}

    # Metrics for use: F1, Recall, Precision, Brier Score
    def print_model_results(
        rf_model: DiabetesModelBase, 
        xgb_model: DiabetesModelBase, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        print_title: str
    ) -> None:
        rf_evaluations: dict = rf_model.get_evaluations(X_test, y_test)
        xgb_evaluations: dict = xgb_model.get_evaluations(X_test, y_test)

        metrics = {
            "Model": ["Random Forest", "XGBoost"],
            "F1": [rf_evaluations.get('f1'), xgb_evaluations.get('f1')],
            "Recall": [rf_evaluations.get('recall'), xgb_evaluations.get('recall')],
            "Precision": [rf_evaluations.get('precision'), xgb_evaluations.get('precision')], 
            "Balanced Accuracy": [rf_evaluations.get('balanced_acc'), xgb_evaluations.get('balanced_acc')], 
            "Brier Score": [rf_evaluations.get('brier'), xgb_evaluations.get('brier')]
        }
        df = pd.DataFrame(metrics)
        
        print("\n", print_title)
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=f".{ROUND_DEC}f"))

        
    # ---- All Features ---- 
    class AllFeatureModel(DiabetesModelBase):
        def __init__(self, model):
            super().__init__(all_features, model)

    rf_model = AllFeatureModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE))
    xgb_model = AllFeatureModel(XGBClassifier(**xgb_params, random_state=RANDOM_STATE))

    # get data
    X_train, X_val, X_test, y_train, y_val, y_test = rand_split_data(X_all_features, y)

    # fit model
    rf_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)
    xgb_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)

    # print results
    eval_print_title = "All Features"
    print_model_results(rf_model, xgb_model, X_test, y_test, eval_print_title)


    # ---- HighCol Removed ----
    features_without_HighChol = [f for f in all_features if f != "HighChol"] # Define features

    class AllButOneFeaturesModel(DiabetesModelBase):
        def __init__(self, model):
            super().__init__(features_without_HighChol, model)

    rf_model = AllButOneFeaturesModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE))
    xgb_model = AllButOneFeaturesModel(XGBClassifier(**xgb_params, random_state=RANDOM_STATE))

    # get data
    new_X = X_all_features[features_without_HighChol]
    X_train, X_val, X_test, y_train, y_val, y_test = rand_split_data(new_X, y)

    # fit model
    rf_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)
    xgb_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)

    # print results
    eval_print_title = "All Features Except HighChol"
    print_model_results(rf_model, xgb_model, X_test, y_test, eval_print_title)

    plot_model_calibration(X_test, y_test, rf_model)
    plot_model_calibration(X_test, y_test, xgb_model)

    # ---- Removing 'Calculation' features ----
    # note, this section also has 'HighChol' feature removed
    features_to_drop = calculation_features + ["HighChol"]
    non_calc_features = [f for f in all_features if f not in features_to_drop]

    class CalculationFeaturesRemovedModel(DiabetesModelBase):
        def __init__(self, model):
            super().__init__(non_calc_features, model)

    rf_model = CalculationFeaturesRemovedModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE))
    xgb_model = CalculationFeaturesRemovedModel(XGBClassifier(**xgb_params, random_state=RANDOM_STATE))

    # get data
    new_X = X_all_features[non_calc_features]
    X_train, X_val, X_test, y_train, y_val, y_test = rand_split_data(new_X, y)

    # fit model
    rf_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)
    xgb_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)

    # print results
    eval_print_title = "HighChol & Calculation Features Removed"
    print_model_results(rf_model, xgb_model, X_test, y_test, eval_print_title)


    # ---- Removing 'Health Survey' features ----
    # note, this section also has 'HighChol' feature removed but includes calculation features.
    features_to_drop = health_survey_features + ["HighChol"]
    non_health_survey_features = [f for f in all_features if f not in features_to_drop]

    class HealthSurveyFeaturesRemovedModel(DiabetesModelBase):
        def __init__(self, model):
            super().__init__(non_health_survey_features, model)

    rf_model = HealthSurveyFeaturesRemovedModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE))
    xgb_model = HealthSurveyFeaturesRemovedModel(XGBClassifier(**xgb_params, random_state=RANDOM_STATE))

    # get data
    new_X = X_all_features[non_health_survey_features]
    X_train, X_val, X_test, y_train, y_val, y_test = rand_split_data(new_X, y)

    # fit model
    rf_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)
    xgb_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)

    # print results
    eval_print_title = "HighChol & Health Survey Features Removed"
    print_model_results(rf_model, xgb_model, X_test, y_test, eval_print_title)


    # ---- Removing 'Health Survey' and 'Calculation' features ---- 
    # note, this section also has 'HighChol' feature removed.
    features_to_drop = health_survey_features + calculation_features + ["HighChol"]
    non_calc_health_survey_features = [f for f in all_features if f not in features_to_drop]

    class NoCalcHealthSurveyFeaturesModel(DiabetesModelBase):
        def __init__(self, model):
            super().__init__(non_calc_health_survey_features, model)

    rf_model = NoCalcHealthSurveyFeaturesModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE))
    xgb_model = NoCalcHealthSurveyFeaturesModel(XGBClassifier(**xgb_params, random_state=RANDOM_STATE))

    # get data
    new_X = X_all_features[non_calc_health_survey_features]
    X_train, X_val, X_test, y_train, y_val, y_test = rand_split_data(new_X, y)

    # fit model
    rf_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)
    xgb_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)

    # print results
    eval_print_title = "HighChol, Calculation, & Health Survey Features Removed"
    print_model_results(rf_model, xgb_model, X_test, y_test, eval_print_title)

    plot_model_calibration(X_test, y_test, rf_model)
    plot_model_calibration(X_test, y_test, xgb_model)

def main():
    representation_test()
    
if __name__ == "__main__":
    main()
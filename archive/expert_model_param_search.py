"""
**Author:** Brayden Edwards  
**Course:** CS 541 – Machine Learning Challenges in the Real World
**School:** Oregon State University
**Professor:** Dr. Kiri Wagstaff  
**Term:** Spring 2025  
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)

from data import get_data 
from config import (
    RANDOM_STATE, 
    TEST_RATIO,
    EXPERT_MODEL_FEATURES,
    VAL_RATIO
) # Global variables

def main():
    subset_ratio = 1

    X, y = get_data(EXPERT_MODEL_FEATURES, subset_ratio=subset_ratio)

    X_train_val, _, y_train_val, _ = train_test_split(
        X, y, 
        test_size=TEST_RATIO, 
        stratify=y, 
        random_state=RANDOM_STATE
    ) # Pulling out the hold-out set. 

    X_train, _, y_train, _ = train_test_split(
        X_train_val, y_train_val, 
        test_size=VAL_RATIO, 
        stratify=y_train_val, 
        random_state=RANDOM_STATE
    ) # Pulling out the hold-out set. 

    param_rf = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 4, 10, 20], 
        "class_weight": [
            "balanced", 
            {0: 1, 1: 4}, 
            {0: 1, 1: 2}, 
            {0: 1, 1:6}
        ]
    }

    params_xgb = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        "scale_pos_weight": [4, 6, 8]
    } # A parameter grid for XGBoost. 
    #   Link: https://www.kaggle.com/code/tilii7/hyperparameter-grid-search-with-xgboost

    rf_base = RandomForestClassifier(random_state=RANDOM_STATE) 
    xgb_base = XGBClassifier(random_state=RANDOM_STATE)

    nested_scores_rf = []
    nested_scores_xgb = []
    
    best_avg_nested_score_rf = -np.inf
    best_avg_nested_score_xgb = -np.inf
    
    best_model_rf = None
    best_model_xgb = None
    
    NUM_TRIALS = 10
    SCORING = "f1"
    for i in range(NUM_TRIALS):
        print(f"Iteration {i}")
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
        outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

        # Random Forest Search
        clf_rf = GridSearchCV(
            rf_base, 
            param_rf, 
            cv=inner_cv, 
            scoring=SCORING, 
            n_jobs=-1
        )
        clf_rf.fit(X_train, y_train)

        nested_score_rf = cross_val_score(
            clf_rf, 
            X=X_train, 
            y=y_train, 
            cv=outer_cv, 
            scoring=SCORING
        )
        average_nested_score_rf = nested_score_rf.mean()
        nested_scores_rf.append(average_nested_score_rf)

        if average_nested_score_rf > best_avg_nested_score_rf:
            best_avg_nested_score_rf = average_nested_score_rf
            best_model_rf = clf_rf

        # XGBoost Search
        clf_xgb = GridSearchCV(
            xgb_base, 
            params_xgb, 
            cv=inner_cv, 
            scoring=SCORING, 
            n_jobs=-1
        )
        clf_xgb.fit(X_train, y_train)

        nested_score_xgb = cross_val_score(
            clf_xgb, 
            X=X_train, 
            y=y_train, 
            cv=outer_cv, 
            scoring=SCORING
        )
        average_nested_score_xgb = nested_score_xgb.mean()
        nested_scores_xgb.append(average_nested_score_xgb)

        if average_nested_score_xgb > best_avg_nested_score_xgb:
            best_avg_nested_score_xgb = average_nested_score_xgb
            best_model_xgb = clf_xgb

    # Results
    if best_model_rf is not None:
        print(f"Best model was {best_model_rf.best_params_}")        
        print(f"Average F1 score was: {best_avg_nested_score_rf:.4f}") 
        print("------------------------")
    if best_model_xgb is not None:
        print(f"Best model was {best_model_xgb.best_params_}")        
        print(f"Average F1 score was: {best_avg_nested_score_xgb:.4f}") 
        print("------------------------")

    with open("best_model_results.txt", "w") as f:
        if best_model_rf is not None:
            f.write("Random Forest Best Params:\n")
            f.write(f"{best_model_rf.best_params_}\n")
            f.write(f"Average F1: {best_avg_nested_score_rf:.4f}\n\n")

        if best_model_xgb is not None:
            f.write("XGBoost Best Params:\n")
            f.write(f"{best_model_xgb.best_params_}\n")
            f.write(f"Average F1: {best_avg_nested_score_xgb:.4f}\n")

if __name__  == "__main__":
    main()
"""
Brayden Edwards
AI541 - Final Project

"""
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, 
)

from data import get_data
from base_model import DiabetesModelBase # Parent model
from config import RANDOM_STATE, HOME_MODEL_FEATURES, TEST_RATIO, VAL_RATIO # Global variables

class AtHomeDiabetesModel(DiabetesModelBase):
    def __init__(self, model):
        super().__init__(HOME_MODEL_FEATURES, model)

def main():
    subset_ratio = 1

    X, y = get_data(HOME_MODEL_FEATURES, subset_ratio=subset_ratio)

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

    
    rf_params = {'class_weight': {0: 1, 1: 4}, 'max_depth': 10, 'n_estimators': 150}
    xgb_params = {'colsample_bytree': 1.0, 'gamma': 5, 'max_depth': 3, 'min_child_weight': 10, 'scale_pos_weight': 4, 'subsample': 1.0}
    
    rf_model = AtHomeDiabetesModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE))
    xgb_model = AtHomeDiabetesModel(XGBClassifier(**xgb_params, random_state=RANDOM_STATE))

    # These parameters and models do nothing to address class imbalance. 
    rf_params_no_class_weight = {'max_depth': 10, 'n_estimators': 150}
    xgb_params_no_class_weight = {'colsample_bytree': 1.0, 'gamma': 5, 'max_depth': 3, 'min_child_weight': 10, 'subsample': 1.0}

    rf_no_class_weight_model = AtHomeDiabetesModel(RandomForestClassifier(**rf_params_no_class_weight, random_state=RANDOM_STATE))
    xgb_no_class_weight_model = AtHomeDiabetesModel(XGBClassifier(**xgb_params_no_class_weight, random_state=RANDOM_STATE))



if __name__  == "__main__":
    main()
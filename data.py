"""
**Author:** Brayden Edwards  
**Course:** CS 541 – Machine Learning Challenges in the Real World
**School:** Oregon State University
**Professor:** Dr. Kiri Wagstaff  
**Term:** Spring 2025  
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo # Import data

from config import (
    RANDOM_STATE, 
    TARGET_COL_NAME, 
    VAL_RATIO, 
    TEST_RATIO
)

def get_data(features: list, subset_ratio: float = 1) -> tuple[pd.DataFrame, pd.Series]:
    # fetch dataset 
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
    
    # data (as pandas dataframes) 
    X = cdc_diabetes_health_indicators.data.features 
    y = cdc_diabetes_health_indicators.data.targets[TARGET_COL_NAME] 

    df: pd.DataFrame = X.join(y)

    # Get a subset of the data. Default is 100%. 
    df = df.sample(frac=subset_ratio, random_state=RANDOM_STATE) 

    X = df[features]
    y = df[TARGET_COL_NAME]
    return X, y

def get_rand_split_data(X, y):
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

def get_class_specific_example(
    X: pd.DataFrame, 
    y: pd.Series, 
    probs: np.ndarray, 
    class_label: int = 1,
    rand_example: int = 0
) -> tuple[pd.DataFrame, int, float]:
    """Return (row-df, true-y, pred-prob) aligned by index."""
    probs = pd.Series(
        probs,           
        index=X.index,
        name="pred_prob"
    )
    matching_idx = y.index[y == class_label] 
    if matching_idx.empty:
        raise ValueError(f"No examples found for class {class_label}.")
    
    rand_example = rand_example % len(matching_idx)
    print("Random value produced get_class_specific_example; not reproduceable.")
    print(f"Random Example Used: {rand_example}")
    idx = matching_idx[rand_example]
    return X.loc[[idx]], int(y.loc[idx]), float(probs.loc[idx])

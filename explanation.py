"""
**Author:** Brayden Edwards  
**Course:** CS 541 – Machine Learning Challenges in the Real World
**School:** Oregon State University
**Professor:** Dr. Kiri Wagstaff  
**Term:** Spring 2025  
"""

import lime.lime_tabular
import pandas as pd
import numpy as np
import shap
from tabulate import tabulate
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from base_model import DiabetesModelBase
from data import get_data, get_rand_split_data
from config import (
    RANDOM_STATE, 
    EXPERT_MODEL_FEATURES, 
    EXPERT_CATEGORICAL_FEATURES,
    HOME_MODEL_FEATURES, 
    ROUND_DEC
)

def produce_three_samples_for_local_expl(
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    probs: np.ndarray, 
    threshold: float
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    high_conf_pos = 0.75
    high_conf_neg = 0.05
    unsure_conf = threshold # Set this to the model threshold

    hi_pos_idx = np.where(probs > high_conf_pos)[0][0]                
    hi_neg_idx = np.where(probs < high_conf_neg)[0][0]         
    border_idx = np.argmin(np.abs(probs - unsure_conf))   

    example_indices = [hi_pos_idx, hi_neg_idx, border_idx]

    return X_test.iloc[example_indices], y_test.iloc[example_indices], example_indices

def get_shap_importance(is_global=False) -> None:
    # Get data
    X, y = get_data(EXPERT_MODEL_FEATURES)
    X_train, X_val, X_test, y_train, y_val, y_test = get_rand_split_data(X, y)

    # Declare model
    class ExpertModel(DiabetesModelBase):
        def __init__(self, model):
            super().__init__(EXPERT_MODEL_FEATURES, model)

    rf_params = {'class_weight': {0: 1, 1: 4}, 'max_depth': 10, 'n_estimators': 150}
    rf_model = ExpertModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE)) 
    rf_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)
    
    xgb_params = {'colsample_bytree': 0.6, 'gamma': 2, 'max_depth': 5, 'min_child_weight': 1, 'scale_pos_weight': 4, 'subsample': 1.0}
    xgb_model = ExpertModel(XGBClassifier(**xgb_params, random_state=RANDOM_STATE))
    xgb_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)

    # Using a sample to ensure reasonable computation time for SHAP algo. 
    # This sample is stratified to ensure that I am gettting good representation. 
    X_test: pd.DataFrame # type hinted for easier use 
    sample_amount = 1000
    X_test_sample, _, _, _ = train_test_split(
        X_test, y_test,
        train_size=sample_amount,
        stratify=y_test,
        random_state=RANDOM_STATE
    )

    """
    docs for explainer: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
    GitHub user "eschibli" posted in a issue log the solution to the error I was getting. 
    His advice was to pass the get preidction function of my custom model directly into
    the explainer isntead of the custom model itself. Here is a link to the issue: 
    https://github.com/shap/shap/issues/2399 
    """
    explainer_rf = shap.Explainer(rf_model.get_predictions, X_test_sample, seed=RANDOM_STATE)
    explainer_xgb = shap.Explainer(xgb_model.get_predictions, X_test_sample, seed=RANDOM_STATE)
    if is_global:
        shap_values_rf = explainer_rf(X_test_sample)
        shap_values_xgb = explainer_xgb(X_test_sample)

        shap.summary_plot(shap_values_rf, X_test_sample, max_display=len(HOME_MODEL_FEATURES))
        shap.summary_plot(shap_values_xgb, X_test_sample, max_display=len(HOME_MODEL_FEATURES))

    else:
        probs = rf_model.get_probabilities(X_test)
        _, _, example_indices = produce_three_samples_for_local_expl(
            X_test, 
            y_test, 
            probs, 
            rf_model.threshold
        )
        for idx in example_indices:
            example = X_test.iloc[idx:idx+1]
            shap_values_rf = explainer_rf(example)
            shap.plots.bar(shap_values_rf, max_display=len(HOME_MODEL_FEATURES))

def get_permutation_importance():
    # Get data
    X, y = get_data(EXPERT_MODEL_FEATURES)
    X_train, X_val, X_test, y_train, y_val, y_test = get_rand_split_data(X, y)

    # Declare model
    class ExpertModel(DiabetesModelBase):
        def __init__(self, model):
            super().__init__(EXPERT_MODEL_FEATURES, model)

    # Becuase method is model agnostic, only looking at the RF. 
    rf_params = {'class_weight': {0: 1, 1: 4}, 'max_depth': 10, 'n_estimators': 150}
    model = ExpertModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE)) 
    model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)

    for random_state in [0, 10, RANDOM_STATE]: # RANDOM_STATE = 42 (in 'config.py')
        print(f"\nRandom State: {random_state}")
        # Code Reference: https://scikit-learn.org/stable/modules/permutation_importance.html
        r = permutation_importance(
            model, 
            X_test, 
            y_test, 
            n_repeats=5, 
            scoring= "f1", 
            random_state=random_state
        )
        for i in r.importances_mean.argsort()[::-1]:
            print(f"{X_test.columns[i]:<8}\t"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}")

def get_lime_explanation() -> None:
    # Get data
    X, y = get_data(EXPERT_MODEL_FEATURES)
    X_train, X_val, X_test, y_train, y_val, y_test = get_rand_split_data(X, y)

    # Declare model
    class ExpertModel(DiabetesModelBase):
        def __init__(self, model):
            super().__init__(EXPERT_MODEL_FEATURES, model)

    rf_params = {'class_weight': {0: 1, 1: 4}, 'max_depth': 10, 'n_estimators': 150}
    rf_model = ExpertModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE))
    rf_model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)

    probs = rf_model.get_probabilities(X_test)
    _, _, example_indices = produce_three_samples_for_local_expl( # get three examples
        X_test, 
        y_test, 
        probs, 
        rf_model.threshold
    )

    feature_names = X_train.columns.tolist() 
    categorical_indices = [feature_names.index(f) for f in EXPERT_CATEGORICAL_FEATURES]
    
    explainer = lime.lime_tabular.LimeTabularExplainer( # producing lime output
        X_train.values, 
        feature_names=feature_names, 
        class_names=['No Diabetes', 'Prediabetes/Diabetes'], 
        categorical_features=categorical_indices, 
        verbose=True, 
        mode="classification", 
        random_state=RANDOM_STATE
    )

    for i, idx in enumerate(example_indices): # printing the lime output for each example
        test_instance = X_test.iloc[idx:idx+1]
        test_array = test_instance.values[0]
        explanation = explainer.explain_instance(
            test_array,
            lambda x: rf_model.get_probabilities(pd.DataFrame(x, columns=EXPERT_MODEL_FEATURES), True),
            num_features=len(EXPERT_MODEL_FEATURES)
        )

        print(f"\n--- LIME Explanation for Example {i+1} ---")
        print(f"True label: {y_test.iloc[idx]}")
        print(tabulate(explanation.as_list(), headers=["Feature", "Weight"], tablefmt="grid", floatfmt=".5f"))


def get_counterfactuals(): 
    # Get data
    X, y = get_data(HOME_MODEL_FEATURES)
    X_train, X_val, X_test, y_train, y_val, y_test = get_rand_split_data(X, y)

    # Declare model
    class AtHomeModel(DiabetesModelBase):
        def __init__(self, model):
            super().__init__(HOME_MODEL_FEATURES, model)

    # Becuase method is model agnostic, only looking at the RF. 
    rf_params = {'class_weight': {0: 1, 1: 4}, 'max_depth': 10, 'n_estimators': 150}
    model = AtHomeModel(RandomForestClassifier(**rf_params, random_state=RANDOM_STATE))
    model.apply_train_calibrate_threshold(X_train, X_val, y_train, y_val)

    # Get probabilities
    probs = model.get_probabilities(X_test)  # class 1 probabilities

    # Attach them to the test set
    X_test_with_probs = X_test.copy()
    X_test_with_probs['prob'] = probs

    # Define a margin around the threshold to ensure counterfactuals are produceable. 
    threshold = model.threshold
    candidate_pool = pd.DataFrame()  
    margin = 0.1
    candidate_pool = X_test_with_probs[
        (X_test_with_probs['prob'] >= threshold - margin) &
        (X_test_with_probs['prob'] <= threshold + margin)
    ]
    
    bounds = [ # I had ChatGPT produce this for my by given it the bounds.
        (0, 1),    # HighBP
        (0, 1),    # CholCheck
        (10, 100),  # BMI
        (0, 1),    # Smoker
        (0, 1),    # Stroke
        (1, 8),    # Income
        (0, 1),    # HeartDiseaseorAttack
        (0, 1),    # PhysActivity
        (0, 1),    # Fruits
        (0, 1),    # Veggies
        (0, 1),    # HvyAlcoholConsump
        (0, 1),    # AnyHealthcare
        (0, 1),    # NoDocbcCost
        (1, 5),    # GenHlth (1 = excellent, 5 = poor)
        (0, 30),   # MentHlth (days per month)
        (0, 30),   # PhysHlth (days per month)
        (0, 1),    # DiffWalk
        (0, 1),    # Sex
        (1, 13),   # Age (ordinal categories: 1 = 18–24, 13 = 80+)
        (1, 6)     # Education
    ] 

    # Sample and print the index (Does not use random seed)
    instance_to_explain = candidate_pool.sample(n=1).drop(columns=['prob']) # This sample process does not use a random seed. See index output. 
    instance_index = instance_to_explain.index.item()  
    print(f"\nSample used is at index {instance_index}\n") # Output the index to the user for reproduceability

    x_prime_init = instance_to_explain.copy()
    np.random.seed(RANDOM_STATE)
    x_prime_init += np.random.normal(scale=0.1, size=x_prime_init.shape)
    x_prime_init = np.clip(x_prime_init, [b[0] for b in bounds], [b[1] for b in bounds])
    
    """Counterfactual method: used Molnar, C. explination for Machter et al. method. 
    
    Full book citation:
        Molnar, C. (2025). Interpretable Machine Learning:
        A Guide for Making Black Box Models Explainable (3rd ed.).
        christophm.github.io/interpretable-ml-book/
    """
    mad_vector = X_test.apply(lambda col: np.median(np.abs(col - np.median(col))), axis=0) # ChatGPT helped me with this line
    mad_vector = mad_vector.replace(0, 1e-6)
    
    f_x = model.get_single_probability(instance_to_explain)
    prob_target = model.threshold
    y_target = int(f_x >= prob_target) # Setting desired outcome to the opposite of the prediction

    print(f"\n Current Outcome: {y_target} with probability: {f_x}")

    best_result = None
    best_loss = np.inf

    orig_class = int(f_x >= prob_target)
    max_iterations = 50

    for _lambda in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50]:
        lambda_found = False
        for _ in range(max_iterations):
            def loss_function(x_prime_list):
                x_prime = pd.DataFrame([x_prime_list], columns=model.features)
                f_x_prime = model.get_single_probability(x_prime)
                distance = np.sum(np.abs(x_prime.iloc[0] - instance_to_explain.iloc[0]) / mad_vector.values)
                loss = _lambda * (f_x_prime - prob_target)**2 + distance
                return loss
            
            # See documentatoin for minimize (nelder-mead recommeneded by Molnar) 
            # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
            result = minimize( # ^^^^^^ Source ^^^^^^
                loss_function, 
                x0=x_prime_init.iloc[0].to_numpy(),
                method="nelder-mead", # Source on bounding: https://stackoverflow.com/questions/19244527/scipy-optimize-how-to-restrict-argument-values
                # bounds=bounds
            ) 
            
            if result.success:
                x_prime = pd.DataFrame([result.x], columns=model.features)
                f_x_prime = model.get_single_probability(x_prime)
                new_class = int(f_x_prime >= prob_target)

                if new_class != orig_class and result.fun <= best_loss:
                    best_loss = result.fun
                    best_result = result
                    lambda_found = True
                    print(f"\nBreaking with Lambda: {_lambda}")
                    break

            x_prime_init = pd.DataFrame([result.x], columns=model.features)

        if best_result and lambda_found:
            pd.set_option('display.max_columns', None) 
            pd.set_option('display.width', 0)   
            counterfactual = pd.DataFrame([best_result.x], columns=model.features)
            comparison = pd.concat(
                [instance_to_explain.reset_index(drop=True), counterfactual], 
                axis=0, 
                ignore_index=True
            )
            comparison.index = ['Original', 'Counterfactual']
            comparison = np.round(comparison, ROUND_DEC)
            comparison[comparison < 1e-6] = 0
            print("\nCounterfactual Comparison:")
            print(comparison.T)
        else:
            print("No successful counterfactual found.")


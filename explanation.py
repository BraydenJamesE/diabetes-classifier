import lime.lime_tabular
import pandas as pd
from base_model import DiabetesModelBase
from config import RANDOM_STATE


def get_lime_explanation(
    test_item: pd.DataFrame,
    y_test: int, 
    X_train: pd.DataFrame, 
    feature_names: list, 
    categorical_features: list, 
    model: DiabetesModelBase,
    is_verbose: bool = True,
    to_html: bool = True 
) -> None:
    
    categorical_indices = [feature_names.index(feat) for feat in categorical_features]
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, 
        feature_names=feature_names, 
        class_names=['No Diabetes', 'Prediabetes/Diabetes'], 
        categorical_features=categorical_indices, 
        verbose=is_verbose, 
        mode="classification", 
        random_state=RANDOM_STATE
    )

    num_features = len(feature_names)
    test_array = test_item.values[0]
    exp = explainer.explain_instance(
        test_array,  
        lambda x : model.get_probabilities(pd.DataFrame(x, columns=feature_names), True), 
        num_features=num_features
    )

    print(f"True y: {y_test}")
    print("\nLIME Explanation:")
    print(exp.as_list())

    if to_html:
        html = exp.as_html()
        with open("lime_explanation.html", "w") as f:
            f.write(html)

def get_counterfactuals(): # TODO: Impliment
    BEHAVIORAL_FEATURES = [
        'BMI', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'GenHlth', 'MentHlth',
        'PhysHlth', 'DiffWalk'
    ]

    STRUCTURAL_OR_DIAGNOSIS_FEATURES = [
        'HighBP', 'HighChol', 'CholCheck',
        'AnyHealthcare', 'NoDocbcCost'
    ]

    return None

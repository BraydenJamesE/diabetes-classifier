"""
**Author:** Brayden Edwards  
**Course:** CS 541 – Machine Learning Challenges in the Real World
**School:** Oregon State University
**Professor:** Dr. Kiri Wagstaff  
**Term:** Spring 2025  
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base_model import DiabetesModelBase
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, roc_auc_score
from config import RANDOM_STATE

def plot_data_series(series: pd.Series, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(series)
    plt.title(title)
    plt.show()

def plot_model_calibration(X_test: pd.DataFrame, y_test: pd.Series, model: DiabetesModelBase) -> None:
    y_prob = model.get_probabilities(X_test)
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    disp = CalibrationDisplay(prob_true, prob_pred, y_prob)
    disp.plot()
    plt.title("Calibration Plot")
    plt.grid(True)
    plt.show()

def plot_probs(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    title: str = "Predicted probability by true class"
) -> None:
    """Box-plot the positive-class probability, split by true label."""
    neg_probs = prob_pos[y_true == 0]
    pos_probs = prob_pos[y_true == 1]
    plt.figure(figsize=(10,6))
    plt.boxplot([neg_probs, pos_probs], labels=["class 0", "class 1"], patch_artist=True)
    plt.title(title)
    plt.show()

def plot_roc(model: DiabetesModelBase, X_test, y_test):
    preds = model.get_predictions(X_test)
    probs = model.get_probabilities(X_test)

    print(f"AUC/ROC: {roc_auc_score(y_test, probs)}")
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="dashed")  
    plt.show()

def get_random_data_sample(full_dataset: pd.Series) -> tuple[pd.DataFrame, int]:
    import random
    random.seed(RANDOM_STATE)
    end_idx = full_dataset.shape[0]
    sample_idx = random.randint(0, end_idx - 1)
    sample = full_dataset.iloc[sample_idx]
    return sample, sample_idx

def plot_feature_importance():
    """Plots feature importance for random seed 42 test.
    
    This function plots the feature importance according to the 
    feature importanace findings in Challenge 2 using a the 
    random seed 42. More information in report. 
    """
    random_seed_used = 42
    feature_importance = [
        ('GenHlth', 0.068),
        ('HighBP', 0.057),
        ('BMI', 0.055),
        ('Age', 0.028),
        ('HighChol', 0.024),
        ('HvyAlchoholConsump', 0.005),
        ('Income', 0.005),
        ('HeartDiseaseorAttack', 0.004),
        ('CholCheck', 0.004),
        ('DiffWalk', 0.001),
        ('Sex', 0),
        ('Education', 0),
        ('AnyHealthcare', 0),
        ('Stroke', 0),
        ('Veggies', 0),
        ('PhysHlth', 0),
        ('Smoker', 0),
        ('Fruits', 0),
        ('PhysActivity', 0),
        ('NoDocbcCost', -0.001),
        ('MentHlth', -0.001)
    ]   
    features, scores = zip(*feature_importance)

    plt.figure(figsize=(10, 6))
    plt.barh(features, scores, color='steelblue')
    plt.xlabel("Mean Importance Score")
    plt.title(f"Permutation Feature Importance\nRandom Seed: {random_seed_used}")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

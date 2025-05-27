"""
Brayden Edwards
AI541 - Final Project
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base_model import DiabetesModelBase
from sklearn.calibration import CalibrationDisplay, calibration_curve

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
    plt.title("Calibration Plot (Validation Set)")
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
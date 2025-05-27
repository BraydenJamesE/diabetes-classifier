import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    recall_score, 
    balanced_accuracy_score,
    precision_score, 
)
from config import ROUND_DEC

class DiabetesModelBase:
    def __init__(self, features, model):
        self.features = features
        self.model = model
        self.calibrated_model = None
        self.threshold = 0.25
        self.evaluations = None

    def train(self, X_train, y_train):
        self.model.fit(X_train[self.features], y_train)

    def calibrate(self, X_val, y_val, method="isotonic"):
        frozen_model = FrozenEstimator(self.model) 
        calibrated_model = CalibratedClassifierCV(estimator=frozen_model, method=method)
        calibrated_model.fit(X_val[self.features], y_val)
        self.calibrated_model = calibrated_model
        
    def get_predictions(self, X):
        probs = self.calibrated_model.predict_proba(X[self.features])[:, 1]
        return (probs >= self.threshold).astype(int)

    def get_evaluations(self, X, y) -> dict: 
        preds = self.get_predictions(X)
        probs = self.get_probabilities(X)
        self.evaluations = {
            "classification_report": classification_report(y, preds),
            "confusion_matrix":  confusion_matrix(y, preds),
            "f1":  round(f1_score(y, preds), ROUND_DEC),
            "accuracy":  round(accuracy_score(y, preds), ROUND_DEC),
            "roc_auc":  round(roc_auc_score(y, probs), ROUND_DEC),
            "balanced_acc":  round(balanced_accuracy_score(y, preds), ROUND_DEC),
            "precision": round(precision_score(y, preds), ROUND_DEC),
            "recall": round(recall_score(y, preds), ROUND_DEC),
        }
        return self.evaluations
    
    def get_probabilities(self, X_test, should_return_full_probs: bool = False):
        assert self.calibrated_model is not None, "Model has not yet been calibrated."
        
        if should_return_full_probs:
            return self.calibrated_model.predict_proba(X_test[self.features])
        else:
            return self.calibrated_model.predict_proba(X_test[self.features])[:,1]
    
    def find_threshold_for_recall(self, y_true, y_probs, target_recall=0.75):
        epsilon = 0.05
        thresholds = np.arange(0, 1.01, 0.01)
        for thresh in thresholds:   
            y_pred = (y_probs >= thresh).astype(int)
            recall = recall_score(y_true, y_pred)
            if abs(recall - target_recall) <= epsilon:
                return thresh, recall
        raise RuntimeError("Unable to find proper threshold.")
    
    def find_threshold_for_f1(self, y_true, y_probs):
        best_f1 = -np.inf
        best_thresh = None
        thresholds = np.arange(0, 1.01, 0.01)
        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int) 
            f1 = f1_score(y_true, y_pred)
            if f1 >= best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        return best_thresh, best_f1

    def set_threshold(self, threshold):
        self.threshold = threshold

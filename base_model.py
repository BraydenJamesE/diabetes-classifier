"""
**Author:** Brayden Edwards  
**Course:** CS 541 – Machine Learning Challenges in the Real World
**School:** Oregon State University
**Professor:** Dr. Kiri Wagstaff  
**Term:** Spring 2025  
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    recall_score, 
    balanced_accuracy_score,
    precision_score, 
    brier_score_loss, 
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
        
    def get_predictions(self, X, custom_threshold: float = None):
        probs = self.calibrated_model.predict_proba(X[self.features])[:, 1]
        if custom_threshold:
            return (probs >= custom_threshold).astype(int)
        else:
            return (probs >= self.threshold).astype(int)
    
    def get_single_prediction(self, X_single: pd.DataFrame):
        if not self.is_model_calibrated():
            raise NotFittedError("Single prediction not possible: model has not been calibrated.")
        prob = self.calibrated_model.predict_proba(X_single[self.features])[:, 1][0]
        return int(prob >= self.threshold)
    
    def get_single_probability(self, X_single: pd.DataFrame):
        if not self.is_model_calibrated():
            raise NotFittedError("Single prediction not possible: model has not been calibrated.")
        prob = self.calibrated_model.predict_proba(X_single[self.features])[:, 1][0]
        return prob

    def get_evaluations(self, X, y) -> dict: 
        if not self.is_model_calibrated():
            raise NotFittedError("Evaluation not possible: model has not been calibrated.")
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
            "brier": round(brier_score_loss(y, probs), ROUND_DEC)
        }
        return self.evaluations
    
    def get_probabilities(self, X_test, should_return_full_probs: bool = False) -> np.ndarray:
        assert self.calibrated_model is not None, "Model has not yet been calibrated."
        
        if should_return_full_probs:
            return self.calibrated_model.predict_proba(X_test[self.features])
        else:
            return self.calibrated_model.predict_proba(X_test[self.features])[:,1]
    
    def find_threshold_for_recall(self, y_true, y_probs, target_recall=0.75):
        epsilon = 0.01
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

    def is_model_calibrated(self):
        return self.calibrated_model != None
    
    def apply_train_calibrate_threshold(
            self, 
            X_train, 
            X_val, 
            y_train, 
            y_val, 
            threshold_opt_type: str = "f1"
        ) -> None:
        """This function trains, calibrates, and sets the threshold for the model"""
        
        error_string = """
            Threshold option type variable not set properly in 'apply_train_calibrate_threshold'
            function. Please either use 'f1' or 'recall'. 
        """
        self.train(X_train, y_train)
        self.calibrate(X_val, y_val)
        
        if threshold_opt_type == "f1":
            probs = self.get_probabilities(X_val)
            threshold, _ = self.find_threshold_for_f1(y_val, probs)
        elif threshold_opt_type == 'recall':
            probs = self.get_probabilities(X_val)
            threshold, _ = self.find_threshold_for_recall(y_val, probs)
        else: 
            raise ValueError(error_string)

        self.set_threshold(threshold)

        
    # Special functions for the permuation importance sklearn function
    def fit(self, X, y):
        self.model.fit(X[self.features], y)
    
    def predict(self, X_test):
        return self.get_predictions(X_test)
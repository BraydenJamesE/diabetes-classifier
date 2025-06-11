# Diabetes Risk Prediction

**Author:** Brayden Edwards  
**Course:** CS 541 – Machine Learning Challenges in the Real World
**School:** Oregon State University
**Professor:** Dr. Kiri Wagstaff  
**Term:** Spring 2025  

This project builds a machine learning system to predict diabetes risk using the CDC Diabetes Health Indicators dataset. The goal is to assist both healthcare professionals and at-home users in assessing risk based on self-reported health factors.

link to dataset: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

Currently, the dataset is accessed through the UCI ML Repo fetch helper. This is done remotely and requires no extra work. If you want to download the data and store it locally, you will need to update the `get_data` function in the `data.py` file, as this is the function where the data is pulled throughout the entire project. This method, however, is not recommended as the remote helper works great. 

## Dependencies
- Python 3

- pandas

- numpy

- scipy

- scikit-learn

- xgboost

- shap

- lime

- matplotlib

- tabulate

- ucimlrepo

- imbalanced-learn

To install these dependencies run the following command in terminal: 

```bash
pip install pandas numpy scipy scikit-learn xgboost shap lime matplotlib tabulate ucimlrepo imbalanced-learn
```

## How to Run the Program
To produce outputs, run the main file by typing the following in your terminal:
```bash
python3 main.py
```

## Prediction Problem 
The prediction problem of my final project is the classification of patients as either not having diabetes or being prediabetic/diabetic. While the goal is primarily based on predicting these two classes, I also aim to create separate models for two distinct user groups: low-context at-home users and domain experts. 

## Models

Two tree-based classifiers were trained and calibrated:

Random Forest
XGBoost

Each classifier's probability outputs were calibrated and a custom decision threshold was decided by optimizing for F1 score. 

## Evaluation

Performance was evaluated using F1, recall, precision, and balanced accuracy. A stratified train/validation/test split was used for robust comparison. 

## Interpretability

To understand model decisions:

Global feature importance was analyzed using permutation importance and SHAP.
Local explanations were generated using LIME and Local SHAP.
Key features across methods: GenHlth, HighBP, and BMI.

An attempt at producing satisfactory counterfactuals was made but had little utility as user outputs due to unrealisitic outputs and optimization issues. 

These exact methodology and results can be found in the linked report. 

## Files
- `main.py` - code for running and obtaining each challenges results.

- `base_model.py` - parent class for custom models.

- `expert_model.py` - file for running class imbalance test with expert model  (challenge 1).

- `home_model.py` - file for running class imbalance test with at-home model  (challenge 1).

- `explanation.py` - file for obtaining interpretability results (challenge 2).

- `representation.py` - file used for obtaining representation results (challenge 3).

- `visualization.py` - file for visualizing results. Not used in code but was used to produce results in my report. 

- `config.py` - file containing important global variables. This file controls the reproduceability of the project. 

- `data.py` - this file is a data handler file including remote download using UCI ML Repo fetch helper.

- `expert_model_param_search.py` - file used to do hyperparameter tuning for expert model on OSU's HPC cluster. 

- `home_model_param_search.py` - file used to do hyperparameter tuning for at-home model on OSU's HPC cluster. 



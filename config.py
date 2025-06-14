"""
**Author:** Brayden Edwards  
**Course:** CS 541 – Machine Learning Challenges in the Real World
**School:** Oregon State University
**Professor:** Dr. Kiri Wagstaff  
**Term:** Spring 2025  
"""

# This file contains all global constants used in the script. Done in the effort 
# of reproducibility. 
ROUND_DEC = 3
RANDOM_STATE = 42 # CANNOT BE CHANGED. This was used to produce my hyperparameters. Change would cause leakage. 
TEST_RATIO = 0.2 # CANNOT BE CHANGED. This was used to produce my hyperparameters. Change would cause leakage. 
VAL_RATIO = 0.2 # CANNOT BE CHANGED. This was used to produce my hyperparameters. Change would cause leakage. 
TARGET_COL_NAME = 'Diabetes_binary'

EXPERT_MODEL_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
    'Income'
]

HOME_MODEL_FEATURES = [
    'HighBP', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Income', 
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education'
]

EXPERT_CATEGORICAL_FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex"
]

HOME_CATEGORICAL_FEATURES = [
    "HighBP",
    "CholCheck",
    "Smoker",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex"
]
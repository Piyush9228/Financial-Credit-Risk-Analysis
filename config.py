"""
Configuration file for the project
"""

# Project Configuration
PROJECT_NAME = "Financial Credit Risk Prediction with Preprocessing Analysis"
VERSION = "1.0.0"
AUTHOR = "[Your Name]"
DATE = "2024"

# Data Configuration
N_SAMPLES = 2000
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Model Configuration
MODELS = {
    'Logistic Regression': {
        'max_iter': 1000,
        'class_weight': 'balanced'
    },
    'Random Forest': {
        'n_estimators': 100,
        'class_weight': 'balanced'
    },
    'Support Vector Machine': {
        'kernel': 'rbf',
        'probability': True,
        'class_weight': 'balanced'
    },
    'Gradient Boosting': {
        'n_estimators': 100
    },
    'K-Nearest Neighbors': {
        'n_neighbors': 5
    }
}

# Evaluation Configuration
CV_SPLITS = 5
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

# Visualization Configuration
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLOR_PALETTE = 'husl'
FIG_SIZE = (12, 8)
DPI = 300

# Output Configuration
OUTPUT_DIR = 'results'
SAVE_FORMATS = ['png', 'csv', 'json']
"""
Model Training Module
Trains and evaluates multiple ML models
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import json

class ModelTrainer:
    """Trains and saves multiple ML models"""
    
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(),
            'svm': SVC(),
            'gradient_boosting': GradientBoostingClassifier(),
            'knn': KNeighborsClassifier()
        }
    
    def train_model(self, model_name, X_train, y_train):
        """Train a specific model"""
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model
    
    def save_model(self, model, filepath):
        """Save trained model"""
        joblib.dump(model, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        return joblib.load(filepath)
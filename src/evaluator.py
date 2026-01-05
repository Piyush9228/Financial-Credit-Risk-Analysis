"""
Evaluation Module
Evaluates model performance and generates metrics
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

class Evaluator:
    """Evaluates model performance"""
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model and return metrics"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        return metrics
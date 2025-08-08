from sklearn.ensemble import IsolationForest
from .base_detector import BaseDetector


class IsolationForestDetector(BaseDetector):
    def __init__(self, model_name):
        model_name = "Isolation Forest"
        self.threshold = 0.5
        super().__init__(model_name=model_name, threshold=self.threshold)

    
    def fit(self, X, y=None):
        """Fit the Isolation Forest model to the training data."""
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def detect(self, X, return_scores=False):
        """Detect anomalies in the given data using the Isolation Forest model."""

        self._check_fitted()
        self._validate_input(X)
        
        scores = self.decision_scores(X)
        predictions = scores > self.threshold
        
        predictions = predictions.astype(int)  # Convert boolean to int (0 or 1)
        if return_scores:
            return predictions, scores
        return predictions
    

    def decision_scores(self, X):
        """Calculate the decision scores for the given data."""
        self._check_fitted()
        self._validate_input(X)
        
        scores = -self.model.decision_function(X)
        return scores
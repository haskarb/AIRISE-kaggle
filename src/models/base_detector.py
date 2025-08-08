from abc import ABC, abstractmethod


class BaseDetector(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.threshold = None

    @abstractmethod
    def fit(self, X, y: None) -> "BaseDetector":
        """
        Fit the model to the training data.
        :param X: Training features.
        :param y: Training labels.
        """
        pass

    @abstractmethod
    def detect(self, X, return_scores: bool = False):
        """
        Predict the labels for the given data.
        :param X: Data to predict.
        :return: Predicted labels.
        """
        pass

    def _check_fitted(self):
        """Check if the model has been fitted.
        :raises RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.model_name} has not been fitted yet. Please call fit() before detect()."
            )

    def _validate_input(self, X):
        """Validate the input data.
        :param X: Data to validate.
        :raises ValueError: If the input data is not valid.
        """
        if not isinstance(X, (list, tuple)):
            raise ValueError(
                f"Input data must be a list or tuple, got {type(X)} instead."
            )
        if len(X) == 0:
            raise ValueError("Input data cannot be empty.")

    @abstractmethod
    def decision_scores(self, X):
        """
        Calculate the decision scores for the given data.
        :param X: Data to calculate decision scores.
        :return: Decision scores.
        """
        pass

    def get_params(self):
        """Get the parameters of the model.
        :return: Dictionary of model parameters.
        """
        return {
            "model_name": self.model_name,
            # "is_fitted": self.is_fitted,
            "threshold": self.threshold,
        }

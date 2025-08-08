from abc import ABC,abstractmethod
from typing import Optional
import polars as pl
import numpy as np

class BaseAD(ABC):

    @abstractmethod
    def fit(self, X:pl.DataFrame, y: Optional[pl.Series] = None) -> "BaseAD":
        pass
    
    @abstractmethod
    def detect(self, X: pl.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def anomaly_scores(self, X: pl.DataFrame) -> np.ndarray:
        pass
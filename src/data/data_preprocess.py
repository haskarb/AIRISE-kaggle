import polars as pl
from src.data.data_loader import DataLoader
from pathlib import Path
import numpy as np

class PreprocessData:
    """
    A class to preprocess data, splitting it into x and y axis dataframes
    and performing integrity checks.
    """
    def __init__(self):
        """
        Initializes the PreprocessData class.
        """
        pass

    def _data_integrity(self, x_data: pl.LazyFrame, y_data: pl.LazyFrame):
        """
        Performs a data integrity check on two lazy dataframes.

        Args:
            x_data: The first Polars LazyFrame.
            y_data: The second Polars LazyFrame.

        Returns:
            A boolean indicating if the data integrity check passed.
        """
        # Collect the lazy frames to compare columns and data
        x_collected = x_data.collect()
        y_collected = y_data.collect()
        
        if x_collected.columns != y_collected.columns:
            print("Columns do not match.")
            return False

        for col in x_collected.columns:
            # Use equals() method instead of series_equal() for column comparison
            if not x_collected[col].equals(y_collected[col]):
                print(f"Data in column '{col}' does not match.")
                return False

        print("Data integrity check passed.")
        return True

    def preprocess_data(self, data: pl.DataFrame):
        """
        Preprocesses the input dataframe using lazy operations.

        Args:
            data: The input Polars DataFrame.

        Returns:
            A numpy array with stacked x and y axis data.
        """
        numeric_columns = ["rpm", "adoc_start", "adoc_end", "rdoc", "fpt"]
        timeseries_columns = list(set(data.columns).difference(numeric_columns + ["axis"]))

        # Convert to lazy frame for efficient operations
        lazy_data = data.lazy()

        # Create lazy queries for x and y data
        data_x_numeric = lazy_data.filter(pl.col('axis') == 'x').select(numeric_columns)
        data_y_numeric = lazy_data.filter(pl.col('axis') == 'y').select(numeric_columns)

        # Data integrity check on numeric columns
        integrity_check_passed = self._data_integrity(data_x_numeric, data_y_numeric)
        if integrity_check_passed:
            print("Numeric data integrity check successful.")
        else:
            print("Numeric data integrity check failed.")

        # Create lazy queries for timeseries data and convert to numpy
        data_x_timeseries = lazy_data.filter(pl.col('axis') == 'x').select(timeseries_columns)
        data_y_timeseries = lazy_data.filter(pl.col('axis') == 'y').select(timeseries_columns)

        # Collect and convert to numpy arrays
        data_x_np = data_x_timeseries.collect().to_numpy()
        data_y_np = data_y_timeseries.collect().to_numpy()

        return np.stack((data_x_np, data_y_np), axis=1)

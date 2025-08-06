from typing import Dict
import polars as pl
from .data_loader import DataLoader
from pathlib import Path
# import num

class PreprocessData:
    def __init__(self):
        pass

    def _integrity_check(self, x_data: pl.DataFrame, y_data: pl.DataFrame) -> bool:
        if len(x_data) != len(y_data):
            return False

        columns_to_check = ["rpm", "adoc_start", "adoc_end"]

        # Check each column individually
        for col in columns_to_check:
            if not (x_data[col] == y_data[col]).all():
                return False

        return True


    def preprocess_data(self, data: pl.DataFrame) -> Dict[str, Dict[str, pl.DataFrame]]:
        numeric_columns = [
            "axis",
            # "label",
            "rpm",
            "adoc_start",
            "adoc_end",
            "rdoc",
            "fpt",
        ]
        timeseries_columns = [col for col in data.columns if col not in numeric_columns]

        data_x = data.filter(pl.col("axis") == "x") # type: ignore
        data_y = data.filter(pl.col("axis") == "y") # type: ignore

        if not self._integrity_check(data_x, data_y):
            raise ValueError("Sensors not matching!!!")

        data_x.drop("axis")
        data_x.drop("axis")

        return {
            "x": {
                "numeric": data_x.select(numeric_columns),
                "timeseries": data_x.select(timeseries_columns),
            },
            "y": {
                "numeric": data_y.select(numeric_columns),
                "timeseries": data_y.select(timeseries_columns),
            },
        }

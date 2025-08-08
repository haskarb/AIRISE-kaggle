import polars as pl
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataLoader:
    file_path: Path 
    

    def get_train_data(self) -> pl.DataFrame:
        return pl.read_csv(self.file_path / "train.csv")

    def get_test_data(self) -> pl.DataFrame:
        return pl.read_csv(self.file_path / "test.csv")

    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        return self.get_train_data(), self.get_test_data()

if __name__ == "__main__":
    data_loader = DataLoader(file_path=Path("data/raw"))
    df_train, df_test = data_loader.load_data()
    print(df_train.head())
    print(df_test.head())

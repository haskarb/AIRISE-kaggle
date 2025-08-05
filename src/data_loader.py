import polars as pl
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataLoader:
    file_path: Path 

    def load_data(self, split="train") -> pl.DataFrame:

        if split == "train":
            file_path = self.file_path / "train.csv"
        elif split == "test":
            file_path = self.file_path / "test.csv"
        else:
            raise ValueError("Invalid split value. Use 'train' or 'test'.")

        return pl.read_csv(file_path)

    def list_files(self) -> List[str]:
        return [str(file) for file in self.file_path.glob("*.csv")]
    


if __name__ == "__main__":
    data_loader = DataLoader(file_path=Path("data/raw"))
    df = data_loader.load_data("train")
    print(df.head())

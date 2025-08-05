import polars as pl
from data_loader import DataLoader
from pathlib import Path

class PreprocessData:
    def __init__(self):
        pass

    def _check_sensor_integrity(self, data: pl.DataFrame) -> bool:
        
        ""

        return data.is_null().sum().sum() == 0

    def preprocess_data(self, data:pl.DataFrame):

        numeric_columns = ["axis", "label", "rpm", "adoc_start", "adoc_end", "rdoc", "fpt"]
        # timeseries_columns 
        timeseries_columns = list(set(data.columns).difference(numeric_columns))

        numeric_data = data.select(numeric_columns)
        timeseries_data = data.select(timeseries_columns)


        data_x = data.filter(pl.col('axis')=='x')
        data_y = data.filter(pl.col('axis')=='y')

        
        return data_x, data_y


if __name__ == "__main__":
    data_loader = DataLoader(file_path=Path("data/raw"))
    df_train, df_test = data_loader.load_data()


    
    preprocessor = PreprocessData()
    x_train, y_train = preprocessor.preprocess_data(df_train)
    # x_test, y_test = preprocessor.preprocess_data(df_test)

    # print("X Train Data:")
    print(x_train.head())
    # print("\nY Train Data:")
    # print(y_train.head())

    # print(x_test.head())

    # Uncomment the following lines to see the unique values in each column


    # print("\nY Train Data:")
    # print(y_train.head())

    # # count unique values in each column
    # print("\nUnique values in each column:")
    # print(x_train.select(pl.all().value_counts()))
    # print(y_train.select(pl.all().value_counts()))
    # print(x_train["label"].value_counts())




        
import polars as pl

class PreprocessData:
    def __init__(self):
        pass

    def preprocess_data(self, data:pl.DataFrame):
        data_x = data.select(pl.col('axis')=='x')
        data_y = data.select(pl.col('axis')=='y')
        
        print(data_x.head())
        
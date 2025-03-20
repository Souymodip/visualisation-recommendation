import pandas as pd
from enum import Enum

class DataType(Enum):
    NUMERIC=0
    CATEGORICAL=1
    TIME_SERIES = 2
    NETWORK=3


def check_df_is_network(df):
    pass


def classify_data_type(df):
    """
    Classifies the dataset into one of the predefined categories:
    1. Only numeric data
    2. Only categorical data
    3. Mix of both categorical and numeric data
    4. Network data (based on heuristic)
    5. Time series data (based on date column detection)
    """
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime']).columns

    type_list = []
    if len(num_cols) > 0:
        type_list.append(DataType.NUMERIC)
    if len(cat_cols) > 0:
        type_list.append(DataType.CATEGORICAL)
    if len(date_cols) > 0:
        type_list.append(DataType.TIME_SERIES)

    return type_list


def test():
    csv_path = "/Users/priyankachakraborti/GIT/time_series/nasdaq/tesla_5y.csv"
    df = pd.read_csv(csv_path)
    classification = classify_data_type(df)
    print(f"Data type category: {classification}")


if __name__ == '__main__':
    test()

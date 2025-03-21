import pandas as pd
from read import detect_index_and_read_csv

def detect_datetime_columns(df):
    datetime_cols = []
    for col in df.select_dtypes(include=['object', 'datetime']):  # Exclude numeric
        if df[col].dtype in ['int64', 'float64']:
            continue 
        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().sum() > len(df) * 0.9:  # 90% threshold
                datetime_cols.append(col)
        except Exception:
            continue
    return datetime_cols


def is_datetime_index(df):
    return isinstance(df.index, pd.DatetimeIndex)


def has_regular_intervals(df, datetime_col):
    if datetime_col:
        time_diffs = df[datetime_col].sort_values().diff().dropna()
        return time_diffs.nunique() == 1  # True if all intervals are the same
    return False

def is_time_series(df, deb=True):
    try:
        datetime_col = detect_datetime_columns(df)
        res = is_datetime_index(df) or len(datetime_col) > 0
        if deb:
            print(f'is_datetime_index? {is_datetime_index(df)}. detect_datetime_columns? {len(datetime_col)}') 
        return res
    except Exception as e:
        print(f"Error in is_time_series: {e}")
        return False

def test():
    path = '/Users/priyankachakraborti/GIT/visualisation-recommendation/time_series_test_cases/mixed_date_formats.csv'
    df = detect_index_and_read_csv(path)
    print(is_time_series(df))


if __name__ == '__main__':
    test()
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from timeseries_decision import is_datetime_index, is_time_series, detect_datetime_columns
# import ace_tools as tools

def generate_test_cases():
    """Generate various test cases for time series detection."""
    test_cases = []
    
    # Test Case 1: Basic time series with datetime index
    dates1 = pd.date_range(start='2023-01-01', periods=100, freq='D')
    values1 = np.random.randn(100).cumsum()  # Random walk
    df1 = pd.DataFrame({'values': values1}, index=dates1)
    test_cases.append(("basic_time_series_with_datetime_index.csv", df1, True, 
                       "Basic time series with DatetimeIndex"))
    
    # Test Case 2: Time series with datetime column
    dates2 = pd.date_range(start='2023-01-01', periods=100, freq='D')
    values2 = np.random.randn(100).cumsum()
    df2 = pd.DataFrame({'date': dates2, 'values': values2})
    test_cases.append(("time_series_with_datetime_column.csv", df2, True,
                       "Time series with datetime column"))
    
    # Test Case 3: Irregular time intervals
    dates3 = [datetime(2023, 1, 1) + timedelta(days=i*np.random.randint(1, 4)) for i in range(50)]
    values3 = np.random.randn(50).cumsum()
    df3 = pd.DataFrame({'date': dates3, 'values': values3})
    test_cases.append(("irregular_time_intervals.csv", df3, True,
                       "Time series with irregular intervals"))
    
    # Test Case 4: Multiple numeric columns
    dates4 = pd.date_range(start='2023-01-01', periods=100, freq='D')
    df4 = pd.DataFrame({
        'date': dates4,
        'temperature': 20 + 5 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100),
        'humidity': 60 + 10 * np.cos(np.linspace(0, 2*np.pi, 100)) + np.random.randn(100),
        'pressure': 1010 + np.random.randn(100).cumsum() * 0.1
    })
    test_cases.append(("multiple_numeric_columns.csv", df4, True,
                       "Time series with multiple numeric columns"))
    
    # Test Case 5: Time series with categorical data
    dates5 = pd.date_range(start='2023-01-01', periods=100, freq='D')
    categories = ['Low', 'Medium', 'High']
    df5 = pd.DataFrame({
        'date': dates5,
        'value': np.random.randn(100).cumsum(),
        'category': [categories[i % 3] for i in range(100)]
    })
    test_cases.append(("time_series_with_categorical.csv", df5, True,
                       "Time series with categorical column"))
    
    # Test Case 6: Time series with missing dates (10% missing)
    dates6 = pd.date_range(start='2023-01-01', periods=100, freq='D')
    # Randomly remove 10% of dates
    mask = np.random.choice([True, False], size=100, p=[0.9, 0.1])
    dates6 = dates6[mask]
    values6 = np.random.randn(len(dates6)).cumsum()
    df6 = pd.DataFrame({'date': dates6, 'values': values6})
    test_cases.append(("time_series_with_missing_dates.csv", df6, True,
                       "Time series with some missing dates"))
    
    # Test Case 7: Non-time series tabular data
    df7 = pd.DataFrame({
        'id': range(1, 101),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    test_cases.append(("non_time_series_tabular.csv", df7, False,
                       "Regular tabular data without datetime"))
    
    # Test Case 8: Date-like strings that aren't real dates
    df8 = pd.DataFrame({
        'id': range(1, 101),
        'fake_date': [f"DATE-{i}" for i in range(1, 101)],
        'value': np.random.randn(100)
    })
    test_cases.append(("fake_date_strings.csv", df8, False,
                       "Data with fake date-like strings"))
    
    # Test Case 9: Just above threshold for datetime detection (91% valid dates)
    dates9 = pd.date_range(start='2023-01-01', periods=91, freq='D')
    invalid_dates = ['not a date'] * 9
    all_dates = list(dates9.astype(str)) + invalid_dates
    np.random.shuffle(all_dates)
    df9 = pd.DataFrame({
        'date': all_dates,
        'value': np.random.randn(100)
    })
    test_cases.append(("just_above_threshold.csv", df9, True,
                       "Just above threshold for datetime detection (91% valid)"))
    
    # Test Case 10: Just below threshold (89% valid dates)
    dates10 = pd.date_range(start='2023-01-01', periods=89, freq='D')
    invalid_dates = ['not a date'] * 11
    all_dates = list(dates10.astype(str)) + invalid_dates
    np.random.shuffle(all_dates)
    df10 = pd.DataFrame({
        'date': all_dates,
        'value': np.random.randn(100)
    })
    test_cases.append(("just_below_threshold.csv", df10, False,
                       "Just below threshold for datetime detection (89% valid)"))
    
    # Test Case 11: Strictly ordered data (not time series)
    df11 = pd.DataFrame({
        'id': range(1, 101),
        'strictly_increasing': range(1, 101),
        'strictly_decreasing': range(100, 0, -1),
        'random_values': np.random.randn(100)
    })
    test_cases.append(("strictly_ordered_data.csv", df11, False,
                       "Strictly ordered data but not time series"))
    
    # Test Case 12: Approximately ordered data
    x = np.linspace(0, 10, 100)
    noise = np.random.randn(100) * 0.1
    trend = x + noise  # mostly increasing but with small fluctuations
    df12 = pd.DataFrame({
        'id': range(1, 101),
        'approx_increasing': trend,
        'random_values': np.random.randn(100)
    })
    test_cases.append(("approximately_ordered_data.csv", df12, False,
                       "Approximately ordered data but not time series"))
    
    # Test Case 13: Date as string format
    date_strings = [f"2023-{i//30+1:02d}-{i%30+1:02d}" for i in range(100)]
    df13 = pd.DataFrame({
        'date_string': date_strings,
        'value': np.random.randn(100).cumsum()
    })
    test_cases.append(("date_as_string.csv", df13, True,
                       "Dates stored as strings that can be parsed"))
    
    # Test Case 14: Mixed date formats
    mixed_dates = []
    for i in range(100):
        if i % 3 == 0:
            mixed_dates.append(f"2023-{i//30+1:02d}-{i%30+1:02d}")
        elif i % 3 == 1:
            mixed_dates.append(f"{i%30+1:02d}/{i//30+1:02d}/2023")
        else:
            mixed_dates.append(f"{i%30+1:02d}-{['Jan', 'Feb', 'Mar', 'Apr'][i//30]}-2023")
    
    df14 = pd.DataFrame({
        'mixed_date_formats': mixed_dates,
        'value': np.random.randn(100).cumsum()
    })
    test_cases.append(("mixed_date_formats.csv", df14, True,
                       "Dates in mixed formats that can still be parsed"))
    
    # Test Case 15: Time series with timestamps
    timestamps = pd.date_range(start='2023-01-01 00:00:00', periods=100, freq='H')
    df15 = pd.DataFrame({
        'timestamp': timestamps,
        'sensor_value': np.sin(np.linspace(0, 8*np.pi, 100)) + np.random.randn(100)*0.2
    })
    test_cases.append(("time_series_with_timestamps.csv", df15, True,
                       "Time series with hourly timestamps"))
    
    return test_cases

# Function to save test cases to folder
def save_test_cases(test_cases, folder="time_series_test_cases"):
    """Save test cases to CSV files in the specified folder."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    saved_paths = []
    for filename, df, expected, description in test_cases:
        filepath = os.path.join(folder, filename)
        df.to_csv(filepath, index=False if 'date' in df.columns or not is_datetime_index(df) else True)
        
        # Save metadata
        meta_filename = os.path.splitext(filename)[0] + "_meta.txt"
        meta_filepath = os.path.join(folder, meta_filename)
        with open(meta_filepath, "w") as f:
            f.write(f"Description: {description}\n")
            f.write(f"Expected time series: {expected}\n")
            f.write(f"Number of rows: {len(df)}\n")
            f.write(f"Number of columns: {len(df.columns)}\n")
            f.write(f"Columns: {', '.join(df.columns)}\n")
            f.write(f"Has DatetimeIndex: {is_datetime_index(df)}\n")
            datetime_col = detect_datetime_columns(df)
            f.write(f"Datetime column: {datetime_col if datetime_col else 'None'}\n")
        
        saved_paths.append(filepath)
    
    return saved_paths


def read_meta_file(meta_filepath):
    """Reads the metadata file and extracts expected time series boolean."""
    expected = None
    with open(meta_filepath, "r") as f:
        for line in f:
            if "Expected time series" in line:
                expected = line.strip().split(": ")[-1].strip() == "True"
    return expected

def validate_time_series(folder="time_series_test_cases"):
    """Reads all CSV files and corresponding metadata, validates is_time_series function."""
    results = []
    
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            csv_filepath = os.path.join(folder, filename)
            meta_filepath = os.path.join(folder, filename.replace(".csv", "_meta.txt"))
            
            if not os.path.exists(meta_filepath):
                print(f"Missing metadata file for {filename}")
                continue
            
            # Read the CSV file as DataFrame
            df = pd.read_csv(csv_filepath)
            
            # Read metadata file
            expected = read_meta_file(meta_filepath)
            
            # Determine if function correctly identifies time series
            detected = is_time_series(df)
            
            # Store results
            results.append({
                "Filename": filename,
                "Expected": expected,
                "Detected": detected,
                "Correct": expected == detected
            })
    
    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)

    # tools.display_dataframe_to_user(name="Time Series Validation Results", dataframe=results_df)
    
    return results_df

def main():
    save_test_cases(generate_test_cases())
    res = validate_time_series()
    print(res)

if __name__ == '__main__':
    main()
            
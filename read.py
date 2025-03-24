import pandas as pd
import chardet
from datetime import datetime
import re
import warnings
# from run_llm import run_llama3

import csv


def detect_header(filename):
    # Try different delimiters
    delimiters = [',', ';', '\t', '|']
    for delimiter in delimiters:
        try:
            with open(filename, newline=delimiter) as csvfile:
                sample = csvfile.read(1024)
                return csv.Sniffer().has_header(sample)
        except:
            continue
    return True


def to_csv_time_cols(file_path):
    # Suppress warnings for this function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Step 1: Detect file encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        print(f"Detected encoding: {encoding}")

        is_header = detect_header(file_path)
        if is_header:
            print("Detected header")
        else:
            print("No header detected")
        
        # Step 2: Read the CSV file with detected encoding
        try:
            df = pd.read_csv(file_path, encoding=encoding, header=0 if is_header else None)
        except Exception as e:
            print(f"Error with detected encoding {encoding}, falling back to utf-8: {e}")
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except Exception as e:
                print(f"Error with utf-8, trying latin-1: {e}")
                df = pd.read_csv(file_path, encoding='latin-1')

        for col in df.columns:
            try:
                s = pd.to_numeric(df[col], errors='raise')
            except Exception:
                continue
            if s.isnull().any() or not s.is_unique:
                continue
            if len(s) < 2 or (s.diff().dropna() == 1).all():
                return df.set_index(col), []

        # if no index found, try for time columns        
        # Step 3: Identify time columns
        time_columns = []
        time_patterns = [
            # Time formats (HH:MM:SS, HH:MM)
            r'^\d{1,2}:\d{2}(:\d{2})?$',
            # Date formats (YYYY-MM-DD, MM/DD/YYYY, etc.)
            r'^\d{4}-\d{1,2}-\d{1,2}$',
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',
            r'^\d{1,2}-\d{1,2}-\d{2,4}$',
            # DateTime formats
            r'^\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{2}(:\d{2})?$',
            r'^\d{1,2}/\d{1,2}/\d{2,4}\s\d{1,2}:\d{2}(:\d{2})?$',
            # Month abbreviation formats (1-Jan, 01-Jan, Jan-01, etc.)
            r'^\d{1,2}[-/]\w{3}([-/]\d{2,4})?$',
            r'^\w{3}[-/]\d{1,2}([-/]\d{2,4})?$'
        ]
        
        # Month abbreviations for pattern matching
        month_abbrs = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        for column in df.columns:
            # Skip columns that are clearly not time-related
            if not pd.api.types.is_string_dtype(df[column]) and not pd.api.types.is_object_dtype(df[column]):
                continue
            # Check if column name suggests time
            # time_related_names = ['time', 'date', 'day', 'year', 'month', 'hour', 'minute', 'second', 'timestamp']
            # if any(time_name in column.lower() for time_name in time_related_names):
            # Try to convert to datetime
            try:
                pd.to_datetime(df[column], errors='raise')
                time_columns.append(column)
                df[column] = pd.to_datetime(df[column])
                continue
            except:
                pass
            
            # If it's not numeric, check if values match time patterns
            if pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
                sample = df[column].dropna().head(10).astype(str)
                if len(sample) > 0:
                    # Check for month abbreviations in the data
                    has_month_abbrs = any(
                        any(abbr in val.lower() for abbr in month_abbrs)
                        for val in sample
                    )
                    
                    pattern_matches = [
                        any(re.match(pattern, val) for val in sample)
                        for pattern in time_patterns
                    ]
                    
                    if any(pattern_matches) or has_month_abbrs:
                        try:
                            # First try with built-in parser
                            time_column = pd.to_datetime(df[column], errors='coerce')
                            if not time_column.isna().all():
                                time_columns.append(column)
                                df[column] = time_column
                                continue
                        except:
                            pass
                        
                        # If standard parsing fails for month abbreviations, try custom approach
                        if has_month_abbrs:
                            try:
                                # Try different date formats with month abbreviations
                                for date_format in ['%d-%b', '%d-%b-%y', '%d-%b-%Y', '%b-%d', '%b-%d-%y', '%b-%d-%Y']:
                                    try:
                                        time_column = pd.to_datetime(df[column], format=date_format, errors='coerce')
                                        if not time_column.isna().all():
                                            time_columns.append(column)
                                            df[column] = time_column
                                            break
                                    except:
                                        continue
                            except:
                                pass
    
    # Set the first time column as index if time columns were found
    if time_columns:
        df.set_index(time_columns[0], inplace=True)

    return df, time_columns

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "/Users/priyankachakraborti/GIT/infographics-data/csv/data/thisdayinhistory.csv"
    dataframe, time_cols = to_csv_time_cols(file_path)
    print(f"DataFrame shape: {dataframe.shape}")
    print(f"Identified time columns: {time_cols}")
    
    # Display first few rows
    # print("\nDataFrame preview:")
    # print(dataframe.head())
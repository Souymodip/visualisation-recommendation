
import pandas as pd

def detect_index_and_read_csv(filepath, n_rows=10, possible_index_names=('id', 'index')):
    try:
        # Read a sample of the CSV to inspect the columns
        sample_df = pd.read_csv(filepath, nrows=n_rows)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading sample: {e}")
        return None


    # Check for a column with a common "index" name
    possible_index_col = None
    for name in possible_index_names:
        if name in sample_df.columns:
            possible_index_col = name
            break
    else:  #If no common name found, check if any columns are unique
        # Check for a unique column
        for col in sample_df.columns:
            if sample_df[col].is_unique:
                possible_index_col = col
                break

    if possible_index_col:
        try:
            # Verify uniqueness on the entire dataset (optional but recommended)
            temp_df = pd.read_csv(filepath, usecols=[possible_index_col]) #Read just the index column
            if not temp_df[possible_index_col].is_unique:
                print(f"Warning: Column '{possible_index_col}' is not unique across the whole dataset. Not using as index.")
                possible_index_col = None #Don't use the index column
        except Exception as e:
            print(f"Error checking full uniqueness: {e}")
            possible_index_col = None #Don't use the index column

    if possible_index_col:
        print(f"Detected index column: '{possible_index_col}'")
        try:
            df = pd.read_csv(filepath, index_col=possible_index_col)
            return df
        except Exception as e:
             print(f"Error reading full CSV with index_col: {e}")
             return None


    # If no suitable index column is found, read the CSV without specifying an index
    print("No suitable index column detected.  Using default integer index.")
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error reading full CSV without index_col: {e}")
        return None
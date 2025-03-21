import pandas as pd
from hierarchy_decision import check_multiple_rooted_hierarchy

def has_one_cat_one_num(df):
    # Select categorical (object or category) and numerical columns.
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    num_cols = df.select_dtypes(include=["number"]).columns
    return (len(cat_cols) == 1) and (len(num_cols) == 1)

def is_categorical_unique(df):
    # Ensure exactly one categorical column exists.
    cat = df.select_dtypes(include=["object", "category"])
    if len(cat.columns) != 1:
        raise ValueError("DataFrame must contain exactly one categorical column.")
    # Return True if all values in the categorical column are unique.
    return cat.iloc[:, 0].is_unique

def has_one_cat_and_several_num(df):
    # Exactly one categorical column and at least one numerical column.
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    num_cols = df.select_dtypes(include=["number"]).columns
    return (len(cat_cols) == 1) and (len(num_cols) >= 1)


def has_several_cat_and_one_num(df):
    # At least one categorical column and one numerical column.
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    num_cols = df.select_dtypes(include=["number"]).columns
    return (len(cat_cols) >= 1) and (len(num_cols) == 1)


def common_ordered_numeric_column(df):
    # Identify the single categorical column and numerical columns.
    cat_col = df.select_dtypes(include=["object", "category"]).squeeze()
    if cat_col.name is None:
        raise ValueError("DataFrame must have exactly one categorical column")
    num_cols = df.select_dtypes(include=["number"]).columns

    # For each group (by the categorical column), determine which numerical columns are ordered
    ordered = df.groupby(cat_col.name).apply(
        lambda g: [col for col in num_cols if g[col].is_monotonic_increasing or g[col].is_monotonic_decreasing]
    )
    
    # Check that each group has exactly one ordered numerical column
    if not ordered.apply(lambda lst: len(lst) == 1).all():
        return False, None
    
    # Check that the ordered column is the same across all groups
    common = ordered.apply(lambda lst: lst[0]).unique()
    return (len(common) == 1, common[0] if len(common) == 1 else None)


def check_categorical_uniqueness(df):
    # Automatically detect categorical columns (object, category, and boolean types)
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # If no categorical columns found, return message
    if not categorical_cols:
        return "No categorical columns found in the dataframe."
    
    # Count occurrences of each combination of categorical values
    # If any combination appears more than once, the data is not unique
    is_unique = df[categorical_cols].duplicated().sum() == 0
    
    return is_unique


def check_categorical_hierarchy(df):
    # Automatically detect categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # If no categorical columns found, return message
    if not categorical_cols:
        return "No categorical columns found in the dataframe."
    
    # Create a new dataframe with only categorical columns
    cat_df = df[categorical_cols].copy()
    
    # Check if the categorical dataframe has a single rooted hierarchy
    result = check_multiple_rooted_hierarchy(cat_df)
    
    return result

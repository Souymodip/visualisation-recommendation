import pandas as pd
from scipy.stats import spearmanr

def is_weakly_ordered(df):
    """Check if a column is weakly increasing or decreasing (allows duplicates)."""
    return any(df[c].is_monotonic_increasing or df[c].is_monotonic_decreasing for c in df.columns)

def is_approximately_ordered(series, threshold=0.9):
    """Check if a column is approximately ordered using Spearman's correlation."""
    if series.nunique() < 3:  # Too few unique values to check ordering
        return False
    rank_corr, _ = spearmanr(series, range(len(series)))
    return abs(rank_corr) >= threshold





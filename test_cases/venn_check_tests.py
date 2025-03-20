import pandas as pd
import numpy as np
import os
from pathlib import Path
from venn_check import check_venn
from read import detect_index_and_read_csv

def generate_random_categorical_data(n_samples=100):
    categories1 = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
    categories2 = np.random.choice(['X', 'Y', 'Z', 'W', 'V'], n_samples)
    return pd.DataFrame({'Category1': categories1, 'Category2': categories2})

def generate_overlapping_categories(n_samples=100):
    categories = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
    categories2 = np.random.choice(['A', 'B', 'C', 'F', 'G'], n_samples)
    return pd.DataFrame({'Category1': categories, 'Category2': categories2})

def generate_category_versions(n_samples=100):
    categories = np.random.choice(['Fruit', 'Vegetable', 'Dairy', 'Meat', 'Grains'], n_samples)
    categories2 = np.random.choice(['Fruit', 'Vegetable', 'Dairy', 'Seafood', 'Nuts'], n_samples)
    return pd.DataFrame({'Category_v1': categories, 'Category_v2': categories2})

def generate_different_attributes(n_samples=100):
    colors = np.random.choice(['Red', 'Blue', 'Green', 'Yellow', 'Purple'], n_samples)
    shapes = np.random.choice(['Circle', 'Triangle', 'Square', 'Rectangle', 'Oval'], n_samples)
    return pd.DataFrame({'Color': colors, 'Shape': shapes})

def generate_user_tags(n_samples=100):
    tags1 = np.random.choice(['Python', 'Java', 'JavaScript', 'SQL', 'R'], n_samples)
    tags2 = np.random.choice(['Python', 'Java', 'C++', 'Go', 'Swift'], n_samples)
    return pd.DataFrame({'TagA': tags1, 'TagB': tags2})

def generate_language_country(n_samples=100):
    languages = np.random.choice(['English', 'Spanish', 'French', 'German', 'Italian'], n_samples)
    countries = np.random.choice(['USA', 'UK', 'France', 'Germany', 'Italy'], n_samples)
    return pd.DataFrame({'Language': languages, 'Country': countries})


def main():
    test_data_dir = Path('test_data')
    test_data_dir.mkdir(exist_ok=True)

    test_cases = [
        ('random_categories.csv', generate_random_categorical_data),
        ('overlapping_categories.csv', generate_overlapping_categories),
        ('category_versions.csv', generate_category_versions),
        ('different_attributes.csv', generate_different_attributes),
        ('user_tags.csv', generate_user_tags),
        ('languages_countries.csv', generate_language_country)
    ]

    for filename, generator in test_cases:
        df = generator()
        df.to_csv(test_data_dir / filename, index=False)
        print(f"\nTesting {filename}:")
        print(f"Venn diagram suitable: {check_venn(str(test_data_dir / filename))}")


if __name__ == "__main__":
    main()

import pandas as pd
import os
from read import detect_index_and_read_csv
from timeseries_decision import is_time_series
from network_decision import check as network_check
from strict_num_rec import flow_chart as strict_num_flow_chart
from strict_num_rec import case_to_recommendation as strict_num_recommendation
from strict_cat_rec import flow_chart as strict_cat_flow_chart
from strict_cat_rec import case_to_recommendation as strict_cat_recommendation  
from cat_num_rec import flow_chart as cat_num_flow_chart
from cat_num_rec import case_to_recommendation as cat_num_recommendation



def top_level_decision(csv_path):
    base_name = os.path.basename(csv_path)
    df = detect_index_and_read_csv(csv_path)

    if df.empty:
        print(f"Empty DataFrame: {base_name}")
        return

    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns  # Anything that's not numerical

    print("=======================numerical_cols=========================")
    if len(numerical_cols) == df.shape[1]:
        
        print(f"Only Numerical DataFrame: {base_name}")
        num_case = strict_num_flow_chart(df)
        recommendations = strict_num_recommendation(num_case)
        print(recommendations)

    print("=======================categorical_cols=========================")
    if len(categorical_cols) == df.shape[1]:    
        print(f"Only Categorical DataFrame: {base_name}")
        cat_case = strict_cat_flow_chart(df, base_name)
        recommendations = strict_cat_recommendation(cat_case)
        print(recommendations)

    print("=======================time_series=========================")
    if is_time_series(df):
        print(f"Time Series DataFrame: {base_name}")
    
    print("================================================")
    if network_check(csv_path):
        print(f"Network DataFrame: {base_name}")

    print("================================================")
    if len(numerical_cols) > 0 and len(categorical_cols) > 0:
        print(f"Mixed DataFrame: {base_name}")
        cat_num_case = cat_num_flow_chart(df, base_name)
        recommendations = cat_num_recommendation(cat_num_case)
        print(recommendations)


def main():
    csv_path = "/Users/priyankachakraborti/Downloads/data.csv"
    top_level_decision(csv_path)

if __name__ == "__main__":
    main()


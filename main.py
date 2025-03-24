import pandas as pd
import os
from read import to_csv_time_cols
from network_decision import check as network_check
from strict_num_rec import flow_chart as strict_num_flow_chart
from strict_num_rec import case_to_recommendation as strict_num_recommendation
from strict_cat_rec import flow_chart as strict_cat_flow_chart
from strict_cat_rec import case_to_recommendation as strict_cat_recommendation  
from cat_num_rec import flow_chart as cat_num_flow_chart
from cat_num_rec import case_to_recommendation as cat_num_recommendation
from aux import print_green, print_yellow, print_red
from chart_types import get_chart_type_name
from sample_plots import plot_chart
from chart_types import ChartType


def show_recommendations(recommendations):
    print_yellow("\t Recommendations are:")
    for k, type in enumerate(recommendations):
        print_yellow(f"\t {k+1}. {get_chart_type_name(type)}")
        plot_chart(type)

def top_level_decision(csv_path):
    base_name = os.path.basename(csv_path)
    df, time_cols = to_csv_time_cols(csv_path)
    # print(f'Columns: {df.columns.tolist()}')
    print(df.head(3))

    # import pdb; pdb.set_trace()
    if df.empty:
        print(f"Empty DataFrame: {base_name}")
        return

    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns  # Anything that's not numerical

    print(f"Plotting recommendations for [{base_name}]")

    if len(numerical_cols) == df.shape[1]:
        print_green(f"[@] Only Numerical DataFrame.")
        recommendations = strict_num_recommendation(strict_num_flow_chart(df))        
        show_recommendations(recommendations)
    else:
        print_red(f"[@] Not a Strict Numerical DataFrame.")
    
    # import pdb; pdb.set_trace() 
    if len(categorical_cols) == df.shape[1]:    
        print_green(f"[@] Only Categorical DataFrame.")
        recommendations = strict_cat_recommendation(strict_cat_flow_chart(df, base_name))
        show_recommendations(recommendations)
    else:
        print_red(f"[@] Not a Strict Categorical DataFrame.")

    if len(numerical_cols) > 0 and len(categorical_cols) > 0:
        print_green(f"[@] Mixed DataFrame.")
        recommendations = cat_num_recommendation(cat_num_flow_chart(df))
        show_recommendations(recommendations)
    else:
        print_red(f"[@] Not a Mixed DataFrame.")

    if len(time_cols) > 0 and len(numerical_cols) == df.shape[1]:
        print_green(f"[@] Time Series Detected.")
        show_recommendations([ChartType.LINE_PLOT, ChartType.AREA_PLOT, ChartType.RIDGE_LINE])
    else:
        print_red(f"[@] Not a Numerical Time Series DataFrame.")

    if network_check(df, base_name):
        print_green(f"[@] Network DataFrame.")
        recommendations = []
        show_recommendations(recommendations)
    else:
        print_red(f"[@] Not a Network DataFrame.")



def main():
    csv_path = "/Users/priyankachakraborti/GIT/infographics-data/csv/data/top10s (version 1).xlsb.csv"
    top_level_decision(csv_path)

if __name__ == "__main__":
    main()


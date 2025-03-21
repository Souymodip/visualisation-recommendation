import pandas as pd
import os
import numpy as np
from cat_num_rec import flow_chart, case_to_recommendation



# Function to generate test cases for each decision branch
def generate_test_cases():
    # Case 0: One categorical, one numeric, categorical is unique
    case0_df = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D'],
        'value': [10, 20, 30, 40]
    })
    case0_df.to_csv("test_cases/case0_one_cat_one_num_unique.csv", index=False)
    
    # Case 1: One categorical, one numeric, categorical is not unique
    case1_df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'value': [10, 20, 30, 40]
    })
    case1_df.to_csv("test_cases/case1_one_cat_one_num_not_unique.csv", index=False)
    
    # Case 2: One categorical, several numeric, categorical is unique
    case2_df = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D'],
        'value1': [10, 20, 30, 40],
        'value2': [100, 200, 300, 400],
        'value3': [1.1, 2.2, 3.3, 4.4]
    })
    case2_df.to_csv("test_cases/case2_one_cat_several_num_unique.csv", index=False)
    
    # Case 3: One categorical, several numeric, categorical not unique, numeric ordered
    # Assuming "ordered" means values follow a consistent pattern/trend for each category
    case3_df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B', 'C', 'C'],
        'year': [2020, 2021, 2020, 2021, 2020, 2021],  # Common ordered column
        'value1': [10, 15, 20, 25, 30, 35],
        'value2': [100, 150, 200, 250, 300, 350]
    })
    case3_df.to_csv("test_cases/case3_one_cat_several_num_not_unique_ordered.csv", index=False)
    
    # Case 4: One categorical, several numeric, categorical not unique, numeric not ordered
    case4_df = pd.DataFrame({
        'category': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value1': [10, 15, 20, 5, 30, 25],  # No consistent ordering
        'value2': [100, 50, 200, 150, 300, 250]
    })
    case4_df.to_csv("test_cases/case4_one_cat_several_num_not_unique_not_ordered.csv", index=False)
    
    # Case 5: Several categorical with hierarchy, one numeric, categorical combination is unique
    case5_df = pd.DataFrame({
        'region': ['North', 'North', 'South', 'South'],
        'city': ['NY', 'Boston', 'Miami', 'Austin'],
        'value': [100, 200, 300, 400]
    })
    case5_df.to_csv("test_cases/case5_several_cat_one_num_hierarchy_unique.csv", index=False)
    
    # Case 6: Several categorical with hierarchy, one numeric, categorical combination is not unique
    case6_df = pd.DataFrame({
        'region': ['North', 'North', 'North', 'South', 'South'],
        'city': ['NY', 'NY', 'Boston', 'Miami', 'Austin'],
        'value': [100, 150, 200, 300, 400]
    })
    case6_df.to_csv("test_cases/case6_several_cat_one_num_hierarchy_not_unique.csv", index=False)
    
    # Case 7: Several categorical without hierarchy, one numeric
    case7_df = pd.DataFrame({
        'fruit': ['Apple', 'Orange', 'Banana', 'Grape'],
        'color': ['Red', 'Orange', 'Yellow', 'Purple'],  # Unrelated to fruit (no hierarchy)
        'value': [10, 20, 30, 40]
    })
    case7_df.to_csv("test_cases/case7_several_cat_one_num_no_hierarchy.csv", index=False)
    
    # Case 8: No categorical, all numeric
    case8_df = pd.DataFrame({
        'value1': [10, 20, 30, 40],
        'value2': [100, 200, 300, 400],
        'value3': [1.1, 2.2, 3.3, 4.4]
    })
    case8_df.to_csv("test_cases/case8_no_cat_all_num.csv", index=False)
    
    print("All test cases generated and saved to 'test_cases' folder.")

# Function to test all CSV files in the test_cases folder
def test_all_cases():
    results = []
    
    # Get all CSV files in the test_cases folder
    csv_files = [f for f in os.listdir("test_cases") if f.endswith(".csv")]
    
    for csv_file in csv_files:
        file_path = os.path.join("test_cases", csv_file)
        base_name = os.path.splitext(csv_file)[0]
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Execute the flow_chart function
        print(f"\nTesting file: {csv_file}")
        result_case = flow_chart(df, base_name)
        
        # Get recommendations
        recommendations = case_to_recommendation(result_case)
        
        # Store results
        results.append({
            'file': csv_file,
            'case': result_case,
            'recommendations': recommendations
        })
    
    # Create a summary dataframe
    summary_df = pd.DataFrame(results)
    print("\nSummary of test results:")
    print(summary_df[['file', 'case']])
    
    return summary_df

# Main function to run the tests
def main():
    # Generate test cases
    generate_test_cases()
    
    # Test all cases
    test_all_cases()
    
    # Save summary to CSV
    # summary.to_csv("test_cases/summary.csv", index=False)
    # print("\nTest summary saved to 'test_cases/summary.csv'")

if __name__ == "__main__":
    # Create a directory to store test CSV files
    os.makedirs("cat_num_rec_test_cases", exist_ok=True)
    main()
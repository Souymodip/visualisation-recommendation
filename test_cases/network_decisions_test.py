import pandas as pd
from network_decision import check_dataframe_structure, check
import os 
import numpy as np

def test_cases():
    # Positive Test Cases

    # Test 1: Basic positive case (already implemented)
    data1 = {
        'C1': ['A', 'B', 'A', 'C'],
        'C2': ['X', 'Y', 'X', 'Z'],
        'Value': [10, 20, 15, 25]
    }
    df1 = pd.DataFrame(data1)
    result1 = check_dataframe_structure(df1)
    print("Test 1 (Basic positive case):", result1)  # Should be True

    # Test 2: First two columns are explicitly categorical
    data2 = {
        'C1': ['A', 'B', 'C', 'D'],
        'C2': ['W', 'X', 'Y', 'Z'],
        'Value1': [1, 2, 3, 4],
        'Value2': [5, 6, 7, 8]
    }
    df2 = pd.DataFrame(data2)
    df2['C1'] = df2['C1'].astype('category')
    df2['C2'] = df2['C2'].astype('category')
    result2 = check_dataframe_structure(df2)
    print("Test 2 (Explicitly categorical):", result2)  # Should be True

    # Test 3: First two columns object, multiple numeric columns
    data3 = {
        'C1': ['A', 'B', 'C', 'D'],
        'C2': ['W', 'X', 'Y', 'Z'],
        'Value1': [1.1, 2.2, 3.3, 4.4],
        'Value2': [5, 6, 7, 8],
        'Value3': [100, 200, 300, 400]
    }
    df3 = pd.DataFrame(data3)
    result3 = check_dataframe_structure(df3)
    print("Test 3 (Multiple numeric columns):", result3)  # Should be True

    # Test 4: Exactly two columns, both categorical
    data4 = {
        'C1': ['A', 'B', 'C'],
        'C2': ['X', 'Y', 'Z']
    }
    df4 = pd.DataFrame(data4)
    result4 = check_dataframe_structure(df4)
    print("Test 4 (Only two categorical columns):", result4)  # Should be True

    # Test 5: Mixed types (strings, None, numeric) in categorical columns
    data5 = {
        'C1': ['A', None, 'C', 1],
        'C2': ['X', 'Y', None, 2],
        'Value': [10, 20, 30, 40]
    }
    df5 = pd.DataFrame(data5)
    result5 = check_dataframe_structure(df5)
    print("Test 5 (Mixed types in categorical):", result5)  # Should be True

    # Negative Test Cases

    # Test 6: Only one column
    data6 = {
        'C1': ['A', 'B', 'C', 'D']
    }
    df6 = pd.DataFrame(data6)
    result6 = check_dataframe_structure(df6)
    print("Test 6 (Only one column):", result6)  # Should be False

    # Test 7: First column is numeric
    data7 = {
        'C1': [1, 2, 3, 4],
        'C2': ['W', 'X', 'Y', 'Z'],
        'Value': [10, 20, 30, 40]
    }
    df7 = pd.DataFrame(data7)
    result7 = check_dataframe_structure(df7)
    print("Test 7 (First column numeric):", result7)  # Should be False

    # Test 8: Second column is numeric
    data8 = {
        'C1': ['A', 'B', 'C', 'D'],
        'C2': [1, 2, 3, 4],
        'Value': [10, 20, 30, 40]
    }
    df8 = pd.DataFrame(data8)
    result8 = check_dataframe_structure(df8)
    print("Test 8 (Second column numeric):", result8)  # Should be False

    # Test 9: Both first two columns are numeric
    data9 = {
        'C1': [1, 2, 3, 4],
        'C2': [5, 6, 7, 8],
        'Value': [10, 20, 30, 40]
    }
    df9 = pd.DataFrame(data9)
    result9 = check_dataframe_structure(df9)
    print("Test 9 (Both columns numeric):", result9)  # Should be False

    # Test 10: Third column is non-numeric
    data10 = {
        'C1': ['A', 'B', 'C', 'D'],
        'C2': ['W', 'X', 'Y', 'Z'],
        'Value': ['10', '20', '30', '40']
    }
    df10 = pd.DataFrame(data10)
    result10 = check_dataframe_structure(df10)
    print("Test 10 (Third column non-numeric):", result10)  # Should be False

    # Test 11: Empty DataFrame
    df11 = pd.DataFrame()
    result11 = check_dataframe_structure(df11)
    print("Test 11 (Empty DataFrame):", result11)  # Should be False

    # Test 12: DataFrame with datetime columns
    data12 = {
        'C1': pd.date_range('2023-01-01', periods=4),
        'C2': ['W', 'X', 'Y', 'Z'],
        'Value': [10, 20, 30, 40]
    }
    df12 = pd.DataFrame(data12)
    result12 = check_dataframe_structure(df12)
    print("Test 12 (First column datetime):", result12)  # Should be False

    # Test 13: Multiple columns with mixed numeric and non-numeric after first two
    data13 = {
        'C1': ['A', 'B', 'C', 'D'],
        'C2': ['W', 'X', 'Y', 'Z'],
        'Value1': [1, 2, 3, 4],
        'Value2': ['a', 'b', 'c', 'd'],
        'Value3': [10, 20, 30, 40]
    }
    df13 = pd.DataFrame(data13)
    result13 = check_dataframe_structure(df13)
    print("Test 13 (Mixed types after first two):", result13)  # Should be False

    # Test 14: Boolean columns
    data14 = {
        'C1': [True, False, True, False],
        'C2': ['W', 'X', 'Y', 'Z'],
        'Value': [10, 20, 30, 40]
    }
    df14 = pd.DataFrame(data14)
    result14 = check_dataframe_structure(df14)
    print("Test 14 (Boolean first column):", result14)  # Should be False

    # Test 15: Float columns that look like objects
    data15 = {
        'C1': ['A', 'B', 'C', 'D'],
        'C2': ['W', 'X', 'Y', 'Z'],
        'Value': [10.1, 20.2, np.nan, 40.4]
    }
    df15 = pd.DataFrame(data15)
    result15 = check_dataframe_structure(df15)
    print("Test 15 (Float with NaN):", result15)  # Should be True

    return {
        "test1": result1,
        "test2": result2,
        "test3": result3,
        "test4": result4,
        "test5": result5,
        "test6": result6,
        "test7": result7,
        "test8": result8,
        "test9": result9,
        "test10": result10,
        "test11": result11,
        "test12": result12,
        "test13": result13,
        "test14": result14,
        "test15": result15
    }


def generate_test_cases(folder):
    os.makedirs(folder, exist_ok=True)
    """Generate test cases for network data detection heuristics."""
    test_cases = []
    
    # POSITIVE TEST CASES - Edge List Format
    
    # Test Case 1: Basic edge list with source/target columns
    edge_list_basic = pd.DataFrame({
        'source': ['A', 'B', 'C', 'D'],
        'target': ['B', 'C', 'D', 'A']
    })
    test_cases.append((f"{folder}/edge_list.csv", edge_list_basic, True, "Basic edge list with source/target columns"))
    edge_list_basic.to_csv(test_cases[-1][0])

    # Test Case 2: Edge list with weight column
    edge_list_weighted = pd.DataFrame({
        'from': ['A', 'B', 'C', 'D'],
        'to': ['B', 'C', 'D', 'A'],
        'weight': [0.5, 0.6, 0.7, 0.8]
    })
    test_cases.append((f"{folder}/network_weighted.csv", edge_list_weighted, True, "Edge list with from/to/weight columns"))
    edge_list_weighted.to_csv(test_cases[-1][0])
    
    # Test Case 3: Edge list with node1/node2 naming
    edge_list_nodes = pd.DataFrame({
        'node1': ['1', '2', '3', '4'],
        'node2': ['2', '3', '4', '1'],
        'strength': [5, 6, 7, 8]
    })
    test_cases.append((f"{folder}/graph_edges.csv", edge_list_nodes, True, "Edge list with node1/node2 naming"))
    edge_list_nodes.to_csv(test_cases[-1][0])

    # Test Case 4: Edge list with numeric IDs
    edge_list_numeric = pd.DataFrame({
        'source_id': [1, 2, 3, 4],
        'target_id': [2, 3, 4, 1]
    })
    test_cases.append((f"{folder}/connections.csv", edge_list_numeric, True, "Edge list with numeric IDs"))
    edge_list_numeric.to_csv(test_cases[-1][0])

    # POSITIVE TEST CASES - Adjacency Matrix Format
    
    # Test Case 5: Basic adjacency matrix
    adj_matrix_basic = pd.DataFrame({
        'node': ['A', 'B', 'C', 'D'],
        'A': [0, 1, 0, 1],
        'B': [1, 0, 1, 0],
        'C': [0, 1, 0, 1],
        'D': [1, 0, 1, 0]
    })
    test_cases.append((f"{folder}/adjacency_matrix.csv", adj_matrix_basic, True, "Basic adjacency matrix"))
    adj_matrix_basic.to_csv(test_cases[-1][0])
    
    # Test Case 6: Adjacency matrix with index column
    adj_matrix_index = pd.DataFrame({
        'index': ['1', '2', '3', '4'],
        '1': [0, 1, 0, 1],
        '2': [1, 0, 1, 0],
        '3': [0, 1, 0, 1],
        '4': [1, 0, 1, 0]
    })
    test_cases.append((f"{folder}/social_network.csv", adj_matrix_index, True, "Adjacency matrix with index column"))
    adj_matrix_index.to_csv(test_cases[-1][0])

    # Test Case 7: Weighted adjacency matrix (float values)
    adj_matrix_weighted = pd.DataFrame({
        'ID': ['A', 'B', 'C', 'D'],
        'A': [0.0, 0.5, 0.0, 0.7],
        'B': [0.5, 0.0, 0.6, 0.0],
        'C': [0.0, 0.6, 0.0, 0.8],
        'D': [0.7, 0.0, 0.8, 0.0]
    })
    test_cases.append((f"{folder}/weighted_links.csv", adj_matrix_weighted, True, "Weighted adjacency matrix"))
    adj_matrix_weighted.to_csv(test_cases[-1][0])
    
    # NEGATIVE TEST CASES - Not network data
    
    # Test Case 8: Standard tabular data
    tabular_data = pd.DataFrame({
        'Name': ['John', 'Jane', 'Bob', 'Alice'],
        'Age': [25, 30, 35, 40],
        'Salary': [50000, 60000, 70000, 80000]
    })
    test_cases.append((f"{folder}/employee_data.csv", tabular_data, False, "Standard tabular data"))
    tabular_data.to_csv(test_cases[-1][0])

    # Test Case 9: Time series data
    time_series = pd.DataFrame({
        'Date': pd.date_range(start='1/1/2023', periods=4),
        'Value': [100, 105, 98, 110]
    })
    test_cases.append((f"{folder}/stock_prices.csv", time_series, False, "Time series data"))
    time_series.to_csv(test_cases[-1][0])

    # Test Case 10: Two columns but not network data
    two_cols_not_network = pd.DataFrame({
        'Product': ['A', 'B', 'C', 'D'],
        'Price': [10, 20, 30, 40]
    })
    test_cases.append((f"{folder}/products.csv", two_cols_not_network, False, "Two columns but not network data"))
    two_cols_not_network.to_csv(test_cases[-1][0])

    # Test Case 11: Three columns but not network data
    three_cols_not_network = pd.DataFrame({
        'Country': ['USA', 'Canada', 'UK', 'Australia'],
        'Population': [331, 38, 68, 26],
        'GDP': [21.4, 1.7, 2.7, 1.4]
    })
    test_cases.append((f"{folder}/country_stats.csv", three_cols_not_network, False, "Three columns but not network data"))
    three_cols_not_network.to_csv(test_cases[-1][0])
    
    # AMBIGUOUS TEST CASES
    
    # Test Case 12: Network filename but regular data
    network_name_regular_data = pd.DataFrame({
        'Student': ['A', 'B', 'C', 'D'],
        'Math': [85, 90, 78, 92],
        'Science': [92, 88, 95, 79]
    })
    test_cases.append((f"{folder}/relations.csv", network_name_regular_data, False, "Network filename but regular data"))
    network_name_regular_data.to_csv(test_cases[-1][0])
    
    # Test Case 13: Regular filename but network-like structure
    regular_name_network_data = pd.DataFrame({
        'Person': ['A', 'B', 'C', 'D'],
        'Friend': ['B', 'C', 'D', 'A']
    })
    test_cases.append((f"{folder}/friends.csv", regular_name_network_data, True, "Regular filename but network-like structure"))
    regular_name_network_data.to_csv(test_cases[-1][0])

    # Test Case 14: Edge list with unusual column names
    edge_list_unusual = pd.DataFrame({
        'entity1': ['A', 'B', 'C', 'D'],
        'entity2': ['B', 'C', 'D', 'A'],
        'correlation': [0.5, 0.6, 0.7, 0.8]
    })
    test_cases.append((f"{folder}/correlations.csv", edge_list_unusual, True, "Edge list with unusual column names"))
    edge_list_unusual.to_csv(test_cases[-1][0])

    # Test Case 15: More complex tabular data with multiple columns
    complex_tabular = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Name': ['John', 'Jane', 'Bob', 'Alice'],
        'Age': [25, 30, 35, 40],
        'Department': ['HR', 'IT', 'Sales', 'Marketing'],
        'Salary': [50000, 60000, 70000, 80000],
        'Years': [3, 5, 7, 2]
    })
    test_cases.append((f"{folder}/employee_details.csv", complex_tabular, False, "Complex tabular data"))
    complex_tabular.to_csv(test_cases[-1][0])

    return test_cases


# Example function to save test cases to files for manual review
def save_test_cases_to_files(directory="network_detection_tests"):
    """Save test cases to CSV files for manual review."""
    test_cases = generate_test_cases()
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for filename, df, is_network, description in test_cases:
        file_path = os.path.join(directory, filename)
        df.to_csv(file_path, index=False)
        
        # Create a metadata file
        meta_file = os.path.join(directory, f"{os.path.splitext(filename)[0]}_meta.txt")
        with open(meta_file, 'w') as f:
            f.write(f"Description: {description}\n")
            f.write(f"Expected to be network data: {is_network}\n")
            f.write(f"Number of columns: {len(df.columns)}\n")
            f.write(f"Column names: {', '.join(df.columns)}\n")
    
    return directory
    

def main():
    # test_cases()
    folder = "network_test_csv"
    csvs = generate_test_cases(folder)
    for d in csvs:
        path = d[0]

        ans = check(csv_path=path)
        print(f'{path} -> {ans}')
        print(f'---------------------------------------------------------------')

if __name__ == '__main__':
    main()
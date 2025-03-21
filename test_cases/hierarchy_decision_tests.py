import pandas as pd
import os
from read import detect_index_and_read_csv
from hierarchy_decision import find_hierarchical_relationships, is_forest_and_count_roots

def create_and_save_dataframe(temp_dir, data, filename):
    """Creates a Pandas DataFrame, saves it as CSV, and stores it."""
    df = pd.DataFrame(data)
    filepath = os.path.join(temp_dir, filename + ".csv")
    df.to_csv(filepath, index=False)
    return df, filepath

def test_find_hierarchical_relationships_positive(temp_dir):
    print("Running find_hierarchical_relationships_positive tests...")
    # Test case 1: Simple hierarchical relationship
    data1 = {'country': ['USA', 'USA', 'Canada', 'Canada', 'Mexico'],
             'state': ['California', 'Texas', 'Ontario', 'Quebec', 'Sonora'],
             'city': ['Los Angeles', 'Austin', 'Toronto', 'Montreal', 'Hermosillo']}
    df1, _ = create_and_save_dataframe(temp_dir, data1, 'hierarchical1')

    expected_relationships1 = [('country', 'city'), ('country', 'state'), ('state', 'city')]
    actual_relationships1 = find_hierarchical_relationships(df1)
    assert set(actual_relationships1) == set(expected_relationships1), f"Test failed: expected {expected_relationships1}, got {actual_relationships1}"
    print("hierarchical1 test passed")

    # Test case 2: Another hierarchical relationship with different column names
    data2 = {'continent': ['Africa', 'Africa', 'Europe', 'Europe'],
             'country': ['Nigeria', 'Egypt', 'France', 'Germany']}
    df2, _ = create_and_save_dataframe(temp_dir, data2, 'hierarchical2')

    expected_relationships2 = [('continent', 'country')]
    actual_relationships2 = find_hierarchical_relationships(df2)
    assert set(actual_relationships2) == set(expected_relationships2), f"Test failed: expected {expected_relationships2}, got {actual_relationships2}"
    print("hierarchical2 test passed")

    # Test case 3: No min_multichild_ratio requirements
    data3 = {'A': ['x','x','y','y','z'],
             'B': ['1','2','3','4','5']}
    df3, _ = create_and_save_dataframe(temp_dir, data3, 'hierarchical3')

    expected_relationships3 = [('A', 'B')]
    actual_relationships3 = find_hierarchical_relationships(df3)
    assert set(actual_relationships3) == set(expected_relationships3), f"Test failed: expected {expected_relationships3}, got {actual_relationships3}"
    print("hierarchical3 test passed")

def test_find_hierarchical_relationships_negative(temp_dir):
    print("Running find_hierarchical_relationships_negative tests...")
    # Test case 1: No hierarchical relationship
    data1 = {'col1': ['A', 'B', 'C', 'D'],
             'col2': ['E', 'F', 'G', 'H']}
    df1, _ = create_and_save_dataframe(temp_dir, data1, 'non_hierarchical1')

    expected_relationships1 = []
    actual_relationships1 = find_hierarchical_relationships(df1)
    assert actual_relationships1 == expected_relationships1, f"Test failed: expected {expected_relationships1}, got {actual_relationships1}"
    print("non_hierarchical1 test passed")

    # Test case 2: High cardinality column
    data2 = {'user_id': ['user1', 'user2', 'user3', 'user4'],
             'product': ['A', 'B', 'C', 'D']}
    df2, _ = create_and_save_dataframe(temp_dir, data2, 'non_hierarchical2')

    expected_relationships2 = []
    actual_relationships2 = find_hierarchical_relationships(df2)
    assert actual_relationships2 == expected_relationships2, f"Test failed: expected {expected_relationships2}, got {actual_relationships2}"
    print("non_hierarchical2 test passed")

    # Test case 3: min_multichild_ratio
    data3 = {'A': ['x','x','y','y','z','z'],
             'B': ['1','1','3','3','5','5']}
    df3, _ = create_and_save_dataframe(temp_dir, data3, 'non_hierarchical3')

    expected_relationships3 = []
    actual_relationships3 = find_hierarchical_relationships(df3)
    assert actual_relationships3 == expected_relationships3, f"Test failed: expected {expected_relationships3}, got {actual_relationships3}"
    print("non_hierarchical3 test passed")

def test_is_forest_and_count_roots_positive():
    print("Running is_forest_and_count_roots_positive tests...")
    # Test case 1: Simple forest with one tree
    edges1 = [('A', 'B'), ('B', 'C'), ('A', 'D')]
    is_forest1, roots1 = is_forest_and_count_roots(edges1)
    assert is_forest1, "Test failed: is_forest1 should be True"
    assert roots1 == 1, f"Test failed: expected 1 root, got {roots1}"
    print("forest_positive1 test passed")

    # Test case 2: Forest with multiple trees
    edges2 = [('A', 'B'), ('C', 'D')]
    is_forest2, roots2 = is_forest_and_count_roots(edges2)
    assert is_forest2, "Test failed: is_forest2 should be True"
    assert roots2 == 2, f"Test failed: expected 2 roots, got {roots2}"
    print("forest_positive2 test passed")

    # Test case 3: Single node (a tree with one node)
    edges3 = []
    is_forest3, roots3 = is_forest_and_count_roots(edges3)
    assert is_forest3, "Test failed: is_forest3 should be True"
    assert roots3 == 0, f"Test failed: expected 0 root, got {roots3}"
    print("forest_positive3 test passed")

def test_is_forest_and_count_roots_negative():
    print("Running is_forest_and_count_roots_negative tests...")
    # Test case 1: Cycle
    edges1 = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    is_forest1, roots1 = is_forest_and_count_roots(edges1)
    assert not is_forest1, "Test failed: is_forest1 should be False"
    assert roots1 == 0, f"Test failed: expected 0 roots, got {roots1}"
    print("forest_negative1 test passed")

    # Test case 2: Multiple parents for one node
    edges2 = [('A', 'B'), ('C', 'B')]
    is_forest2, roots2 = is_forest_and_count_roots(edges2)
    assert not is_forest2, "Test failed: is_forest2 should be False"
    assert roots2 == 0, f"Test failed: expected 0 roots, got {roots2}"
    print("forest_negative2 test passed")

    # Test case 3: Self-loop
    edges3 = [('A', 'A')]
    is_forest3, roots3 = is_forest_and_count_roots(edges3)
    assert not is_forest3, "Test failed: is_forest3 should be False"
    assert roots3 == 0, f"Test failed: expected 0 roots, got {roots3}"
    print("forest_negative3 test passed")


def main():
    temp_dir = 'hierarchy_decision_tests_csv'
    os.makedirs(temp_dir, exist_ok=True)
    try:
        test_find_hierarchical_relationships_positive(temp_dir)
        test_find_hierarchical_relationships_negative(temp_dir)
        test_is_forest_and_count_roots_positive()
        test_is_forest_and_count_roots_negative()
    finally: # clean up even if a test fails
        # for filename in os.listdir(temp_dir):
        #     filepath = os.path.join(temp_dir, filename)
        #     os.remove(filepath)
        # os.rmdir(temp_dir) # only remove the dir if it is empty
        print("All tests completed.")


if __name__ == "__main__":
    main()
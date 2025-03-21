import pandas as pd
import itertools
import networkx as nx

def get_relation_pairs(df, unique_threshold=0.95, min_multichild_ratio=0.5):
    relationships = []
    n_rows = len(df)
    
    # Iterate over each unique pair of columns
    for col1, col2 in itertools.combinations(df.columns, 2):
        if col1 == col2:
            continue
        # if col1 == 'state' and col2 == 'city':
        #     import pdb; pdb.set_trace()
        # Skip high-cardinality columns (almost unique)
        # if (df[col1].nunique() / n_rows) > unique_threshold or (df[col2].nunique() / n_rows) > unique_threshold:
        #     continue
        
        # Candidate: col1 as parent, col2 as child
        if df[col1].nunique() < df[col2].nunique() or (min_multichild_ratio == 1.0 and df[col1].nunique() == df[col2].nunique()):
            # Verify that each child (col2) has a unique parent (col1)
            if df.groupby(col2)[col1].nunique().eq(1).all():
                # Count unique children per parent
                group_counts = df.groupby(col1)[col2].nunique()
                # If fewer than min_multichild_ratio of the parent groups are one-to-one, add as hierarchical
                if (group_counts == 1).mean() <= min_multichild_ratio:
                    relationships.append((col1, col2))
        # Candidate: col2 as parent, col1 as child
        elif df[col2].nunique() < df[col1].nunique():
            if df.groupby(col1)[col2].nunique().eq(1).all():
                group_counts = df.groupby(col2)[col1].nunique()
                if (group_counts == 1).mean() <= min_multichild_ratio:
                    relationships.append((col2, col1))
    return relationships


def find_hierarchical_relationships(df):
    rels = get_relation_pairs(df)
    if (len(rels) > 0):
        rels2 = get_relation_pairs(df, unique_threshold=0.99, min_multichild_ratio=1.0)
        rels =  list(set(rels + rels2))
    return rels

def is_forest_and_count_roots(edges):
    G = nx.DiGraph(edges)
    # Check that each node has at most one parent
    if not all(d <= 1 for d in dict(G.in_degree()).values()):
        return False, 0
    # Ensure the graph is acyclic
    if not nx.is_directed_acyclic_graph(G):
        return False, 0
    # Count roots (nodes with in-degree 0)
    roots = sum(1 for _, d in G.in_degree() if d == 0)
    return True, roots

def check_single_rooted_hierarchy(df):
    rels = get_relation_pairs(df)
    is_forest, num_roots = is_forest_and_count_roots(rels)
    if (is_forest and num_roots == 1):
        return True
    return False


def check_multiple_rooted_hierarchy(df):
    rels = get_relation_pairs(df)
    is_forest, num_roots = is_forest_and_count_roots(rels)
    if (is_forest and num_roots >= 1):
        return True
    return False

# Example usage:
if __name__ == "__main__":
    # Example DataFrame representing a potential hierarchical structure.
    # data = {'country': ['USA', 'USA', 'Canada', 'Canada', 'Mexico'],
    #          'state': ['California', 'Texas', 'Ontario', 'Quebec', 'Sonora'],
    #          'city': ['Los Angeles', 'Austin', 'Toronto', 'Montreal', 'Hermosillo']}
    # df = pd.DataFrame(data)

    df = pd.DataFrame({
        'region': ['North', 'North', 'South', 'South'],
        'city': ['NY', 'Boston', 'Miami', 'Austin'],
        'value': [100, 200, 300, 400]
    })
    
    rels = find_hierarchical_relationships(df)
    print("Hierarchical relationships (parent -> child):")
    for parent, child in rels:
        print(f"{parent} -> {child}")
    print(is_forest_and_count_roots(rels))

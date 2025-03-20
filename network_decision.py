import os
import pandas as pd
import numpy as np
from run_llm import run_llama3
from read import detect_index_and_read_csv

context_prompt = """
Heuristics for Detecting Network Data (Edge List or Adjacency Matrix)
1. CSV File Name Heuristics
If the file name contains keywords like "network", "edges", "graph", "links", "connections", "adjacency", "nodes", or "relations", it might indicate network data.
Example names:
social_network.csv
graph_edges.csv
adjacency_matrix.csv

Answer: Yes

2. Column Name Heuristics
For network data, we typically see:

Edge List Format (Common in Graphs)

A CSV with exactly 2 or 3 columns, often named something like:
source, target
node1, node2
from, to

Example:
Filename: filename
Number of columns: 2
Column names and types: name and type of column data 

source,target
A,B
B,C
C,A
Edge list Format
Answer: Yes

Sometimes an extra weight column (weight, strength, cost, etc.).
Example:
Filename: 
Number of columns: 3
Column names and types: name and type of column data

source,target,weight
A,B,0.5
B,C,0.8
C,A,0.6
Weighted edge list Format
Answer: Yes

The first column is often called "node", "index", or "ID", followed by several numeric columns representing connections.
Example:
Filename: 
Number of columns: Many columns (one for each nodes)
Column names and types: name and type of column data

node,A,B,C,D
A,0,1,0,1
B,1,0,1,0
C,0,1,0,1
D,1,0,1,0
Column names match row labels, indicating an adjacency matrix.
Answer: Yes

If the given information does not justifies the existence of network data,
Answer: No
Given the following CSV metadata:
"""

def check_dataframe_structure(df: pd.DataFrame) -> bool:
    if df.shape[1] == 2:
        return all(df.dtypes[:2].astype(str).isin(['category', 'object']))
    elif df.shape[1] == 3:
        return sum(df.dtypes.astype(str).isin(['category', 'object'])) == 2 and sum(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns) == 1
    return False


def create_csv_meta_data(csv_path):
    base_name = os.path.basename(csv_path)
    df = detect_index_and_read_csv(csv_path)
    columns = df.columns.values.tolist()
    column_types = [str(df[col].dtype) for col in columns]
    column_info = [f"{col} ({typ})" for col, typ in zip(columns, column_types)]
    sample_data = df.head(3).to_string(index=False)
    metadata = f"""
Filename: {base_name}
Number of columns: {len(columns)}
Column names and types: {', '.join(column_info)}
First few rows:
{sample_data}
"""
    return metadata, df


# def create_test_llm_queries():
#     """Create LLM queries for each test case."""
#     test_cases = generate_test_cases()
#     llm_queries = []
    
#     for filename, df, is_network, description in test_cases:
#         # Get column information
#         columns = df.columns.tolist()
#         column_types = [str(df[col].dtype) for col in columns]
#         column_info = [f"{col} ({typ})" for col, typ in zip(columns, column_types)]
        
#         # Get sample data (first 3 rows as string representation)
#         sample_data = df.head(3).to_string(index=False)
        
#         # Create metadata string
#         metadata = f"""
# Filename: {filename}
# Number of columns: {len(columns)}
# Column names and types: {', '.join(column_info)}
# First few rows:
# {sample_data}
# """
        
#         # Create full prompt
#         query = f"""
# Context:
# {context_prompt}

# {metadata}

# Question: Is this CSV likely to contain network data (edge list or adjacency matrix)? Explain your reasoning.
# """
        
#         llm_queries.append({
#             "query": query,
#             "filename": filename,
#             "expected_is_network": is_network,
#             "description": description
#         })
    
#     return llm_queries


def parse_llm_answer(ans):
    last_line = ans.split('\n')
    answer = last_line[-1].lower()
    res = None 
    if 'yes' in answer: 
        res = True
    if 'no' in answer:
        res = False
    if res is None:
        res = parse_llm_bool(ans)
    assert res is not None, f'No decision parsed: {answer}'
    return res

def parse_llm_bool(ans):
    querry = f'No explanation necessary only 1 or 0. Check the following paragraph, and answer only 1 if the paragraph confirms existence of network data, and answer only 0 other wise: \n {ans}. No explanation necessary only 1 or 0.\n Your answer ?'
    print(f'------- sub querry : Sentiment -----------')
    out = run_llama3(querry)
    print(out)
    print("------------------------------------------- ")
    return '1' in out.lower()
    


def check(csv_path):
    meta, df = create_csv_meta_data(csv_path=csv_path)
    if check_dataframe_structure(df):
        # import pdb; pdb.set_trace()
        cmd = context_prompt + meta + "\n Give reason for your decision. In the end, final answer in the last line in the format. \n Answer: Yes (or No)\n"
        out = run_llama3(cmd)
        print(f'----------------------- {os.path.basename(csv_path)} ---------------------')
        print(out)
        return parse_llm_answer(out)
    else:
        return False
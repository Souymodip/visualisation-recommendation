import pandas as pd
from run_llm import run_llama3
from read import detect_index_and_read_csv


def generate_venn_prompt(csv_filename, col1_name, col2_name):
    prompt = f"""
Act as a data visualization expert. I have a CSV file named "{csv_filename}" containing two columns: "{col1_name}" and "{col2_name}". I want to decide if these two columns represent similar sets with potential overlaps (where a Venn diagram would be useful) or if they are distinct attributes (where a Venn diagram might not be appropriate).

Consider the following:
1. The columns have possibly similar naming conventions (e.g., both have "category" in their name) or other indicators that they might represent variations of the same conceptual set.
2. These columns may contain overlapping values (shared category labels).
3. If the columns represent completely different attributes (e.g., "Color" vs. "Shape"), a Venn diagram wouldn't be meaningful.

Examples:

- **Example 1**: 
  - Columns: "Category_v1" and "Category_v2" 
  - Both columns list items such as "Fruit", "Vegetable", "Dairy". 
  - This suggests they overlap in many values, so a Venn diagram could show how much each column includes "Fruit", "Vegetable", and so on.

- **Example 2**:
  - Columns: "Color" and "Shape"
  - One column has entries like "Red", "Blue", "Green", and the other has "Circle", "Triangle", "Square".
  - These describe different properties (no overlap), so a Venn diagram wouldn't make sense.

- **Example 3**:
  - Columns: "TagA" and "TagB"
  - The CSV file is named "user_tags.csv", implying both columns might store user-generated tags.
  - Often these tags can overlap if users used similar keywords. A Venn diagram might reveal common tags versus unique tags between the two columns.

- **Example 4**:
  - Columns: "Language" and "Country"
  - The CSV file is named "languages_countries.csv". The columns might represent a person's native language and home country. 
  - Unless these columns are coded in a way that they share the same values (e.g., short codes like "EN" for English and "EN" for England—which is unusual), there's likely no overlap.

**Task**:
1. Examine the CSV filename "{csv_filename}" and the column names "{col1_name}" and "{col2_name}" for clues about whether these columns may overlap in values.
2. Provide a **verbose reasoning** paragraph discussing why or why not a Venn diagram is appropriate.
3. Conclude with a **single line** that says either "Yes" (a Venn diagram is likely suitable) or "No" (a Venn diagram is not suitable).

Answer format:
<reasoning>
[Your verbose reasoning here]
</reasoning>
<answer>
[Yes or No]
</answer>
"""
    return prompt


def parse_answer(answer):
    last_lines = answer.lower().split('\n')[-3:]
    last_line = ''.join(last_lines)
    if 'yes' in last_line:
        return True
    elif 'no' in last_line:
        return False
    else:
        raise ValueError(f"Invalid answer: {last_line}")
    

def check_venn(csv_file):
    df = detect_index_and_read_csv(csv_file)
    col1_name = df.columns[0]
    col2_name = df.columns[1]
    prompt = generate_venn_prompt(csv_file, col1_name, col2_name)
    answer = run_llama3(prompt)
    return parse_answer(answer)


import pandas as pd
from read import detect_index_and_read_csv
from venn_check import check_venn
from hierarchy_decision import find_hierarchical_relationships
from chart_types import ChartType

def flow_chart(df:pd.DataFrame, filename:str):
    num_columns = len(df.columns)
    if num_columns == 1:
        print(f'Recommendation is to convert to frequency table')
        return 0
    elif num_columns == 2  and check_venn(filename, df.columns[0], df.columns[1]):
        return 1
    else:
        rels = find_hierarchical_relationships(df)
        if len(rels) > 0:
            return 2
        else:
            print(f'Recommendation is to convert to frequency table')
            return 3
        
def case_to_recommendation(case:int):
    if case == 0 or case == 3:
        return [
            ChartType.BAR_CHART, ChartType.LOLLIPOP_CHART, ChartType.WORD_CLOUD,
            ChartType.DONUT_CHART, ChartType.PIE_CHART, ChartType.TREE_MAP,
            ChartType.CIRCLE_PACKING, ChartType.SUNBURST_CHART
        ]
    elif case == 1:
        return [ChartType.VENN_DIAGRAM]
    elif case == 2:
        return [
            ChartType.TREE_MAP, ChartType.CIRCLE_PACKING, ChartType.SUNBURST_CHART,
            ChartType.DENDROGRAM
        ]
    else:
        return []

 
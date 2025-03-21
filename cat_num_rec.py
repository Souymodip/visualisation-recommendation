import pandas as pd
import os
from mixed_decisions import has_one_cat_one_num, has_one_cat_and_several_num, is_categorical_unique
from mixed_decisions import common_ordered_numeric_column, has_several_cat_and_one_num, check_categorical_hierarchy, check_categorical_uniqueness
from chart_types import ChartType

def flow_chart(df, base_name):
    # if df has one cat and one num, return base_name
    if has_one_cat_one_num(df):
        # if cat is unique
        if is_categorical_unique(df):
            print('One cat and one num, cat is unique')
            return 0
        # if cat is not unique
        else:
            print('One cat and one num, cat is not unique')
            return 1
    elif has_one_cat_and_several_num(df):
        # Cat col is unique
        if is_categorical_unique(df):
            print('One cat and several num, cat row is unique')
            return 2
        # Cat col is not unique
        elif common_ordered_numeric_column(df):
            print('One cat and several num, cat is not unique and num is ordered')
            return 3
        # Cat col is not unique and not ordered
        else:
            print('One cat and several num, cat is not unique and num is not ordered')
            return 4
    elif has_several_cat_and_one_num(df):
        # has hierarchy
        if check_categorical_hierarchy(df):
            # each row for the cat col is unique
            if check_categorical_uniqueness(df):
                print('Has hierarchy, Several cat and one num, cat row is unique')
                return 5
            else:
                print('Has hierarchy, Several cat and one num, cat is not unique')
                return 6
        else:
            print('Has no hierarchy, Several cat and one num')
            return 7
    else:
        print('No cat and one num')
        return 8
    

def case_to_recommendation(case:int):
    if case == 0:
        return [
            ChartType.BAR_CHART, ChartType.LOLLIPOP_CHART, ChartType.BOX_PLOT, ChartType.DONUT_CHART,
            ChartType.PIE_CHART, ChartType.TREE_MAP, ChartType.CIRCLE_PACKING, ChartType.WORD_CLOUD
        ]
    elif case == 1:
        return [
            ChartType.BOX_PLOT, ChartType.VIOLIN_PLOT, ChartType.RIDGE_LINE, ChartType.DENSITY_PLOT_1D,
            ChartType.HISTOGRAM
        ]
    elif case == 2:
        return [
            ChartType.GROUPED_SCATTER_PLOT, ChartType.GROUPED_BAR_CHART, ChartType.STACKED_BAR_CHART,
            ChartType.PARALLEL_PLOT, ChartType.SPIDER_CHART, ChartType.SANKEY_CHART
        ]
    elif case == 3:
        return [
            ChartType.STACKED_AREA_PLOT, ChartType.AREA_PLOT, ChartType.STREAM_GRAPH,
            ChartType.LINE_PLOT, ChartType.CONNECTED_SCATTER_PLOT
        ]
    elif case == 4:
        return [
            ChartType.GROUPED_SCATTER_PLOT, ChartType.DENSITY_PLOT_2D, ChartType.BOX_PLOT,
            ChartType.VIOLIN_PLOT, ChartType.PCA, ChartType.CORRELOGRAM
        ]
    elif case == 5:
        return [
            ChartType.BAR_CHART, ChartType.DENDROGRAM, ChartType.SUNBURST_CHART, ChartType.TREE_MAP,
            ChartType.CIRCLE_PACKING,
        ]
    elif case == 6:
        return [
            ChartType.BOX_PLOT, ChartType.VIOLIN_PLOT
        ]
    else:
        return []


def test():
    df = pd.DataFrame({
        'region': ['North', 'North', 'South', 'South'],
        'city': ['NY', 'Boston', 'Miami', 'Austin'],
        'value': [100, 200, 300, 400]
    })

    print(check_categorical_hierarchy(df))
    print(check_categorical_uniqueness(df))

if __name__ == "__main__":
    test()
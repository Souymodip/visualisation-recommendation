import pandas as pd
from ordered_decision import is_weakly_ordered
from chart_types import ChartType

def is_small(df, tau=10):
    return len(df) <= 10

def check_ordered(df):
    return is_weakly_ordered(df)


def flow_chart(df:pd.DataFrame):
    num_columns = len(df.columns)
    is_ordered = check_ordered(df)
    df_is_small = is_small(df)
    
    if num_columns == 1:
        return 0
    elif num_columns == 2:
        if not is_ordered and df_is_small:
            return 1
        elif not is_ordered and not df_is_small:
            return 2
        elif is_ordered:
            return 3
    elif num_columns == 3:
        if not is_ordered:
            return 4
        else:
            return 5
    elif num_columns > 3:
        if is_ordered:
            return 6
        else:
            return 7
    
    return "No matching case found"

def case_to_recommendation(case):
    if case == 0:
        return [
            ChartType.HISTOGRAM, ChartType.DENSITY_PLOT_1D
        ]
    elif case == 1:
        return [
            ChartType.BOX_PLOT, ChartType.HISTOGRAM, ChartType.SCATTER_PLOT_2D
        ]
    elif case == 2:
        return [
            ChartType.VIOLIN_PLOT, ChartType.DENSITY_PLOT_1D, ChartType.SCATTER_PLOT_WITH_MARGINALS, ChartType.DENSITY_PLOT_2D
        ]
    elif case == 3:
        return [
            ChartType.CONNECTED_SCATTER_PLOT, ChartType.AREA_PLOT, ChartType.LINE_PLOT
        ]
    elif case == 4:
        return [
            ChartType.BOX_PLOT, ChartType.VIOLIN_PLOT, ChartType.BUBBLE_PLOT, ChartType.SCATTER_PLOT_3D, ChartType.SURFACE_PLOT
        ]
    elif case == 5 or case == 6:
        return [
            ChartType.STACKED_AREA_PLOT, ChartType.STREAM_GRAPH, ChartType.LINE_PLOT, ChartType.AREA_PLOT
        ]
    elif case == 7:
        return [
            ChartType.BOX_PLOT, ChartType.VIOLIN_PLOT, ChartType.RIDGE_LINE, ChartType.PCA, ChartType.CORRELOGRAM, ChartType.HEATMAP, ChartType.DENDROGRAM
        ]
    else:
        return []
    
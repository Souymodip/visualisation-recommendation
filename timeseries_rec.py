import pandas as pd
from chart_types import ChartType


def flow_chart(df: pd.DataFrame) -> list:
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) == 1:
        return 0
    else:
        return 1
    

def case_to_recommendation(case: int) -> list:
    if case == 0:
        return [ChartType.LINE_PLOT, ChartType.BOX_PLOT, ChartType.VIOLIN_PLOT, ChartType.RIDGE_LINE, ChartType.AREA_PLOT]
    else:
        return [ChartType.LINE_PLOT, ChartType.BOX_PLOT, ChartType.VIOLIN_PLOT, ChartType.RIDGE_LINE, ChartType.STACKED_AREA_PLOT, ChartType.HEATMAP, ChartType.STREAM_GRAPH]


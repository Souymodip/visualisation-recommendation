from enum import Enum

class ChartType(Enum):
    HISTOGRAM=1
    DENSITY_PLOT_1D=2
    BOX_PLOT=3
    SCATTER_PLOT_2D=4
    VIOLIN_PLOT=5
    SCATTER_PLOT_WITH_MARGINALS=6
    DENSITY_PLOT_2D=7
    CONNECTED_SCATTER_PLOT=8
    AREA_PLOT=9
    LINE_PLOT=10
    BUBBLE_PLOT=11
    SCATTER_PLOT_3D=12
    SURFACE_PLOT=13
    STACKED_AREA_PLOT=14
    STREAM_GRAPH=15
    RIDGE_LINE=16
    PCA=17
    CORRELOGRAM=18
    HEATMAP=19
    DENDROGRAM=20
    VENN_DIAGRAM=21
    BAR_CHART=22
    WAFFLE_CHART=23
    WORD_CLOUD=24
    DONUT_CHART=25
    PIE_CHART=26
    TREE_MAP=27
    CIRCLE_PACKING=28
    SUNBURST_CHART=29
    LOLLIPOP_CHART=30
    GROUPED_SCATTER_PLOT=31
    GROUPED_BAR_CHART=32
    PARALLEL_PLOT=33
    SPIDER_CHART=34
    STACKED_BAR_CHART=35
    SANKEY_CHART=36
    
def get_chart_type_name(chart_type: ChartType) -> str:
    return chart_type.name.lower().replace('_', ' ')

# If you need to get all chart type names as a list, you can also add this function:
def get_all_chart_type_names(types:list[ChartType]) -> list[str]:
    return [get_chart_type_name(chart_type) for chart_type in types]
    
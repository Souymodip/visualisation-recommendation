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
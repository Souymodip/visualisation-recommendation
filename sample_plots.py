import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from matplotlib_venn import venn2
from pandas.plotting import parallel_coordinates
from chart_types import ChartType
import squarify
from wordcloud import WordCloud

sns.set_theme(style="whitegrid")

def get_sample_df():
    # Generate some sample data
    np.random.seed(42)
    n = 500
    x = np.random.normal(0, 1, n)
    y = x * 0.5 + np.random.normal(0, 1, n)
    z = x * 0.3 + y * 0.5 + np.random.normal(0, 1, n)
    categorical = np.random.choice(['A', 'B', 'C', 'D'], n)
    time_data = pd.date_range(start='2022-01-01', periods=100, freq='D')
    time_values = np.cumsum(np.random.randn(100))
    time_series = pd.DataFrame({'date': time_data, 'value': time_values})
    sizes = np.random.uniform(10, 100, n)
    groups = np.random.choice(['Group 1', 'Group 2', 'Group 3'], n)

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'category': categorical,
        'size': sizes,
        'group': groups
    })
    return {
        'df': df,
        'time_series': time_series
    }

# Function to save each plot
def save_plot(plt_obj, name):
    plt_obj.tight_layout()
    plt_obj.show()
    # plt_obj.savefig(f"{name}.png")
    plt_obj.close()

# 1. HISTOGRAM (already fixed)
def plot_histogram():
    sample = get_sample_df()
    x = sample['df']['x']
    plt.figure(figsize=(8, 6))
    sns.histplot(x, kde=True)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    save_plot(plt, "histogram")

# 2. DENSITY_PLOT_1D (already fixed)
def plot_density_1d():
    sample = get_sample_df()
    x = sample['df']['x']
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x, fill=True)
    plt.title('1D Density Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')
    save_plot(plt, "density_plot_1d")

# 3. BOX PLOT
def plot_box_plot():
    sample = get_sample_df()
    df = sample['df']
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df['category'], y=df['x'])
    plt.title('Box Plot')
    plt.xlabel('Category')
    plt.ylabel('Value')
    save_plot(plt, "box_plot")

# 4. SCATTER_PLOT_2D
def plot_scatter_2d():
    sample = get_sample_df()
    df = sample['df']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='x', y='y', hue='category')
    plt.title('2D Scatter Plot')
    plt.xlabel('X value')
    plt.ylabel('Y value')
    save_plot(plt, "scatter_plot_2d")

# 5. VIOLIN_PLOT
def plot_violin():
    sample = get_sample_df()
    df = sample['df']
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=df['category'], y=df['x'])
    plt.title('Violin Plot')
    plt.xlabel('Category')
    plt.ylabel('Value')
    save_plot(plt, "violin_plot")

# 6. SCATTER_PLOT_WITH_MARGINALS
def plot_scatter_with_marginals():
    sample = get_sample_df()
    df = sample['df']
    g = sns.JointGrid(data=df, x="x", y="y", height=8)
    g.plot_joint(sns.scatterplot, hue=df['category'])
    g.plot_marginals(sns.histplot)
    g.fig.suptitle('Scatter Plot with Marginals', y=1.02)
    save_plot(plt, "scatter_with_marginals")

# 7. DENSITY_PLOT_2D
def plot_density_2d():
    sample = get_sample_df()
    df = sample['df']
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x='x', y='y', fill=True, cmap="viridis")
    plt.title('2D Density Plot')
    plt.xlabel('X value')
    plt.ylabel('Y value')
    save_plot(plt, "density_plot_2d")

# 8. CONNECTED_SCATTER_PLOT
def plot_connected_scatter():
    sample = get_sample_df()
    time_series = sample['time_series']
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=time_series, x='date', y='value', marker='o')
    plt.title('Connected Scatter Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    save_plot(plt, "connected_scatter_plot")

# 9. AREA_PLOT
def plot_area():
    sample = get_sample_df()
    time_series = sample['time_series']
    plt.figure(figsize=(8, 6))
    plt.fill_between(time_series['date'], time_series['value'])
    plt.title('Area Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    save_plot(plt, "area_plot")

# 10. LINE_PLOT
def plot_line():
    sample = get_sample_df()
    time_series = sample['time_series']
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=time_series, x='date', y='value')
    plt.title('Line Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    save_plot(plt, "line_plot")

# 11. BUBBLE_PLOT
def plot_bubble():
    sample = get_sample_df()
    df = sample['df']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='x', y='y', size='size', hue='category', sizes=(20, 200))
    plt.title('Bubble Plot')
    plt.xlabel('X value')
    plt.ylabel('Y value')
    save_plot(plt, "bubble_plot")

# 12. SCATTER_PLOT_3D
def plot_scatter_3d():
    sample = get_sample_df()
    df = sample['df']
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'purple'}
    for category in df['category'].unique():
        category_data = df[df['category'] == category]
        ax.scatter(category_data['x'], category_data['y'], category_data['z'], 
                   c=colors[category], label=category, alpha=0.7)
    
    ax.set_xlabel('X value')
    ax.set_ylabel('Y value')
    ax.set_zlabel('Z value')
    ax.set_title('3D Scatter Plot')
    ax.legend()
    save_plot(plt, "scatter_plot_3d")

# 13. SURFACE_PLOT
def plot_surface():
    # For consistency, get sample (although not used in this plot)
    sample = get_sample_df()
    # Generate grid data for surface plot
    x_grid = np.linspace(-3, 3, 50)
    y_grid = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.sin(X) * np.cos(Y)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Surface Plot')
    save_plot(plt, "surface_plot")

# 14. STACKED_AREA_PLOT
def plot_stacked_area():
    # Although this function generates its own data, you could add:
    sample = get_sample_df()  # Sample extraction (not used here)
    # Generate data for stacked area
    dates = pd.date_range(start='2022-01-01', periods=50, freq='D')
    data = np.random.rand(50, 4) * 100
    data = np.cumsum(data, axis=1)
    stacked_data = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'], index=dates)
    
    plt.figure(figsize=(10, 6))
    plt.stackplot(dates, stacked_data['A'], stacked_data['B'], 
                  stacked_data['C'], stacked_data['D'],
                  labels=['A', 'B', 'C', 'D'])
    plt.title('Stacked Area Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    save_plot(plt, "stacked_area_plot")

# 15. STREAM_GRAPH
def plot_stream_graph():
    # Although this function generates its own data, you could add:
    sample = get_sample_df()  # Sample extraction (not used here)
    # Using the same data as stacked area plot but centering around zero
    dates = pd.date_range(start='2022-01-01', periods=50, freq='D')
    data = np.random.randn(50, 4) * 10
    data = data.cumsum(axis=0) + 20  # Making sure values are mostly positive
    stream_data = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'], index=dates)
    
    # Using plotly for better stream graph
    fig = go.Figure()
    for column in stream_data.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=stream_data[column],
            mode='lines',
            stackgroup='one',
            name=column
        ))
    
    fig.update_layout(
        title='Stream Graph',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white'
    )
    fig.write_image("stream_graph.png")

# 16. RIDGE_LINE
def plot_ridge_line():
    # Create data for ridgeline plot
    pos = np.arange(4)
    data = []
    for i in pos:
        data.append(np.random.normal(i, 1, 500))
    
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("mako_r", 4)
    
    for i, d in enumerate(data):
        sns.kdeplot(d, fill=True, alpha=0.5, linewidth=0.5, color=palette[i], label=f'Group {i+1}')
    
    plt.title('Ridge Line Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    save_plot(plt, "ridge_line_plot")

# 17. PCA
def plot_pca():
    sample = get_sample_df()
    df = sample['df']
    # Run PCA on our dataset
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df[['x', 'y', 'z']])
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['category'] = df['category']
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='category')
    plt.title('PCA: First Two Principal Components')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    save_plot(plt, "pca_plot")

# 18. CORRELOGRAM
def plot_correlogram():
    # Create a multi-variate dataset
    multi_df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(0, 1, 100),
        'C': np.random.normal(0, 1, 100),
        'D': np.random.normal(0, 1, 100),
        'E': np.random.normal(0, 1, 100)
    })
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(multi_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlogram / Correlation Matrix')
    save_plot(plt, "correlogram")

# 19. HEATMAP
def plot_heatmap():
    # Create a dataset for heatmap
    data = np.random.rand(10, 12)
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, cmap='hot', annot=True)
    plt.title('Heatmap')
    plt.xlabel('Column')
    plt.ylabel('Row')
    save_plot(plt, "heatmap")

# 20. DENDROGRAM
def plot_dendrogram():
    # Generate cluster data
    np.random.seed(42)
    X = np.random.rand(20, 5)
    
    # Calculate the distance matrix
    Z = hierarchy.linkage(X, 'ward')
    
    plt.figure(figsize=(12, 8))
    hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=10)
    plt.title('Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    save_plot(plt, "dendrogram")


# 21. VENN_DIAGRAM

def plot_venn_diagram():
    plt.figure(figsize=(6,6))
    # Example: two sets with some overlap.
    set1 = set(['A', 'B', 'C', 'D'])
    set2 = set(['C', 'D', 'E', 'F'])
    venn2([set1, set2], set_labels=('Set1', 'Set2'))
    plt.title('Venn Diagram')
    save_plot(plt, "venn_diagram")

# 21. BAR_CHART
def plot_bar_chart():
    sample = get_sample_df()
    df = sample['df']
    plt.figure(figsize=(8, 6))
    # Count frequency of each category.
    order = df['category'].value_counts().index
    sns.countplot(x='category', data=df, order=order)
    plt.title('Bar Chart of Category Frequency')
    plt.xlabel('Category')
    plt.ylabel('Count')
    save_plot(plt, "bar_chart")

# 22. WAFFLE_CHART
def plot_waffle_chart():
    # Simple waffle chart from sample categorical frequency.
    sample = get_sample_df()
    df = sample['df']
    counts = df['category'].value_counts().to_dict()
    total = sum(counts.values())
    # Define grid dimensions.
    rows, cols = 10, 10
    total_tiles = rows * cols
    # Calculate number of tiles for each category.
    tiles = {cat: int(count/total * total_tiles) for cat, count in counts.items()}
    
    # Create grid.
    waffle = np.zeros((rows, cols))
    flat_index = 0
    for cat, tile_count in tiles.items():
        for _ in range(tile_count):
            r, c = divmod(flat_index, cols)
            waffle[r, c] = ord(cat[0])  # use ascii code as a stand-in for color/label
            flat_index += 1
    
    plt.figure(figsize=(6, 6))
    plt.matshow(waffle, cmap="viridis", fignum=1)
    plt.title('Waffle Chart')
    plt.xticks([])
    plt.yticks([])
    save_plot(plt, "waffle_chart")

# 23. WORD_CLOUD
def plot_word_cloud():
    # Create a simple word cloud from sample text.
    text = "data visualization chart plot graph analysis python matplotlib seaborn plotly"
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Word Cloud')
    save_plot(plt, "word_cloud")

# 24. DONUT_CHART
def plot_donut_chart():
    sample = get_sample_df()
    df = sample['df']
    counts = df['category'].value_counts()
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    # Draw a white circle in the center.
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Donut Chart')
    save_plot(plt, "donut_chart")

# 25. PIE_CHART
def plot_pie_chart():
    sample = get_sample_df()
    df = sample['df']
    counts = df['category'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Pie Chart')
    save_plot(plt, "pie_chart")

# 26. TREE_MAP
def plot_tree_map():
    sample = get_sample_df()
    df = sample['df']
    counts = df['category'].value_counts().reset_index()
    counts.columns = ['category', 'value']
    plt.figure(figsize=(8, 6))
    squarify.plot(sizes=counts['value'], label=counts['category'], alpha=.8)
    plt.title('Tree Map')
    plt.axis('off')
    save_plot(plt, "tree_map")

# 27. CIRCLE_PACKING
def plot_circle_packing():
    sample = get_sample_df()
    df = sample['df']
    # Aggregate data by category.
    agg = df['category'].value_counts().reset_index()
    agg.columns = ['category', 'value']
    fig = px.scatter(agg, x='category', y='value', size='value', text='category',
                     title="Circle Packing (Bubble Chart Approximation)")
    fig.update_traces(textposition='middle center')
    fig.show()

# 28. SUNBURST_CHART
def plot_sunburst_chart():
    # Create sample hierarchical data.
    df = pd.DataFrame({
        'region': ['North', 'North', 'South', 'South', 'East', 'East'],
        'country': ['USA', 'Canada', 'USA', 'Mexico', 'China', 'Japan'],
        'value': [10, 15, 10, 5, 20, 10]
    })
    fig = px.sunburst(df, path=['region', 'country'], values='value', title="Sunburst Chart")
    fig.show()

# 29. LOLLIPOP_CHART
def plot_lollipop_chart():
    sample = get_sample_df()
    df = sample['df'].groupby('category', as_index=False)['x'].mean()
    df = df.sort_values('x')
    plt.figure(figsize=(8, 6))
    plt.stem(df['category'], df['x'], basefmt=" ")
    plt.title('Lollipop Chart')
    plt.xlabel('Category')
    plt.ylabel('Average X')
    save_plot(plt, "lollipop_chart")

# 30. GROUPED_SCATTER_PLOT
def plot_grouped_scatter_plot():
    sample = get_sample_df()
    df = sample['df']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='x', y='y', hue='group', style='category', data=df)
    plt.title('Grouped Scatter Plot')
    save_plot(plt, "grouped_scatter_plot")

# 31. GROUPED_BAR_CHART
def plot_grouped_bar_chart():
    sample = get_sample_df()
    df = sample['df']
    # Aggregate mean x by category and group.
    agg = df.groupby(['category', 'group'], as_index=False)['x'].mean()
    plt.figure(figsize=(8, 6))
    sns.barplot(x='category', y='x', hue='group', data=agg)
    plt.title('Grouped Bar Chart')
    plt.xlabel('Category')
    plt.ylabel('Average X')
    save_plot(plt, "grouped_bar_chart")

# 32. PARALLEL_PLOT
def plot_parallel_plot():
    sample = get_sample_df()
    df = sample['df'][['x', 'y', 'z', 'group']]
    plt.figure(figsize=(10, 6))
    parallel_coordinates(df, class_column='group', colormap=plt.get_cmap("Set2"))
    plt.title('Parallel Plot')
    save_plot(plt, "parallel_plot")

# 33. SPIDER_CHART
def plot_spider_chart():
    sample = get_sample_df()
    df = sample['df'].groupby('category', as_index=False).mean()
    # Categories for spider chart axes.
    categories = list(df.columns[1:])  # skip category column
    N = len(categories)
    # Compute angles for each axis.
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for i, row in df.iterrows():
        values = row[1:].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row['category'])
        ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title('Spider Chart')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    save_plot(plt, "spider_chart")

# 34. STACKED_BAR_CHART
def plot_stacked_bar_chart():
    sample = get_sample_df()
    df = sample['df']
    # Create a pivot table: count of rows per category and group.
    pivot = df.pivot_table(index='category', columns='group', aggfunc='size', fill_value=0)
    pivot.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.title('Stacked Bar Chart')
    plt.xlabel('Category')
    plt.ylabel('Count')
    save_plot(plt, "stacked_bar_chart")

# 35. SANKEY_CHART
def plot_sankey_chart():
    # Create a simple Sankey diagram using Plotly.
    node_labels = ["A", "B", "C", "D"]
    # Example links between nodes.
    source = [0, 0, 1, 1]
    target = [2, 3, 2, 3]
    values = [8, 4, 2, 6]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels
        ),
        link=dict(
            source=source,
            target=target,
            value=values
        ))])
    fig.update_layout(title_text="Sankey Chart", font_size=10)
    fig.show()


# def 

def test():
    # Execute all plotting functions
    plot_histogram()
    plot_density_1d()
    plot_box_plot()
    plot_scatter_2d()
    plot_violin()
    plot_scatter_with_marginals()
    plot_density_2d()
    plot_connected_scatter()
    plot_area()
    plot_line()
    plot_bubble()
    plot_scatter_3d()
    plot_surface()
    plot_stacked_area()
    plot_stream_graph()
    plot_ridge_line()
    plot_pca()
    plot_correlogram()
    plot_heatmap()
    plot_dendrogram()
    plot_venn_diagram()
    plot_bar_chart()
    plot_waffle_chart()
    plot_word_cloud()
    plot_donut_chart()
    plot_pie_chart()
    plot_tree_map()
    plot_circle_packing()
    plot_sunburst_chart()
    plot_lollipop_chart()
    plot_grouped_scatter_plot()
    plot_grouped_bar_chart()
    plot_parallel_plot()
    plot_spider_chart()
    plot_stacked_bar_chart()
    plot_sankey_chart()


if __name__ == "__main__":
    plot_venn_diagram()

def plot_chart(chart_type: ChartType):
    """
    Plots a specific chart based on the provided ChartType enum value.
    
    Args:
        chart_type (ChartType): The type of chart to plot
    """
    plot_functions = {
        ChartType.HISTOGRAM: plot_histogram,
        ChartType.DENSITY_PLOT_1D: plot_density_1d,
        ChartType.BOX_PLOT: plot_box_plot,
        ChartType.SCATTER_PLOT_2D: plot_scatter_2d,
        ChartType.VIOLIN_PLOT: plot_violin,
        ChartType.SCATTER_PLOT_WITH_MARGINALS: plot_scatter_with_marginals,
        ChartType.DENSITY_PLOT_2D: plot_density_2d,
        ChartType.CONNECTED_SCATTER_PLOT: plot_connected_scatter,
        ChartType.AREA_PLOT: plot_area,
        ChartType.LINE_PLOT: plot_line,
        ChartType.BUBBLE_PLOT: plot_bubble,
        ChartType.SCATTER_PLOT_3D: plot_scatter_3d,
        ChartType.SURFACE_PLOT: plot_surface,
        ChartType.STACKED_AREA_PLOT: plot_stacked_area,
        ChartType.STREAM_GRAPH: plot_stream_graph,
        ChartType.RIDGE_LINE: plot_ridge_line,
        ChartType.PCA: plot_pca,
        ChartType.CORRELOGRAM: plot_correlogram,
        ChartType.HEATMAP: plot_heatmap,
        ChartType.DENDROGRAM: plot_dendrogram,
        ChartType.VENN_DIAGRAM: plot_venn_diagram,
        ChartType.BAR_CHART: plot_bar_chart,
        ChartType.WAFFLE_CHART: plot_waffle_chart,
        ChartType.WORD_CLOUD: plot_word_cloud,
        ChartType.DONUT_CHART: plot_donut_chart,
        ChartType.PIE_CHART: plot_pie_chart,
        ChartType.TREE_MAP: plot_tree_map,
        ChartType.CIRCLE_PACKING: plot_circle_packing,
        ChartType.SUNBURST_CHART: plot_sunburst_chart,
        ChartType.LOLLIPOP_CHART: plot_lollipop_chart,
        ChartType.GROUPED_SCATTER_PLOT: plot_grouped_scatter_plot,
        ChartType.GROUPED_BAR_CHART: plot_grouped_bar_chart,
        ChartType.PARALLEL_PLOT: plot_parallel_plot,
        ChartType.SPIDER_CHART: plot_spider_chart,
        ChartType.STACKED_BAR_CHART: plot_stacked_bar_chart,
        ChartType.SANKEY_CHART: plot_sankey_chart
    }
    
    if chart_type not in plot_functions:
        raise ValueError(f"Chart type {chart_type} is not supported")
    
    plot_functions[chart_type]()
    plt.close('all')

# Example usage:
# plot_chart(ChartType.HISTOGRAM)

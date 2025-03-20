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


if __name__ == "__main__":
    test()

"""
model_analyzer.py - Clustering Analysis for Undervalued Player Detection

This module applies K-Means clustering to identify patterns and groups of players
based on their performance metrics and salary. The goal is to automatically
discover which cluster represents "undervalued" players (high performance, low salary).

Why K-Means?
- Unsupervised learning: No need for pre-labeled "undervalued" tags
- Interpretable: Creates clear, distinct groups
- Scalable: Works efficiently with thousands of players
- Standard for player profiling in sports analytics
"""

import logging
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Output directory for analysis results
ANALYSIS_DIR = os.path.join(DATA_DIR, 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def load_cleaned_data():
    """
    Load the cleaned dataset from data_cleaner.py output.

    Returns:
        DataFrame with cleaned player data
    """
    cleaned_path = os.path.join(DATA_DIR, 'cleaned_undervalued_data.csv')

    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(
            f"Cleaned data not found at {cleaned_path}. "
            f"Please run data_cleaner.py first."
        )

    df = pd.read_csv(cleaned_path)
    logger.info(f"Loaded cleaned data: {len(df):,} rows")
    return df


def prepare_features_for_clustering(df):
    """
    Select and prepare features for K-Means clustering.

    We choose features that capture:
    - Player value (WAR, wOBA)
    - Player cost (salary)
    - Luck indicators (BABIP)
    - Power metrics (ISO)
    - Contact ability (K%)

    Args:
        df (DataFrame): Cleaned player data

    Returns:
        DataFrame: Features ready for clustering
        DataFrame: Original data with player identifiers
        list: Names of feature columns used
    """
    logger.info("Preparing features for clustering...")

    # Define core features for clustering
    # These are the dimensions that define player profiles
    feature_columns = [
        'WAR',  # Wins Above Replacement - overall value
        'salary',  # Salary in millions - cost
        'BABIP',  # Batting Average on Balls In Play - luck vs skill
        'wOBA',  # Weighted On-Base Average - offensive contribution
        'ISO',  # Isolated Power - raw power
        'K%'  # Strikeout rate - contact ability
    ]

    # Check which features actually exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(f"Available features for clustering: {available_features}")

    if len(available_features) < 3:
        logger.warning("Not enough features for clustering. Using default features.")
        # Fallback to basic features if advanced metrics missing
        available_features = ['WAR', 'salary']
        if 'hits_per_ab' in df.columns:
            available_features.append('hits_per_ab')

    # Create feature matrix
    # Drop rows with missing values in feature columns
    feature_df = df[available_features].dropna()

    # Keep track of which rows were kept
    kept_indices = feature_df.index

    # Also keep identifying information for these rows
    id_columns = []
    for col in ['playerID', 'Name', 'yearID', 'teamID']:
        if col in df.columns:
            id_columns.append(col)

    player_info = df.loc[kept_indices, id_columns].copy()

    logger.info(f"Feature matrix shape: {feature_df.shape}")
    logger.info(f"Kept {len(kept_indices):,} rows after dropping missing values")

    return feature_df, player_info, available_features


def determine_optimal_k(feature_matrix, max_k=10):
    """
    Determine optimal number of clusters using Elbow Method and Silhouette Score.

    The elbow method looks for the "bend" in inertia (within-cluster sum of squares).
    The silhouette score measures how similar points are to their own cluster vs other clusters.

    Args:
        feature_matrix (array-like): Scaled features for clustering
        max_k (int): Maximum number of clusters to try

    Returns:
        dict: Results of k evaluation
        int: Recommended number of clusters
    """
    logger.info(f"Determining optimal K (1 to {max_k})...")

    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    from sklearn.metrics import silhouette_score

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)
        inertias.append(kmeans.inertia_)

        # Silhouette score requires at least 2 clusters and more samples than clusters
        if len(set(labels)) > 1 and len(feature_matrix) > k:
            sil_score = silhouette_score(feature_matrix, labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(-1)

        logger.info(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")

    # Find best k based on silhouette score (higher is better)
    if silhouette_scores:
        best_k = k_range[np.argmax(silhouette_scores)]
        logger.info(f"Optimal K based on silhouette score: {best_k}")
    else:
        # Fallback: look for elbow in inertia
        # Calculate rate of change in inertia
        inertia_diffs = np.diff(inertias)
        inertia_diffs_2 = np.diff(inertia_diffs)

        # The elbow is where the second derivative is highest
        if len(inertia_diffs_2) > 0:
            best_k = np.argmax(inertia_diffs_2) + 2  # +2 because we start at k=2
        else:
            best_k = 4  # Default if can't determine

    # Create visualization of elbow curve
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_k + 1), [inertias[0]] + inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster SSE)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score by k')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'k_optimization.png'), dpi=150)
    plt.close()

    return {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'best_k': best_k
    }, best_k


def apply_kmeans_clustering(feature_matrix, n_clusters=4):
    """
    Apply K-Means clustering to player features.

    K-Means steps:
    1. Initialize k random cluster centers
    2. Assign each point to nearest center
    3. Recalculate centers as mean of assigned points
    4. Repeat until convergence

    Args:
        feature_matrix (array-like): Scaled features
        n_clusters (int): Number of clusters

    Returns:
        array: Cluster labels for each player
        KMeans: Trained KMeans model
    """
    logger.info(f"Applying K-Means clustering with k={n_clusters}...")

    # Initialize K-Means with fixed random state for reproducibility
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,  # Run 10 times with different centroid seeds
        max_iter=300,  # Maximum iterations per run
        algorithm='lloyd'  # Standard K-Means algorithm
    )

    # Fit the model and predict cluster labels
    cluster_labels = kmeans.fit_predict(feature_matrix)

    # Count players per cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        logger.info(f"Cluster {cluster}: {count} players ({count / len(cluster_labels) * 100:.1f}%)")

    return cluster_labels, kmeans


def analyze_clusters(df, player_info, cluster_labels, feature_names, kmeans_model):
    """
    Analyze each cluster to identify characteristics and find undervalued group.

    Calculates:
    - Mean values for each feature per cluster
    - Value ratio (WAR/salary) per cluster
    - Identifies which cluster has highest value ratio (undervalued)

    Args:
        df (DataFrame): Original feature matrix
        player_info (DataFrame): Player identifying information
        cluster_labels (array): Cluster assignments
        feature_names (list): Names of features used
        kmeans_model (KMeans): Trained K-Means model

    Returns:
        DataFrame: Players with cluster assignments
        DataFrame: Cluster statistics
        int: Cluster ID identified as undervalued
    """
    logger.info("Analyzing cluster characteristics...")

    # Add cluster labels to player info
    result_df = player_info.copy()
    result_df['cluster'] = cluster_labels

    # Add back the feature values for analysis
    for i, col in enumerate(feature_names):
        result_df[col] = df[col].values

    # Calculate cluster statistics
    cluster_stats = []

    for cluster in sorted(result_df['cluster'].unique()):
        cluster_data = result_df[result_df['cluster'] == cluster]

        # Calculate mean values for key metrics
        stats = {
            'cluster': cluster,
            'count': len(cluster_data),
            'pct': len(cluster_data) / len(result_df) * 100
        }

        # Add mean for each feature
        for col in feature_names:
            stats[f'avg_{col}'] = cluster_data[col].mean()

        # Calculate value ratio (WAR per million $)
        if 'WAR' in feature_names and 'salary' in feature_names:
            avg_war = cluster_data['WAR'].mean()
            avg_salary = cluster_data['salary'].mean()
            stats['avg_value_ratio'] = avg_war / (avg_salary + 0.1)  # Avoid division by zero
            stats['total_war'] = cluster_data['WAR'].sum()
            stats['total_salary'] = cluster_data['salary'].sum()

        cluster_stats.append(stats)

    # Create statistics dataframe
    stats_df = pd.DataFrame(cluster_stats)
    stats_df = stats_df.sort_values('avg_value_ratio', ascending=False)

    # Identify undervalued cluster (highest value ratio)
    undervalued_cluster = stats_df.iloc[0]['cluster']

    logger.info("=" * 60)
    logger.info("CLUSTER STATISTICS (sorted by value ratio):")
    logger.info("=" * 60)

    # Display cluster statistics
    display_cols = ['cluster', 'count', 'avg_WAR', 'avg_salary', 'avg_value_ratio']
    display_cols = [col for col in display_cols if col in stats_df.columns]
    logger.info(f"\n{stats_df[display_cols].to_string()}")

    logger.info(f"\n✅ Undervalued cluster identified: Cluster {int(undervalued_cluster)}")
    logger.info(f"   This group has the highest WAR per dollar ratio")

    return result_df, stats_df, undervalued_cluster


def visualize_clusters(result_df, feature_names, undervalued_cluster):
    """
    Create visualizations of clustering results.

    Generates:
    1. PCA 2D projection of clusters
    2. Pairplot of key features
    3. Boxplots comparing clusters

    Args:
        result_df (DataFrame): Players with cluster assignments
        feature_names (list): Features used in clustering
        undervalued_cluster (int): ID of undervalued cluster
    """
    logger.info("Creating cluster visualizations...")

    # Set style
    sns.set_style("whitegrid")

    # 1. PCA 2D Projection
    # Reduce dimensions to 2 for visualization
    feature_matrix = result_df[feature_names].values
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(feature_matrix)

    result_df['PC1'] = pca_result[:, 0]
    result_df['PC2'] = pca_result[:, 1]

    # Calculate variance explained
    variance_explained = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(12, 8))

    # Create color map highlighting undervalued cluster
    colors = result_df['cluster'].map(
        lambda x: 'red' if x == undervalued_cluster else
        ('blue' if x == 0 else 'green' if x == 1 else 'orange' if x == 2 else 'purple')
    )

    scatter = plt.scatter(
        result_df['PC1'],
        result_df['PC2'],
        c=colors,
        alpha=0.6,
        s=30,
        edgecolors='black',
        linewidth=0.5
    )

    # Highlight cluster centers (approximated)
    for cluster in result_df['cluster'].unique():
        cluster_data = result_df[result_df['cluster'] == cluster]
        center_pc1 = cluster_data['PC1'].mean()
        center_pc2 = cluster_data['PC2'].mean()

        plt.scatter(
            center_pc1, center_pc2,
            c='black',
            s=200,
            marker='X',
            edgecolors='white',
            linewidth=2,
            label=f'Cluster {cluster} Center'
        )

    plt.xlabel(f'PC1 ({variance_explained[0]:.1f}% variance)')
    plt.ylabel(f'PC2 ({variance_explained[1]:.1f}% variance)')
    plt.title('Player Clusters - PCA 2D Projection\n(Red = Undervalued Cluster)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'cluster_pca.png'), dpi=150)
    plt.close()

    # 2. Boxplots comparing key metrics across clusters
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    metrics_to_plot = ['WAR', 'salary', 'BABIP', 'wOBA', 'ISO', 'K%']
    metrics_to_plot = [m for m in metrics_to_plot if m in result_df.columns]

    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes):
            sns.boxplot(
                x='cluster',
                y=metric,
                data=result_df,
                ax=axes[i],
                palette='Set2'
            )
            axes[i].set_title(f'{metric} by Cluster')
            # Highlight undervalued cluster
            if undervalued_cluster in result_df['cluster'].unique():
                axes[i].axvline(x=undervalued_cluster, color='red', linestyle='--', alpha=0.5)

    # Hide unused subplots
    for i in range(len(metrics_to_plot), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'cluster_boxplots.png'), dpi=150)
    plt.close()

    # 3. Scatter plot: WAR vs Salary colored by cluster
    plt.figure(figsize=(12, 8))

    for cluster in sorted(result_df['cluster'].unique()):
        cluster_data = result_df[result_df['cluster'] == cluster]
        label = f'Cluster {cluster}'
        if cluster == undervalued_cluster:
            label += ' (UNDERVALUED)'

        plt.scatter(
            cluster_data['salary'],
            cluster_data['WAR'],
            label=label,
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel('Salary (Millions $)')
    plt.ylabel('WAR (Wins Above Replacement)')
    plt.title('WAR vs Salary by Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'war_vs_salary.png'), dpi=150)
    plt.close()

    logger.info(f"Visualizations saved to {ANALYSIS_DIR}")


def identify_undervalued_players(result_df, undervalued_cluster, top_n=50):
    """
    Extract and rank players from the undervalued cluster.

    Args:
        result_df (DataFrame): Players with cluster assignments
        undervalued_cluster (int): ID of undervalued cluster
        top_n (int): Number of top players to return

    Returns:
        DataFrame: Top undervalued players
    """
    logger.info(f"Identifying top {top_n} undervalued players from cluster {undervalued_cluster}...")

    # Filter to undervalued cluster
    undervalued_df = result_df[result_df['cluster'] == undervalued_cluster].copy()

    # Calculate value ratio if not already present
    if 'WAR' in undervalued_df.columns and 'salary' in undervalued_df.columns:
        undervalued_df['value_ratio'] = undervalued_df['WAR'] / (undervalued_df['salary'] + 0.1)
        undervalued_df = undervalued_df.sort_values('value_ratio', ascending=False)
    else:
        # Fallback to WAR if no salary
        undervalued_df = undervalued_df.sort_values('WAR', ascending=False)

    # Select top N
    top_undervalued = undervalued_df.head(top_n)

    logger.info(f"Found {len(undervalued_df)} players in undervalued cluster")
    logger.info(f"Top {top_n} saved to CSV")

    return top_undervalued


def save_results(result_df, stats_df, top_undervalued, undervalued_cluster):
    """
    Save all analysis results to CSV files.

    Args:
        result_df (DataFrame): All players with cluster assignments
        stats_df (DataFrame): Cluster statistics
        top_undervalued (DataFrame): Top undervalued players
        undervalued_cluster (int): ID of undervalued cluster
    """
    # Save all players with cluster assignments
    result_path = os.path.join(ANALYSIS_DIR, 'players_with_clusters.csv')
    result_df.to_csv(result_path, index=False)
    logger.info(f"Saved players with clusters to: {result_path}")

    # Save cluster statistics
    stats_path = os.path.join(ANALYSIS_DIR, 'cluster_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"Saved cluster statistics to: {stats_path}")

    # Save top undervalued players
    undervalued_path = os.path.join(DATA_DIR, 'top_undervalued_players.csv')
    top_undervalued.to_csv(undervalued_path, index=False)
    logger.info(f"Saved top undervalued players to: {undervalued_path}")

    # Save cluster summary as text
    summary_path = os.path.join(ANALYSIS_DIR, 'cluster_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CLUSTER ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Undervalued Cluster: {undervalued_cluster}\n")
        f.write(f"Total players analyzed: {len(result_df)}\n\n")

        f.write("Cluster Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(stats_df.to_string())

        f.write("\n\nTop 10 Undervalued Players:\n")
        f.write("-" * 40 + "\n")
        display_cols = ['Name', 'teamID', 'yearID', 'WAR', 'salary', 'value_ratio']
        display_cols = [col for col in display_cols if col in top_undervalued.columns]
        f.write(top_undervalued.head(10)[display_cols].to_string())

    logger.info(f"Saved summary to: {summary_path}")


def main():
    """
    Main execution function for clustering analysis.

    Steps:
    1. Load cleaned data
    2. Prepare features for clustering
    3. Determine optimal number of clusters
    4. Apply K-Means clustering
    5. Analyze cluster characteristics
    6. Visualize results
    7. Identify undervalued players
    8. Save results
    """
    logger.info("=" * 60)
    logger.info("STARTING CLUSTERING ANALYSIS FOR UNDERVALUED PLAYERS")
    logger.info("=" * 60)

    try:
        # Step 1: Load data
        df = load_cleaned_data()

        # Step 2: Prepare features
        feature_df, player_info, feature_names = prepare_features_for_clustering(df)

        # Step 3: Scale features (crucial for K-Means)
        # K-Means is distance-based, so features must be on similar scales
        scaler = StandardScaler()
        feature_scaled = scaler.fit_transform(feature_df)

        # Step 4: Determine optimal K
        k_results, best_k = determine_optimal_k(feature_scaled, max_k=8)

        # Step 5: Apply K-Means with optimal K
        cluster_labels, kmeans_model = apply_kmeans_clustering(feature_scaled, n_clusters=best_k)

        # Step 6: Analyze clusters
        result_df, stats_df, undervalued_cluster = analyze_clusters(
            feature_df, player_info, cluster_labels, feature_names, kmeans_model
        )

        # Step 7: Visualize results
        visualize_clusters(result_df, feature_names, undervalued_cluster)

        # Step 8: Identify top undervalued players
        top_undervalued = identify_undervalued_players(result_df, undervalued_cluster, top_n=50)

        # Step 9: Save results
        save_results(result_df, stats_df, top_undervalued, undervalued_cluster)

        # Step 10: Display top 10 undervalued players
        logger.info("=" * 60)
        logger.info("TOP 10 UNDERVALUED PLAYERS (Final Results):")
        logger.info("=" * 60)

        display_cols = ['Name', 'teamID', 'yearID', 'WAR', 'salary', 'value_ratio', 'BABIP', 'wOBA']
        display_cols = [col for col in display_cols if col in top_undervalued.columns]

        logger.info(f"\n{top_undervalued.head(10)[display_cols].to_string()}")

        logger.info("=" * 60)
        logger.info("CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("Results saved to 'data/analysis/' directory")
        logger.info("Top undervalued players saved to 'data/top_undervalued_players.csv'")
        logger.info("Next step: Create app.py for Streamlit dashboard")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error in clustering analysis: {e}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    main()
"""
model_analyzer.py - Enhanced Clustering Analysis for Undervalued Player Detection

VERSION 3: Uses clean model_data.csv with unified 'team' column
Focus: Identify undervalued players using K-Means clustering

This module:
1. Loads clean model_data.csv from data pipeline
2. Applies K-Means clustering to identify player profiles
3. Identifies undervalued cluster (highest WAR/salary ratio)
4. Saves results for dashboard use
5. Generates visualizations and statistics

Author: Moneyball Analytics
Date: 2026
"""

import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ANALYSIS_DIR = os.path.join(DATA_DIR, 'analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def load_clean_data():
    """
    Load the clean model-ready dataset from data pipeline.

    Returns:
        DataFrame with clean data for clustering
    """
    logger.info("="*60)
    logger.info("LOADING CLEAN DATA")
    logger.info("="*60)

    # Try multiple possible paths (prioritize new model_data.csv)
    possible_paths = [
        os.path.join(DATA_DIR, 'model_data.csv'),           # New clean data
        os.path.join(DATA_DIR, 'model_ready_data.csv'),     # Previous version
        os.path.join(DATA_DIR, 'cleaned_undervalued_data.csv')  # Old version
    ]

    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            logger.info(f"✅ Loaded: {os.path.basename(path)}")
            logger.info(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            logger.info(f"   Years: {sorted(df['yearID'].unique())}")
            logger.info(f"   Team column: {'team' if 'team' in df.columns else 'NOT FOUND'}")
            break

    if df is None:
        logger.error("❌ No data file found. Please run data_cleaner_complete.py first.")
        raise FileNotFoundError("No data file available for clustering")

    return df


def prepare_features_for_clustering(df):
    """
    Select and prepare features for K-Means clustering.

    Args:
        df (DataFrame): Clean player data

    Returns:
        DataFrame: Features ready for clustering
        DataFrame: Player information (with team)
        list: Feature column names
    """
    logger.info("="*60)
    logger.info("PREPARING FEATURES FOR CLUSTERING")
    logger.info("="*60)

    # Core features that define player profiles
    feature_columns = [
        'WAR',           # Wins Above Replacement - overall value
        'salary',        # Salary in millions - cost
        'BABIP',         # Batting Average on Balls In Play - luck vs skill
        'wOBA',          # Weighted On-Base Average - offensive contribution
        'ISO',           # Isolated Power - raw power
        'K%'             # Strikeout rate - contact ability
    ]

    # Check which features actually exist
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(f"Available features for clustering: {available_features}")

    if len(available_features) < 3:
        logger.error("Insufficient features for clustering")
        raise ValueError("Need at least 3 features for clustering")

    # Create feature matrix (drop rows with missing values)
    feature_df = df[available_features].dropna()

    # Keep track of which rows were kept
    kept_indices = feature_df.index
    logger.info(f"Rows before dropping NA: {len(df)}")
    logger.info(f"Rows after dropping NA: {len(feature_df)}")
    logger.info(f"Dropped {len(df) - len(feature_df)} rows with missing values")

    # Keep player information for later reference
    id_columns = []
    for col in ['Name', 'team', 'yearID']:
        if col in df.columns:
            id_columns.append(col)

    player_info = df.loc[kept_indices, id_columns].copy()

    # Feature statistics
    logger.info("\nFeature statistics:")
    logger.info(f"\n{feature_df.describe().round(3)}")

    return feature_df, player_info, available_features


def determine_optimal_k(feature_matrix, max_k=8):
    """
    Determine optimal number of clusters using Silhouette Score.

    Args:
        feature_matrix (array-like): Scaled features
        max_k (int): Maximum clusters to try

    Returns:
        int: Optimal number of clusters
        dict: Results for visualization
    """
    logger.info("="*60)
    logger.info(f"DETERMINING OPTIMAL K (2 to {max_k})")
    logger.info("="*60)

    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)

        if len(set(labels)) > 1 and len(feature_matrix) > k:
            sil_score = silhouette_score(feature_matrix, labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(-1)

        logger.info(f"K={k}: Silhouette={silhouette_scores[-1]:.4f}")

    # Find best k (highest silhouette score)
    best_k = k_range[np.argmax(silhouette_scores)]
    logger.info(f"\n✅ Optimal K: {best_k} (Silhouette={max(silhouette_scores):.4f})")

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={best_k}')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score by Number of Clusters', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'optimal_k.png'), dpi=150)
    plt.close()

    return best_k, {'k_range': list(k_range), 'silhouette_scores': silhouette_scores}


def apply_kmeans_clustering(feature_matrix, n_clusters):
    """
    Apply K-Means clustering to player features.

    Args:
        feature_matrix (array-like): Scaled features
        n_clusters (int): Number of clusters

    Returns:
        array: Cluster labels
        KMeans: Trained model
        StandardScaler: Fitted scaler
    """
    logger.info("="*60)
    logger.info(f"APPLYING K-MEANS CLUSTERING (K={n_clusters})")
    logger.info("="*60)

    # Scale features (crucial for K-Means)
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_matrix)

    # Apply K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )

    cluster_labels = kmeans.fit_predict(feature_scaled)

    # Count players per cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        percentage = count / len(cluster_labels) * 100
        logger.info(f"Cluster {cluster}: {count} players ({percentage:.1f}%)")

    return cluster_labels, kmeans, scaler, feature_scaled


def analyze_clusters(feature_df, player_info, cluster_labels, feature_names):
    """
    Analyze each cluster to identify characteristics and find undervalued group.

    Args:
        feature_df (DataFrame): Original feature values
        player_info (DataFrame): Player information
        cluster_labels (array): Cluster assignments
        feature_names (list): Feature column names

    Returns:
        DataFrame: Players with cluster assignments
        DataFrame: Cluster statistics
        int: Undervalued cluster ID
        dict: Centroid of undervalued cluster
    """
    logger.info("="*60)
    logger.info("ANALYZING CLUSTERS")
    logger.info("="*60)

    # Add cluster labels to player info
    result_df = player_info.copy()
    result_df['cluster'] = cluster_labels

    # Add feature values back
    for i, col in enumerate(feature_names):
        result_df[col] = feature_df[col].values

    # Calculate cluster statistics
    cluster_stats = []

    for cluster in sorted(result_df['cluster'].unique()):
        cluster_data = result_df[result_df['cluster'] == cluster]

        stats = {
            'cluster': cluster,
            'count': len(cluster_data),
            'percentage': len(cluster_data) / len(result_df) * 100
        }

        # Calculate mean for each feature
        for col in feature_names:
            stats[f'avg_{col}'] = cluster_data[col].mean()

        # Calculate value ratio (WAR per million $)
        if 'WAR' in feature_names and 'salary' in feature_names:
            avg_war = cluster_data['WAR'].mean()
            avg_salary = cluster_data['salary'].mean()
            stats['avg_value_ratio'] = avg_war / (avg_salary + 0.1)
            stats['total_WAR'] = cluster_data['WAR'].sum()
            stats['total_salary'] = cluster_data['salary'].sum()

        cluster_stats.append(stats)

    # Create statistics dataframe
    stats_df = pd.DataFrame(cluster_stats)
    stats_df = stats_df.sort_values('avg_value_ratio', ascending=False)

    # Identify undervalued cluster (highest WAR per $)
    undervalued_cluster = int(stats_df.iloc[0]['cluster'])

    logger.info("\n" + "="*60)
    logger.info("CLUSTER STATISTICS (sorted by value ratio)")
    logger.info("="*60)

    display_cols = ['cluster', 'count', 'avg_WAR', 'avg_salary', 'avg_value_ratio']
    logger.info(f"\n{stats_df[display_cols].round(3).to_string()}")

    logger.info(f"\n✅ Undervalued cluster: {undervalued_cluster}")
    logger.info(f"   Highest WAR per dollar ratio")

    # Calculate centroid of undervalued cluster
    undervalued_data = result_df[result_df['cluster'] == undervalued_cluster]
    centroid = {}
    for col in feature_names:
        centroid[col] = undervalued_data[col].mean()

    return result_df, stats_df, undervalued_cluster, centroid


def visualize_clusters_pca(result_df, feature_names, undervalued_cluster):
    """
    Visualize clusters using PCA 2D projection.

    Args:
        result_df (DataFrame): Players with cluster assignments
        feature_names (list): Feature names
        undervalued_cluster (int): ID of undervalued cluster
    """
    logger.info("="*60)
    logger.info("CREATING PCA VISUALIZATION")
    logger.info("="*60)

    # Get feature matrix
    feature_matrix = result_df[feature_names].values

    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(feature_matrix)

    result_df['PC1'] = pca_result[:, 0]
    result_df['PC2'] = pca_result[:, 1]

    # Variance explained
    var_explained = pca.explained_variance_ratio_ * 100
    logger.info(f"PC1 explains {var_explained[0]:.1f}% of variance")
    logger.info(f"PC2 explains {var_explained[1]:.1f}% of variance")

    # Create color palette
    n_clusters = len(result_df['cluster'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    plt.figure(figsize=(12, 8))

    for cluster in range(n_clusters):
        cluster_data = result_df[result_df['cluster'] == cluster]
        label = f'Cluster {cluster}'
        if cluster == undervalued_cluster:
            label += ' (UNDERVALUED)'

        plt.scatter(
            cluster_data['PC1'],
            cluster_data['PC2'],
            c=[colors[cluster]],
            label=label,
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel(f'PC1 ({var_explained[0]:.1f}%)')
    plt.ylabel(f'PC2 ({var_explained[1]:.1f}%)')
    plt.title('Player Clusters - PCA Projection')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'cluster_pca.png'), dpi=150)
    plt.close()

    logger.info(f"✅ PCA visualization saved to {ANALYSIS_DIR}")


def visualize_feature_distributions(result_df, feature_names, undervalued_cluster):
    """
    Create boxplots comparing feature distributions across clusters.

    Args:
        result_df (DataFrame): Players with cluster assignments
        feature_names (list): Feature names
        undervalued_cluster (int): ID of undervalued cluster
    """
    logger.info("="*60)
    logger.info("CREATING FEATURE DISTRIBUTION PLOTS")
    logger.info("="*60)

    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(feature_names):
        if i < len(axes):
            # Prepare data for boxplot
            data_to_plot = []
            cluster_labels = sorted(result_df['cluster'].unique())
            for cluster in cluster_labels:
                data_to_plot.append(result_df[result_df['cluster'] == cluster][feature])

            # Create boxplot
            bp = axes[i].boxplot(data_to_plot, patch_artist=True, labels=[f'C{c}' for c in cluster_labels])

            # Color boxes (highlight undervalued cluster)
            for j, patch in enumerate(bp['boxes']):
                if cluster_labels[j] == undervalued_cluster:
                    patch.set_facecolor('#ff6b6b')
                    patch.set_alpha(0.7)
                else:
                    patch.set_facecolor('#4ecdc4')
                    patch.set_alpha(0.7)

            axes[i].set_title(f'{feature} by Cluster', fontsize=12)
            axes[i].set_ylabel(feature)
            axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'cluster_boxplots.png'), dpi=150)
    plt.close()

    logger.info(f"✅ Feature distribution plots saved to {ANALYSIS_DIR}")


def create_2025_analysis(result_df, centroid, undervalued_cluster):
    """
    Create focused analysis for 2025 players.

    Args:
        result_df (DataFrame): All players with clusters
        centroid (dict): Undervalued cluster centroid
        undervalued_cluster (int): ID of undervalued cluster

    Returns:
        DataFrame: 2025 players with scores
    """
    logger.info("="*60)
    logger.info("ANALYZING 2025 PLAYERS")
    logger.info("="*60)

    # Filter to 2025 players
    players_2025 = result_df[result_df['yearID'] == 2025].copy()
    logger.info(f"Found {len(players_2025)} players in 2025")

    if len(players_2025) == 0:
        logger.warning("No 2025 players found!")
        return pd.DataFrame()

    # Calculate similarity to undervalued centroid
    if centroid:
        # Normalize features for distance calculation
        distances = []
        for _, player in players_2025.iterrows():
            dist = 0
            for feature, value in centroid.items():
                if feature in player:
                    # Simple Euclidean distance (unweighted)
                    dist += (player[feature] - value) ** 2
            dist = np.sqrt(dist)
            distances.append(dist)

        # Convert to similarity score (0-100, higher = more similar)
        min_dist = min(distances)
        max_dist = max(distances)
        if max_dist > min_dist:
            similarity = 100 * (1 - (np.array(distances) - min_dist) / (max_dist - min_dist))
        else:
            similarity = np.ones(len(distances)) * 50

        players_2025['similarity_score'] = similarity
        players_2025['value_ratio'] = players_2025['WAR'] / (players_2025['salary'] + 0.1)

    # Identify undervalued 2025 players (those in undervalued cluster)
    undervalued_2025 = players_2025[players_2025['cluster'] == undervalued_cluster].copy()
    undervalued_2025 = undervalued_2025.sort_values('value_ratio', ascending=False)

    logger.info(f"\nTop 10 undervalued players in 2025:")
    display_cols = ['Name', 'team', 'WAR', 'salary', 'value_ratio']
    display_cols = [c for c in display_cols if c in undervalued_2025.columns]
    logger.info(f"\n{undervalued_2025.head(10)[display_cols].to_string()}")

    return players_2025, undervalued_2025


def save_results(result_df, stats_df, players_2025, undervalued_2025, centroid):
    """
    Save all analysis results to CSV files.

    Args:
        result_df (DataFrame): All players with clusters
        stats_df (DataFrame): Cluster statistics
        players_2025 (DataFrame): 2025 players with scores
        undervalued_2025 (DataFrame): Undervalued 2025 players
        centroid (dict): Undervalued cluster centroid
    """
    logger.info("="*60)
    logger.info("SAVING RESULTS")
    logger.info("="*60)

    # Save all players with clusters
    all_players_path = os.path.join(ANALYSIS_DIR, 'all_players_with_clusters.csv')
    result_df.to_csv(all_players_path, index=False)
    logger.info(f"✅ Saved: {all_players_path}")

    # Save cluster statistics
    stats_path = os.path.join(ANALYSIS_DIR, 'cluster_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"✅ Saved: {stats_path}")

    # Save 2025 analysis
    if not players_2025.empty:
        players_2025_path = os.path.join(ANALYSIS_DIR, 'players_2025_analysis.csv')
        players_2025.to_csv(players_2025_path, index=False)
        logger.info(f"✅ Saved: {players_2025_path}")

    # Save top undervalued 2025
    if not undervalued_2025.empty:
        top_path = os.path.join(DATA_DIR, 'top_undervalued_2025.csv')
        undervalued_2025.head(50).to_csv(top_path, index=False)
        logger.info(f"✅ Saved: {top_path}")

    # Save centroid
    if centroid:
        centroid_df = pd.DataFrame([centroid])
        centroid_path = os.path.join(ANALYSIS_DIR, 'undervalued_centroid.csv')
        centroid_df.to_csv(centroid_path, index=False)
        logger.info(f"✅ Saved: {centroid_path}")


def main():
    """
    Main execution function for clustering analysis.
    """
    logger.info("="*70)
    logger.info("ENHANCED CLUSTERING ANALYSIS V3")
    logger.info("="*70)

    try:
        # Step 1: Load clean data
        df = load_clean_data()

        # Step 2: Prepare features
        feature_df, player_info, feature_names = prepare_features_for_clustering(df)

        # Step 3: Scale features
        scaler = StandardScaler()
        feature_scaled = scaler.fit_transform(feature_df)

        # Step 4: Determine optimal K
        best_k, _ = determine_optimal_k(feature_scaled, max_k=8)

        # Step 5: Apply K-Means
        cluster_labels, kmeans, _, _ = apply_kmeans_clustering(feature_scaled, best_k)

        # Step 6: Analyze clusters
        result_df, stats_df, undervalued_cluster, centroid = analyze_clusters(
            feature_df, player_info, cluster_labels, feature_names
        )

        # Step 7: Create visualizations
        visualize_clusters_pca(result_df, feature_names, undervalued_cluster)
        visualize_feature_distributions(result_df, feature_names, undervalued_cluster)

        # Step 8: Analyze 2025 players
        players_2025, undervalued_2025 = create_2025_analysis(
            result_df, centroid, undervalued_cluster
        )

        # Step 9: Save results
        save_results(result_df, stats_df, players_2025, undervalued_2025, centroid)

        logger.info("="*70)
        logger.info("✅ CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("1. Run app.py to visualize results in dashboard")
        logger.info("2. Check top_undervalued_2025.csv for scouting targets")

    except Exception as e:
        logger.error(f"Error in clustering analysis: {e}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    main()
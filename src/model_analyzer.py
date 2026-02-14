"""
model_analyzer_v2.py - Enhanced Clustering Analysis for Undervalued Player Detection

VERSION 2: Hybrid approach combining clustering with distance metrics and trends
FOCUS: 2025 players with similarity scores to the undervalued cluster

This module:
1. Loads cleaned data and applies K-Means clustering
2. Identifies the undervalued cluster (highest WAR/salary ratio)
3. Calculates distance from each 2025 player to the undervalued centroid
4. Computes trends (2023→2024→2025) to find improving players
5. Generates composite scores (0-100) for ranking
6. Saves focused outputs for 2025 players only

Author: Moneyball Scouting Project
Date: 2026
"""

import logging
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
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
    logger.info(f"Years present: {sorted(df['yearID'].unique())}")
    return df


def prepare_features_for_clustering(df):
    """
    Select and prepare features for K-Means clustering.

    Args:
        df (DataFrame): Cleaned player data

    Returns:
        DataFrame: Features ready for clustering
        DataFrame: Original data with player identifiers
        list: Names of feature columns used
    """
    logger.info("Preparing features for clustering...")

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
        logger.warning("Not enough features for clustering. Using WAR and salary only.")
        available_features = ['WAR', 'salary']

    # Create feature matrix (drop rows with missing values)
    feature_df = df[available_features].dropna()

    # Keep track of which rows were kept
    kept_indices = feature_df.index

    # Also keep identifying information
    id_columns = []
    for col in ['playerID', 'Name', 'yearID', 'teamID', 'Age']:
        if col in df.columns:
            id_columns.append(col)

    player_info = df.loc[kept_indices, id_columns].copy()

    logger.info(f"Feature matrix shape: {feature_df.shape}")
    logger.info(f"Kept {len(kept_indices):,} rows after dropping missing values")

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
    logger.info(f"Determining optimal K (2 to {max_k})...")

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

        logger.info(f"K={k}: Silhouette={silhouette_scores[-1]:.3f}")

    # Find best k (highest silhouette score)
    best_k = k_range[np.argmax(silhouette_scores)]
    logger.info(f"Optimal K based on silhouette score: {best_k}")

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
    plt.savefig(os.path.join(ANALYSIS_DIR, 'optimal_k_silhouette.png'), dpi=150)
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
    logger.info(f"Applying K-Means clustering with k={n_clusters}...")

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
        logger.info(f"Cluster {cluster}: {count} players ({count/len(cluster_labels)*100:.1f}%)")

    return cluster_labels, kmeans, scaler, feature_scaled


def identify_undervalued_cluster(feature_df, cluster_labels):
    """
    Identify which cluster represents undervalued players (highest WAR/salary ratio).

    Args:
        feature_df (DataFrame): Original feature values
        cluster_labels (array): Cluster assignments

    Returns:
        int: Cluster ID of undervalued group
        DataFrame: Cluster statistics
        dict: Centroid of undervalued cluster (original scale)
    """
    logger.info("Identifying undervalued cluster...")

    # Add cluster labels to features
    df_with_clusters = feature_df.copy()
    df_with_clusters['cluster'] = cluster_labels

    # Calculate statistics per cluster
    cluster_stats = []

    for cluster in sorted(df_with_clusters['cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]

        # Calculate mean values
        stats = {
            'cluster': cluster,
            'count': len(cluster_data),
            'pct': len(cluster_data) / len(df_with_clusters) * 100,
            'avg_WAR': cluster_data['WAR'].mean(),
            'avg_salary': cluster_data['salary'].mean(),
            'avg_BABIP': cluster_data['BABIP'].mean() if 'BABIP' in cluster_data.columns else 0,
            'avg_wOBA': cluster_data['wOBA'].mean() if 'wOBA' in cluster_data.columns else 0,
            'avg_ISO': cluster_data['ISO'].mean() if 'ISO' in cluster_data.columns else 0,
            'avg_K': cluster_data['K%'].mean() if 'K%' in cluster_data.columns else 0
        }

        # Calculate value ratio (WAR per million dollars)
        stats['avg_value_ratio'] = stats['avg_WAR'] / (stats['avg_salary'] + 0.1)
        stats['total_WAR'] = cluster_data['WAR'].sum()
        stats['total_salary'] = cluster_data['salary'].sum()

        cluster_stats.append(stats)

    # Create statistics dataframe
    stats_df = pd.DataFrame(cluster_stats)
    stats_df = stats_df.sort_values('avg_value_ratio', ascending=False)

    # Undervalued cluster = highest value ratio
    undervalued_cluster = int(stats_df.iloc[0]['cluster'])

    logger.info("=" * 60)
    logger.info("CLUSTER STATISTICS (sorted by value ratio):")
    logger.info("=" * 60)

    display_cols = ['cluster', 'count', 'avg_WAR', 'avg_salary', 'avg_value_ratio']
    logger.info(f"\n{stats_df[display_cols].to_string()}")

    logger.info(f"\n✅ Undervalued cluster identified: Cluster {undervalued_cluster}")
    logger.info(f"   This group has the highest WAR per dollar ratio")

    # Extract centroid of undervalued cluster (original scale)
    undervalued_data = df_with_clusters[df_with_clusters['cluster'] == undervalued_cluster]
    centroid = {
        'WAR': undervalued_data['WAR'].mean(),
        'salary': undervalued_data['salary'].mean(),
        'BABIP': undervalued_data['BABIP'].mean() if 'BABIP' in undervalued_data.columns else 0,
        'wOBA': undervalued_data['wOBA'].mean() if 'wOBA' in undervalued_data.columns else 0,
        'ISO': undervalued_data['ISO'].mean() if 'ISO' in undervalued_data.columns else 0,
        'K%': undervalued_data['K%'].mean() if 'K%' in undervalued_data.columns else 0
    }

    return undervalued_cluster, stats_df, centroid


def calculate_distances_to_centroid(feature_df, centroid, weights=None):
    """
    Calculate weighted Euclidean distance from each player to the undervalued centroid.

    Args:
        feature_df (DataFrame): Feature values for all players
        centroid (dict): Centroid of undervalued cluster
        weights (dict): Feature weights (closer to centroid = more similar)

    Returns:
        array: Distance scores (lower = more similar)
        array: Normalized similarity scores (0-100, higher = more similar)
    """
    logger.info("Calculating distances to undervalued centroid...")

    # Default weights if not provided
    if weights is None:
        weights = {
            'WAR': 0.30,
            'salary': 0.25,  # Lower salary is better (inverse relationship)
            'BABIP': 0.15,
            'wOBA': 0.15,
            'ISO': 0.10,
            'K%': 0.05
        }

    # Initialize distance array
    distances = np.zeros(len(feature_df))

    # Calculate weighted Euclidean distance
    for feature, weight in weights.items():
        if feature in feature_df.columns and feature in centroid:
            # For salary, we want LOW salary to be closer to centroid (inverse)
            if feature == 'salary':
                # Transform salary so that lower values are closer to centroid
                # We use absolute difference but centroid has low salary already
                diff = np.abs(feature_df[feature].values - centroid[feature])
            else:
                diff = np.abs(feature_df[feature].values - centroid[feature])

            # Normalize the difference by feature range to make weights comparable
            feature_range = feature_df[feature].max() - feature_df[feature].min()
            if feature_range > 0:
                diff_normalized = diff / feature_range
            else:
                diff_normalized = diff

            distances += weight * diff_normalized

    # Convert distances to similarity scores (0-100, higher = more similar)
    # Invert and normalize: closer distance = higher score
    min_dist = distances.min()
    max_dist = distances.max()

    if max_dist > min_dist:
        similarity_scores = 100 * (1 - (distances - min_dist) / (max_dist - min_dist))
    else:
        similarity_scores = np.ones_like(distances) * 50

    logger.info(f"Distance range: [{min_dist:.3f}, {max_dist:.3f}]")
    logger.info(f"Similarity range: [{similarity_scores.min():.1f}, {similarity_scores.max():.1f}]")

    return distances, similarity_scores


def calculate_trends(df, player_col='playerID', years=[2023, 2024, 2025]):
    """
    Calculate performance trends for players with multiple years.

    Args:
        df (DataFrame): Full dataset with multiple years
        player_col (str): Column identifying players
        years (list): Years to consider for trend

    Returns:
        dict: Trend scores by player
    """
    logger.info(f"Calculating trends for years {years}...")

    # Filter to years of interest
    df_years = df[df['yearID'].isin(years)].copy()

    # Pivot to get WAR by year for each player
    war_pivot = df_years.pivot_table(
        index=player_col,
        columns='yearID',
        values='WAR',
        aggfunc='mean'
    ).reset_index()

    # Ensure all years are present
    for year in years:
        if year not in war_pivot.columns:
            war_pivot[year] = np.nan

    # Calculate trends
    trends = {}

    for _, row in war_pivot.iterrows():
        player = row[player_col]
        war_values = []

        for year in years:
            if year in row.index and pd.notna(row[year]):
                war_values.append(row[year])
            else:
                war_values.append(np.nan)

        # Need at least 2 years to calculate trend
        valid_values = [v for v in war_values if not np.isnan(v)]

        if len(valid_values) >= 2:
            # Simple linear trend: slope over available years
            x = list(range(len(valid_values)))
            y = valid_values

            if len(x) > 1 and np.std(x) > 0:
                # Calculate slope using simple formula
                x_mean = np.mean(x)
                y_mean = np.mean(y)

                numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
                denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))

                if denominator != 0:
                    slope = numerator / denominator
                else:
                    slope = 0

                # Normalize slope to a reasonable range
                # Positive slope = improving
                trend_score = np.clip(slope * 10, -5, 5)  # Scale factor
            else:
                trend_score = 0
        else:
            trend_score = 0

        trends[player] = trend_score

    logger.info(f"Calculated trends for {len(trends)} players")

    return trends


def calculate_composite_score(similarity, war_value, trend_score, age=None, weights=None):
    """
    Calculate final composite score (0-100) for ranking undervalued players.

    Args:
        similarity (float): Similarity to undervalued centroid (0-100)
        war_value (float): WAR value
        trend_score (float): Trend score (-5 to 5)
        age (float, optional): Player age
        weights (dict, optional): Feature weights

    Returns:
        float: Composite score (0-100)
    """
    if weights is None:
        # Opción A: Balanceada
        weights = {
            'similarity': 0.40,  # How similar to undervalued profile
            'war': 0.30,          # Current performance
            'trend': 0.20,        # Improving trajectory
            'age': 0.10            # Youth = upside (if age provided)
        }

    # Normalize WAR to 0-100 scale (assuming WAR typically 0-10)
    war_normalized = np.clip(war_value * 10, 0, 100)

    # Normalize trend (-5 to 5) to 0-100
    trend_normalized = np.clip((trend_score + 5) * 10, 0, 100)

    # Start with base score
    score = (
        weights['similarity'] * similarity +
        weights['war'] * war_normalized +
        weights['trend'] * trend_normalized
    )

    # Add age factor if provided
    if age is not None and 'age' in weights and weights['age'] > 0:
        # Younger = better (18-35 scale, 18 = 100, 35 = 0)
        age_factor = np.clip(100 * (35 - age) / 17, 0, 100)
        score += weights['age'] * age_factor
        score = score / (1 + weights['age'])  # Renormalize

    return np.clip(score, 0, 100)


def analyze_2025_players(df, player_info_with_clusters, centroid, feature_names):
    """
    Focused analysis on 2025 players with distance and similarity metrics.

    Args:
        df (DataFrame): Original full dataset
        player_info_with_clusters (DataFrame): Players with cluster assignments
        centroid (dict): Undervalued cluster centroid
        feature_names (list): Feature columns used

    Returns:
        DataFrame: 2025 players with all scores
    """
    logger.info("=" * 60)
    logger.info("FOCUSED ANALYSIS: 2025 PLAYERS ONLY")
    logger.info("=" * 60)

    # Filter to 2025 players
    players_2025 = player_info_with_clusters[player_info_with_clusters['yearID'] == 2025].copy()
    logger.info(f"Found {len(players_2025)} players in 2025")

    if len(players_2025) == 0:
        logger.warning("No 2025 players found in dataset!")
        return pd.DataFrame()

    # Get feature values for these players
    feature_cols = [col for col in feature_names if col in players_2025.columns]

    # Calculate distances and similarity scores
    distances, similarity = calculate_distances_to_centroid(
        players_2025[feature_cols],
        centroid
    )

    players_2025['distance_to_centroid'] = distances
    players_2025['similarity_score'] = similarity

    # Calculate trends
    trends = calculate_trends(df, years=[2023, 2024, 2025])
    players_2025['trend_score'] = players_2025['playerID'].map(trends).fillna(0)

    # Calculate composite score
    composite_scores = []

    for _, row in players_2025.iterrows():
        score = calculate_composite_score(
            similarity=row['similarity_score'],
            war_value=row['WAR'],
            trend_score=row['trend_score'],
            age=row.get('Age', None)
        )
        composite_scores.append(score)

    players_2025['composite_score'] = composite_scores

    # Sort by composite score (higher = better prospect)
    players_2025 = players_2025.sort_values('composite_score', ascending=False)

    logger.info(f"Top 5 players by composite score:")
    for i in range(min(5, len(players_2025))):
        row = players_2025.iloc[i]
        logger.info(f"  {i+1}. {row.get('Name', 'Unknown')}: Score={row['composite_score']:.1f}, "
                   f"WAR={row['WAR']:.2f}, Similarity={row['similarity_score']:.1f}")

    return players_2025


def save_centroid(centroid, output_path):
    """Save centroid to CSV for use in dashboard."""
    centroid_df = pd.DataFrame([centroid])
    centroid_df.to_csv(output_path, index=False)
    logger.info(f"Saved centroid to: {output_path}")


def save_2025_analysis(players_2025, output_path):
    """Save 2025 players with all scores to CSV."""
    # Select columns for output
    output_cols = ['playerID', 'Name', 'yearID', 'teamID', 'Age',
                   'WAR', 'salary', 'BABIP', 'wOBA', 'ISO', 'K%',
                   'cluster', 'distance_to_centroid', 'similarity_score',
                   'trend_score', 'composite_score']

    existing_cols = [col for col in output_cols if col in players_2025.columns]

    players_2025[existing_cols].to_csv(output_path, index=False)
    logger.info(f"Saved 2025 analysis to: {output_path}")


def save_top_undervalued(players_2025, n=50, output_path=None):
    """Save top N undervalued players (by composite score)."""
    if output_path is None:
        output_path = os.path.join(DATA_DIR, 'top_undervalued_2025.csv')

    top_n = players_2025.head(n).copy()

    # Select display columns
    display_cols = ['Name', 'teamID', 'Age', 'WAR', 'salary',
                    'similarity_score', 'trend_score', 'composite_score']
    existing_cols = [col for col in display_cols if col in top_n.columns]

    top_n[existing_cols].to_csv(output_path, index=False)
    logger.info(f"Saved top {n} undervalued 2025 players to: {output_path}")

    return top_n


def visualize_2025_results(players_2025, centroid, output_dir):
    """Create visualizations focused on 2025 players."""
    logger.info("Creating 2025-focused visualizations...")

    # 1. Distribution of composite scores
    plt.figure(figsize=(10, 6))
    plt.hist(players_2025['composite_score'], bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=players_2025['composite_score'].mean(), color='red',
                linestyle='--', label=f"Mean: {players_2025['composite_score'].mean():.1f}")
    plt.xlabel('Composite Score (0-100)', fontsize=12)
    plt.ylabel('Number of Players', fontsize=12)
    plt.title('Distribution of Composite Scores - 2025 Players', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2025_score_distribution.png'), dpi=150)
    plt.close()

    # 2. WAR vs Salary scatter (colored by composite score)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        players_2025['salary'],
        players_2025['WAR'],
        c=players_2025['composite_score'],
        cmap='viridis',
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    plt.colorbar(scatter, label='Composite Score')
    plt.xlabel('Salary (Millions $)', fontsize=12)
    plt.ylabel('WAR', fontsize=12)
    plt.title('2025 Players: WAR vs Salary (colored by score)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Mark centroid
    plt.scatter(centroid['salary'], centroid['WAR'],
                c='red', s=200, marker='X', edgecolors='white',
                linewidth=2, label='Undervalued Centroid')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2025_war_vs_salary.png'), dpi=150)
    plt.close()

    # 3. Top 10 players radar (comparison to centroid)
    from math import pi

    top10 = players_2025.head(10)
    metrics = ['WAR', 'BABIP', 'wOBA', 'ISO']
    metrics = [m for m in metrics if m in players_2025.columns]

    if len(metrics) >= 3:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()

        for i, (idx, player) in enumerate(top10.iterrows()):
            if i >= len(axes):
                break

            ax = axes[i]

            # Number of variables
            N = len(metrics)

            # Angle of each axis
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            # Player values
            values = [player[m] for m in metrics]
            # Normalize by centroid
            centroid_values = [centroid.get(m, 1) for m in metrics]
            values_normalized = [v / cv if cv > 0 else v for v, cv in zip(values, centroid_values)]
            values_normalized += values_normalized[:1]

            # Plot
            ax.plot(angles, values_normalized, 'o-', linewidth=2, color='blue')
            ax.fill(angles, values_normalized, alpha=0.25, color='blue')

            # Add centroid reference line (1.0)
            ax.plot(angles, [1]*len(angles), '--', color='red', alpha=0.5, linewidth=1)

            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=8)
            ax.set_ylim(0, 2)
            ax.set_title(f"{player.get('Name', 'Unknown')[:15]}", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2025_top10_radar.png'), dpi=150)
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """
    Main execution function for enhanced clustering analysis.

    NEW IN VERSION 2:
    - Saves undervalued cluster centroid
    - Calculates distances for 2025 players
    - Computes trends and composite scores
    - Focuses output on 2025 prospects
    """
    logger.info("=" * 70)
    logger.info("ENHANCED CLUSTERING ANALYSIS V2 - FOCUS ON 2025 PLAYERS")
    logger.info("=" * 70)

    try:
        # Step 1: Load data
        df = load_cleaned_data()

        # Step 2: Prepare features for clustering
        feature_df, player_info, feature_names = prepare_features_for_clustering(df)

        # Step 3: Determine optimal K
        best_k, k_results = determine_optimal_k(
            StandardScaler().fit_transform(feature_df),
            max_k=8
        )

        # Step 4: Apply K-Means
        cluster_labels, kmeans_model, scaler, feature_scaled = apply_kmeans_clustering(
            feature_df, best_k
        )

        # Step 5: Add cluster labels to player info
        player_info_with_clusters = player_info.copy()
        player_info_with_clusters['cluster'] = cluster_labels

        # Add feature values back
        for col in feature_names:
            player_info_with_clusters[col] = feature_df[col].values

        # Step 6: Identify undervalued cluster and get centroid
        undervalued_cluster, cluster_stats, centroid = identify_undervalued_cluster(
            feature_df, cluster_labels
        )

        # Step 7: Save centroid for later use
        centroid_path = os.path.join(ANALYSIS_DIR, 'undervalued_centroid.csv')
        save_centroid(centroid, centroid_path)

        # Step 8: Analyze 2025 players specifically
        players_2025 = analyze_2025_players(
            df, player_info_with_clusters, centroid, feature_names
        )

        if len(players_2025) > 0:
            # Step 9: Save 2025 analysis
            analysis_2025_path = os.path.join(ANALYSIS_DIR, 'players_2025_analysis.csv')
            save_2025_analysis(players_2025, analysis_2025_path)

            # Step 10: Save top undervalued 2025 players
            top_2025 = save_top_undervalued(players_2025, n=50)

            # Step 11: Visualize 2025 results
            visualize_2025_results(players_2025, centroid, ANALYSIS_DIR)

            # Step 12: Display top 10
            logger.info("=" * 70)
            logger.info("TOP 10 UNDERVALUED PLAYERS - 2025 SEASON")
            logger.info("=" * 70)

            display_cols = ['Name', 'teamID', 'Age', 'WAR', 'salary',
                           'similarity_score', 'trend_score', 'composite_score']
            existing_display = [col for col in display_cols if col in top_2025.columns]

            logger.info(f"\n{top_2025.head(10)[existing_display].to_string()}")

            # Step 13: Summary statistics for 2025
            logger.info("=" * 70)
            logger.info("2025 SUMMARY STATISTICS")
            logger.info("=" * 70)
            logger.info(f"Total 2025 players analyzed: {len(players_2025)}")
            logger.info(f"Average composite score: {players_2025['composite_score'].mean():.1f}")
            logger.info(f"Top score: {players_2025['composite_score'].max():.1f}")
            logger.info(f"Players with similarity > 80: {(players_2025['similarity_score'] > 80).sum()}")
        else:
            logger.warning("No 2025 players found. Check your data source.")

        # Save all players with clusters for reference
        all_players_path = os.path.join(ANALYSIS_DIR, 'all_players_with_clusters.csv')
        player_info_with_clusters.to_csv(all_players_path, index=False)

        # Save cluster statistics
        stats_path = os.path.join(ANALYSIS_DIR, 'cluster_statistics.csv')
        cluster_stats.to_csv(stats_path, index=False)

        logger.info("=" * 70)
        logger.info("ENHANCED ANALYSIS V2 COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info("Output files generated:")
        logger.info(f"  • Centroid: {centroid_path}")
        logger.info(f"  • All players with clusters: {all_players_path}")
        logger.info(f"  • Cluster statistics: {stats_path}")
        logger.info(f"  • 2025 players analysis: {analysis_2025_path}")
        logger.info(f"  • Top 50 undervalued 2025: {os.path.join(DATA_DIR, 'top_undervalued_2025.csv')}")
        logger.info(f"  • Visualizations: {ANALYSIS_DIR}/*.png")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error in enhanced analysis: {e}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    main()
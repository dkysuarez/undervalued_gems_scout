"""
app.py - Undervalued Gems Scout Interactive Dashboard
WITH DYNAMIC COLUMN DETECTION - CORREGIDO
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from datetime import datetime
import base64

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Undervalued Gems Scout",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a1928 0%, #1a3a4a 100%);
        min-height: 100vh;
    }
    
    .main-title {
        font-size: 3.8rem;
        background: linear-gradient(120deg, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0 5px 0;
        font-weight: 800;
        margin-bottom: 0px;
        animation: glow 3s infinite;
    }
    
    @keyframes glow {
        0% { text-shadow: 0 0 20px rgba(78, 205, 196, 0.3); }
        50% { text-shadow: 0 0 40px rgba(78, 205, 196, 0.6); }
        100% { text-shadow: 0 0 20px rgba(78, 205, 196, 0.3); }
    }
    
    .sub-title {
        text-align: center;
        color: #a8e6cf !important;
        font-size: 1.3rem;
        margin-bottom: 40px;
        font-style: italic;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 25px 15px;
        border: 1px solid rgba(78, 205, 196, 0.2);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: all 0.4s ease;
        height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin: 10px 0;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 45px rgba(78, 205, 196, 0.25);
        border: 1px solid rgba(78, 205, 196, 0.5);
        background: rgba(255, 255, 255, 0.1);
    }
    
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #a8e6cf !important;
        margin-bottom: 12px;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: white !important;
        line-height: 1.2;
        text-shadow: 0 0 20px rgba(78, 205, 196, 0.5);
        margin-bottom: 5px;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        color: #4ecdc4 !important;
    }
    
    .sidebar-header {
        color: #4ecdc4 !important;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 2px;
        margin-top: 20px;
        margin-bottom: 10px;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(78, 205, 196, 0.3);
        padding-bottom: 5px;
    }
    
    .footer {
        text-align: center;
        padding: 30px 20px 10px 20px;
        color: rgba(168, 230, 207, 0.6) !important;
        font-size: 0.9rem;
        border-top: 1px solid rgba(78, 205, 196, 0.15);
        margin-top: 60px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding: 10px 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 30px;
        padding: 10px 25px;
        color: #a8e6cf !important;
        border: 1px solid rgba(78, 205, 196, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4ecdc4, #45b7d1) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOADING SCREEN
# =============================================================================
def show_loading_screen():
    loading = st.empty()
    with loading.container():
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 500px; flex-direction: column;">
            <div style="width: 150px; height: 150px; position: relative; animation: bounce 1.2s infinite;">
                <div style="width: 100%; height: 100%; background: white; border-radius: 50%; 
                    position: relative; overflow: hidden; box-shadow: 0 0 40px rgba(78, 205, 196, 0.8); 
                    animation: roll 2.5s linear infinite;"></div>
            </div>
            <div style="margin-top: 40px; font-size: 1.5rem; color: #4ecdc4; animation: pulse 1.8s infinite;">
                UNDERVALUED GEMS SCOUT
            </div>
            <div style="color: #a8e6cf; margin-top: 20px;">Loading baseball analytics...</div>
        </div>
        <style>
            @keyframes roll { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            @keyframes bounce { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-30px); } }
            @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
        </style>
        """, unsafe_allow_html=True)

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress.progress(i + 1)

    loading.empty()
    progress.empty()

# =============================================================================
# LOAD ALL DATA
# =============================================================================
@st.cache_data
def load_all_data():
    """Load ALL available data files"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    analysis_dir = os.path.join(data_dir, 'analysis')

    data = {}

    # 1. Top undervalued 2025
    top_path = os.path.join(data_dir, 'top_undervalued_2025.csv')
    if os.path.exists(top_path):
        data['top_2025'] = pd.read_csv(top_path)
        st.sidebar.success(f"✅ Loaded top_2025: {len(data['top_2025'])} players")
    else:
        data['top_2025'] = pd.DataFrame()
        st.sidebar.warning("⚠️ top_undervalued_2025.csv not found")

    # 2. Full 2025 analysis
    full_path = os.path.join(analysis_dir, 'players_2025_analysis.csv')
    if os.path.exists(full_path):
        data['full_2025'] = pd.read_csv(full_path)
        st.sidebar.success(f"✅ Loaded full_2025: {len(data['full_2025'])} players")
    else:
        data['full_2025'] = pd.DataFrame()

    # 3. All historical players
    all_path = os.path.join(analysis_dir, 'all_players_with_clusters.csv')
    if os.path.exists(all_path):
        data['all_players'] = pd.read_csv(all_path)
        st.sidebar.success(f"✅ Loaded historical: {len(data['all_players'])} players")
    else:
        data['all_players'] = pd.DataFrame()

    # 4. Cluster statistics
    stats_path = os.path.join(analysis_dir, 'cluster_statistics.csv')
    if os.path.exists(stats_path):
        data['cluster_stats'] = pd.read_csv(stats_path)

    # 5. Centroid
    centroid_path = os.path.join(analysis_dir, 'undervalued_centroid.csv')
    if os.path.exists(centroid_path):
        data['centroid'] = pd.read_csv(centroid_path).iloc[0].to_dict()
    else:
        data['centroid'] = {}

    return data

# =============================================================================
# GET AVAILABLE COLUMNS (HELPER FUNCTION)
# =============================================================================
def get_available_columns(df, column_list):
    """Return only columns that exist in the dataframe"""
    return [col for col in column_list if col in df.columns]

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
def render_sidebar(data):
    with st.sidebar:
        st.markdown("## ⚾ Filters")
        st.markdown("---")

        # Dataset selector
        st.markdown('<div class="sidebar-header">📋 DATASET</div>', unsafe_allow_html=True)
        dataset_options = []
        if not data['top_2025'].empty:
            dataset_options.append("Top 50 - 2025")
        if not data['full_2025'].empty:
            dataset_options.append("All 2025 Players")
        if not data['all_players'].empty:
            dataset_options.append("Historical (All Years)")

        if not dataset_options:
            st.error("No datasets found!")
            return None

        selected_dataset = st.selectbox("Select Dataset", dataset_options)

        # Get current df
        if selected_dataset == "Top 50 - 2025":
            df = data['top_2025'].copy()
        elif selected_dataset == "All 2025 Players":
            df = data['full_2025'].copy()
        else:
            df = data['all_players'].copy()

        # Calculate WAR_per_M if possible
        if 'WAR' in df.columns and 'salary' in df.columns:
            df['WAR_per_M'] = df['WAR'] / (df['salary'] + 0.1)

        # Team filter - CHECK IF COLUMN EXISTS
        st.markdown('<div class="sidebar-header">🏟️ TEAM</div>', unsafe_allow_html=True)
        team_cols = get_available_columns(df, ['teamID', 'team', 'Team', 'teamId'])
        if team_cols:
            team_col = team_cols[0]
            teams = ['All Teams'] + sorted(df[team_col].dropna().unique().tolist())
            selected_team = st.selectbox("Team", teams)
        else:
            selected_team = 'All Teams'
            st.info("No team data available")

        # Year filter
        st.markdown('<div class="sidebar-header">📅 YEAR</div>', unsafe_allow_html=True)
        year_cols = get_available_columns(df, ['yearID', 'Year', 'year'])
        if year_cols:
            year_col = year_cols[0]
            years = sorted(df[year_col].unique())
            selected_year = st.selectbox("Year", ['All Years'] + years)
        else:
            selected_year = 'All Years'

        # Metric filters
        st.markdown('<div class="sidebar-header">📊 METRICS</div>', unsafe_allow_html=True)

        if 'WAR' in df.columns:
            min_war = st.slider("Min WAR", 0.0, float(df['WAR'].max()), 1.0, 0.5)
        else:
            min_war = 0

        if 'wOBA' in df.columns:
            wOBA_min = st.slider("Min wOBA", 0.200, 0.500, 0.300, 0.010, format="%.3f")
        else:
            wOBA_min = 0

        if 'BABIP' in df.columns:
            babip_min = st.slider("Min BABIP", 0.200, 0.400, 0.200, 0.010, format="%.3f")
        else:
            babip_min = 0

        # Price filters
        st.markdown('<div class="sidebar-header">💰 PRICE</div>', unsafe_allow_html=True)

        if 'salary' in df.columns:
            max_salary = st.slider("Max Salary ($M)", 0.0, float(df['salary'].max()), 2.0, 0.1)
        else:
            max_salary = 10

        if 'WAR_per_M' in df.columns:
            min_war_per_m = st.slider("Min WAR per $1M", 0.0, 20.0, 2.0, 0.5)
        else:
            min_war_per_m = 0

        # Sort by
        st.markdown('<div class="sidebar-header">📈 SORT BY</div>', unsafe_allow_html=True)
        sort_options = []
        if 'composite_score' in df.columns:
            sort_options.append('composite_score')
        if 'WAR' in df.columns:
            sort_options.append('WAR')
        if 'similarity_score' in df.columns:
            sort_options.append('similarity_score')
        if 'WAR_per_M' in df.columns:
            sort_options.append('WAR_per_M')

        if not sort_options:
            sort_options = [df.columns[0]]

        sort_by = st.selectbox("Sort by", sort_options)

        # Number of players
        top_n = st.number_input("Players to show", 5, 100, 20, 5)

        st.markdown("---")
        st.metric("Players in view", len(df))

        filters = {
            'dataset': selected_dataset,
            'df': df,
            'team': selected_team,
            'team_col': team_cols[0] if team_cols else None,
            'year': selected_year,
            'year_col': year_cols[0] if year_cols else None,
            'min_war': min_war,
            'wOBA_min': wOBA_min,
            'babip_min': babip_min,
            'max_salary': max_salary,
            'min_war_per_m': min_war_per_m,
            'sort_by': sort_by,
            'top_n': top_n
        }

        return filters

# =============================================================================
# APPLY FILTERS
# =============================================================================
def apply_filters(filters):
    if filters is None:
        return pd.DataFrame()

    df = filters['df'].copy()

    # Team filter
    if filters['team'] != 'All Teams' and filters['team_col'] and filters['team_col'] in df.columns:
        df = df[df[filters['team_col']] == filters['team']]

    # Year filter
    if filters['year'] != 'All Years' and filters['year_col'] and filters['year_col'] in df.columns:
        df = df[df[filters['year_col']] == filters['year']]

    # Metric filters
    if 'WAR' in df.columns:
        df = df[df['WAR'] >= filters['min_war']]

    if 'wOBA' in df.columns:
        df = df[df['wOBA'] >= filters['wOBA_min']]

    if 'BABIP' in df.columns:
        df = df[df['BABIP'] >= filters['babip_min']]

    if 'salary' in df.columns:
        df = df[df['salary'] <= filters['max_salary']]

    if 'WAR_per_M' in df.columns:
        df = df[df['WAR_per_M'] >= filters['min_war_per_m']]

    # Sort
    if filters['sort_by'] in df.columns:
        df = df.sort_values(filters['sort_by'], ascending=False)

    return df

# =============================================================================
# VISUALIZATIONS - WITH DYNAMIC COLUMN DETECTION
# =============================================================================
def create_scatter_plot(df, centroid):
    """WAR vs Salary scatter plot"""
    if 'WAR' not in df.columns or 'salary' not in df.columns:
        return go.Figure()

    # Build hover data dynamically
    hover_data = {}
    if 'teamID' in df.columns:
        hover_data['teamID'] = True
    if 'yearID' in df.columns:
        hover_data['yearID'] = True
    if 'wOBA' in df.columns:
        hover_data['wOBA'] = ':.3f'
    if 'BABIP' in df.columns:
        hover_data['BABIP'] = ':.3f'

    fig = px.scatter(
        df,
        x='salary',
        y='WAR',
        hover_name='Name' if 'Name' in df.columns else None,
        hover_data=hover_data,
        title='WAR vs Salary',
        labels={'salary': 'Salary ($M)', 'WAR': 'WAR'},
        color='WAR_per_M' if 'WAR_per_M' in df.columns else None,
        color_continuous_scale='Tealgrn',
        size='WAR' if 'WAR' in df.columns else None,
        size_max=15
    )

    # Add centroid if available
    if centroid and 'WAR' in centroid and 'salary' in centroid:
        fig.add_trace(
            go.Scatter(
                x=[centroid['salary']],
                y=[centroid['WAR']],
                mode='markers',
                marker=dict(symbol='x', size=20, color='red', line=dict(width=2, color='white')),
                name='Profile Centroid'
            )
        )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6f3ff',
        height=500
    )

    return fig

def create_radar_chart(player, centroid):
    """Radar chart comparing player to centroid"""
    categories = ['WAR', 'BABIP', 'wOBA', 'ISO']

    player_values = []
    centroid_values = []

    for cat in categories:
        if cat in player and cat in centroid:
            max_val = max(player[cat], centroid[cat])
            if max_val > 0:
                player_values.append(player[cat] / max_val * 100)
                centroid_values.append(centroid[cat] / max_val * 100)
            else:
                player_values.append(0)
                centroid_values.append(0)
        else:
            player_values.append(0)
            centroid_values.append(0)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=player_values,
        theta=categories,
        fill='toself',
        name=player.get('Name', 'Player'),
        line_color='#4ecdc4',
        fillcolor='rgba(78,205,196,0.3)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=centroid_values,
        theta=categories,
        fill='toself',
        name='Undervalued Profile',
        line_color='#ff6b6b',
        fillcolor='rgba(255,107,107,0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6f3ff',
        height=400
    )

    return fig

def create_trend_chart(df, top_n=10):
    """Bar chart of top players"""
    if 'Name' not in df.columns:
        return go.Figure()

    top_df = df.head(top_n).copy()
    y_col = 'composite_score' if 'composite_score' in df.columns else 'WAR' if 'WAR' in df.columns else df.columns[0]

    fig = px.bar(
        top_df,
        x='Name',
        y=y_col,
        color=y_col,
        title=f'Top {top_n} Players',
        labels={y_col: 'Score'},
        color_continuous_scale='Tealgrn'
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6f3ff',
        xaxis_tickangle=-45,
        height=400
    )

    return fig

def create_team_distribution(df):
    """Pie chart of team distribution"""
    team_cols = [col for col in ['teamID', 'team', 'Team'] if col in df.columns]
    if not team_cols:
        return go.Figure()

    team_col = team_cols[0]
    team_counts = df[team_col].value_counts().head(8)

    fig = px.pie(
        values=team_counts.values,
        names=team_counts.index,
        title='Top Teams',
        color_discrete_sequence=px.colors.sequential.Tealgrn
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6f3ff',
        height=400
    )

    return fig

def create_year_distribution(df):
    """Bar chart of year distribution"""
    year_cols = [col for col in ['yearID', 'Year', 'year'] if col in df.columns]
    if not year_cols:
        return go.Figure()

    year_col = year_cols[0]
    year_counts = df[year_col].value_counts().sort_index()

    fig = px.bar(
        x=year_counts.index,
        y=year_counts.values,
        title='Players by Year',
        labels={'x': 'Year', 'y': 'Count'},
        color=year_counts.values,
        color_continuous_scale='Tealgrn'
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6f3ff',
        height=400
    )

    return fig

def create_correlation_heatmap(df):
    """Correlation heatmap of metrics"""
    corr_cols = ['WAR', 'salary', 'BABIP', 'wOBA', 'ISO', 'K%',
                 'similarity_score', 'composite_score', 'WAR_per_M']
    corr_cols = [col for col in corr_cols if col in df.columns]

    if len(corr_cols) < 2:
        return go.Figure()

    corr_matrix = df[corr_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='Tealgrn',
        title="Metric Correlations"
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6f3ff',
        height=500
    )

    return fig

def create_histogram(df, column, title):
    """Create histogram for any column"""
    if column not in df.columns:
        return go.Figure()

    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=title,
        color_discrete_sequence=['#4ecdc4']
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6f3ff'
    )
    return fig

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    show_loading_screen()

    with st.spinner(''):
        data = load_all_data()

    st.markdown('<h1 class="main-title">⚾ UNDERVALUED GEMS SCOUT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Complete Baseball Analytics Platform</p>', unsafe_allow_html=True)

    filters = render_sidebar(data)

    if filters is None:
        st.error("No data available. Please check your data files.")
        return

    filtered_df = apply_filters(filters)

    if filtered_df.empty:
        st.warning("No players match the selected filters.")
        return

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">PLAYERS</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(filtered_df)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-delta">in view</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">AVG WAR</div>', unsafe_allow_html=True)
        if 'WAR' in filtered_df.columns and len(filtered_df) > 0:
            st.markdown(f'<div class="metric-value">{filtered_df["WAR"].mean():.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-delta">Max: {filtered_df["WAR"].max():.1f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">AVG WAR/$M</div>', unsafe_allow_html=True)
        if 'WAR_per_M' in filtered_df.columns and len(filtered_df) > 0:
            st.markdown(f'<div class="metric-value">{filtered_df["WAR_per_M"].mean():.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-delta">Best: {filtered_df["WAR_per_M"].max():.1f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">TEAMS</div>', unsafe_allow_html=True)
        team_cols = [col for col in ['teamID', 'team', 'Team'] if col in filtered_df.columns]
        if team_cols:
            team_col = team_cols[0]
            st.markdown(f'<div class="metric-value">{filtered_df[team_col].nunique()}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-delta">represented</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Player Rankings",
        "📈 Visual Analytics",
        "🎯 Player Deep Dive",
        "📊 Correlations",
        "📤 Export Data"
    ])

    with tab1:
        st.header("🏆 Player Rankings")

        # Table
        display_cols = []
        for col in ['Name', 'teamID', 'yearID', 'WAR', 'salary', 'wOBA', 'BABIP', 'ISO', 'WAR_per_M', 'composite_score']:
            if col in filtered_df.columns:
                display_cols.append(col)

        if display_cols:
            display_df = filtered_df[display_cols].head(filters['top_n']).copy()

            # Format
            if 'salary' in display_df.columns:
                display_df['salary'] = display_df['salary'].apply(lambda x: f"${x:.2f}M")
            if 'WAR_per_M' in display_df.columns:
                display_df['WAR_per_M'] = display_df['WAR_per_M'].apply(lambda x: f"{x:.2f}")

            st.dataframe(display_df, use_container_width=True, height=400)

        # Bar chart
        st.plotly_chart(create_trend_chart(filtered_df, filters['top_n']), use_container_width=True)

    with tab2:
        st.header("📈 Visual Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_scatter_plot(filtered_df, data['centroid']), use_container_width=True)

        with col2:
            st.plotly_chart(create_team_distribution(filtered_df), use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(create_year_distribution(filtered_df), use_container_width=True)

        with col4:
            if 'WAR_per_M' in filtered_df.columns:
                st.plotly_chart(create_histogram(filtered_df, 'WAR_per_M', 'WAR per $1M Distribution'), use_container_width=True)

    with tab3:
        st.header("🎯 Player Deep Dive")

        if 'Name' in filtered_df.columns and len(filtered_df) > 0:
            selected = st.selectbox("Select Player", filtered_df['Name'].tolist())

            if selected:
                player = filtered_df[filtered_df['Name'] == selected].iloc[0]

                col1, col2 = st.columns([1, 1.5])

                with col1:
                    st.markdown('<div class="metric-card" style="height: auto;">', unsafe_allow_html=True)
                    st.subheader(f"📋 {selected}")

                    # Show available metrics
                    if 'teamID' in player:
                        st.write(f"**Team:** {player['teamID']}")
                    if 'yearID' in player:
                        st.write(f"**Year:** {player['yearID']}")
                    if 'WAR' in player:
                        st.write(f"**WAR:** {player['WAR']:.2f}")
                    if 'salary' in player:
                        st.write(f"**Salary:** ${player['salary']:.2f}M")
                    if 'wOBA' in player:
                        st.write(f"**wOBA:** {player['wOBA']:.3f}")
                    if 'BABIP' in player:
                        st.write(f"**BABIP:** {player['BABIP']:.3f}")
                    if 'ISO' in player:
                        st.write(f"**ISO:** {player['ISO']:.3f}")
                    if 'composite_score' in player:
                        st.write(f"**Score:** {player['composite_score']:.1f}")

                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    if data['centroid']:
                        fig = create_radar_chart(player, data['centroid'])
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a player to see detailed analysis")

    with tab4:
        st.header("📊 Correlation Analysis")
        st.plotly_chart(create_correlation_heatmap(filtered_df), use_container_width=True)

        if not data['cluster_stats'].empty:
            st.subheader("Cluster Statistics")
            st.dataframe(data['cluster_stats'], use_container_width=True)

    with tab5:
        st.header("📤 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="metric-card" style="height: auto;">', unsafe_allow_html=True)
            st.subheader("Export Current View")

            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="baseball_data.csv" style="color: #4ecdc4;">📥 Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.markdown(f"**Rows:** {len(filtered_df)}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card" style="height: auto;">', unsafe_allow_html=True)
            st.subheader("Dataset Info")
            st.write(f"**Dataset:** {filters['dataset']}")
            st.write(f"**Total:** {len(filters['df'])} players")
            st.write(f"**Filtered:** {len(filtered_df)} players")
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    total_players = len(data.get('all_players', [])) + len(data.get('full_2025', []))
    st.markdown(f"""
    <div class="footer">
        <p>⚾ Undervalued Gems Scout - Complete Baseball Analytics Platform</p>
        <p>Data: Lahman Database, FanGraphs • Model: K-Means Clustering</p>
        <p>📊 {total_players} total players • {len(data.get('full_2025', []))} 2025 players</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
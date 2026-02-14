"""
app.py - Undervalued Gems Scout Interactive Dashboard
WITH DYNAMIC COLUMN DETECTION - PROFESSIONAL LOADING SCREEN
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
# PAGE CONFIGURATION - MUST BE FIRST
# =============================================================================
st.set_page_config(
    page_title="Undervalued Gems Scout",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PROFESSIONAL LOADING SCREEN
# =============================================================================
def show_professional_loading_screen():
    """Professional full-screen loading animation"""

    # CSS for loading screen
    st.markdown("""
    <style>
    /* HIDE ALL STREAMLIT ELEMENTS DURING LOADING */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* FULLSCREEN LOADING OVERLAY */
    .loading-fullscreen {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #0a1928 0%, #1a3a4a 100%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        color: white;
    }
    
    /* WELCOME MESSAGE */
    .welcome-message {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 10px;
        text-align: center;
        background: linear-gradient(120deg, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: glow 3s infinite;
    }
    
    @keyframes glow {
        0% { text-shadow: 0 0 20px rgba(78, 205, 196, 0.3); }
        50% { text-shadow: 0 0 40px rgba(78, 205, 196, 0.6); }
        100% { text-shadow: 0 0 20px rgba(78, 205, 196, 0.3); }
    }
    
    .welcome-subtitle {
        font-size: 20px;
        font-weight: 300;
        margin-bottom: 40px;
        text-align: center;
        color: #a8e6cf;
        opacity: 0.9;
        font-style: italic;
    }
    
    .pulse-container { 
        display: flex; 
        justify-content: center; 
        align-items: center; 
        margin: 30px 0; 
    }
    
    .pulse-circle {
        width: 120px; 
        height: 120px; 
        background: linear-gradient(135deg, #4ecdc4, #45b7d1);
        border-radius: 50%; 
        position: relative;
        animation: pulse 2s infinite; 
        display: flex; 
        align-items: center; 
        justify-content: center;
        font-size: 60px; 
        color: white;
        box-shadow: 0 10px 30px rgba(78, 205, 196, 0.5);
    }
    
    .pulse-circle::before, .pulse-circle::after {
        content: ''; 
        position: absolute; 
        border: 2px solid #4ecdc4; 
        border-radius: 50%;
        width: 100%; 
        height: 100%; 
        animation: ripple 2s infinite;
        opacity: 0.7;
    }
    
    .pulse-circle::after { 
        animation-delay: 0.5s; 
    }
    
    @keyframes pulse { 
        0% { 
            transform: scale(0.95); 
            box-shadow: 0 0 0 0 rgba(78, 205, 196, 0.7); 
        }
        70% { 
            transform: scale(1); 
            box-shadow: 0 0 0 30px rgba(78, 205, 196, 0); 
        }
        100% { 
            transform: scale(0.95); 
            box-shadow: 0 0 0 0 rgba(78, 205, 196, 0); 
        } 
    }
    
    @keyframes ripple { 
        0% { 
            transform: scale(1); 
            opacity: 1; 
        } 
        100% { 
            transform: scale(2); 
            opacity: 0; 
        } 
    }
    
    .loading-text { 
        font-size: 24px; 
        font-weight: 300; 
        margin-top: 30px; 
        text-align: center; 
        color: #4ecdc4;
    }
    
    .progress-container { 
        width: 500px; 
        height: 8px; 
        background: rgba(255,255,255,0.1); 
        border-radius: 4px; 
        margin-top: 40px; 
        overflow: hidden; 
        border: 1px solid rgba(78, 205, 196, 0.3);
    }
    
    .progress-bar { 
        height: 100%; 
        background: linear-gradient(90deg, #4ecdc4, #45b7d1); 
        width: 0%; 
        border-radius: 4px; 
        animation: progress 2.5s ease-in-out forwards; 
        box-shadow: 0 0 20px rgba(78, 205, 196, 0.8);
    }
    
    @keyframes progress { 
        0% { 
            width: 0%; 
        } 
        100% { 
            width: 100%; 
        } 
    }
    
    .steps-container {
        display: flex;
        gap: 30px;
        margin-top: 30px;
        color: #a8e6cf;
        font-size: 14px;
    }
    
    .step {
        text-align: center;
        padding: 10px 20px;
        border-radius: 30px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(78, 205, 196, 0.2);
    }
    
    .step-active {
        background: rgba(78, 205, 196, 0.2);
        border-color: #4ecdc4;
        color: #4ecdc4;
        font-weight: 600;
    }
    
    /* HIDE CONTENT WHILE LOADING */
    .app-content {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

    # Loading animation container
    loading_placeholder = st.empty()

    with loading_placeholder.container():
        st.markdown("""
        <div class='loading-fullscreen'>
            <div class='welcome-message'>⚾ UNDERVALUED GEMS SCOUT</div>
            <div class='welcome-subtitle'>Advanced Baseball Analytics Platform</div>
            <div class='pulse-container'>
                <div class='pulse-circle'>⚾</div>
            </div>
            <div class='loading-text' id="loading-text">Initializing analytics engine...</div>
            <div class='progress-container'>
                <div class='progress-bar'></div>
            </div>
            <div class='steps-container'>
                <div class='step' id="step1">📊 Loading Data</div>
                <div class='step' id="step2">⚙️ Processing</div>
                <div class='step' id="step3">📈 Preparing Visualizations</div>
                <div class='step' id="step4">🚀 Ready</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Simulate loading steps with delays
        time.sleep(1)

        # Update step 1
        st.markdown("""
        <script>
        setTimeout(function() {
            document.getElementById('step1').style.background = 'rgba(78,205,196,0.2)';
            document.getElementById('step1').style.borderColor = '#4ecdc4';
            document.getElementById('loading-text').innerText = 'Loading player data...';
        }, 1000);
        </script>
        """, unsafe_allow_html=True)

        time.sleep(1)

        # Update step 2
        st.markdown("""
        <script>
        setTimeout(function() {
            document.getElementById('step2').style.background = 'rgba(78,205,196,0.2)';
            document.getElementById('step2').style.borderColor = '#4ecdc4';
            document.getElementById('loading-text').innerText = 'Processing statistics...';
        }, 2000);
        </script>
        """, unsafe_allow_html=True)

        time.sleep(1)

        # Update step 3
        st.markdown("""
        <script>
        setTimeout(function() {
            document.getElementById('step3').style.background = 'rgba(78,205,196,0.2)';
            document.getElementById('step3').style.borderColor = '#4ecdc4';
            document.getElementById('loading-text').innerText = 'Creating visualizations...';
        }, 3000);
        </script>
        """, unsafe_allow_html=True)

        time.sleep(1)

    # Clear loading screen
    loading_placeholder.empty()

    # Show brief success message
    success_msg = st.empty()
    success_msg.success("✅ System loaded successfully | Undervalued Gems Scout")
    time.sleep(1.5)
    success_msg.empty()


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
    
    /* Animation for content after loading */
    .app-content {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Estilo para la pelota - mantiene su color original */
    .baseball-icon {
        display: inline-block;
        background: none !important;
        -webkit-text-fill-color: initial !important;
        color: initial !important;
        filter: drop-shadow(0 0 10px rgba(78, 205, 196, 0.5));
        margin-right: 5px;
    }
    
    /* Títulos de gráficos en blanco con glow */
    .chart-title {
        color: white !important;
        font-size: 1.4rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 15px;
        margin-top: 10px;
        text-shadow: 0 0 15px rgba(78, 205, 196, 0.5);
        letter-spacing: 1px;
        border-bottom: 1px solid rgba(78, 205, 196, 0.3);
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


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

        selected_dataset = st.selectbox("Select Dataset", dataset_options, key="dataset_select")

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

        # Team filter
        st.markdown('<div class="sidebar-header">🏟️ TEAM</div>', unsafe_allow_html=True)
        team_cols = get_available_columns(df, ['teamID', 'team', 'Team', 'teamId'])
        if team_cols:
            team_col = team_cols[0]
            teams = ['All Teams'] + sorted(df[team_col].dropna().unique().tolist())
            selected_team = st.selectbox("Team", teams, key="team_select")
        else:
            selected_team = 'All Teams'
            st.info("No team data available")

        # Year filter
        st.markdown('<div class="sidebar-header">📅 YEAR</div>', unsafe_allow_html=True)
        year_cols = get_available_columns(df, ['yearID', 'Year', 'year'])
        if year_cols:
            year_col = year_cols[0]
            years = sorted(df[year_col].unique())
            selected_year = st.selectbox("Year", ['All Years'] + years, key="year_select")
        else:
            selected_year = 'All Years'

        # Metric filters
        st.markdown('<div class="sidebar-header">📊 METRICS</div>', unsafe_allow_html=True)

        if 'WAR' in df.columns:
            min_war = st.slider("Min WAR", 0.0, float(df['WAR'].max()), 1.0, 0.5, key="war_slider")
        else:
            min_war = 0

        if 'wOBA' in df.columns:
            wOBA_min = st.slider("Min wOBA", 0.200, 0.500, 0.300, 0.010, format="%.3f", key="woba_slider")
        else:
            wOBA_min = 0

        if 'BABIP' in df.columns:
            babip_min = st.slider("Min BABIP", 0.200, 0.400, 0.200, 0.010, format="%.3f", key="babip_slider")
        else:
            babip_min = 0

        # Price filters
        st.markdown('<div class="sidebar-header">💰 PRICE</div>', unsafe_allow_html=True)

        if 'salary' in df.columns:
            max_salary = st.slider("Max Salary ($M)", 0.0, float(df['salary'].max()), 2.0, 0.1, key="salary_slider")
        else:
            max_salary = 10

        if 'WAR_per_M' in df.columns:
            min_war_per_m = st.slider("Min WAR per $1M", 0.0, 20.0, 2.0, 0.5, key="war_per_m_slider")
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

        sort_by = st.selectbox("Sort by", sort_options, key="sort_select")

        # Number of players
        top_n = st.number_input("Players to show", 5, 100, 20, 5, key="top_n_input")

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
    """WAR vs Salary scatter plot - SIN TÍTULO INTERNO"""
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
        title=None,
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
        font_color='white',
        height=500,
        xaxis=dict(title_font_color='white', tickfont_color='white'),
        yaxis=dict(title_font_color='white', tickfont_color='white'),
        coloraxis_colorbar=dict(title_font_color='white', tickfont_color='white')
    )

    return fig


def create_radar_chart(player, centroid):
    """Radar chart comparing player to centroid - CON TEXTO BLANCO"""
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

    # Traza del jugador
    fig.add_trace(go.Scatterpolar(
        r=player_values,
        theta=categories,
        fill='toself',
        name=player.get('Name', 'Player'),
        line_color='#4ecdc4',
        fillcolor='rgba(78,205,196,0.3)'
    ))

    # Traza del perfil undervalued
    fig.add_trace(go.Scatterpolar(
        r=centroid_values,
        theta=categories,
        fill='toself',
        name='Undervalued Profile',
        line_color='#ff6b6b',
        fillcolor='rgba(255,107,107,0.3)'
    ))

    # Configuración del layout con TODO en blanco
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='white', size=10),  # Números del eje en blanco
                gridcolor='rgba(255,255,255,0.2)',  # Grid más suave
                linecolor='rgba(255,255,255,0.3)'  # Línea del eje
            ),
            angularaxis=dict(
                tickfont=dict(color='white', size=11, weight='bold'),  # Categorías en blanco
                gridcolor='rgba(255,255,255,0.2)',
                linecolor='rgba(255,255,255,0.3)'
            ),
            bgcolor='rgba(0,0,0,0)'  # Fondo transparente
        ),
        showlegend=True,
        legend=dict(
            font=dict(color='white', size=11),  # Leyenda en blanco
            bgcolor='rgba(0,0,0,0.5)',  # Fondo semi-transparente para la leyenda
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            x=0.8,
            y=1.1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  # Texto general en blanco
        height=450,
        margin=dict(l=80, r=80, t=40, b=40)  # Márgenes para mejor visualización
    )

    # Actualizar colores de las líneas de la cuadrícula y ejes
    fig.update_polars(
        radialaxis_gridcolor='rgba(255,255,255,0.15)',
        angularaxis_gridcolor='rgba(255,255,255,0.15)'
    )

    return fig


def create_trend_chart(df, top_n=10):
    """Bar chart of top players - SIN TÍTULO INTERNO Y CON TEXTO BLANCO"""
    if 'Name' not in df.columns:
        return go.Figure()

    top_df = df.head(top_n).copy()
    y_col = 'composite_score' if 'composite_score' in df.columns else 'WAR' if 'WAR' in df.columns else df.columns[0]

    fig = px.bar(
        top_df,
        x='Name',
        y=y_col,
        color=y_col,
        title=None,  # QUITAMOS EL TÍTULO INTERNO
        labels={y_col: 'Score', 'Name': 'Player'},
        color_continuous_scale='Tealgrn',
        text=y_col  # Mostrar el valor encima de cada barra
    )

    fig.update_traces(
        texttemplate='%{text:.2f}',  # Formato del texto en las barras
        textposition='outside',       # Posición del texto
        textfont=dict(color='white', size=10),  # Texto de las barras en blanco
        marker_line_color='rgba(255,255,255,0.2)',  # Borde de las barras
        marker_line_width=1,
        hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>'  # Tooltip personalizado
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  # TODO el texto en blanco
        height=450,
        margin=dict(l=50, r=50, t=30, b=100),  # Márgenes ajustados
        xaxis=dict(
            title=dict(text='Player', font=dict(color='white')),  # Título del eje X en blanco
            tickfont=dict(color='white', size=10),  # Nombres de jugadores en blanco
            tickangle=-45,  # Rotación para que se lean mejor
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.3)'
        ),
        yaxis=dict(
            title=dict(text='Score', font=dict(color='white')),  # Título del eje Y en blanco
            tickfont=dict(color='white'),  # Números del eje Y en blanco
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.3)'
        ),
        coloraxis_colorbar=dict(
            title=dict(text='Score', font=dict(color='white')),  # Barra de color
            tickfont=dict(color='white'),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)'
        ),
        hoverlabel=dict(
            bgcolor='#1a3a4a',  # Fondo del tooltip
            font=dict(color='white', size=12),  # Texto del tooltip en blanco
            bordercolor='#4ecdc4'
        )
    )

    return fig


def create_team_distribution(df):
    """Pie chart of team distribution - SIN TÍTULO INTERNO"""
    team_cols = [col for col in ['teamID', 'team', 'Team'] if col in df.columns]
    if not team_cols:
        return go.Figure()

    team_col = team_cols[0]
    team_counts = df[team_col].value_counts().head(8)

    fig = px.pie(
        values=team_counts.values,
        names=team_counts.index,
        title=None,
        color_discrete_sequence=px.colors.sequential.Tealgrn,
        hole=0.3
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400,
        legend=dict(font_color='white')
    )

    fig.update_traces(textfont_color='white')

    return fig


def create_year_distribution(df):
    """Bar chart of year distribution - SIN TÍTULO INTERNO"""
    year_cols = [col for col in ['yearID', 'Year', 'year'] if col in df.columns]
    if not year_cols:
        return go.Figure()

    year_col = year_cols[0]
    year_counts = df[year_col].value_counts().sort_index()

    fig = px.bar(
        x=year_counts.index,
        y=year_counts.values,
        title=None,
        labels={'x': 'Year', 'y': 'Count'},
        color=year_counts.values,
        color_continuous_scale='Tealgrn'
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400,
        xaxis=dict(title_font_color='white', tickfont_color='white'),
        yaxis=dict(title_font_color='white', tickfont_color='white'),
        coloraxis_colorbar=dict(title_font_color='white', tickfont_color='white')
    )

    return fig


def create_correlation_heatmap(df):
    """Correlation heatmap of metrics - CON TEXTO BLANCO"""
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
        title=None  # Sin título interno
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),  # Texto general en blanco
        height=500,
        xaxis=dict(
            tickfont=dict(color='white'),  # Etiquetas del eje X en blanco
            title_font=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            tickfont=dict(color='white'),  # Etiquetas del eje Y en blanco
            title_font=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        coloraxis_colorbar=dict(
            title_font=dict(color='white'),  # Título de la barra de color en blanco
            tickfont=dict(color='white'),  # Números de la barra en blanco
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)'
        )
    )

    # Actualizar el color de los números dentro de las celdas
    fig.update_traces(
        textfont=dict(color='white', size=10),  # Números en blanco
        hoverinfo='none'
    )

    return fig


def create_histogram(df, column, title=None):
    """Create histogram for any column - SIN TÍTULO INTERNO"""
    if column not in df.columns:
        return go.Figure()

    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        title=None,
        color_discrete_sequence=['#4ecdc4'],
        labels={column: column.replace('_', ' ')}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=400,
        xaxis=dict(title_font_color='white', tickfont_color='white'),
        yaxis=dict(title_font_color='white', tickfont_color='white')
    )
    return fig


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # ========== INITIAL LOADING SCREEN ==========
    if 'initial_loaded' not in st.session_state:
        # Show professional loading screen
        show_professional_loading_screen()
        st.session_state.initial_loaded = True
        st.rerun()

    # ========== LOAD DATA ==========
    with st.spinner('Loading data...'):
        data = load_all_data()

    # ========== MAIN TITLE ==========
    st.markdown("""
    <h1 class="main-title">
        <span class="baseball-icon">⚾</span> UNDERVALUED GEMS SCOUT
    </h1>
    """, unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Complete Baseball Analytics Platform</p>', unsafe_allow_html=True)

    # ========== SIDEBAR FILTERS ==========
    filters = render_sidebar(data)

    if filters is None:
        st.error("No data available. Please check your data files.")
        return

    # ========== APPLY FILTERS ==========
    filtered_df = apply_filters(filters)

    if filtered_df.empty:
        st.warning("No players match the selected filters.")
        return

    # ========== WRAP CONTENT FOR ANIMATION ==========
    st.markdown('<div class="app-content">', unsafe_allow_html=True)

    # ========== METRICS ROW ==========
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>PLAYERS</div>
            <div class='metric-value'>{len(filtered_df)}</div>
            <div class='metric-delta'>in view</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if 'WAR' in filtered_df.columns and len(filtered_df) > 0:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>AVG WAR</div>
                <div class='metric-value'>{filtered_df['WAR'].mean():.2f}</div>
                <div class='metric-delta'>Max: {filtered_df['WAR'].max():.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>AVG WAR</div>
                <div class='metric-value'>N/A</div>
                <div class='metric-delta'>—</div>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        if 'WAR_per_M' in filtered_df.columns and len(filtered_df) > 0:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>AVG WAR/$M</div>
                <div class='metric-value'>{filtered_df['WAR_per_M'].mean():.2f}</div>
                <div class='metric-delta'>Best: {filtered_df['WAR_per_M'].max():.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>AVG WAR/$M</div>
                <div class='metric-value'>N/A</div>
                <div class='metric-delta'>—</div>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        team_cols = [col for col in ['teamID', 'team', 'Team'] if col in filtered_df.columns]
        if team_cols:
            team_col = team_cols[0]
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>TEAMS</div>
                <div class='metric-value'>{filtered_df[team_col].nunique()}</div>
                <div class='metric-delta'>represented</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>TEAMS</div>
                <div class='metric-value'>N/A</div>
                <div class='metric-delta'>—</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ========== TABS WITH UNIQUE KEYS ==========
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Player Rankings",
        "📈 Visual Analytics",
        "🎯 Player Deep Dive",
        "📊 Correlations",
        "📤 Export Data"
    ])

    with tab1:
        st.markdown("""
        <h2 style="color: #a8e6cf; border-bottom: 2px solid rgba(78, 205, 196, 0.3); padding-bottom: 8px;">
            🏆 Player Rankings
        </h2>
        """, unsafe_allow_html=True)

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

        # Bar chart with unique key
        st.plotly_chart(create_trend_chart(filtered_df, filters['top_n']),
                       use_container_width=True,
                       key="trend_chart_main")

    with tab2:
        st.markdown("""
        <h2 style="color: #a8e6cf; border-bottom: 2px solid rgba(78, 205, 196, 0.3); padding-bottom: 8px; margin-bottom: 20px;">
            📈 Visual Analytics
        </h2>
        """, unsafe_allow_html=True)

        # Verificar qué gráficos están disponibles
        has_scatter = 'WAR' in filtered_df.columns and 'salary' in filtered_df.columns
        has_year = len([col for col in ['yearID', 'Year', 'year'] if col in filtered_df.columns]) > 0
        has_war_per_m = 'WAR_per_M' in filtered_df.columns
        has_team = len([col for col in ['teamID', 'team', 'Team'] if col in filtered_df.columns]) > 0

        # PRIMERA FILA - WAR vs Salary
        if has_scatter:
            st.markdown('<p class="chart-title">⚾ WAR vs Salary</p>', unsafe_allow_html=True)
            st.plotly_chart(create_scatter_plot(filtered_df, data['centroid']),
                           use_container_width=True,
                           key="scatter_plot_main")
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.markdown('<p class="chart-title">⚾ WAR vs Salary</p>', unsafe_allow_html=True)
            st.info("WAR and Salary data not available for this dataset")
            st.markdown("<br>", unsafe_allow_html=True)

        # SEGUNDA FILA - Dos columnas
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p class="chart-title">📅 Players by Year</p>', unsafe_allow_html=True)
            if has_year:
                st.plotly_chart(create_year_distribution(filtered_df),
                               use_container_width=True,
                               key="year_dist_main")
            else:
                st.info("No year data available")

        with col2:
            st.markdown('<p class="chart-title">💰 WAR per $1M Distribution</p>', unsafe_allow_html=True)
            if has_war_per_m:
                st.plotly_chart(create_histogram(filtered_df, 'WAR_per_M'),
                               use_container_width=True,
                               key="hist_war_main")
            else:
                st.info("No WAR per $1M data available")

        # TERCERA FILA - Team Distribution
        if has_team:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="chart-title">🏟️ Top Teams Distribution</p>', unsafe_allow_html=True)
            st.plotly_chart(create_team_distribution(filtered_df),
                           use_container_width=True,
                           key="team_dist_main")

    with tab3:
        st.markdown("""
        <h2 style="color: #a8e6cf; border-bottom: 2px solid rgba(78, 205, 196, 0.3); padding-bottom: 8px; margin-bottom: 20px;">
            🎯 Player Deep Dive
        </h2>
        """, unsafe_allow_html=True)

        if 'Name' in filtered_df.columns and len(filtered_df) > 0:
            selected = st.selectbox("Select Player", filtered_df['Name'].tolist(), key="player_select_dive")

            if selected:
                player = filtered_df[filtered_df['Name'] == selected].iloc[0]

                col1, col2 = st.columns([1, 1.5])

                with col1:
                    # SOLO el título del jugador - nada más
                    st.markdown(f"""
                    <h3 style="color: white; text-shadow: 0 0 10px rgba(78, 205, 196, 0.5); margin-bottom: 20px;">
                        📋 {selected}
                    </h3>
                    """, unsafe_allow_html=True)

                    # Crear un contenedor limpio para las métricas
                    metrics_container = st.container()

                    with metrics_container:
                        # Team (solo si existe)
                        if 'teamID' in player and pd.notna(player['teamID']):
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.06); border-radius: 10px; padding: 10px; border-left: 4px solid #4ecdc4; margin-bottom: 8px;'>
                                <span style='color: #a8e6cf; font-size: 0.8rem; text-transform: uppercase;'>TEAM</span>
                                <div style='color: white; font-size: 1.2rem; font-weight: 600;'>{player['teamID']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Year (solo si existe)
                        if 'yearID' in player and pd.notna(player['yearID']):
                            year_val = int(player['yearID']) if isinstance(player['yearID'], (int, float)) else player[
                                'yearID']
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.06); border-radius: 10px; padding: 10px; border-left: 4px solid #4ecdc4; margin-bottom: 8px;'>
                                <span style='color: #a8e6cf; font-size: 0.8rem; text-transform: uppercase;'>YEAR</span>
                                <div style='color: white; font-size: 1.2rem; font-weight: 600;'>{year_val}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # WAR
                        if 'WAR' in player and pd.notna(player['WAR']):
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.06); border-radius: 10px; padding: 10px; border-left: 4px solid #4ecdc4; margin-bottom: 8px;'>
                                <span style='color: #a8e6cf; font-size: 0.8rem; text-transform: uppercase;'>WAR</span>
                                <div style='color: white; font-size: 1.2rem; font-weight: 600;'>{player['WAR']:.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Salary
                        if 'salary' in player and pd.notna(player['salary']):
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.06); border-radius: 10px; padding: 10px; border-left: 4px solid #4ecdc4; margin-bottom: 8px;'>
                                <span style='color: #a8e6cf; font-size: 0.8rem; text-transform: uppercase;'>SALARY</span>
                                <div style='color: white; font-size: 1.2rem; font-weight: 600;'>${player['salary']:.2f}M</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # wOBA
                        if 'wOBA' in player and pd.notna(player['wOBA']):
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.06); border-radius: 10px; padding: 10px; border-left: 4px solid #4ecdc4; margin-bottom: 8px;'>
                                <span style='color: #a8e6cf; font-size: 0.8rem; text-transform: uppercase;'>wOBA</span>
                                <div style='color: white; font-size: 1.2rem; font-weight: 600;'>{player['wOBA']:.3f}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # BABIP
                        if 'BABIP' in player and pd.notna(player['BABIP']):
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.06); border-radius: 10px; padding: 10px; border-left: 4px solid #4ecdc4; margin-bottom: 8px;'>
                                <span style='color: #a8e6cf; font-size: 0.8rem; text-transform: uppercase;'>BABIP</span>
                                <div style='color: white; font-size: 1.2rem; font-weight: 600;'>{player['BABIP']:.3f}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # ISO
                        if 'ISO' in player and pd.notna(player['ISO']):
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.06); border-radius: 10px; padding: 10px; border-left: 4px solid #4ecdc4; margin-bottom: 8px;'>
                                <span style='color: #a8e6cf; font-size: 0.8rem; text-transform: uppercase;'>ISO</span>
                                <div style='color: white; font-size: 1.2rem; font-weight: 600;'>{player['ISO']:.3f}</div>
                            </div>
                            """, unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <h3 style="color: white; text-shadow: 0 0 10px rgba(78, 205, 196, 0.5); margin-bottom: 20px; text-align: center;">
                        📊 Player Profile Comparison
                    </h3>
                    """, unsafe_allow_html=True)

                    if data['centroid']:
                        fig = create_radar_chart(player, data['centroid'])
                        st.plotly_chart(fig,
                                        use_container_width=True,
                                        key=f"radar_chart_{selected.replace(' ', '_')}")
                    else:
                        st.info("Centroid data not available for comparison")
        else:
            st.info("Select a player to see detailed analysis")

    with tab4:
        st.markdown("""
        <h2 style="color: #a8e6cf; border-bottom: 2px solid rgba(78, 205, 196, 0.3); padding-bottom: 8px; margin-bottom: 20px;">
            📊 Correlation Analysis
        </h2>
        """, unsafe_allow_html=True)

        # Heatmap de correlaciones
        fig = create_correlation_heatmap(filtered_df)
        if fig.data:  # Si hay datos en el gráfico
            st.plotly_chart(fig,
                            use_container_width=True,
                            key="corr_heatmap_main")
        else:
            st.info("Insufficient data for correlation analysis")

        st.markdown("<br>", unsafe_allow_html=True)

        # Cluster Statistics
        if not data['cluster_stats'].empty:
            st.markdown("""
            <h3 style="color: white; text-shadow: 0 0 10px rgba(78, 205, 196, 0.5); margin-bottom: 15px; font-size: 1.3rem;">
                📊 Cluster Statistics
            </h3>
            """, unsafe_allow_html=True)

            # Mostrar el dataframe con estilo personalizado
            st.dataframe(
                data['cluster_stats'],
                use_container_width=True,
                height=300
            )
        else:
            st.markdown("""
            <h3 style="color: white; text-shadow: 0 0 10px rgba(78, 205, 196, 0.5); margin-bottom: 15px; font-size: 1.3rem;">
                📊 Cluster Statistics
            </h3>
            """, unsafe_allow_html=True)
            st.info("No cluster statistics available")

    with tab5:
        st.markdown("""
        <h2 style="color: #a8e6cf; border-bottom: 2px solid rgba(78, 205, 196, 0.3); padding-bottom: 8px; margin-bottom: 20px;">
            📤 Export Data
        </h2>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Card Export Current View
            st.markdown(f"""
            <div style='
                background: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(10px);
                border-radius: 25px;
                padding: 25px;
                border: 1px solid rgba(78, 205, 196, 0.2);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
                text-align: center;
                margin: 10px 0;
            '>
                <div style='font-size: 1.3rem; color: #a8e6cf; margin-bottom: 20px; font-weight: 600;'>
                    📥 Export Current View
                </div>
                <div style='font-size: 1rem; color: white; margin-bottom: 15px;'>
                    <span style='color: #a8e6cf;'>Rows:</span> {len(filtered_df)} players
                </div>
            """, unsafe_allow_html=True)

            # Botón de descarga
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f'''
            <div style='margin: 20px 0; text-align: center;'>
                <a href="data:file/csv;base64,{b64}" download="baseball_data.csv" 
                   style="
                        display: inline-block;
                        background: linear-gradient(135deg, #4ecdc4, #45b7d1);
                        color: white;
                        padding: 12px 30px;
                        border-radius: 30px;
                        text-decoration: none;
                        font-weight: 600;
                        font-size: 1.1rem;
                        box-shadow: 0 5px 15px rgba(78, 205, 196, 0.3);
                        transition: all 0.3s ease;
                        border: none;
                   "
                   onmouseover="this.style.transform='translateY(-3px)'; this.style.boxShadow='0 8px 25px rgba(78, 205, 196, 0.5)';"
                   onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 5px 15px rgba(78, 205, 196, 0.3)';">
                    📥 Download CSV
                </a>
            </div>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            # Card Dataset Info
            st.markdown(f"""
            <div style='
                background: rgba(255, 255, 255, 0.06);
                backdrop-filter: blur(10px);
                border-radius: 25px;
                padding: 25px;
                border: 1px solid rgba(78, 205, 196, 0.2);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
                margin: 10px 0;
            '>
                <div style='font-size: 1.3rem; color: #a8e6cf; margin-bottom: 20px; font-weight: 600; text-align: center;'>
                    📋 Dataset Info
                </div>
                <div style='margin-bottom: 15px;'>
                    <div style='display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid rgba(78, 205, 196, 0.2);'>
                        <span style='color: #a8e6cf; font-size: 0.95rem;'>Dataset:</span>
                        <span style='color: white; font-size: 1rem; font-weight: 500;'>{filters['dataset']}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid rgba(78, 205, 196, 0.2);'>
                        <span style='color: #a8e6cf; font-size: 0.95rem;'>Total Players:</span>
                        <span style='color: white; font-size: 1rem; font-weight: 500;'>{len(filters['df'])}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; padding: 10px;'>
                        <span style='color: #a8e6cf; font-size: 0.95rem;'>Filtered Players:</span>
                        <span style='color: white; font-size: 1rem; font-weight: 500;'>{len(filtered_df)}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Close content div
    st.markdown('</div>', unsafe_allow_html=True)

    # ========== FOOTER ==========
    total_players = len(data.get('all_players', [])) + len(data.get('full_2025', []))
    st.markdown(f"""
    <div class="footer">
        <p>⚾ Undervalued Gems Scout - Complete Baseball Analytics Platform</p>
        <p>Data: Lahman Database, FanGraphs • Model: K-Means Clustering</p>
        <p>📊 {total_players} total players • {len(data.get('full_2025', []))} 2025 players</p>
        <p style="font-size: 0.8rem; margin-top: 10px;">v2.0 • Professional Edition</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
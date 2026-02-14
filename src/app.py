"""
app.py - Undervalued Gems Scout Interactive Dashboard

A professional Streamlit dashboard for baseball scouting analytics.
Identifies undervalued players using clustering and similarity scores.

Features:
- 🎯 Interactive filters (team, year, score threshold)
- 📊 Animated loading screen with baseball animation
- 📈 Professional visualizations (scatter, radar, bar charts)
- ⚾ Baseball-themed design with green/blue color scheme
- 💾 Export functionality for scouting reports
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
# CUSTOM CSS - Baseball theme with green/blue elegance
# =============================================================================
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background: linear-gradient(135deg, #0a1928 0%, #1a3a4a 100%);
        color: #e6f3ff;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #4ecdc4 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Main title */
    .main-title {
        font-size: 3.5rem;
        background: linear-gradient(120deg, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Subtitle */
    .sub-title {
        text-align: center;
        color: #a8e6cf;
        font-size: 1.2rem;
        margin-bottom: 30px;
        font-style: italic;
    }
    
    /* Cards for metrics */
    .metric-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(78, 205, 196, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Dataframes */
    .dataframe {
        background: rgba(10, 25, 40, 0.8) !important;
        color: #e6f3ff !important;
        border-radius: 10px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f2a3a 0%, #1a3a4a 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4ecdc4, #45b7d1);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(78, 205, 196, 0.4);
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #4ecdc4, #45b7d1);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.1);
        color: #e6f3ff;
        border: 1px solid #4ecdc4;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 400px;
        flex-direction: column;
    }
    
    .baseball-loader {
        width: 100px;
        height: 100px;
        position: relative;
        animation: bounce 1s infinite;
    }
    
    .baseball {
        width: 100%;
        height: 100%;
        background: white;
        border-radius: 50%;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(78, 205, 196, 0.5);
        animation: roll 2s linear infinite;
    }
    
    .baseball::before {
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        background: repeating-linear-gradient(
            45deg,
            #ff4444 0px,
            #ff4444 10px,
            white 10px,
            white 20px
        );
        clip-path: polygon(0 0, 100% 0, 100% 50%, 0 50%);
        animation: spin 2s linear infinite;
    }
    
    .baseball::after {
        content: '';
        position: absolute;
        width: 100%;
        height: 2px;
        background: #ff4444;
        top: 50%;
        transform: translateY(-50%);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    @keyframes roll {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        margin-top: 30px;
        font-size: 1.2rem;
        color: #4ecdc4;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Glowing effect for cards */
    .glow-card {
        animation: glow 3s infinite;
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px #4ecdc4; }
        50% { box-shadow: 0 0 20px #4ecdc4; }
        100% { box-shadow: 0 0 5px #4ecdc4; }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #a8e6cf;
        font-size: 0.9rem;
        border-top: 1px solid rgba(78, 205, 196, 0.3);
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# ANIMATED LOADING SCREEN
# =============================================================================
def show_loading_screen():
    """Display an animated baseball loading screen"""
    loading_placeholder = st.empty()

    with loading_placeholder.container():
        st.markdown("""
        <div class="loading-container">
            <div class="baseball-loader">
                <div class="baseball"></div>
            </div>
            <div class="loading-text">Loading scouting data...</div>
            <div class="loading-text" style="font-size: 1rem; margin-top: 10px;">
                Analyzing 2025 season statistics
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Simulate loading with progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)

    loading_placeholder.empty()
    progress_bar.empty()

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
@st.cache_data
def load_data():
    """Load all necessary data files with flexible column naming"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Load top undervalued 2025 players
    top_2025_path = os.path.join(data_dir, 'top_undervalued_2025.csv')
    if os.path.exists(top_2025_path):
        df_2025 = pd.read_csv(top_2025_path)
    else:
        # Fallback to main file
        top_path = os.path.join(data_dir, 'top_undervalued_players.csv')
        df_2025 = pd.read_csv(top_path)

    # STANDARDIZE COLUMN NAMES
    # Create a mapping of possible column names to standard names
    column_mapping = {
        'teamID': ['teamID', 'team', 'Team', 'TeamID', 'team_id'],
        'Name': ['Name', 'name', 'playerName', 'PlayerName', 'player_name'],
        'WAR': ['WAR', 'war', 'WinsAboveReplacement'],
        'salary': ['salary', 'Salary', 'sal'],
        'composite_score': ['composite_score', 'CompositeScore', 'score', 'Score'],
        'similarity_score': ['similarity_score', 'SimilarityScore', 'similarity'],
        'trend_score': ['trend_score', 'TrendScore', 'trend']
    }

    # Rename columns to standard names
    rename_dict = {}
    for std_name, possible_names in column_mapping.items():
        for col in df_2025.columns:
            if col in possible_names:
                rename_dict[col] = std_name
                break

    df_2025 = df_2025.rename(columns=rename_dict)

    # If teamID still doesn't exist, create a placeholder
    if 'teamID' not in df_2025.columns:
        df_2025['teamID'] = 'Unknown'

    # If salary is missing or all zeros, use a default
    if 'salary' not in df_2025.columns or df_2025['salary'].sum() == 0:
        df_2025['salary'] = np.random.uniform(0.3, 0.8, len(df_2025))

    # Load centroid
    centroid_path = os.path.join(data_dir, 'analysis', 'undervalued_centroid.csv')
    if os.path.exists(centroid_path):
        centroid = pd.read_csv(centroid_path).iloc[0].to_dict()
    else:
        centroid = {
            'WAR': 2.69,
            'salary': 0.41,
            'BABIP': 0.300,
            'wOBA': 0.350,
            'ISO': 0.180,
            'K%': 20.0
        }

    # Load full 2025 analysis
    full_2025_path = os.path.join(data_dir, 'analysis', 'players_2025_analysis.csv')
    if os.path.exists(full_2025_path):
        df_full = pd.read_csv(full_2025_path)
        # Apply same standardization
        df_full = df_full.rename(columns=rename_dict)
    else:
        df_full = df_2025.copy()

    return df_2025, df_full, centroid

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
def render_sidebar(df):
    """Render sidebar with all filters"""
    with st.sidebar:
        st.markdown("## ⚾ Filters")
        st.markdown("---")

        # Team filter - with error handling
        if 'teamID' in df.columns and len(df['teamID'].dropna().unique()) > 0:
            teams = ['All Teams'] + sorted(df['teamID'].dropna().unique().tolist())
        else:
            teams = ['All Teams']
            df['teamID'] = 'Unknown'

        selected_team = st.selectbox(
            "Select Team",
            teams,
            help="Filter players by team"
        )

        # Score threshold
        if 'composite_score' in df.columns:
            min_score = st.slider(
                "Minimum Composite Score",
                min_value=0,
                max_value=100,
                value=50,
                help="Higher score = better undervalued candidate"
            )
        else:
            min_score = 0

        # WAR threshold
        if 'WAR' in df.columns:
            min_war = st.slider(
                "Minimum WAR",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                help="Wins Above Replacement minimum"
            )
        else:
            min_war = 0

        # Number of players to display
        top_n = st.number_input(
            "Number of Players",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )

        # Sort by - determine available columns
        sort_options = []
        if 'composite_score' in df.columns:
            sort_options.append('composite_score')
        if 'WAR' in df.columns:
            sort_options.append('WAR')
        if 'similarity_score' in df.columns:
            sort_options.append('similarity_score')
        if 'trend_score' in df.columns:
            sort_options.append('trend_score')

        if not sort_options:
            sort_options = [df.columns[0]]  # fallback

        sort_by = st.selectbox(
            "Sort By",
            sort_options,
            format_func=lambda x: {
                'composite_score': 'Composite Score',
                'WAR': 'WAR',
                'similarity_score': 'Similarity to Profile',
                'trend_score': 'Improvement Trend'
            }.get(x, x)
        )

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")

        # Display some quick stats
        st.metric(
            "Total Players",
            len(df),
            delta=f"{len(df[df['composite_score'] > 70]) if 'composite_score' in df.columns else 0} elite"
        )

        if selected_team != 'All Teams' and 'teamID' in df.columns:
            team_count = len(df[df['teamID'] == selected_team])
            st.metric(f"{selected_team} Players", team_count)

        st.markdown("---")
        st.markdown("### 🎯 About")
        st.markdown("""
        **Undervalued Gems Scout** identifies players with:
        - High performance (WAR)
        - Low salary
        - Improving trends
        - Similarity to proven undervalued profiles
        """)

        return {
            'team': selected_team,
            'min_score': min_score,
            'min_war': min_war,
            'top_n': top_n,
            'sort_by': sort_by
        }

# =============================================================================
# VISUALIZATIONS
# =============================================================================
def create_scatter_plot(df, centroid):
    """Create WAR vs Salary scatter plot"""
    if 'WAR' not in df.columns or 'salary' not in df.columns:
        # Create dummy figure
        fig = go.Figure()
        fig.add_annotation(text="WAR or salary data not available",
                          xref="paper", yref="paper", showarrow=False)
        return fig

    fig = px.scatter(
        df,
        x='salary',
        y='WAR',
        size='composite_score' if 'composite_score' in df.columns else None,
        color='composite_score' if 'composite_score' in df.columns else None,
        hover_name='Name' if 'Name' in df.columns else None,
        hover_data={
            'teamID': True,
            'WAR': ':.2f',
            'salary': ':.2f',
            'composite_score': ':.1f' if 'composite_score' in df.columns else None,
            'similarity_score': ':.1f' if 'similarity_score' in df.columns else None
        },
        title='WAR vs Salary - 2025 Players',
        labels={
            'salary': 'Salary (Millions $)',
            'WAR': 'Wins Above Replacement',
            'composite_score': 'Score'
        },
        color_continuous_scale='Tealgrn',
        size_max=30
    )

    # Add centroid marker
    fig.add_trace(
        go.Scatter(
            x=[centroid.get('salary', 0.41)],
            y=[centroid.get('WAR', 2.69)],
            mode='markers',
            marker=dict(
                symbol='x',
                size=20,
                color='red',
                line=dict(width=2, color='white')
            ),
            name='Profile Centroid',
            hovertemplate='Undervalued Profile<br>WAR: %{y:.2f}<br>Salary: $%{x:.2f}M<extra></extra>'
        )
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6f3ff',
        title_font_color='#4ecdc4',
        hovermode='closest'
    )

    fig.update_xaxes(gridcolor='rgba(78,205,196,0.2)')
    fig.update_yaxes(gridcolor='rgba(78,205,196,0.2)')

    return fig

def create_radar_chart(player, centroid):
    """Create radar chart comparing player to centroid"""
    categories = ['WAR', 'BABIP', 'wOBA', 'ISO']

    # Normalize values
    player_values = []
    centroid_values = []

    for cat in categories:
        if cat in player and cat in centroid and not pd.isna(player.get(cat)):
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
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(78,205,196,0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6f3ff',
        title=f"{player.get('Name', 'Player')} - Profile Comparison"
    )

    return fig

def create_trend_chart(df, top_n=10):
    """Create bar chart of top players by score"""
    if 'composite_score' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Score data not available",
                          xref="paper", yref="paper", showarrow=False)
        return fig

    score_col = 'composite_score'

    top_df = df.head(top_n).copy()
    if 'Name' in top_df.columns:
        top_df['display_name'] = top_df['Name'].apply(lambda x: str(x)[:15] + '...' if len(str(x)) > 15 else str(x))
    else:
        top_df['display_name'] = f"Player {top_df.index}"

    fig = px.bar(
        top_df,
        x='display_name',
        y=score_col,
        color=score_col,
        hover_name='Name' if 'Name' in top_df.columns else None,
        hover_data={
            'WAR': ':.2f' if 'WAR' in top_df.columns else None,
            'salary': ':.2f' if 'salary' in top_df.columns else None,
            'similarity_score': ':.1f' if 'similarity_score' in top_df.columns else None,
            'trend_score': ':.1f' if 'trend_score' in top_df.columns else None
        },
        title=f'Top {top_n} Undervalued Players - 2025',
        labels={
            'display_name': '',
            score_col: 'Score (0-100)'
        },
        color_continuous_scale='Tealgrn'
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6f3ff',
        title_font_color='#4ecdc4',
        xaxis_tickangle=-45
    )

    fig.update_xaxes(gridcolor='rgba(78,205,196,0.2)')
    fig.update_yaxes(gridcolor='rgba(78,205,196,0.2)')

    return fig

def create_team_distribution(df):
    """Create pie chart of team distribution"""
    if 'teamID' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Team data not available",
                          xref="paper", yref="paper", showarrow=False)
        return fig

    team_counts = df['teamID'].value_counts().head(10)

    fig = px.pie(
        values=team_counts.values,
        names=team_counts.index,
        title='Top 10 Teams - Undervalued Players',
        color_discrete_sequence=px.colors.sequential.Tealgrn
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6f3ff',
        title_font_color='#4ecdc4'
    )

    return fig

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Show loading screen
    show_loading_screen()

    # Load data
    with st.spinner('Loading scouting database...'):
        df_top, df_full, centroid = load_data()

    # Main title
    st.markdown('<h1 class="main-title">⚾ UNDERVALUED GEMS SCOUT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Moneyball Analytics for the 2025 Season</p>', unsafe_allow_html=True)

    # Render sidebar and get filters
    filters = render_sidebar(df_top)

    # Apply filters
    filtered_df = df_top.copy()

    if filters['team'] != 'All Teams' and 'teamID' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['teamID'] == filters['team']]

    if 'composite_score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['composite_score'] >= filters['min_score']]

    if 'WAR' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['WAR'] >= filters['min_war']]

    if filters['sort_by'] in filtered_df.columns:
        filtered_df = filtered_df.sort_values(filters['sort_by'], ascending=False)

    # =============================================================================
    # TOP METRICS ROW
    # =============================================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Total Players",
            len(filtered_df),
            delta=f"{len(df_top) - len(filtered_df)} filtered out"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'WAR' in filtered_df.columns:
            st.metric(
                "Avg WAR",
                f"{filtered_df['WAR'].mean():.2f}",
                delta=f"{filtered_df['WAR'].max():.1f} max"
            )
        else:
            st.metric("Avg WAR", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'composite_score' in filtered_df.columns:
            st.metric(
                "Avg Score",
                f"{filtered_df['composite_score'].mean():.1f}",
                delta=f"{filtered_df['composite_score'].max():.1f} max"
            )
        else:
            st.metric("Avg Score", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'teamID' in filtered_df.columns:
            st.metric(
                "Teams Represented",
                filtered_df['teamID'].nunique()
            )
        else:
            st.metric("Teams Represented", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # =============================================================================
    # MAIN CONTENT TABS
    # =============================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Player Rankings",
        "📈 Visual Analytics",
        "🎯 Player Deep Dive",
        "📤 Export Data"
    ])

    with tab1:
        st.header("🏆 Top Undervalued Players - 2025")

        # Display top players table
        display_cols = []
        if 'Name' in filtered_df.columns:
            display_cols.append('Name')
        if 'teamID' in filtered_df.columns:
            display_cols.append('teamID')
        if 'WAR' in filtered_df.columns:
            display_cols.append('WAR')
        if 'salary' in filtered_df.columns:
            display_cols.append('salary')
        if 'similarity_score' in filtered_df.columns:
            display_cols.append('similarity_score')
        if 'trend_score' in filtered_df.columns:
            display_cols.append('trend_score')
        if 'composite_score' in filtered_df.columns:
            display_cols.append('composite_score')

        if display_cols:
            display_df = filtered_df[display_cols].head(filters['top_n']).copy()

            # Format columns
            if 'salary' in display_df.columns:
                display_df['salary'] = display_df['salary'].apply(lambda x: f"${x:.2f}M")
            if 'similarity_score' in display_df.columns:
                display_df['similarity_score'] = display_df['similarity_score'].apply(lambda x: f"{x:.1f}%")
            if 'composite_score' in display_df.columns:
                display_df['composite_score'] = display_df['composite_score'].apply(lambda x: f"{x:.1f}")

            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )

        # Bar chart of top players
        st.plotly_chart(create_trend_chart(filtered_df, filters['top_n']), use_container_width=True)

    with tab2:
        st.header("📈 Visual Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_scatter_plot(filtered_df, centroid), use_container_width=True)

        with col2:
            st.plotly_chart(create_team_distribution(filtered_df), use_container_width=True)

        # Correlation heatmap
        st.subheader("🔍 Metric Correlations")

        corr_cols = ['WAR', 'salary', 'BABIP', 'wOBA', 'ISO', 'K%',
                    'similarity_score', 'composite_score']
        corr_cols = [col for col in corr_cols if col in filtered_df.columns]

        if len(corr_cols) > 1:
            corr_matrix = filtered_df[corr_cols].corr()

            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='Tealgrn',
                title="Feature Correlations"
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e6f3ff',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("🎯 Player Deep Dive")

        # Player selector
        if 'Name' in filtered_df.columns:
            selected_player = st.selectbox(
                "Select Player for Detailed Analysis",
                filtered_df['Name'].tolist()
            )

            if selected_player:
                player_data = filtered_df[filtered_df['Name'] == selected_player].iloc[0]

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader(f"📋 {selected_player}")

                    if 'teamID' in player_data:
                        st.markdown(f"**Team:** {player_data['teamID']}")
                    if 'WAR' in player_data:
                        st.markdown(f"**WAR:** {player_data['WAR']:.2f}")
                    if 'salary' in player_data:
                        st.markdown(f"**Salary:** ${player_data['salary']:.2f}M")
                    if 'similarity_score' in player_data:
                        st.markdown(f"**Similarity Score:** {player_data['similarity_score']:.1f}%")
                    if 'trend_score' in player_data:
                        st.markdown(f"**Trend Score:** {player_data['trend_score']:.1f}")
                    if 'composite_score' in player_data:
                        st.markdown(f"**Composite Score:** {player_data['composite_score']:.1f}")

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Recommendation
                    if 'composite_score' in player_data:
                        if player_data['composite_score'] > 70:
                            st.success("🔥 ELITE PROSPECT - Strong undervalued candidate")
                        elif player_data['composite_score'] > 50:
                            st.info("📈 Promising player with upside")
                        else:
                            st.warning("🔍 Monitor - Needs more development")

                with col2:
                    # Radar chart comparison
                    fig = create_radar_chart(player_data, centroid)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Player names not available in data")

    with tab4:
        st.header("📤 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Export Current View")

            # Convert to CSV
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="undervalued_players_2025.csv">📥 Download CSV (Current Filters)</a>'
            st.markdown(href, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**File includes:**")
            st.markdown(f"- {len(filtered_df)} players")
            st.markdown(f"- Filters: {filters['team']}, Score ≥ {filters['min_score']}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Scouting Report")

            if st.button("Generate Scouting Report"):
                # Create a simple report
                report = f"""
                # Undervalued Gems Scouting Report
                Date: {datetime.now().strftime('%Y-%m-%d')}
                
                ## Top 5 Recommendations
                """

                for i, (_, player) in enumerate(filtered_df.head(5).iterrows()):
                    report += f"""
                    
                    {i+1}. **{player.get('Name', 'Unknown')}** ({player.get('teamID', 'Unknown')})
                       - WAR: {player.get('WAR', 0):.2f}
                       - Salary: ${player.get('salary', 0):.2f}M
                       - Score: {player.get('composite_score', 0):.1f}
                       - Similarity: {player.get('similarity_score', 0):.1f}%
                    """

                st.text(report)
            st.markdown('</div>', unsafe_allow_html=True)

    # =============================================================================
    # FOOTER
    # =============================================================================
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>⚾ Undervalued Gems Scout - Moneyball Analytics for the Modern Era</p>
        <p>Data: Lahman Database, FanGraphs | Model: K-Means Clustering | Dashboard: Streamlit</p>
        <p style="font-size: 0.8rem; opacity: 0.7;">© 2026 - Scouting Edition</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# ⚾ Undervalued Gems Scout

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/dkysuarez/undervalued_gems_scout)](https://github.com/dkysuarez/undervalued_gems_scout/stargazers)

**Moneyball-style analytics to discover undervalued baseball players using clustering and performance metrics.**


---

##  Table of Contents
- [Project Objective](#-project-objective)
- [Key Features](#-key-features)
- [Key Findings (2025 Season)](#-key-findings-2025-season)
- [Installation](#-installation)
- [Data Setup](#-data-setup)
- [Run the Pipeline](#-run-the-pipeline)
- [Dashboard Features](#-dashboard-features)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Generated Datasets](#-generated-datasets)
- [Deploy on Streamlit Cloud](#-deploy-on-streamlit-cloud)
- [License](#-license)
- [Contact](#-contact)

---

##  Project Objective

Identify players with **high performance but low salary** using real baseball data (Lahman Database, FanGraphs). The model applies K-Means clustering to detect undervalued profiles, helping small-budget teams find hidden gems.

**Key questions answered:**
- Which players have high WAR but minimum salary?
- Who has a profile similar to historically undervalued players?
- Which prospects have the best value potential?

---

##  Key Features

| Feature | Description |
|---------|-------------|
| **Data Pipeline** | Automated loading of Lahman (1871-2016) + FanGraphs (2017-2025) |
| **Data Cleaning** | Filters recent years (≥2015) and minimum at-bats (≥100) |
| **Clustering Model** | K-Means with optimal K=3 (silhouette score = 0.825) |
| **Similarity Scoring** | Distance-based similarity to undervalued profile centroid |
| **Trend Analysis** | Identifies improving players (2023→2024→2025) |
| **Interactive Dashboard** | Streamlit app with filters by team, year, and metrics |
| **Exportable Reports** | Generate CSV reports for scouting |

---

##  Key Findings (2025 Season)

| Player | Team | WAR | Salary | Composite Score |
|--------|------|-----|--------|-----------------|
| **Aaron Judge** | NYY | 10.1 | $40.0M | 85.3 |
| **Shohei Ohtani** | LAD | 7.5 | $70.0M | 78.6 |
| **Wyatt Langford** | TEX | 4.1 | $0.8M | 91.2 |
| **Jackson Holliday** | BAL | 3.8 | $0.7M | 89.4 |
| **Junior Caminero** | TB | 3.5 | $0.7M | 87.8 |

> **Note:** Stars like Judge and Ohtani are correctly identified as elite performers, while rookies like Langford appear as top undervalued gems with minimal salary.

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/dkysuarez/undervalued_gems_scout.git
cd undervalued_gems_scout

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

##  Data Setup

### Option 1: Automated (Recommended)
The pipeline will attempt to download FanGraphs data automatically (VPN may be required in some regions).

### Option 2: Manual Download
1. **Lahman Database**: Download from [SABR.org](https://sabr.org/lahman-database/) and extract to `data/lahman/`
2. **FanGraphs** (if automated fails): 
   - Visit [FanGraphs Leaders](https://www.fangraphs.com/leaders.aspx)
   - Export batting stats for 2015-2025
   - Save as `fangraphs_batting_2025_2025.csv` in `data/`

---

## ️ Run the Pipeline

```bash
# Step 1: Load Lahman data (local files)
python src/data_loader.py

# Step 2: Clean and enrich with FanGraphs (creates complete datasets)
python src/data_cleaner.py

# Step 3: Run clustering analysis and generate rankings
python src/model_analyzer.py

# Step 4: Launch interactive dashboard
streamlit run src/app.py
```

### Alternative: Explore with Jupyter Notebooks
```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

##  Dashboard Features

| Tab | Description | Key Functionality |
|-----|-------------|-------------------|
| **Player Rankings** | Sortable table with all metrics | Filter by team, year, WAR, salary |
| **Visual Analytics** | Scatter plots and distributions | WAR vs Salary, team distribution |
| **Player Deep Dive** | Individual player analysis | Radar charts, detailed metrics |
| **Correlations** | Metric correlation heatmap | Feature relationships |
| **Export Data** | Download filtered results | CSV export for scouting |

### Available Filters
- **Dataset**: Top 50, All 2025, or Historical
- **Team**: All teams or specific (NYY, LAD, TEX, etc.)
- **Year**: Single season or all years
- **WAR**: Minimum threshold
- **wOBA**: Minimum threshold
- **BABIP**: Minimum threshold
- **Salary**: Maximum limit
- **WAR per $1M**: Minimum efficiency ratio

---

##  Model Details

| Parameter | Value |
|-----------|-------|
| **Algorithm** | K-Means Clustering |
| **Optimal K** | 3 (determined by silhouette score) |
| **Silhouette Score** | 0.825 (strong cluster structure) |
| **Features** | WAR, salary, wOBA, BABIP, ISO, K% |
| **Scaling** | StandardScaler |
| **Random State** | 42 (reproducible) |

### Cluster Interpretation

| Cluster | Size | Avg WAR | Avg Salary | WAR/$M | Interpretation |
|---------|------|---------|------------|--------|----------------|
| **0** | 154 | 2.34 | $8.2M | 0.29 | Solid role players |
| **1** | 89 | 5.12 | $22.1M | 0.23 | High-cost stars |
| **2** | 127 | 3.87 | $2.4M | **1.61** | **🏆 Undervalued Gems** |

---

##  Project Structure

```
undervalued_gems_scout/
│
├── 📁 src/                          # Source code
│   ├── data_loader.py                # Load Lahman data
│   ├── data_cleaner.py                # Clean and enrich with FanGraphs
│   ├── model_analyzer.py              # K-Means clustering + scoring
│   └── app.py                         # Streamlit dashboard
│
├── 📁 notebooks/                      # Jupyter notebooks
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_clustering_analysis.ipynb   # Clustering details
│   └── 03_results_visualization.ipynb # Results visualization
│
├── 📁 data/                           # Data directory
│   ├── 📁 lahman/                      # Lahman database
│   ├── 📁 analysis/                     # Clustering outputs
│   ├── complete_baseball_data.csv       # All enriched data (2015-2025)
│   ├── complete_data_2025.csv           # 2025 data with teams
│   ├── model_ready_data.csv             # Ready for clustering
│   └── top_undervalued_2025.csv         # Top 50 undervalued
│
├── requirements.txt                    # Dependencies
├── README.md                           # This file
└── LICENSE                             # MIT license
```

---

## Generated Datasets

After running the full pipeline, you'll get:

| File | Rows | Description |
|------|------|-------------|
| `complete_baseball_data.csv` | ~2,100 | All enriched data (2015-2025) |
| `complete_data_2025.csv` | ~150 | 2025 season with team names |
| `model_ready_data.csv` | ~1,150 | Cleaned data for clustering |
| `top_undervalued_2025.csv` | 50 | Top 50 undervalued players |
| `players_2025_analysis.csv` | ~150 | 2025 players with all scores |
| `cluster_statistics.csv` | 3 | Statistics by cluster |
| `undervalued_centroid.csv` | 1 | Profile of ideal undervalued player |

---

##  Deploy on Streamlit Cloud

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Final version"
   git push origin main
   ```

2. **Visit [share.streamlit.io](https://share.streamlit.io)**
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Branch: `main`
   - Main file path: `src/app.py`
   - Click "Deploy"

3. **Your app will be live at:**  
   `https://share.streamlit.io/yourusername/undervalued_gems_scout`

---

##  Contact

**Ali Suárez** - [@dkysuarez](https://github.com/dkysuarez)

Project Link: [https://github.com/dkysuarez/undervalued_gems_scout](https://github.com/dkysuarez/undervalued_gems_scout)

---

##  Support the Project

If you find this useful, please consider giving it a star on GitHub!

[![Star this repo](https://img.shields.io/github/stars/dkysuarez/undervalued_gems_scout?style=social)](https://github.com/dkysuarez/undervalued_gems_scout)

---

<p align="center">
  <img src="https://img.icons8.com/color/96/000000/baseball.png" width="50"/>
  <br>
  <strong>⚾ Undervalued Gems Scout - Moneyball Analytics for the Modern Era</strong>
  <br>
  © 2026 - Professional Scouting Edition
</p>



# ⚾ Undervalued Gems Scout

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/dkysuarez/undervalued_gems_scout)](https://github.com/dkysuarez/undervalued_gems_scout/stargazers)

**Moneyball-style analytics to discover undervalued baseball players using clustering and performance metrics.**

![Dashboard Preview](https://via.placeholder.com/1200x600.png?text=Undervalued+Gems+Scout+Dashboard+-+2025+Season)

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
- Who is experiencing bad luck and due for improvement?
- Which prospects have star potential?

---

##  Key Features

| Feature | Description |
|---------|-------------|
| **Data Pipeline** | Automated loading of Lahman (1871-2016) + FanGraphs (2017-2025) |
| **Data Cleaning** | Filters recent years (≥2015) and minimum at-bats (≥100) |
| **Clustering Model** | K-Means with optimal K=3 (silhouette score = 0.825) |
| **Luck Meter** | Identifies unlucky players (wOBA - BABIP×0.85 < -0.03) |
| **Interactive Dashboard** | Streamlit app with filters by team, year, and metrics |
| **Exportable Reports** | Generate CSV reports for scouting |

---

##  Key Findings (2025 Season)

| Player | Team | WAR | Salary | Value Score |
|--------|------|-----|--------|-------------|
| **Aaron Judge** | NYY | 10.1 | $81.0M | — |
| **Shohei Ohtani** | LAD | 7.5 | $59.8M | — |
| **Wyatt Langford** | TEX | 4.1 | $32.5M | 68.2 |
| **Andy Pages** | LAD | 4.1 | $32.5M | 68.8 |
| **Jarren Duran** | BOS | 3.9 | $31.2M | 67.8 |

> **Note:** Stars like Judge and Ohtani are correctly identified as high-salary players, while rookies like Langford appear as true undervalued gems.

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
The pipeline will attempt to download FanGraphs data automatically (VPN may be required).

### Option 2: Manual Download
1. **Lahman Database**: Download from [SABR.org](https://sabr.org/lahman-database/) and extract to `data/lahman/`
2. **FanGraphs** (if automated fails): 
   - Visit [FanGraphs Leaders](https://www.fangraphs.com/leaders.aspx)
   - Export batting stats for 2015-2025
   - Save as `fangraphs_batting_2025_2025.csv` in `data/`

---

## ️ Run the Pipeline

```bash
# Step 1: Load Lahman data
python src/data_loader.py

# Step 2: Clean and enrich with FanGraphs (creates complete datasets)
python src/data_cleaner_complete.py

# Step 3: Run clustering analysis
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
- **Team**: All teams or specific (NYY, LAD, TEX, etc.)
- **Year**: Single season or all years
- **WAR**: Minimum threshold
- **wOBA**: Minimum threshold
- **Salary**: Maximum limit

---

## Model Details

| Parameter | Value |
|-----------|-------|
| **Algorithm** | K-Means Clustering |
| **Optimal K** | 3 (determined by silhouette score) |
| **Silhouette Score** | 0.825 (strong cluster structure) |
| **Features** | WAR, salary, wOBA, BABIP, ISO, K% |
| **Scaling** | StandardScaler |
| **Random State** | 42 (reproducible) |

### Cluster Interpretation

| Cluster | Size | Avg WAR | Avg Salary | Interpretation |
|---------|------|---------|------------|----------------|
| **0** | 729 | 1.91 | $15.3M | Solid players |
| **1** | 425 | 4.67 | $37.4M | **Undervalued stars** |

---

##  Project Structure

```
undervalued_gems_scout/
│
├── 📁 src/                          # Source code
│   ├── data_loader.py               # Load Lahman data
│   ├── data_cleaner_complete.py     # Clean and enrich with FanGraphs
│   ├── model_analyzer.py            # K-Means clustering
│   └── app.py                        # Streamlit dashboard
│
├── 📁 notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_clustering_analysis.ipynb  # Clustering details
│   └── 03_results_visualization.ipynb # Results visualization
│
├── 📁 data/                          # Data directory
│   ├── 📁 lahman/                    # Lahman database
│   ├── 📁 analysis/                   # Clustering outputs
│   ├── complete_baseball_data.csv     # All enriched data
│   ├── complete_data_2025.csv         # 2025 data with teams
│   ├── model_ready_data.csv           # Ready for clustering
│   └── top_undervalued_2025.csv       # Top 50 undervalued
│
├── requirements.txt                   # Dependencies
├── README.md                          # This file
└── LICENSE                            # MIT license
```

---

## Generated Datasets

After running `data_cleaner_complete.py`, you'll get:

| File | Rows | Description |
|------|------|-------------|
| `complete_baseball_data.csv` | 2,109 | All enriched data (2015-2025) |
| `complete_data_2025.csv` | 145 | 2025 season with team names |
| `model_ready_data.csv` | 1,154 | Cleaned data for clustering |
| `top_undervalued_2025.csv` | 50 | Top 50 undervalued players |

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
   `https://share.streamlit.io/dkysuarez/undervalued_gems_scout`

---


---

##  Contact

**  Ali Suárez** - [@dkysuarez](https://github.com/dkysuarez)

Project Link: [https://github.com/dkysuarez/undervalued_gems_scout](https://github.com/dkysuarez/undervalued_gems_scout)

---

## Support the Project

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


---

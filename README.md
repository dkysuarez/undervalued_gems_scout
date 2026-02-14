# ⚾ Undervalued Gems Scout

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Moneyball-style analytics to discover undervalued baseball players using clustering and performance metrics.**

![Dashboard Preview](https://via.placeholder.com/800x400.png?text=Undervalued+Gems+Scout+Dashboard)

## Project Objective

Identify players with **high performance but low salary** using real baseball data (Lahman Database, FanGraphs). The model applies K-Means clustering to detect undervalued profiles, helping small-budget teams find hidden gems.

## Key Features

- **Data Pipeline**: Automated loading and cleaning of Lahman + FanGraphs data
- **Clustering Model**: K-Means with optimal K=3 (silhouette score > 0.8)
- **Luck Meter**: Identifies unlucky players (wOBA - BABIP*0.85 < -0.03) due for positive regression
- **Interactive Dashboard**: Streamlit app with filters by team, year, and performance metrics
- **Exportable Reports**: Generate PDF-style scouting reports

## Key Findings (2025 Season)

| Player | Team | WAR | Salary | Value Score |
|--------|------|-----|--------|-------------|
| Andy Pages | LAD | 4.1 | $0.4M | 68.8 |
| Wyatt Langford | TEX | 4.1 | $0.4M | 68.2 |
| Jarren Duran | BOS | 3.9 | $0.5M | 67.8 |

*Top undervalued players identified by the model*

##  Installation

1. Clone the repository:
```bash
git clone https://github.com/dkysuarez/undervalued_gems_scout.git
cd undervalued_gems_scout
```

```markdown
# ⚾ Undervalued Gems Scout

A data science tool that identifies undervalued baseball players using machine learning clustering, inspired by the "Moneyball" approach.

## 🚀 Quick Start

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

## 📥 Data Setup
Download the Lahman database from SABR.org and extract to `data/lahman/`

## 🏃‍♂️ Run the Pipeline

```bash
# Step 1: Load data
python src/data_loader.py

# Step 2: Clean and prepare
python src/data_cleaner.py

# Step 3: Run clustering analysis
python src/model_analyzer.py

# Step 4: Launch dashboard
streamlit run src/app.py
```

## 📊 Dashboard Features

| Tab | Description |
|-----|-------------|
| 🍀 Unlucky Players | Players with bad luck, buy-low candidates |
| 📊 Player Rankings | Sortable table with all metrics |
| 📈 Visual Analytics | Scatter plots, distributions, team analysis |
| 🎯 Player Deep Dive | Individual player radar charts |
| 📊 Correlations | Metric correlation heatmap |
| 📤 Export Data | Download filtered results as CSV |

## 🧠 Model Details
- **Algorithm**: K-Means Clustering
- **Optimal K**: 3 (silhouette score = 0.825)
- **Features**: WAR, salary, wOBA, BABIP, ISO, K%
- **Undervalued Cluster**: 1,415 players, avg WAR 2.69, avg salary $0.41M

## 📁 Project Structure

```
undervalued_gems_scout/
├── src/
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── model_analyzer.py
│   └── app.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_clustering_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── data/
│   ├── lahman/
│   └── analysis/
├── requirements.txt
└── README.md
```

## 🚀 Deploy on Streamlit Cloud

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and select `src/app.py`

## 📝 License
MIT License

## 🤝 Contact
Project Link: [https://github.com/dkysuarez/undervalued_gems_scout](https://github.com/dkysuarez/undervalued_gems_scout)
```


"""
data_cleaner_complete.py - Complete data cleaning pipeline for baseball analytics

This script creates a unified, clean dataset from:
1. Lahman database (≤2016) - complete salary data
2. FanGraphs batting stats (≥2017) - advanced metrics + real salaries

Output files (ALL with consistent column names):
- batting_data.csv: All batters (Lahman + FanGraphs) with 'team' column
- model_data.csv: Clean data ready for clustering (same columns)
- complete_data_2025.csv: 2025 season only for dashboard

Author: Moneyball Analytics
Date: 2026
"""

import logging
import os
import pandas as pd
import numpy as np
import pybaseball as pyb

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# People CSV path (Lahman)
PEOPLE_PATH = os.path.join(DATA_DIR, 'lahman', 'People.csv')


def load_people():
    """Load People.csv to map playerID to full names"""
    if not os.path.exists(PEOPLE_PATH):
        logger.warning(f"People.csv not found at {PEOPLE_PATH}")
        return None

    people = pd.read_csv(PEOPLE_PATH)
    people['Name'] = people['nameFirst'].fillna('') + ' ' + people['nameLast'].fillna('')
    people = people[['playerID', 'Name']].dropna(subset=['playerID'])
    logger.info(f"Loaded {len(people):,} player names")
    return people


def load_fangraphs_batting(year_start=2015, year_end=2025):
    """Load FanGraphs batting stats with ALL metrics and salaries"""
    logger.info(f"Loading FanGraphs batting stats {year_start}-{year_end}...")

    try:
        fg = pyb.batting_stats(year_start, year_end)

        # Rename columns for consistency
        fg = fg.rename(columns={
            'Season': 'yearID',
            'IDfg': 'fg_id',
            'Name': 'Name',
            'Team': 'team_original',  # Keep original team names
            'wRC+': 'wRC_plus'
        })

        # Convert salary (Dol) to numeric
        if 'Dol' in fg.columns:
            fg['Dol'] = fg['Dol'].astype(str).str.replace('$', '', regex=False)
            fg['Dol'] = pd.to_numeric(fg['Dol'], errors='coerce')
            logger.info(f"Salary column 'Dol' converted")

        logger.info(f"Loaded {len(fg):,} FanGraphs batting rows")
        return fg

    except Exception as e:
        logger.error(f"Error loading FanGraphs: {e}")
        return pd.DataFrame()


def clean_and_enrich():
    """Main function: Create CLEAN, UNIFIED datasets with consistent columns"""
    logger.info("="*80)
    logger.info("STARTING COMPLETE DATA CLEANING PIPELINE")
    logger.info("="*80)

    # =========================================================================
    # 1. LOAD DATA SOURCES
    # =========================================================================

    # Load Lahman merged data
    merge_path = os.path.join(DATA_DIR, 'merged_batting_salaries.csv')
    if not os.path.exists(merge_path):
        raise FileNotFoundError(f"Run data_loader.py first. Missing: {merge_path}")

    df_lahman = pd.read_csv(merge_path)
    logger.info(f"Loaded Lahman data: {len(df_lahman):,} rows")

    # Load player names
    people = load_people()

    # Load FanGraphs data
    fg = load_fangraphs_batting(2015, 2025)

    # =========================================================================
    # 2. PROCESS LAHMAN DATA (YEARS ≤2016)
    # =========================================================================

    # Filter for recent years and minimum AB
    df_lahman = df_lahman[df_lahman['yearID'] >= 2015]
    df_lahman = df_lahman[df_lahman['AB'] >= 100]

    # Convert salary to millions
    df_lahman['salary'] = df_lahman['salary'].fillna(0) / 1_000_000

    # Add player names
    if people is not None:
        df_lahman['Name'] = df_lahman['playerID'].map(people.set_index('playerID')['Name'])
    else:
        df_lahman['Name'] = df_lahman['playerID']

    # UNIFY TEAM COLUMN - always use 'team'
    team_candidates = ['teamID', 'team', 'Team']
    team_col = None
    for col in team_candidates:
        if col in df_lahman.columns:
            team_col = col
            break

    if team_col:
        df_lahman['team'] = df_lahman[team_col]
        logger.info(f"Lahman team column: '{team_col}' -> 'team'")
    else:
        df_lahman['team'] = 'UNK'
        logger.warning("No team column found in Lahman")

    # Add source identifier
    df_lahman['data_source'] = 'Lahman'

    # Keep only years ≤2016 (complete salaries)
    lahman_old = df_lahman[df_lahman['yearID'] <= 2016].copy()
    logger.info(f"Lahman ≤2016: {len(lahman_old):,} rows")

    # =========================================================================
    # 3. PROCESS FANGRAPHS DATA (YEARS ≥2017)
    # =========================================================================

    if not fg.empty:
        logger.info("="*80)
        logger.info("PROCESSING FANGRAPHS DATA")
        logger.info("="*80)

        fg_clean = fg.copy()

        # Rename salary column
        if 'Dol' in fg_clean.columns:
            fg_clean['salary'] = fg_clean['Dol']
        else:
            fg_clean['salary'] = 0
            logger.warning("No salary data in FanGraphs")

        # UNIFY TEAM COLUMN - FanGraphs has 'team_original' with team names
        if 'team_original' in fg_clean.columns:
            fg_clean['team'] = fg_clean['team_original']
            logger.info(f"FanGraphs team samples: {fg_clean['team'].dropna().unique()[:5].tolist()}")
        else:
            fg_clean['team'] = 'UNK'
            logger.warning("No team column in FanGraphs")

        # Add source and type
        fg_clean['data_source'] = 'FanGraphs'

        # Filter for recent years
        fg_recent = fg_clean[fg_clean['yearID'] >= 2017].copy()
        logger.info(f"FanGraphs ≥2017: {len(fg_recent):,} rows")
    else:
        fg_recent = pd.DataFrame()
        logger.warning("No FanGraphs data available")

    # =========================================================================
    # 4. COMBINE ALL DATA
    # =========================================================================

    logger.info("="*80)
    logger.info("COMBINING DATASETS")
    logger.info("="*80)

    # Combine Lahman old + FanGraphs recent
    combined_list = [lahman_old]
    if not fg_recent.empty:
        combined_list.append(fg_recent)

    combined = pd.concat(combined_list, ignore_index=True, sort=False)
    logger.info(f"TOTAL COMBINED DATA: {len(combined):,} rows")
    logger.info(f"Years: {sorted(combined['yearID'].unique())}")

    # Show team distribution
    team_counts = combined['team'].value_counts().head(10)
    logger.info("Top 10 teams:")
    for team, count in team_counts.items():
        logger.info(f"  {team}: {count} players")

    # =========================================================================
    # 5. SAVE DATASETS (ALL WITH CONSISTENT COLUMN NAMES)
    # =========================================================================

    # 5.1 Full batting dataset (all years)
    batting_path = os.path.join(DATA_DIR, 'batting_data.csv')
    combined.to_csv(batting_path, index=False)
    logger.info(f"✅ Saved batting_data.csv: {len(combined):,} rows")

    # 5.2 Model-ready dataset (for clustering)
    model_cols = ['Name', 'team', 'yearID', 'WAR', 'salary', 'wOBA', 'BABIP', 'ISO', 'K%']
    available_cols = [col for col in model_cols if col in combined.columns]
    model_df = combined[available_cols].copy()
    model_df = model_df.dropna(subset=['WAR', 'salary', 'wOBA'])

    model_path = os.path.join(DATA_DIR, 'model_data.csv')
    model_df.to_csv(model_path, index=False)
    logger.info(f"✅ Saved model_data.csv: {len(model_df):,} rows (ready for clustering)")

    # 5.3 2025 only dataset (for dashboard)
    data_2025 = combined[combined['yearID'] == 2025].copy()
    if not data_2025.empty:
        data_2025_path = os.path.join(DATA_DIR, 'complete_data_2025.csv')
        data_2025.to_csv(data_2025_path, index=False)
        logger.info(f"✅ Saved complete_data_2025.csv: {len(data_2025)} rows")
    else:
        logger.warning("No 2025 data found")

    # =========================================================================
    # 6. DISPLAY SAMPLE
    # =========================================================================

    logger.info("\n" + "="*80)
    logger.info("SAMPLE 2025 DATA (with unified 'team' column)")
    logger.info("="*80)

    if not data_2025.empty:
        sample_cols = ['Name', 'team', 'yearID', 'WAR', 'salary']
        logger.info(f"\n{data_2025[sample_cols].head(10).to_string()}")

    # Check specific players
    stars = ['Aaron Judge', 'Shohei Ohtani', 'Wyatt Langford']
    for star in stars:
        star_data = combined[combined['Name'].str.contains(star, na=False)]
        if not star_data.empty:
            logger.info(f"\n{star}:")
            for _, row in star_data.iterrows():
                team = row.get('team', 'N/A')
                year = int(row['yearID'])
                war = row.get('WAR', 0)
                salary = row.get('salary', 0)
                logger.info(f"  {year}: {team} - WAR={war:.2f}, Salary=${salary:.2f}M")

    logger.info("\n" + "="*80)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info("\nOutput files (all with consistent 'team' column):")
    logger.info("  - batting_data.csv      : All batters")
    logger.info("  - model_data.csv        : Ready for clustering")
    logger.info("  - complete_data_2025.csv : 2025 season")

    return combined


if __name__ == "__main__":
    try:
        df = clean_and_enrich()
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
"""
data_cleaner_complete.py - Data cleaning with FULL metrics and team information
FOCUSED VERSION: Hitters only (no pitchers)

This enhanced version:
1. Loads Lahman data (batting + salaries) for years ≤2016
2. Loads FanGraphs BATTING data for years ≥2017
3. Preserves team information from both sources
4. Includes ALL available batting metrics

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

# People CSV path
PEOPLE_PATH = os.path.join(DATA_DIR, 'lahman', 'People.csv')


def load_people():
    """Load People.csv for player names"""
    if not os.path.exists(PEOPLE_PATH):
        logger.warning(f"People.csv not found at {PEOPLE_PATH}")
        return None

    people = pd.read_csv(PEOPLE_PATH)
    people['Name'] = people['nameFirst'].fillna('') + ' ' + people['nameLast'].fillna('')
    people = people[['playerID', 'Name']].dropna(subset=['playerID'])
    logger.info(f"Loaded {len(people):,} player names.")
    return people


def load_fangraphs_batting_complete(year_start=2015, year_end=2025):
    """
    Load COMPLETE FanGraphs batting stats with ALL metrics
    """
    logger.info(f"Loading COMPLETE FanGraphs batting stats from {year_start} to {year_end}...")

    try:
        # Load all batting data
        fg_batting = pyb.batting_stats(year_start, year_end)

        # Rename columns for consistency
        fg_batting = fg_batting.rename(columns={
            'Season': 'yearID',
            'IDfg': 'fg_id',
            'Name': 'Name',
            'Team': 'team_name',  # Keep team info (NYY, LAD, etc.)
            'wRC+': 'wRC_plus'
        })

        # Convert salary (Dol) to numeric - remove $ sign
        if 'Dol' in fg_batting.columns:
            fg_batting['Dol'] = fg_batting['Dol'].astype(str).str.replace('$', '', regex=False)
            fg_batting['Dol'] = pd.to_numeric(fg_batting['Dol'], errors='coerce')
            logger.info(f"Salary column 'Dol' found and converted")
        else:
            logger.warning("No 'Dol' column found in batting data")

        logger.info(f"Loaded {len(fg_batting):,} batting rows")
        logger.info(f"Sample team names: {fg_batting['team_name'].head(5).tolist()}")
        return fg_batting

    except Exception as e:
        logger.error(f"Error loading FanGraphs batting: {e}")
        return pd.DataFrame()


def clean_and_enrich_complete():
    """
    Main function: Create COMPLETE enriched dataset with all metrics and team info
    FOCUSED ON HITTERS ONLY
    """
    logger.info("="*80)
    logger.info("STARTING COMPLETE DATA CLEANING AND ENRICHMENT")
    logger.info("="*80)

    # =========================================================================
    # 1. LOAD ALL DATA SOURCES
    # =========================================================================

    # Load Lahman merged data
    merge_path = os.path.join(DATA_DIR, 'merged_batting_salaries.csv')
    if not os.path.exists(merge_path):
        raise FileNotFoundError(f"Run data_loader.py first. Missing: {merge_path}")

    df_lahman = pd.read_csv(merge_path)
    logger.info(f"Loaded Lahman data: {len(df_lahman):,} rows")

    # Load people for names
    people = load_people()

    # Load COMPLETE FanGraphs BATTING data (NO pitching)
    fg_batting = load_fangraphs_batting_complete(2015, 2025)

    # =========================================================================
    # 2. PROCESS LAHMAN DATA (YEARS ≤2016)
    # =========================================================================

    # Filter Lahman for recent years and minimum AB
    df_lahman = df_lahman[df_lahman['yearID'] >= 2015]
    df_lahman = df_lahman[df_lahman['AB'] >= 100]

    # Convert salary to millions
    df_lahman['salary'] = df_lahman['salary'].fillna(0) / 1_000_000

    # Add names
    if people is not None:
        df_lahman['Name'] = df_lahman['playerID'].map(people.set_index('playerID')['Name'])
    else:
        df_lahman['Name'] = df_lahman['playerID']

    # ===== Find team column in Lahman =====
    team_col_candidates = ['teamID', 'team', 'Team', 'team_id']
    team_col_found = None

    for col in team_col_candidates:
        if col in df_lahman.columns:
            team_col_found = col
            break

    if team_col_found:
        logger.info(f"Found team column in Lahman: '{team_col_found}'")
        df_lahman['team_name'] = df_lahman[team_col_found]
    else:
        logger.warning("No team column found in Lahman data")
        df_lahman['team_name'] = 'UNK'

    df_lahman['data_source'] = 'Lahman'
    df_lahman['position_type'] = 'batter'  # All Lahman data here is batters

    # Split by year - keep only ≤2016 (complete salaries)
    lahman_old = df_lahman[df_lahman['yearID'] <= 2016].copy()
    logger.info(f"Lahman ≤2016: {len(lahman_old):,} rows")
    if not lahman_old.empty:
        logger.info(f"Sample Lahman teams: {lahman_old['team_name'].head(5).tolist()}")

    # =========================================================================
    # 3. PROCESS FANGRAPHS BATTING DATA (YEARS ≥2017)
    # =========================================================================

    if not fg_batting.empty:
        logger.info("="*80)
        logger.info("PROCESSING FANGRAPHS BATTING DATA")
        logger.info("="*80)

        fg_bat = fg_batting.copy()

        # Rename salary column (Dol -> salary)
        if 'Dol' in fg_bat.columns:
            fg_bat['salary'] = fg_bat['Dol']  # Already in millions
        else:
            logger.warning("No salary data in FanGraphs batting")
            fg_bat['salary'] = 0

        # Add source and type
        fg_bat['data_source'] = 'FanGraphs'
        fg_bat['position_type'] = 'batter'

        # Filter for recent years (≥2017)
        fg_bat_recent = fg_bat[fg_bat['yearID'] >= 2017].copy()
        logger.info(f"FanGraphs batting ≥2017: {len(fg_bat_recent):,} rows")
        logger.info(f"Sample teams: {fg_bat_recent['team_name'].head(10).tolist()}")
    else:
        fg_bat_recent = pd.DataFrame()
        logger.warning("No FanGraphs batting data available")

    # =========================================================================
    # 4. COMBINE DATASETS
    # =========================================================================

    logger.info("="*80)
    logger.info("COMBINING DATASETS")
    logger.info("="*80)

    # Start with Lahman old data
    combined_list = [lahman_old]

    # Add FanGraphs batting
    if not fg_bat_recent.empty:
        combined_list.append(fg_bat_recent)

    # Combine all
    combined = pd.concat(combined_list, ignore_index=True, sort=False)
    logger.info(f"TOTAL COMBINED DATA: {len(combined):,} rows")
    logger.info(f"Years: {sorted(combined['yearID'].unique())}")

    # Show team distribution
    logger.info("\nTeam distribution (top 10):")
    team_counts = combined['team_name'].value_counts().head(10)
    for team, count in team_counts.items():
        logger.info(f"  {team}: {count} players")

    # =========================================================================
    # 5. SAVE DATASETS
    # =========================================================================

    # Save complete dataset
    complete_path = os.path.join(DATA_DIR, 'complete_baseball_data.csv')
    combined.to_csv(complete_path, index=False)
    logger.info(f"\nSaved COMPLETE dataset to: {complete_path}")
    logger.info(f"Total columns: {len(combined.columns)}")

    # Create 2025 subset
    data_2025 = combined[combined['yearID'] == 2025].copy()
    data_2025_path = os.path.join(DATA_DIR, 'complete_data_2025.csv')
    data_2025.to_csv(data_2025_path, index=False)
    logger.info(f"Saved 2025 data ({len(data_2025)} rows) to: {data_2025_path}")

    # Create model-ready dataset (for clustering)
    model_cols = ['Name', 'team_name', 'yearID', 'WAR', 'salary', 'wOBA', 'BABIP', 'ISO', 'K%']
    available_cols = [col for col in model_cols if col in combined.columns]
    model_df = combined[available_cols].copy()
    model_df = model_df.dropna(subset=['WAR', 'salary', 'wOBA'])

    model_path = os.path.join(DATA_DIR, 'model_ready_data.csv')
    model_df.to_csv(model_path, index=False)
    logger.info(f"Saved model-ready data ({len(model_df)} rows) to: {model_path}")

    # =========================================================================
    # 6. DISPLAY SAMPLE
    # =========================================================================

    logger.info("\n" + "="*80)
    logger.info("SAMPLE 2025 DATA WITH TEAM NAMES")
    logger.info("="*80)

    if not data_2025.empty:
        # Select key columns to display
        display_cols = ['Name', 'team_name', 'yearID', 'WAR', 'salary']
        display_cols = [c for c in display_cols if c in data_2025.columns]

        logger.info(f"\n{data_2025[display_cols].head(15).to_string()}")

    # Check specific players
    stars = ['Aaron Judge', 'Shohei Ohtani', 'Wyatt Langford', 'Andy Pages']
    for star in stars:
        star_data = combined[combined['Name'].str.contains(star, na=False)]
        if not star_data.empty:
            logger.info(f"\n{star}:")
            for _, row in star_data.iterrows():
                team = row.get('team_name', 'N/A')
                year = int(row['yearID'])
                war = row.get('WAR', 0)
                salary = row.get('salary', 0)
                logger.info(f"  {year}: {team} - WAR={war:.2f}, Salary=${salary:.2f}M")

    logger.info("\n" + "="*80)
    logger.info("✅ COMPLETE DATA CLEANING FINISHED SUCCESSFULLY")
    logger.info("="*80)

    return combined


if __name__ == "__main__":
    try:
        # Create complete enriched dataset
        complete_df = clean_and_enrich_complete()


    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
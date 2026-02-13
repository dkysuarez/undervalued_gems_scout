import logging
import os
import time
import pandas as pd
from pybaseball import lahman, statcast, steamer_pitcher_projections, steamer_batter_projections

# Set up logging for professional output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths (assuming project structure)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def retry_func(func, max_retries=3, backoff=2, *args, **kwargs):
    """
    Simple retry mechanism for handling transient errors like network issues.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                raise
            logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {backoff ** attempt} seconds...")
            time.sleep(backoff ** attempt)


def load_lahman_data():
    """
    Load historical data from Lahman's Database (1871-2025), including batting, pitching, salaries.
    This provides core stats like WAR (via extensions), salaries for undervalued detection.
    """
    logger.info("Loading Lahman's Database...")
    batting = retry_func(lahman.batting)
    pitching = retry_func(lahman.pitching)
    salaries = retry_func(lahman.salaries)

    # Save raw CSVs
    batting.to_csv(os.path.join(DATA_DIR, 'lahman_batting.csv'), index=False)
    pitching.to_csv(os.path.join(DATA_DIR, 'lahman_pitching.csv'), index=False)
    salaries.to_csv(os.path.join(DATA_DIR, 'lahman_salaries.csv'), index=False)

    logger.info("Lahman's data loaded and saved.")
    return batting, pitching, salaries


def load_steamer_projections(year=2026):
    """
    Load Steamer projections for the given year (e.g., 2026 preseason).
    Includes projected WAR, wOBA, OPS+, ISO, K%, BABIP, etc. for patterns in future undervalued players.
    No salaries here, merge with historical.
    """
    logger.info(f"Loading Steamer projections for {year}...")
    bat_proj = retry_func(steamer_batter_projections, season=year)
    pit_proj = retry_func(steamer_pitcher_projections, season=year)

    # Save raw CSVs
    bat_proj.to_csv(os.path.join(DATA_DIR, f'steamer_batting_{year}.csv'), index=False)
    pit_proj.to_csv(os.path.join(DATA_DIR, f'steamer_pitching_{year}.csv'), index=False)

    logger.info(f"Steamer projections for {year} loaded and saved.")
    return bat_proj, pit_proj


def load_statcast_data(start_date='2025-03-27', end_date='2025-10-05'):
    """
    Load Statcast data for advanced metrics (exit velocity, launch angle, hard hit %, BABIP).
    Default to 2025 full season (adjust for 2026 when available).
    Crucial for patterns like high exit vel / low BABIP (undervalued due to luck).
    """
    logger.info(f"Loading Statcast data from {start_date} to {end_date}...")
    statcast_df = retry_func(statcast, start_dt=start_date, end_dt=end_date)

    # Save raw CSV (can be large, but pandas handles it)
    statcast_df.to_csv(os.path.join(DATA_DIR, 'statcast_2025.csv'), index=False)

    logger.info("Statcast data loaded and saved.")
    return statcast_df


def merge_datasets(batting, pitching, salaries, bat_proj, pit_proj, statcast_df):
    """
    Merge all datasets for a unified DF.
    Key: Merge on playerID/year where possible. This enables pattern detection (e.g., WAR/salary ratio).
    For simplicity, merge historical batting/pitching with salaries, and append projections/statcast separately.
    Full merge logic can be expanded in data_cleaner.py.
    """
    logger.info("Merging datasets...")

    # Example merge: Historical batting with salaries
    historical = pd.merge(batting, salaries, on=['playerID', 'yearID', 'teamID'], how='left')
    historical = pd.merge(historical, pitching, on=['playerID', 'yearID', 'teamID'], suffixes=('_bat', '_pit'),
                          how='outer')

    # Append projections (projections have 'mlbid', map to playerID if needed)
    # For now, save separately; advanced merging in cleaner script

    # Merge Statcast (has 'player_id', needs mapping)
    # Placeholder: Full integration in next steps

    historical.to_csv(os.path.join(DATA_DIR, 'merged_historical.csv'), index=False)
    bat_proj.to_csv(os.path.join(DATA_DIR, 'merged_projections_bat.csv'), index=False)  # Temp
    pit_proj.to_csv(os.path.join(DATA_DIR, 'merged_projections_pit.csv'), index=False)
    statcast_df.to_csv(os.path.join(DATA_DIR, 'merged_statcast.csv'), index=False)

    logger.info("Datasets merged and saved.")
    return historical  # Return main DF for further use


if __name__ == "__main__":
    try:
        # Load all sources
        batting, pitching, salaries = load_lahman_data()
        bat_proj, pit_proj = load_steamer_projections(2026)
        statcast_df = load_statcast_data()  # Adjust dates if needed for 2026 preseason

        # Merge
        merged = merge_datasets(batting, pitching, salaries, bat_proj, pit_proj, statcast_df)

        logger.info("Data loading complete. Check 'data/' folder for CSVs.")
    except Exception as e:
        logger.error(f"Critical error in data loading: {e}")
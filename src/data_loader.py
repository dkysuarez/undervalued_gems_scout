import logging
import os
import time
import pandas as pd
import pybaseball as pyb  # Only for Statcast and FanGraphs (Lahman is now local)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directory (relative to project)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def retry_func(func, max_retries=3, backoff=2, *args, **kwargs):
    """Retry wrapper for flaky network calls (Statcast/FanGraphs)."""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                raise
            logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {backoff ** attempt} seconds...")
            time.sleep(backoff ** attempt)


def load_lahman_local(lahman_dir=os.path.join(DATA_DIR, 'lahman')):
    """
    Load Lahman CSVs from local unzipped folder.
    Manual download required from: https://sabr.org/lahman-database
    Expected version: 2025 release (January 2026), stats up to 2025 season.
    """
    if not os.path.exists(lahman_dir):
        raise FileNotFoundError(
            f"Lahman directory not found at {lahman_dir}.\n"
            "Fix steps:\n"
            "1. Download the ZIP from https://sabr.org/lahman-database\n"
            "2. Unzip ALL contents directly into 'data/lahman/'\n"
            "   (you should see Batting.csv, Salaries.csv, Pitching.csv, People.csv, etc.)\n"
            "3. Run this script again."
        )

    logger.info(f"Loading local Lahman data from: {lahman_dir}")

    batting_path = os.path.join(lahman_dir, 'Batting.csv')
    salaries_path = os.path.join(lahman_dir, 'Salaries.csv')
    pitching_path = os.path.join(lahman_dir, 'Pitching.csv')
    people_path = os.path.join(lahman_dir, 'People.csv')

    # Check required files
    missing = [p for p in [batting_path, salaries_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}. Check unzip process.")

    batting = pd.read_csv(batting_path)
    salaries = pd.read_csv(salaries_path)
    pitching = pd.read_csv(pitching_path) if os.path.exists(pitching_path) else pd.DataFrame()
    people = pd.read_csv(people_path) if os.path.exists(people_path) else None

    # Save copies for convenience
    batting.to_csv(os.path.join(DATA_DIR, 'lahman_batting.csv'), index=False)
    salaries.to_csv(os.path.join(DATA_DIR, 'lahman_salaries.csv'), index=False)

    logger.info(f"Lahman loaded successfully! Batting rows: {len(batting):,}, Salaries rows: {len(salaries):,}")
    return batting, pitching, salaries, people


def load_statcast_data(start_date='2025-03-01', end_date='2025-11-01'):
    """Load Statcast data for advanced metrics (exit velocity, launch angle, hard hit %, BABIP)."""
    logger.info(f"Loading Statcast data ({start_date} to {end_date})...")
    statcast_df = retry_func(pyb.statcast, start_dt=start_date, end_dt=end_date)

    statcast_df.to_csv(os.path.join(DATA_DIR, 'statcast_2025.csv'), index=False)
    logger.info(f"Statcast loaded: {len(statcast_df):,} rows.")
    return statcast_df


def load_fangraphs_stats(year=2025):
    """Load FanGraphs batting and pitching stats (includes WAR, wOBA, OPS+, etc.) for the given year."""
    logger.info(f"Loading FanGraphs stats for batting and pitching ({year})...")
    batting_fg = retry_func(pyb.batting_stats, year, year)
    pitching_fg = retry_func(pyb.pitching_stats, year, year)

    batting_fg.to_csv(os.path.join(DATA_DIR, f'fangraphs_batting_{year}.csv'), index=False)
    pitching_fg.to_csv(os.path.join(DATA_DIR, f'fangraphs_pitching_{year}.csv'), index=False)

    logger.info(f"FanGraphs loaded: Batting {len(batting_fg):,} rows.")
    return batting_fg, pitching_fg


def merge_core_datasets(batting, salaries):
    """Basic merge of batting + salaries to start detecting undervalued players."""
    logger.info("Merging core batting + salaries datasets...")
    merged = pd.merge(batting, salaries, on=['playerID', 'yearID'], how='left')
    merged.to_csv(os.path.join(DATA_DIR, 'merged_batting_salaries.csv'), index=False)
    logger.info(f"Merge saved: {len(merged):,} rows.")
    return merged


if __name__ == "__main__":
    try:
        # Main step: Load local Lahman
        batting, pitching, salaries, people = load_lahman_local()

        # Optional: Load Statcast (may take time, comment out if testing fast)
        # statcast_df = load_statcast_data()

        # Optional: Load FanGraphs for WAR/advanced metrics (highly recommended)
        # fang_bat, fang_pit = load_fangraphs_stats(2025)

        # Initial merge
        merge_core_datasets(batting, salaries)

        logger.info("Data loading completed! Check 'data/' folder for CSVs.")
        logger.info("Next step: Create data_cleaner.py to compute patterns (WAR/salary ratio, etc.).")
    except Exception as e:
        logger.error(f"General error: {e}")
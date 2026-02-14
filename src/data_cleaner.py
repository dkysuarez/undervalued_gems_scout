import logging
import os
import pandas as pd
import pybaseball as pyb

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# People CSV path (from Lahman)
PEOPLE_PATH = os.path.join(DATA_DIR, 'lahman', 'People.csv')  # Adjust if not in subfolder


def load_people():
    """
    Load People.csv for player names (playerID to Name).

    Returns:
        DataFrame with playerID and Name columns, or None if file not found
    """
    if not os.path.exists(PEOPLE_PATH):
        logger.warning(f"People.csv not found at {PEOPLE_PATH}. Names won't be added.")
        return None

    people = pd.read_csv(PEOPLE_PATH)
    # Create full name column by combining first and last name
    people['Name'] = people['nameFirst'].fillna('') + ' ' + people['nameLast'].fillna('')
    people = people[['playerID', 'Name']].dropna(subset=['playerID'])
    logger.info(f"Loaded {len(people):,} player names.")
    return people


def load_fangraphs_advanced(year_start=2015, year_end=2025):
    """
    Load advanced batting stats from FanGraphs.

    Includes key metrics: WAR, wOBA, wRC+, BABIP, ISO, K%, etc.

    Args:
        year_start (int): Starting year for data
        year_end (int): Ending year for data

    Returns:
        DataFrame with FanGraphs advanced stats, or empty DataFrame if error
    """
    logger.info(f"Loading FanGraphs batting stats from {year_start} to {year_end}...")

    try:
        fg_batting = pyb.batting_stats(year_start, year_end)

        # Rename columns for consistency and to avoid special characters
        fg_batting = fg_batting.rename(columns={
            'Season': 'yearID',
            'IDfg': 'fg_id',
            'Name': 'Name',  # Keep as is
            'wRC+': 'wRC_plus'  # Rename to avoid + character in column name
        })

        # Define desired columns (these are the key metrics we want)
        desired_columns = ['yearID', 'Name', 'WAR', 'wOBA', 'wRC_plus', 'BABIP', 'ISO', 'K%']

        # Check which columns actually exist in the dataframe
        existing_columns = [col for col in desired_columns if col in fg_batting.columns]

        logger.info(f"Found columns: {existing_columns}")

        # If wRC_plus doesn't exist, try to use OPS as a fallback
        if 'wRC_plus' not in existing_columns and 'OPS' in fg_batting.columns:
            logger.info("wRC+ not found, using OPS as alternative")
            existing_columns.append('OPS')

        # Select only the columns that exist
        fg_df = fg_batting[existing_columns].copy()

        logger.info(f"FanGraphs loaded successfully: {len(fg_df):,} rows.")
        return fg_df

    except Exception as e:
        logger.error(f"Error loading FanGraphs: {e}. Skipping advanced stats.")
        return pd.DataFrame()


def clean_and_enrich_data():
    """
    Main function: Clean data, enrich with advanced metrics, and compute patterns.

    This function:
    1. Loads merged Lahman batting + salary data
    2. Applies filters (year >= 2015, AB >= 100)
    3. Loads FanGraphs advanced stats
    4. Merges datasets
    5. Computes key metrics and patterns for undervalued player detection
    6. Saves cleaned data to CSV

    Returns:
        DataFrame with cleaned and enriched data
    """
    logger.info("Starting data cleaning and enrichment...")

    # Load merged batting + salaries data from data_loader.py output
    merge_path = os.path.join(DATA_DIR, 'merged_batting_salaries.csv')
    if not os.path.exists(merge_path):
        raise FileNotFoundError(f"Merged file not found: {merge_path}. Run data_loader.py first.")

    df = pd.read_csv(merge_path)
    logger.info(f"Loaded merged data: {len(df):,} rows")

    # Load people data for player names
    people = load_people()

    # Load FanGraphs advanced stats
    fg_df = load_fangraphs_advanced(2015, 2025)

    # Basic cleaning on Lahman merge
    # Filter for recent years only
    df = df[df['yearID'] >= 2015]
    logger.info(f"After year filter (>=2015): {len(df):,} rows")

    # Minimum at-bats to avoid small sample sizes
    df = df[df['AB'] >= 100]
    logger.info(f"After AB filter (>=100): {len(df):,} rows")

    # Convert salary to millions USD and fill NaN values with 0
    df['salary'] = df['salary'].fillna(0) / 1_000_000

    # Compute basic patterns from Lahman data
    df['hits_per_ab'] = df['H'] / df['AB']
    df['iso_lahman'] = (df['2B'] + 2 * df['3B'] + 3 * df['HR']) / df['AB']  # Isolated Power approximation
    df['k_rate_lahman'] = df['SO'] / df['AB'] if 'SO' in df.columns else 0

    # If FanGraphs data is available, merge it with Lahman data
    if not fg_df.empty:
        # Add Name column to Lahman data using people lookup
        if people is not None:
            df['Name'] = df['playerID'].map(people.set_index('playerID')['Name'])
        else:
            df['Name'] = df['playerID']  # Fallback to playerID if no names available

        # Merge on yearID and Name (approximate matching - may miss some players)
        logger.info("Merging Lahman data with FanGraphs advanced stats...")
        merged = pd.merge(df, fg_df, on=['yearID', 'Name'], how='left', suffixes=('_lahman', '_fg'))
        logger.info(f"After merge: {len(merged):,} rows")

        # Fill missing FanGraphs values with 0 or mean as appropriate
        for col in ['WAR', 'wOBA', 'BABIP', 'ISO', 'K%']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)

        # Handle wRC_plus or OPS based on what's available
        if 'wRC_plus' in merged.columns:
            merged['wRC_plus'] = merged['wRC_plus'].fillna(100)  # 100 is league average
        elif 'OPS' in merged.columns:
            merged['OPS'] = merged['OPS'].fillna(merged['OPS'].mean())
    else:
        # If no FanGraphs data, use Lahman data with placeholder values
        merged = df.copy()
        merged['WAR'] = 0  # Placeholder
        merged['wOBA'] = 0
        merged['BABIP'] = 0
        merged['ISO'] = 0
        merged['K%'] = 0
        logger.warning("Using placeholder values for advanced stats (FanGraphs data not available)")

    # KEY PATTERNS FOR UNDERVALUED PLAYER DETECTION

    # 1. Value ratio: WAR per million dollars of salary
    # Higher ratio means more value for less money
    merged['value_ratio'] = merged['WAR'] / (merged['salary'] + 0.1)  # Add 0.1 to avoid division by zero

    # 2. "Bad luck, good talent" pattern: Low BABIP but good wOBA
    # Indicates player is hitting well but balls aren't falling for hits
    if 'BABIP' in merged.columns and 'wOBA' in merged.columns:
        merged['babip_low_high_woba'] = (merged['BABIP'] < 0.280) & (merged['wOBA'] > 0.320)
    else:
        merged['babip_low_high_woba'] = False

    # 3. "Power with contact" pattern: High ISO, low strikeout rate
    # Indicates player makes contact and hits for power
    if 'ISO' in merged.columns and 'K%' in merged.columns:
        merged['power_contact'] = (merged['ISO'] > 0.160) & (merged['K%'] < 0.22)
    else:
        merged['power_contact'] = False

    # 4. "xStats outperforming" would go here if we had expected stats
    # (Would need Statcast data for exit velocity, xBA, xSLG, etc.)

    # 5. Combined score: Weighted combination of multiple factors
    # Normalize components to 0-1 scale where possible
    merged['undervalued_score'] = 0

    # Add value ratio component (normalized)
    if merged['value_ratio'].max() > merged['value_ratio'].min():
        merged['undervalued_score'] += (merged['value_ratio'] - merged['value_ratio'].min()) / \
                                       (merged['value_ratio'].max() - merged['value_ratio'].min()) * 0.4

    # Add pattern indicators
    merged['undervalued_score'] += merged['babip_low_high_woba'].astype(int) * 0.3
    merged['undervalued_score'] += merged['power_contact'].astype(int) * 0.3

    # Sort by value_ratio descending to find most undervalued players
    undervalued = merged.sort_values('value_ratio', ascending=False)

    # Save cleaned data to CSV
    output_path = os.path.join(DATA_DIR, 'cleaned_undervalued_data.csv')
    undervalued.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to: {output_path}")

    # Display top 10 undervalued players
    logger.info("=" * 60)
    logger.info("TOP 10 UNDERVALUED PLAYERS (by value_ratio):")
    logger.info("=" * 60)

    # Select columns to display (only those that exist)
    display_cols = []
    for col in ['Name', 'yearID', 'teamID', 'WAR', 'salary', 'value_ratio', 'BABIP', 'wOBA', 'undervalued_score']:
        if col in undervalued.columns:
            display_cols.append(col)

    logger.info(f"\n{undervalued.head(10)[display_cols].to_string()}")

    return undervalued


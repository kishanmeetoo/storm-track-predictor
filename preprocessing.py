import pandas as pd
from utils import calculate_speed


def load_and_clean_data():
    """
    Load and clean storm dataset from 'storms.csv'.

    - Standardizes column names
    - Combines year/month/day/hour into a datetime column
    - Drops rows missing wind radii info (for reliable modeling)
    - Fills missing categories
    - Sorts the data by storm name and time
    """
    df = pd.read_csv('storms.csv')

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Create datetime column
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']], errors='coerce')

    # Drop rows missing critical data (2004+ storms)
    df = df.dropna(subset=['tropicalstorm_force_diameter', 'hurricane_force_diameter'])

    # Fill missing hurricane categories
    df['category'] = df['category'].fillna('Not a Hurricane')

    # Sort chronologically by storm name and time
    df = df.sort_values(by=['name', 'datetime'])

    return df


def generate_features(df):
    """
    Generate lag features, storm speed, interactions, and lead targets for modeling.

    Returns:
        DataFrame with engineered features ready for training.
    """
    df = df.sort_values(by=['name', 'datetime']).copy()

    # Lag features
    df['lat_lag1'] = df.groupby('name')['lat'].shift(1)
    df['long_lag1'] = df.groupby('name')['long'].shift(1)
    df['wind_lag1'] = df.groupby('name')['wind'].shift(1)
    df['pressure_lag1'] = df.groupby('name')['pressure'].shift(1)

    # Time difference (in hours)
    df['time_diff'] = (df['datetime'] - df.groupby('name')['datetime'].shift(1)).dt.total_seconds() / 3600

    # Drop rows missing lagged values or time_diff
    df = df.dropna(subset=[
        'lat_lag1', 'long_lag1', 'wind_lag1', 'pressure_lag1', 'time_diff'
    ]).copy()

    # Calculate storm speed using geodesic distance
    df['storm_speed'] = df.groupby('name').apply(calculate_speed).explode().astype(float).values

    # Interaction feature
    df['wind_pressure_interaction'] = df['wind'] * df['pressure']

    # Lead targets
    df['lat_lead1'] = df.groupby('name')['lat'].shift(-1)
    df['long_lead1'] = df.groupby('name')['long'].shift(-1)

    # Drop rows missing targets
    df = df.dropna(subset=['lat_lead1', 'long_lead1'])

    return df

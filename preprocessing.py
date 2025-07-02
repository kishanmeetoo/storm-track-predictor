import pandas as pd
import numpy as np
from geopy.distance import geodesic
from utils import calculate_speed


def load_and_clean_data():
    # Load the dataset
    df = pd.read_csv('storms.csv')

    # Check basic info and preview
    print(df.info())
    print(df.head())

    # Strip any extra spaces in column names
    df.columns = df.columns.str.strip().str.lower()

    # Combine year, month, day, hour into one datetime column
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']], errors='coerce')

    # Drop original columns if you want (optional)
    # df.drop(columns=['year', 'month', 'day', 'hour'], inplace=True)

    # Preview the result
    print(df[['datetime', 'lat', 'long', 'wind', 'pressure']].head())

    # Check for missing values
    print(df.isnull().sum())

    # Drop rows missing wind radii info (2004 onwards only)
    df = df.dropna(subset=['tropicalstorm_force_diameter', 'hurricane_force_diameter'])

    # Replace missing hurricane category with label
    df['category'] = df['category'].fillna('Not a Hurricane')

    # Sort data by storm name and datetime
    df = df.sort_values(by=['name', 'year', 'month', 'day', 'hour'])

    return df

def generate_features(df):
    df = df.sort_values(by=['name', 'datetime']).copy()

    # Lag features
    df['lat_lag1'] = df.groupby('name')['lat'].shift(1)
    df['long_lag1'] = df.groupby('name')['long'].shift(1)
    df['wind_lag1'] = df.groupby('name')['wind'].shift(1)
    df['pressure_lag1'] = df.groupby('name')['pressure'].shift(1)

    # Time difference (in hours)
    df['time_diff'] = df['datetime'] - df.groupby('name')['datetime'].shift(1)
    df['time_diff'] = df['time_diff'].dt.total_seconds() / 3600

    # Drop rows with missing lag/time_diff values
    df = df.dropna(subset=[
        'lat_lag1', 'long_lag1', 'wind_lag1', 'pressure_lag1', 'time_diff'
    ]).copy()

    # Calculate storm speed (km/h) using geodesic distance
    from geopy.distance import geodesic

    def calculate_speed(row):
        if (
                pd.notna(row['lat_lag1']) and pd.notna(row['long_lag1']) and
                pd.notna(row['time_diff']) and row['time_diff'] > 0
        ):
            coord1 = (row['lat_lag1'], row['long_lag1'])
            coord2 = (row['lat'], row['long'])
            distance_km = geodesic(coord1, coord2).km
            return distance_km / row['time_diff']  # speed = distance / time
        return None

    df['storm_speed'] = df.apply(calculate_speed, axis=1)

    # Interaction feature
    df['wind_pressure_interaction'] = df['wind'] * df['pressure']

    # Lead targets
    df['lat_lead1'] = df.groupby('name')['lat'].shift(-1)
    df['long_lead1'] = df.groupby('name')['long'].shift(-1)

    # Drop rows with missing targets
    df = df.dropna(subset=['lat_lead1', 'long_lead1'])

    # Sanity check
    print("Missing values after feature generation:\n", df.isnull().sum())

    return df
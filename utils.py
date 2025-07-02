import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, label="Model"):
    """
    Print the Mean Squared Error and R² score of a model.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} MSE: {mse:.4f}")
    print(f"{label} R²: {r2:.4f}")


def calculate_speed(df_group):
    """
    Calculate storm speed between consecutive records (in nautical miles per hour).
    To be used with groupby('name').apply().
    """
    speeds = []
    for i in range(len(df_group)):
        if i == 0:
            speeds.append(0)
            continue

        prev_coords = (df_group.iloc[i - 1]['lat'], df_group.iloc[i - 1]['long'])
        curr_coords = (df_group.iloc[i]['lat'], df_group.iloc[i]['long'])
        time_diff = df_group.iloc[i]['time_diff']

        if pd.isna(time_diff) or time_diff == 0:
            speeds.append(0)
        else:
            distance = geodesic(prev_coords, curr_coords).nautical
            speed = distance / time_diff
            speeds.append(speed)

    return speeds

def get_storm_track_by_id(df, storm_name, year):
    """
    Filter a storm DataFrame by name and year.
    """
    storm_df = df[(df['name'].str.upper() == storm_name.upper()) & (df['year'] == year)].copy()

    if storm_df.empty:
        raise ValueError(f"No data found for storm: {storm_name}, {year}")

    if 'datetime' in storm_df.columns:
        storm_df = storm_df.sort_values('datetime')

    return storm_df.reset_index(drop=True)



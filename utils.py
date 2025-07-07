import pandas as pd
from geopy.distance import geodesic
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, label="Model"):
    """
    Prints the MSE and R² score for a model's predictions.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} MSE: {mse:.4f}")
    print(f"{label} R²: {r2:.4f}")


def calculate_speed(group):
    """
    Calculates storm speed between each pair of consecutive points
    in a storm group (in km/h). To be used with groupby().apply().
    """
    speeds = [0]  # First entry has no previous point

    for i in range(1, len(group)):
        lat1, lon1 = group.iloc[i - 1][['lat', 'long']]
        lat2, lon2 = group.iloc[i][['lat', 'long']]
        time_diff = group.iloc[i]['time_diff']

        if pd.isna(time_diff) or time_diff == 0:
            speeds.append(0)
        else:
            dist_km = geodesic((lat1, lon1), (lat2, lon2)).km
            speeds.append(dist_km / time_diff)

    return pd.Series(speeds, index=group.index)


def get_storm_track_by_id(df, storm_name, year):
    """
    Filters the DataFrame to return a specific storm by name and year,
    sorted by datetime.
    """
    storm_df = df[
        (df['name'].str.upper() == storm_name.upper()) & (df['year'] == year)
    ].copy()

    if storm_df.empty:
        raise ValueError(f"No data found for storm: {storm_name} ({year})")

    return storm_df.sort_values('datetime').reset_index(drop=True)

import seaborn as sns
import matplotlib.pyplot as plt
import folium


def visualize_data(df):
    """
    Basic EDA visualizations for storm dataset:
    - Histogram of storm speed
    - Wind vs Pressure scatter plot
    - Correlation heatmap
    """

    # Storm speed distribution
    sns.histplot(df['storm_speed'], bins=50, kde=True)
    plt.title('Storm Speed Distribution')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Wind vs. Pressure scatter
    sns.scatterplot(data=df, x='pressure', y='wind')
    plt.title('Wind Speed vs. Central Pressure')
    plt.xlabel('Pressure (mb)')
    plt.ylabel('Wind Speed (knots)')
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_predicted_vs_actual_map(actual_df, predicted_df, storm_name):
    """
    Creates and saves a folium map comparing actual vs predicted storm paths.
    """
    start_coords = [actual_df['lat'].iloc[0], actual_df['long'].iloc[0]]
    m = folium.Map(location=start_coords, zoom_start=5)

    # Actual path (blue)
    folium.PolyLine(
        list(zip(actual_df['lat'], actual_df['long'])),
        color='blue', weight=4, opacity=0.6, tooltip='Actual Path'
    ).add_to(m)

    # Predicted path (red)
    folium.PolyLine(
        list(zip(predicted_df['pred_lat'], predicted_df['pred_long'])),
        color='red', weight=4, opacity=0.6, tooltip='Predicted Path'
    ).add_to(m)

    # Start marker
    folium.Marker(
        location=start_coords,
        popup='Start',
        icon=folium.Icon(color='green')
    ).add_to(m)

    filename = f"{storm_name.lower()}_predicted_vs_actual_map.html"
    m.save(filename)
    print(f"Map saved: {filename}")


def plot_florence_wind_prediction(florence_df):
    """
    Line plot of actual vs predicted wind speed for Hurricane Florence.
    """
    _plot_wind(florence_df, title='Florence: Actual vs Predicted Wind Speed')


def plot_wind_prediction(df, storm_name):
    """
    Line plot of actual vs predicted wind speed for a given storm.
    """
    _plot_wind(df, title=f'{storm_name}: Actual vs Predicted Wind Speed')


def _plot_wind(df, title):
    """
    Helper function to plot actual vs predicted wind speed over time.
    """
    if 'pred_wind' not in df.columns:
        raise ValueError("Missing 'pred_wind' in DataFrame.")

    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['wind'], label='Actual', color='blue', marker='o')
    plt.plot(df['datetime'], df['pred_wind'], label='Predicted', color='red', linestyle='--', marker='x')
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('Wind Speed (knots)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

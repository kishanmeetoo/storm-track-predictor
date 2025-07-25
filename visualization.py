import seaborn as sns
import matplotlib.pyplot as plt
import folium
import os


def visualize_data(df):
    """
    Basic EDA visualizations for storm dataset:
    - Histogram of storm speed
    - Wind vs Pressure scatter plot
    - Correlation heatmap
    """
    os.makedirs("plots", exist_ok=True)

    # Storm speed distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['storm_speed'], bins=50, kde=True)
    plt.title('Storm Speed Distribution', fontsize=16)
    plt.xlabel('Speed (km/h)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/storm_speed_distribution.png", dpi=300)
    plt.show()

    # Wind vs. Pressure scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='pressure', y='wind')
    plt.title('Wind Speed vs. Central Pressure', fontsize=16)
    plt.xlabel('Pressure (mb)', fontsize=14)
    plt.ylabel('Wind Speed (knots)', fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/wind_vs_pressure.png", dpi=300)
    plt.show()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png", dpi=300)
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

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['wind'], label='Actual', color='blue', marker='o')
    plt.plot(df['datetime'], df['pred_wind'], label='Predicted', color='red', linestyle='--', marker='x')
    plt.title(title, fontsize=16)
    plt.xlabel('Datetime', fontsize=14)
    plt.ylabel('Wind Speed (knots)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    safe_title = title.lower().replace(" ", "_").replace(":", "")
    plt.savefig(f"plots/{safe_title}.png", dpi=300)
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(df):
    """
    Perform basic visualizations on the storm dataset:
    - Histogram of storm speed
    - Scatter plot of wind vs pressure
    - Correlation matrix heatmap
    """

    # Plot histogram of storm speed to understand its distribution
    sns.histplot(df['storm_speed'], bins=50, kde=True)
    plt.title('Storm Speed Distribution')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    # Scatter plot to explore relationship between wind speed and pressure
    sns.scatterplot(data=df, x='pressure', y='wind')
    plt.title('Wind Speed vs. Central Pressure')
    plt.xlabel('Pressure (millibars)')
    plt.ylabel('Wind Speed (knots)')
    plt.tight_layout()
    plt.show()

    # Select only numeric columns for the correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Matrix of Storm Data')
    plt.tight_layout()
    plt.show()

def plot_predicted_vs_actual_map(actual_df, predicted_df, storm_name):
    import folium

    m = folium.Map(location=[actual_df['lat'].iloc[0], actual_df['long'].iloc[0]], zoom_start=5)

    # Actual path
    actual_coords = list(zip(actual_df['lat'], actual_df['long']))
    folium.PolyLine(actual_coords, color='blue', weight=4, opacity=0.6, tooltip='Actual Path').add_to(m)

    # Predicted path
    predicted_coords = list(zip(predicted_df['pred_lat'], predicted_df['pred_long']))
    folium.PolyLine(predicted_coords, color='red', weight=4, opacity=0.6, tooltip='Predicted Path').add_to(m)

    # Mark start point
    folium.Marker(location=actual_coords[0], popup='Start', icon=folium.Icon(color='green')).add_to(m)

    # Save map with storm name
    filename = f"{storm_name.lower()}_predicted_vs_actual_map.html"
    m.save(filename)
    print(f"Map saved to {filename}")


import matplotlib.pyplot as plt

def plot_florence_wind_prediction(florence_df):
    """
    Plots actual vs predicted wind speed for Hurricane Florence.
    Assumes 'wind' (actual) and 'pred_wind' (predicted) are in the DataFrame.
    """
    if 'pred_wind' not in florence_df.columns:
        raise ValueError("Missing 'pred_wind' column in DataFrame.")

    plt.figure(figsize=(12, 6))
    plt.plot(florence_df['datetime'], florence_df['wind'], label='Actual Wind Speed', color='blue', marker='o')
    plt.plot(florence_df['datetime'], florence_df['pred_wind'], label='Predicted Wind Speed', color='red', linestyle='--', marker='x')
    plt.xlabel('Datetime')
    plt.ylabel('Wind Speed (knots)')
    plt.title('Florence: Actual vs Predicted Wind Speed Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_wind_prediction(df, storm_name):
    """
    Line plot of actual vs predicted wind speed for a given storm.
    """
    if 'pred_wind' not in df.columns:
        raise ValueError("Missing 'pred_wind' in DataFrame.")

    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['wind'], label='Actual Wind Speed', color='blue', marker='o')
    plt.plot(df['datetime'], df['pred_wind'], label='Predicted Wind Speed', color='red', linestyle='--', marker='x')
    plt.xlabel('Datetime')
    plt.ylabel('Wind Speed (knots)')
    plt.title(f'{storm_name}: Actual vs Predicted Wind Speed')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


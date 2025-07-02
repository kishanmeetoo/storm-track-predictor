from preprocessing import load_and_clean_data, generate_features
from visualization import visualize_data
from modeling import (
    train_all_models,
    run_stacking_model,
    predict_wind_speed,
    predict_lat_lon_path_leave_one_out_stacked,
    predict_florence_wind_speed
)
from advanced_models import tune_gradient_boosting
from utils import get_storm_track_by_id
from visualization import (
    plot_predicted_vs_actual_map,
    plot_florence_wind_prediction
)
from modeling import predict_wind_speed_for_storm
from visualization import plot_wind_prediction


# ---- Main Execution ----

# Load and preprocess data
raw_df = load_and_clean_data()
df = generate_features(raw_df)


# Visualize data
visualize_data(df)

# Train individual models and stacking model
train_all_models(df)
run_stacking_model(df)

# Predict wind speed using best model
predict_wind_speed(df)

# Optional: Tune Gradient Boosting
tune_gradient_boosting(df)

# ---- Florence Track Prediction ----

# Florence's ID in HURDAT2
florence_df = get_storm_track_by_id(df, 'Florence', 2018)
florence_predicted = predict_lat_lon_path_leave_one_out_stacked(df, 'Florence', 2018)
plot_predicted_vs_actual_map(florence_df, florence_predicted,  'Florence')

# ---- Florence Wind Speed Prediction ----

# Predict and plot wind speed for Florence
florence_wind_df = predict_florence_wind_speed(df)
plot_predicted_vs_actual_map(florence_df, florence_predicted, 'Florence')

isabel_df = get_storm_track_by_id(raw_df, 'Dorian', 2019)
isabel_predicted = predict_lat_lon_path_leave_one_out_stacked(df, 'Dorian', 2019)
plot_predicted_vs_actual_map(isabel_df, isabel_predicted, 'Dorian')

# Dorian Wind Speed Prediction
dorian_wind_df = predict_wind_speed_for_storm(df, 'Dorian', 2019)
plot_wind_prediction(dorian_wind_df, 'Dorian')




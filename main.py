from preprocessing import load_and_clean_data, generate_features
from visualization import (
    visualize_data,
    plot_predicted_vs_actual_map,
    plot_florence_wind_prediction,
    plot_wind_prediction
)
from modeling import (
    train_all_models,
    run_stacking_model,
    predict_wind_speed,
    predict_lat_lon_path_leave_one_out_stacked,
    predict_florence_wind_speed,
    predict_wind_speed_for_storm
)
from advanced_models import tune_gradient_boosting
from utils import get_storm_track_by_id


def main():
    # Load and preprocess dataset
    raw_df = load_and_clean_data()
    df = generate_features(raw_df)

    # Visualize dataset
    visualize_data(df)

    # Train baseline and stacking models
    train_all_models(df)
    run_stacking_model(df)

    # Predict general wind speeds
    predict_wind_speed(df)

    # Optional: Hyperparameter tuning
    tune_gradient_boosting(df)

    # Predict and plot Florence track
    florence_df = get_storm_track_by_id(df, 'Florence', 2018)
    florence_pred = predict_lat_lon_path_leave_one_out_stacked(df, 'Florence', 2018)
    plot_predicted_vs_actual_map(florence_df, florence_pred, 'Florence')

    # Predict and plot Florence wind speed
    florence_wind = predict_florence_wind_speed(df)
    plot_florence_wind_prediction(florence_wind)

    # Predict and plot Dorian track
    dorian_df = get_storm_track_by_id(raw_df, 'Dorian', 2019)
    dorian_pred = predict_lat_lon_path_leave_one_out_stacked(df, 'Dorian', 2019)
    plot_predicted_vs_actual_map(dorian_df, dorian_pred, 'Dorian')

    # Predict and plot Dorian wind speed
    dorian_wind = predict_wind_speed_for_storm(df, 'Dorian', 2019)
    plot_wind_prediction(dorian_wind, 'Dorian')


if __name__ == '__main__':
    main()

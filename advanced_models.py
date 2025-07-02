from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def tune_gradient_boosting(df):
    """
    Perform hyperparameter tuning using GridSearchCV
    for predicting latitude and longitude with GradientBoostingRegressor.
    """

    features = ['lat', 'long', 'wind', 'pressure', 'storm_speed', 'lat_lag1', 'long_lag1']
    df_model = df.dropna(subset=features + ['lat_lead1', 'long_lead1'])

    X = df_model[features]
    y_lat = df_model['lat_lead1']
    y_long = df_model['long_lead1']

    # Train-test split
    X_train, X_test, y_lat_train, y_lat_test, y_long_train, y_long_test = train_test_split(
        X, y_lat, y_long, test_size=0.2, random_state=42
    )

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }

    print("Tuning Latitude Model...")

    # GridSearch for latitude
    grid_search_lat = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search_lat.fit(X_train, y_lat_train)
    best_lat_model = grid_search_lat.best_estimator_

    lat_preds = best_lat_model.predict(X_test)
    lat_mse = mean_squared_error(y_lat_test, lat_preds)
    lat_r2 = r2_score(y_lat_test, lat_preds)

    print(f"Best Latitude Model MSE: {lat_mse:.4f}")
    print(f"Best Latitude Model R²: {lat_r2:.4f}")

    print("\nTuning Longitude Model...")

    # GridSearch for longitude
    grid_search_long = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search_long.fit(X_train, y_long_train)
    best_long_model = grid_search_long.best_estimator_

    long_preds = best_long_model.predict(X_test)
    long_mse = mean_squared_error(y_long_test, long_preds)
    long_r2 = r2_score(y_long_test, long_preds)

    print(f"Best Longitude Model MSE: {long_mse:.4f}")
    print(f"Best Longitude Model R²: {long_r2:.4f}")

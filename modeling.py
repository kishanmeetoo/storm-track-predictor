import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import evaluate_model


def train_all_models(df):
    """
    Trains and evaluates several models to predict next-step latitude and longitude.
    Models: Linear Regression, Random Forest, Gradient Boosting, SVM, KNN
    """
    features = ['lat', 'long', 'wind', 'pressure', 'storm_speed', 'lat_lag1', 'long_lag1']
    df_model = df.dropna(subset=features + ['lat_lead1', 'long_lead1'])

    X = df_model[features]
    y_lat = df_model['lat_lead1']
    y_long = df_model['long_lead1']

    X_train, X_test, y_lat_train, y_lat_test, y_long_train, y_long_test = train_test_split(
        X, y_lat, y_long, test_size=0.2, random_state=42
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVM': SVR(kernel='rbf'),
        'KNN': KNeighborsRegressor(n_neighbors=8)
    }

    for name, model in models.items():
        # Predict latitude
        model.fit(X_train, y_lat_train)
        lat_preds = model.predict(X_test)
        evaluate_model(y_lat_test, lat_preds, f"{name} - Latitude")

        # Predict longitude
        model.fit(X_train, y_long_train)
        long_preds = model.predict(X_test)
        evaluate_model(y_long_test, long_preds, f"{name} - Longitude")


def run_stacking_model(df):
    """
    Trains and evaluates a stacking model to predict next-step lat/long.
    Base models: GBR, RF, KNN. Meta-model: Linear Regression.
    """
    features = ['lat', 'long', 'wind', 'pressure', 'storm_speed', 'lat_lag1', 'long_lag1']
    df_model = df.dropna(subset=features + ['lat_lead1', 'long_lead1'])

    X = df_model[features]
    y_lat = df_model['lat_lead1']
    y_long = df_model['long_lead1']

    X_train, X_test, y_lat_train, y_lat_test, y_long_train, y_long_test = train_test_split(
        X, y_lat, y_long, test_size=0.2, random_state=42
    )

    base_models = [
        ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('knn', KNeighborsRegressor(n_neighbors=8))
    ]
    meta_model = LinearRegression()

    for label, y_train, y_test in [('Latitude', y_lat_train, y_lat_test), ('Longitude', y_long_train, y_long_test)]:
        stack_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
        stack_model.fit(X_train, y_train)
        preds = stack_model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Stacked {label} MSE: {mse:.4f}, R²: {r2:.4f}")


def predict_wind_speed(df):
    """
    Predicts wind speed using Gradient Boosting on filtered data (post-1979).
    """
    features = ['year', 'month', 'day', 'hour', 'lat', 'long', 'pressure',
                'tropicalstorm_force_diameter', 'hurricane_force_diameter']

    df_wind = df[df['year'] >= 1979].dropna(subset=features + ['wind'])

    X = df_wind[features]
    y = df_wind['wind']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    evaluate_model(y_test, preds, label="Wind Speed Prediction")


def predict_lat_lon_path_leave_one_out_stacked(df, storm_name, year):
    """
    Trains a stacked model excluding one storm, then predicts that storm’s path.
    Returns a DataFrame with predictions.
    """
    features = ['lat', 'long', 'wind', 'pressure', 'storm_speed', 'lat_lag1', 'long_lag1']
    targets = ['lat_lead1', 'long_lead1']

    df = df.dropna(subset=features + targets).copy()

    # Make a deep copy here to avoid SettingWithCopyWarning
    test_df = df[(df['name'].str.upper() == storm_name.upper()) & (df['year'] == year)].copy()
    train_df = df[~((df['name'].str.upper() == storm_name.upper()) & (df['year'] == year))]

    X_train = train_df[features]
    y_lat_train = train_df['lat_lead1']
    y_long_train = train_df['long_lead1']
    X_test = test_df[features]

    base_models = [
        ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('knn', KNeighborsRegressor(n_neighbors=8))
    ]
    meta_model = LinearRegression()

    stack_lat = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    stack_long = StackingRegressor(estimators=base_models, final_estimator=meta_model)

    stack_lat.fit(X_train, y_lat_train)
    stack_long.fit(X_train, y_long_train)

    test_df.loc[:, 'pred_lat'] = stack_lat.predict(X_test)
    test_df.loc[:, 'pred_long'] = stack_long.predict(X_test)

    return test_df


def predict_florence_wind_speed(df):
    """
    Predicts Florence's wind speed using all other storms for training.
    Returns Florence DataFrame with predictions.
    """
    return predict_wind_speed_for_storm(df, 'Florence', 2018)


def predict_wind_speed_for_storm(df, storm_name, year):
    """
    Predicts wind speed for a specific storm using leave-one-storm-out approach.
    Returns storm DataFrame with predicted wind column.
    """
    features = ['year', 'month', 'day', 'hour', 'lat', 'long', 'pressure',
                'tropicalstorm_force_diameter', 'hurricane_force_diameter']

    storm_df = df[(df['name'].str.upper() == storm_name.upper()) & (df['year'] == year)].copy()
    train_df = df[~((df['name'].str.upper() == storm_name.upper()) & (df['year'] == year))].copy()

    train_df = train_df.dropna(subset=features + ['wind'])
    storm_df = storm_df.dropna(subset=features)

    X_train = train_df[features]
    y_train = train_df['wind']
    X_test = storm_df[features]

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    storm_df['pred_wind'] = model.predict(X_test)
    return storm_df

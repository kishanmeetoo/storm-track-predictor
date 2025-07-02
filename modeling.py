from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from utils import evaluate_model



def train_all_models(df):
    """
    Trains and evaluates several models (Linear Regression, RF, GB, SVM, KNN)
    to predict next-step latitude and longitude.
    """

    # --- STEP 1: Define features and targets ---
    features = ['lat', 'long', 'wind', 'pressure', 'storm_speed', 'lat_lag1', 'long_lag1']
    df_model = df.dropna(subset=features + ['lat_lead1', 'long_lead1'])

    X = df_model[features]
    y_lat = df_model['lat_lead1']
    y_long = df_model['long_lead1']

    # --- STEP 2: Train-test split ---
    X_train, X_test, y_lat_train, y_lat_test, y_long_train, y_long_test = train_test_split(
        X, y_lat, y_long, test_size=0.2, random_state=42
    )

    # --- STEP 3: Define models ---
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Machine': SVR(kernel='rbf'),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=8)
    }

    # --- STEP 4: Train and evaluate each model ---
    for name, model in models.items():
        model.fit(X_train, y_lat_train)
        lat_preds = model.predict(X_test)
        lat_mse = mean_squared_error(y_lat_test, lat_preds)
        lat_r2 = r2_score(y_lat_test, lat_preds)

        model.fit(X_train, y_long_train)
        long_preds = model.predict(X_test)
        long_mse = mean_squared_error(y_long_test, long_preds)
        long_r2 = r2_score(y_long_test, long_preds)

        evaluate_model(y_lat_test, lat_preds, f"{name} Latitude")
        evaluate_model(y_long_test, long_preds, f"{name} Longitude")


def run_stacking_model(df):
    """
    Trains a stacking model using GBR, RF, and KNN as base models
    and Linear Regression as the meta-model.
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
        ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('knn', KNeighborsRegressor(n_neighbors=8))
    ]
    meta_model = LinearRegression()

    # -------- Latitude Stacking --------
    stack_lat_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    stack_lat_model.fit(X_train, y_lat_train)
    lat_preds_stack = stack_lat_model.predict(X_test)
    mse_lat_stack = mean_squared_error(y_lat_test, lat_preds_stack)
    r2_lat_stack = r2_score(y_lat_test, lat_preds_stack)

    print(f"Stacked Latitude MSE: {mse_lat_stack:.4f}, R²: {r2_lat_stack:.4f}")

    # -------- Longitude Stacking --------
    stack_long_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    stack_long_model.fit(X_train, y_long_train)
    long_preds_stack = stack_long_model.predict(X_test)
    mse_long_stack = mean_squared_error(y_long_test, long_preds_stack)
    r2_long_stack = r2_score(y_long_test, long_preds_stack)

    print(f"Stacked Longitude MSE: {mse_long_stack:.4f}, R²: {r2_long_stack:.4f}")


def predict_wind_speed(df):
    """
    Predicts wind speed using Gradient Boosting Regressor
    based on storm attributes and wind radii.
    """

    # --- Load Cleaned Hurricane Dataset ---
    df_wind = df.copy()

    # --- Filter out early years (done previously) ---
    df_wind = df_wind[df_wind['year'] >= 1979]

    # --- Define Features and Label ---
    features_wind = ['year', 'month', 'day', 'hour', 'lat', 'long', 'pressure',
                     'tropicalstorm_force_diameter', 'hurricane_force_diameter']

    df_wind = df_wind.dropna(subset=features_wind + ['wind'])
    X = df_wind[features_wind]
    y = df_wind['wind']

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train Gradient Boosting Regressor ---
    wind_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    wind_model.fit(X_train, y_train)

    # --- Predictions and Metrics ---
    wind_preds = wind_model.predict(X_test)
    wind_mse = mean_squared_error(y_test, wind_preds)
    wind_r2 = r2_score(y_test, wind_preds)

    print(f"Wind Speed Prediction MSE: {wind_mse:.4f}")
    print(f"Wind Speed Prediction R²: {wind_r2:.4f}")

def predict_lat_lon_path_leave_one_out_stacked(full_df, test_storm_name, test_year):
    """
    Train on all storms except one using stacked model, then predict the left-out storm's path.
    Returns prediction DataFrame for the test storm.
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor

    features = ['lat', 'long', 'wind', 'pressure', 'storm_speed', 'lat_lag1', 'long_lag1']
    target_cols = ['lat_lead1', 'long_lead1']

    # Drop rows with missing features or targets
    full_df = full_df.dropna(subset=features + target_cols).copy()

    # Separate test storm
    test_df = full_df[(full_df['name'].str.upper() == test_storm_name.upper()) & (full_df['year'] == test_year)].copy()
    train_df = full_df[~((full_df['name'].str.upper() == test_storm_name.upper()) & (full_df['year'] == test_year))].copy()

    X_train = train_df[features]
    y_lat_train = train_df['lat_lead1']
    y_long_train = train_df['long_lead1']

    X_test = test_df[features]

    # Define base and meta models
    base_models = [
        ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('knn', KNeighborsRegressor(n_neighbors=8))
    ]
    meta_model = LinearRegression()

    # Train stacking models
    stack_lat_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    stack_long_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

    stack_lat_model.fit(X_train, y_lat_train)
    stack_long_model.fit(X_train, y_long_train)

    test_df['pred_lat'] = stack_lat_model.predict(X_test)
    test_df['pred_long'] = stack_long_model.predict(X_test)

    return test_df

from sklearn.ensemble import GradientBoostingRegressor

def predict_florence_wind_speed(df):
    """
    Trains a wind speed model on all storms except Florence,
    and returns a DataFrame of Florence with predicted wind speed.
    """
    features = ['year', 'month', 'day', 'hour', 'lat', 'long', 'pressure',
                'tropicalstorm_force_diameter', 'hurricane_force_diameter']

    # Extract Florence and training data
    florence_df = df[(df['name'].str.upper() == 'FLORENCE') & (df['year'] == 2018)].copy()
    train_df = df[~((df['name'].str.upper() == 'FLORENCE') & (df['year'] == 2018))].copy()

    # Clean data
    train_df = train_df.dropna(subset=features + ['wind'])
    florence_df = florence_df.dropna(subset=features)

    X_train = train_df[features]
    y_train = train_df['wind']
    X_test = florence_df[features]

    # Train model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    florence_df['pred_wind'] = model.predict(X_test)

    return florence_df

from sklearn.ensemble import GradientBoostingRegressor

def predict_wind_speed_for_storm(df, storm_name, year):
    """
    Trains a wind speed model on all storms except the specified one,
    and returns the storm's DataFrame with predicted wind speed.
    """
    features = ['year', 'month', 'day', 'hour', 'lat', 'long', 'pressure',
                'tropicalstorm_force_diameter', 'hurricane_force_diameter']

    # Filter storm and training data
    storm_df = df[(df['name'].str.upper() == storm_name.upper()) & (df['year'] == year)].copy()
    train_df = df[~((df['name'].str.upper() == storm_name.upper()) & (df['year'] == year))].copy()

    # Clean
    train_df = train_df.dropna(subset=features + ['wind'])
    storm_df = storm_df.dropna(subset=features)

    X_train = train_df[features]
    y_train = train_df['wind']
    X_test = storm_df[features]

    # Train model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    storm_df['pred_wind'] = model.predict(X_test)

    return storm_df

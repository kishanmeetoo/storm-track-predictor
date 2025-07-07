from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def tune_gradient_boosting(df):
    """
    Performs GridSearchCV to tune Gradient Boosting Regressor
    for predicting next-step latitude and longitude.
    """
    features = ['lat', 'long', 'wind', 'pressure', 'storm_speed', 'lat_lag1', 'long_lag1']
    targets = ['lat_lead1', 'long_lead1']
    df = df.dropna(subset=features + targets)

    X = df[features]
    y_lat = df['lat_lead1']
    y_long = df['long_lead1']

    X_train, X_test, y_lat_train, y_lat_test, y_long_train, y_long_test = train_test_split(
        X, y_lat, y_long, test_size=0.2, random_state=42
    )

    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }

    def tune_target(y_train, y_test, label):
        grid = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Best {label} Model MSE: {mse:.4f}")
        print(f"Best {label} Model R²: {r2:.4f}")

    print("Tuning Latitude Model...")
    tune_target(y_lat_train, y_lat_test, 'Latitude')

    print("\nTuning Longitude Model...")
    tune_target(y_long_train, y_long_test, 'Longitude')

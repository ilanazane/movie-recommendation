from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from data_processing import load_data
import joblib

# def train(train_config):

# Process data
train_data, test_data = load_data(
    "data/ml-latest-small/ratings.csv", "data/ml-latest-small/movies.csv"
)

# define features and target variables
x_train, x_test = train_data[["userId", "movieId"]], test_data[["userId", "movieId"]]
y_train, y_test = train_data["rating"], test_data["rating"]

# Train part (HPT/Optimize/compute metrics)

# train random forest regressor model
model = RandomForestRegressor()

# set the parameters that we want to search
params = {
    "max_depth": [10, 20, 25],
    "min_samples_leaf": [5, 10, 20, 25, 30],
    "n_estimators": [10, 15, 20, 25, 30],
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=4,
    n_jobs=-1,
    verbose=1,
    scoring="neg_mean_squared_error",
)
grid_search.fit(x_train, y_train)

print("finished trainings")


# evaluate model
y_pred = grid_search.predict(x_test)
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)


# Save model
joblib.dump(grid_search, "movie_model.pkl")
print("model saved")

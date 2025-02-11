import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# read in datasets
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
movies = pd.read_csv("data/ml-latest-small/movies.csv")

# merge datasets
data = pd.merge(ratings, movies, on="movieId")
data = data.drop(columns=["timestamp"])  # drop timestamp column

# encode categorical features
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

# fit and transform data
data["userId"] = user_encoder.fit_transform(data["userId"])
data["movieId"] = movie_encoder.fit_transform(data["movieId"])

# create train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print("train:", len(train_data))
print("test:", len(test_data))

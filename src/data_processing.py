import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(
    ratings_file: str = "data/ml-latest-small/ratings.csv",
    movies_file: str = "data/ml-latest-small/movies.csv",
):
    """
    Description of this function.

    :param datafile: descriptioin
    :returns: Loaded data used for training.
    """

    # read in datasets
    ratings = pd.read_csv(ratings_file)
    movies = pd.read_csv(movies_file)

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

    return train_data, test_data

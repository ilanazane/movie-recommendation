from fastapi import FastAPI
import joblib
import pandas as pd
from data_processing import load_data

# load trained model and data
model = joblib.load("movie_recommender.pkl")

ratings, movies = load_data("ml-latest-small/ratings.csv", "ml-latest-small/movies.csv")

print("model loaded")


app = FastAPI()


@app.get("/")
def home():
    return {"message": "Movie Recommendation API"}


@app.get("/recommend/")
def recommend(user_id: int, n: int = 5):
    all_movies = ratings["movieId"].unique()
    # find all movies that user has rated
    user_movies = ratings[ratings["userId"] == user_id]["movieId"].values
    unseen_movies = [m for m in all_movies if m not in user_movies]
    print("hello")

    # predict ratings for unseen movies
    predictions = model.predict(
        pd.DataFrame(
            {"userId": [user_id] * len(unseen_movies), "movieId": unseen_movies}
        )
    )

    # get top N recommendations
    top_movies = sorted(
        zip(unseen_movies, predictions), key=lambda x: x[1], reverse=True
    )[:n]
    recommended_movie_ids = [m[0] for m in top_movies]

    # map back to movie titles
    recommended_movies = movies[movies["movieId"].isin(recommended_movie_ids)][
        ["movieId", "title"]
    ]
    return recommended_movies.to_dict(orient="records")

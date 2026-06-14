import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_data(movies_path="data/movies.csv", ratings_path="data/ratings.csv"):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


def build_user_movie_matrix(ratings):
    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    return matrix


def get_similar_users(user_id, matrix, top_n=5):
    if user_id not in matrix.index:
        return pd.Series(dtype=float)
    filled = matrix.fillna(0)
    sim_matrix = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim_matrix, index=matrix.index, columns=matrix.index)
    similar = sim_df[user_id].drop(user_id).sort_values(ascending=False)
    return similar.head(top_n)


def recommend_movies(user_id, matrix, movies_df, top_n=10):
    similar_users = get_similar_users(user_id, matrix, top_n=20)

    if similar_users.empty:
        return pd.DataFrame()

    # Movies our user has already seen
    seen = set(matrix.loc[user_id].dropna().index)

    # Weighted average rating from similar users for unseen movies
    scores = {}
    for other_user, similarity in similar_users.items():
        other_ratings = matrix.loc[other_user].dropna()
        for movie_id, rating in other_ratings.items():
            if movie_id not in seen:
                if movie_id not in scores:
                    scores[movie_id] = {"weighted_sum": 0, "sim_sum": 0}
                scores[movie_id]["weighted_sum"] += similarity * rating
                scores[movie_id]["sim_sum"] += similarity

    if not scores:
        return pd.DataFrame()

    predictions = {
        mid: data["weighted_sum"] / data["sim_sum"]
        for mid, data in scores.items()
        if data["sim_sum"] > 0
    }

    pred_series = pd.Series(predictions).sort_values(ascending=False).head(top_n)
    result = movies_df[movies_df["movieId"].isin(pred_series.index)].copy()
    result["predicted_rating"] = result["movieId"].map(pred_series)
    result = result.sort_values("predicted_rating", ascending=False)
    return result[["title", "genres", "predicted_rating"]]

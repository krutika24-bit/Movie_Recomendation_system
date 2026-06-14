"""
Run this script to generate sample data for testing.
Or download the real MovieLens dataset from:
https://grouplens.org/datasets/movielens/latest/
and place movies.csv and ratings.csv in the data/ folder.
"""
import pandas as pd
import numpy as np
import os

def generate_sample_data():
    os.makedirs("data", exist_ok=True)

    movies_data = {
        "movieId": list(range(1, 21)),
        "title": [
            "Inception (2010)", "The Dark Knight (2008)", "Interstellar (2014)",
            "RRR (2022)", "3 Idiots (2009)", "Dangal (2016)",
            "Avengers: Endgame (2019)", "The Matrix (1999)", "Parasite (2019)",
            "Joker (2019)", "The Shawshank Redemption (1994)", "Forrest Gump (1994)",
            "The Godfather (1972)", "Pulp Fiction (1994)", "Fight Club (1999)",
            "Goodfellas (1990)", "The Silence of the Lambs (1991)", "Schindler's List (1993)",
            "Whiplash (2014)", "La La Land (2016)",
        ],
        "genres": [
            "Action|Sci-Fi|Thriller", "Action|Crime|Drama",
            "Adventure|Drama|Sci-Fi", "Action|Drama",
            "Comedy|Drama", "Biography|Drama|Sport",
            "Action|Adventure|Sci-Fi", "Action|Sci-Fi",
            "Drama|Thriller", "Crime|Drama|Thriller",
            "Drama", "Drama|Romance",
            "Crime|Drama", "Crime|Drama",
            "Drama|Thriller", "Biography|Crime|Drama",
            "Crime|Drama|Thriller", "Biography|Drama|History",
            "Drama|Music", "Comedy|Drama|Music",
        ],
    }
    movies = pd.DataFrame(movies_data)
    movies.to_csv("data/movies.csv", index=False)

    np.random.seed(42)
    n_users, n_movies = 50, 20
    ratings_list = []
    for user_id in range(1, n_users + 1):
        n_rated = np.random.randint(5, 15)
        movie_ids = np.random.choice(range(1, n_movies + 1), n_rated, replace=False)
        for movie_id in movie_ids:
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])
            ratings_list.append({"userId": user_id, "movieId": int(movie_id), "rating": float(rating)})

    ratings = pd.DataFrame(ratings_list)
    ratings.to_csv("data/ratings.csv", index=False)
    print("Sample data created: data/movies.csv and data/ratings.csv")
    print(f"  {len(movies)} movies, {len(ratings)} ratings from {n_users} users")


if __name__ == "__main__":
    generate_sample_data()

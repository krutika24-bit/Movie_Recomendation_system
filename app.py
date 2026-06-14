import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.collaborative import load_data, build_user_movie_matrix, recommend_movies
from src.content_based import recommend_similar_movies

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.caption("Collaborative filtering + Content-based filtering")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    if not os.path.exists("data/movies.csv"):
        st.error("data/movies.csv not found. Run: python src/generate_sample_data.py")
        st.stop()
    return load_data()

@st.cache_data
def get_matrix(ratings):
    return build_user_movie_matrix(ratings)

movies, ratings = get_data()
matrix = get_matrix(ratings)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
mode = st.sidebar.radio(
    "Recommendation type",
    ["👥 Collaborative filtering", "🎭 Content-based filtering"],
)

# ── Collaborative filtering tab ────────────────────────────────────────────────
if mode == "👥 Collaborative filtering":
    st.subheader("👥 Collaborative Filtering")
    st.markdown(
        "Finds users with similar taste to yours, then recommends movies they loved that you haven't seen."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        all_users = sorted(matrix.index.tolist())
        user_id = st.selectbox("Select a user ID", all_users)
        top_n = st.slider("Number of recommendations", 5, 20, 10)

    with col2:
        user_ratings = matrix.loc[user_id].dropna()
        st.markdown(f"**User {user_id} has rated {len(user_ratings)} movies**")
        if len(user_ratings) > 0:
            sample = movies[movies["movieId"].isin(user_ratings.index)].copy()
            sample["your_rating"] = sample["movieId"].map(user_ratings)
            sample = sample.sort_values("your_rating", ascending=False).head(5)
            st.dataframe(
                sample[["title", "genres", "your_rating"]].rename(
                    columns={"title": "Movie", "genres": "Genres", "your_rating": "Rating ★"}
                ),
                hide_index=True,
                use_container_width=True,
            )

    if st.button("Get recommendations", type="primary"):
        with st.spinner("Finding similar users..."):
            recs = recommend_movies(user_id, matrix, movies, top_n=top_n)

        if recs.empty:
            st.warning("Not enough data to recommend movies for this user.")
        else:
            st.success(f"Top {len(recs)} recommendations for User {user_id}")
            recs_display = recs.copy()
            recs_display["predicted_rating"] = recs_display["predicted_rating"].round(2)
            st.dataframe(
                recs_display.rename(
                    columns={
                        "title": "Movie",
                        "genres": "Genres",
                        "predicted_rating": "Predicted Rating ★",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

# ── Content-based tab ──────────────────────────────────────────────────────────
else:
    st.subheader("🎭 Content-Based Filtering")
    st.markdown(
        "Finds movies with similar genres/themes to a movie you already like."
    )

    movie_list = sorted(movies["title"].tolist())
    selected_movie = st.selectbox("Pick a movie you like", movie_list)
    top_n = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("Find similar movies", type="primary"):
        with st.spinner("Analysing movie content..."):
            recs, matched = recommend_similar_movies(selected_movie, movies, top_n=top_n)

        if recs.empty:
            st.warning("No similar movies found.")
        else:
            st.success(f"Movies similar to **{matched}**")
            st.dataframe(
                recs.rename(
                    columns={
                        "title": "Movie",
                        "genres": "Genres",
                        "similarity": "Similarity Score",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Dataset: MovieLens (grouplens.org) or sample data from `src/generate_sample_data.py`"
)

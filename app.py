import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# -----------------------------
# Load Data
# -----------------------------
ratings = pd.read_csv("ml-latest-small/ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/ml-latest-small/movies.csv")

st.title("🎬 Movie Recommender System (SVD - Mean Centered)")

# -----------------------------
# Create User-Item Matrix
# -----------------------------
user_item_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

# -----------------------------
# Mean Centering (IMPORTANT)
# -----------------------------
user_means = user_item_matrix.mean(axis=1)

centered_matrix = user_item_matrix.sub(user_means, axis=0)
centered_matrix = centered_matrix.fillna(0)

movie_ids = centered_matrix.columns.tolist()

# -----------------------------
# Train SVD (cached)
# -----------------------------
@st.cache_resource
def train_svd(matrix):
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd.fit(matrix)
    return svd

svd = train_svd(centered_matrix)

# -----------------------------
# UI Input
# -----------------------------
st.subheader("🎯 Rate movies you like")

selected_movies = st.multiselect(
    "Select movies:",
    movies['title'].values
)

user_ratings = {}

for movie in selected_movies:
    rating = st.slider(f"Rate '{movie}'", 1, 5, 3)
    user_ratings[movie] = rating

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_svd(user_ratings, n=10):

    # Step 1: Create user vector
    user_vector = np.zeros(len(movie_ids))

    for movie, rating in user_ratings.items():
        movie_row = movies[movies['title'] == movie]
        if len(movie_row) == 0:
            continue

        movie_id = movie_row['movieId'].values[0]

        if movie_id in movie_ids:
            idx = movie_ids.index(movie_id)
            user_vector[idx] = rating

    # Step 2: Compute user's mean
    if len(user_ratings) > 0:
        user_mean = np.mean(list(user_ratings.values()))
    else:
        user_mean = 3.0

    # Step 3: Mean-center user vector
    user_vector_centered = user_vector.copy()
    user_vector_centered[user_vector_centered != 0] -= user_mean

    # Step 4: Project to latent space
    user_vector_centered = user_vector_centered.reshape(1, -1)
    latent_user = svd.transform(user_vector_centered)

    # Step 5: Reconstruct ratings
    predicted_centered = latent_user.dot(svd.components_)
    predicted = predicted_centered + user_mean

    scores = predicted.flatten()

    # Step 6: Remove already rated movies
    for movie in user_ratings.keys():
        movie_id = movies[movies['title'] == movie]['movieId'].values[0]
        if movie_id in movie_ids:
            idx = movie_ids.index(movie_id)
            scores[idx] = -1

    # Step 7: Clip to valid range
    scores = np.clip(scores, 0, 5)

    # Step 8: Top N
    top_indices = np.argsort(scores)[::-1][:n]

    recommended = []
    for i in top_indices:
        title = movies[movies['movieId'] == movie_ids[i]]['title'].values[0]
        recommended.append({
            "title": title,
            "score": round(scores[i], 2)
        })

    return recommended

# -----------------------------
# Button
# -----------------------------
if st.button("🚀 Recommend"):

    if len(user_ratings) == 0:
        st.warning("Please select and rate at least one movie")

    else:
        recs = recommend_svd(user_ratings)

        st.subheader("🎯 Recommendations")

        for rec in recs:
            st.write(f"🎬 {rec['title']}  ⭐ {rec['score']}")
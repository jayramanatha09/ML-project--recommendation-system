import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Data
# -----------------------------
ratings = pd.read_csv("ml-latest-small/ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/ml-latest-small/movies.csv")

st.title("🎬 Movie Recommender System")

# -----------------------------
# Sidebar
# -----------------------------
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["SVD", "Similarity", "Baseline", "Compare All"]
)

top_k = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# -----------------------------
# User Input
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
# Build User-Item Matrix
# -----------------------------
user_item_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

# -----------------------------
# SIMILARITY MODEL (Item-based)
# -----------------------------
user_item_filled = user_item_matrix.fillna(0)
similarity_matrix = cosine_similarity(user_item_filled.T)

movie_ids = user_item_matrix.columns.tolist()

def recommend_similarity(user_ratings, n=10):
    scores = np.zeros(len(movie_ids))

    for movie, rating in user_ratings.items():
        movie_row = movies[movies['title'] == movie]
        if len(movie_row) == 0:
            continue

        movie_id = movie_row['movieId'].values[0]

        if movie_id in movie_ids:
            idx = movie_ids.index(movie_id)
            sim_scores = similarity_matrix[idx]

            adjusted = rating - 3  # mean-centering
            scores += sim_scores * adjusted

    # remove watched
    for movie in user_ratings:
        movie_id = movies[movies['title'] == movie]['movieId'].values[0]
        if movie_id in movie_ids:
            idx = movie_ids.index(movie_id)
            scores[idx] = -1

    top_indices = np.argsort(scores)[::-1][:n]

    recs = []
    for i in top_indices:
        title = movies[movies['movieId'] == movie_ids[i]]['title'].values[0]
        recs.append({"title": title, "score": round(scores[i], 3)})

    return recs

# -----------------------------
# BASELINE MODEL
# -----------------------------
def recommend_baseline(n=10):
    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count'])
    movie_stats = movie_stats[movie_stats['count'] >= 50]

    top = movie_stats.sort_values(by='mean', ascending=False).head(n)

    recs = []
    for movie_id, row in top.iterrows():
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        recs.append({"title": title, "score": round(row['mean'], 2)})

    return recs

# -----------------------------
# SVD MODEL (Mean-Centered)
# -----------------------------
user_means = user_item_matrix.mean(axis=1)
centered_matrix = user_item_matrix.sub(user_means, axis=0).fillna(0)

@st.cache_resource
def train_svd(matrix):
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd.fit(matrix)
    return svd

svd = train_svd(centered_matrix)

def recommend_svd(user_ratings, n=10):
    user_vector = np.zeros(len(movie_ids))

    for movie, rating in user_ratings.items():
        movie_row = movies[movies['title'] == movie]
        if len(movie_row) == 0:
            continue

        movie_id = movie_row['movieId'].values[0]

        if movie_id in movie_ids:
            idx = movie_ids.index(movie_id)
            user_vector[idx] = rating

    # user mean
    user_mean = np.mean(list(user_ratings.values())) if user_ratings else 3.0

    # mean-center
    user_vector_centered = user_vector.copy()
    user_vector_centered[user_vector_centered != 0] -= user_mean

    # project
    latent_user = svd.transform(user_vector_centered.reshape(1, -1))

    # reconstruct
    predicted_centered = latent_user.dot(svd.components_)
    scores = (predicted_centered + user_mean).flatten()

    # remove watched
    for movie in user_ratings:
        movie_id = movies[movies['title'] == movie]['movieId'].values[0]
        if movie_id in movie_ids:
            idx = movie_ids.index(movie_id)
            scores[idx] = -1

    scores = np.clip(scores, 0, 5)

    top_indices = np.argsort(scores)[::-1][:n]

    recs = []
    for i in top_indices:
        title = movies[movies['movieId'] == movie_ids[i]]['title'].values[0]
        recs.append({"title": title, "score": round(scores[i], 2)})

    return recs

# -----------------------------
# Display Helper
# -----------------------------
def display(title, recs):
    st.subheader(title)
    for r in recs:
        st.write(f"🎬 {r['title']}  ⭐ {r['score']}")

# -----------------------------
# MAIN BUTTON
# -----------------------------
if st.button("🚀 Recommend"):

    if len(user_ratings) == 0:
        st.warning("Please rate at least one movie")

    else:

        if model_choice == "SVD":
            display("SVD Recommendations", recommend_svd(user_ratings, top_k))

        elif model_choice == "Similarity":
            display("Similarity Recommendations", recommend_similarity(user_ratings, top_k))

        elif model_choice == "Baseline":
            display("Baseline Recommendations", recommend_baseline(top_k))

        elif model_choice == "Compare All":

            col1, col2, col3 = st.columns(3)

            with col1:
                display("SVD", recommend_svd(user_ratings, top_k))

            with col2:
                display("Similarity", recommend_similarity(user_ratings, top_k))

            with col3:
                display("Baseline", recommend_baseline(top_k))
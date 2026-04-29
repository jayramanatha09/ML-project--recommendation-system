import streamlit as st
import pickle
import pandas as pd

# Load your data/model
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended = []
    for i in movie_list:
        recommended.append(movies.iloc[i[0]].title)
    return recommended

# UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Choose a movie",
    movies['title'].values
)

if st.button("Recommend"):
    results = recommend(selected_movie)
    for movie in results:
        st.write(movie)
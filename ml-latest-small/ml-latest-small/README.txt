# 🎬 Movie Recommender System (SVD-Based)

## 📌 Overview

This project implements a personalized movie recommendation system using Collaborative Filtering with Singular Value Decomposition (SVD).

Users can:
- Select multiple movies
- Provide ratings (1–5)
- Receive personalized movie recommendations

The system predicts how much a user would like unseen movies using latent factor modeling.

---

## 🚀 Features

- Multi-movie rating input
- SVD-based collaborative filtering
- Predicted ratings (0–5 scale)
- Real-time recommendations using Streamlit
- Handles new users (cold-start problem)

---

## 🧠 How It Works

### 1. Data Representation

A user-item matrix is created from the dataset:

User × Movie → Rating

---

### 2. Mean Centering

To remove user bias:

centered_matrix = user_item_matrix - user_mean

This ensures users who rate consistently high or low are normalized.

---

### 3. SVD Decomposition

The matrix is factorized into:

User Factors × Movie Factors

This helps learn hidden preferences such as:
- genre liking
- style preference
- viewing patterns

---

### 4. New User Input

When a user provides ratings:
- A user vector is created
- It is mean-centered
- Projected into latent space using SVD

---

### 5. Prediction

Ratings are reconstructed using:

predicted = latent_user × components + user_mean

This gives predicted ratings for all movies.

---

### 6. Recommendation

- Already rated movies are removed
- Movies are sorted by predicted rating
- Top N movies are recommended

---

## 📊 Output

The system displays:
- Movie title
- Predicted rating (0–5)

---

## 📁 Dataset

MovieLens Latest Small Dataset

- 100,836 ratings
- 9,742 movies
- 610 users
- Rating scale: 0.5 to 5.0

Files used:
- ratings.csv
- movies.csv

---

## ⚙️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn (TruncatedSVD)

---

## ▶️ How to Run

1. Install dependencies:

pip install pandas numpy scikit-learn streamlit

2. Run the app:

streamlit run app.py

---

## 🧪 Example Workflow

1. Select movies:
   Toy Story, Jumanji

2. Rate them:
   Toy Story → 5
   Jumanji → 4

3. Click "Recommend"

4. Output:
   Shrek → 4.3
   Finding Nemo → 4.1

---

## 🔍 Key Concepts

- Collaborative Filtering: Uses user behavior instead of content
- Latent Factors: Hidden patterns learned by SVD
- Cold Start: New users handled using input ratings

---

## ⚠️ Limitations

- Does not use movie content (genres, tags)
- Cannot recommend completely new movies without ratings
- Depends on dataset quality

---

## 🚀 Future Improvements

- Hybrid model (SVD + content-based)
- Add movie posters
- Add evaluation metrics (Precision@K, RMSE)
- Improve UI/UX

---

## 🧠 Conclusion

This system demonstrates a collaborative filtering approach using SVD to learn user preferences and generate personalized recommendations.

---

## 📚 Reference

F. Maxwell Harper and Joseph A. Konstan.  
The MovieLens Datasets: History and Context.  
ACM Transactions on Interactive Intelligent Systems (TiiS), 2015.
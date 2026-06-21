# Anime Recommendation System

A content-based recommendation system that suggests similar anime using cosine similarity based on genre, rating, episodes, and member count.

## Files

* `code.py` — Python code for the recommendation system
* `app.py` — Streamlit web application
* `anime.csv` — Dataset
* `requirements.txt` — Required Python libraries

The application will open in your browser and recommend similar anime based on your input.

## How It Works

* Handles missing values in the dataset.
* Converts genres into numerical features using TF-IDF.
* Scales rating, episodes, and members using MinMaxScaler.
* Computes cosine similarity between anime.
* Recommends the most similar anime.


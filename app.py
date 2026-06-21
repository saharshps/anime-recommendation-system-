import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.title("Anime Recommendation System")

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\sahar\OneDrive\Desktop\ds and ml files\anime.csv")

    df["genre"] = df["genre"].fillna("")
    df["rating"] = df["rating"].fillna(df["rating"].mean())

    df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
    df["episodes"] = df["episodes"].fillna(df["episodes"].median())

    df["members"] = df["members"].fillna(df["members"].median())

    return df

@st.cache_data
def build_similarity(df):
    tfidf = TfidfVectorizer()
    genre_matrix = tfidf.fit_transform(df["genre"])

    scaler = MinMaxScaler()
    num_features = scaler.fit_transform(df[["rating", "episodes", "members"]])

    features = np.hstack((genre_matrix.toarray(), num_features))

    similarity = cosine_similarity(features)
    return similarity

df = load_data()
similarity = build_similarity(df)

def recommend_anime(anime_name, threshold, top_n):
    match = df[df["name"].str.lower() == anime_name.lower()]

    if len(match) == 0:
        return None

    idx = match.index[0]

    scores = list(enumerate(similarity[idx]))

    recommendations = []

    for i, score in scores:
        if score >= threshold and i != idx:
            recommendations.append((df.iloc[i]["name"], score))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    return recommendations[:top_n]

anime_name = st.text_input("Enter Anime Name")
threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.05)
top_n = st.slider("Number of Recommendations", 1, 20, 5)

if st.button("Recommend"):
    if anime_name.strip() == "":
        st.warning("Please enter an anime name")
    else:
        results = recommend_anime(anime_name, threshold, top_n)

        if results is None:
            st.error("Anime not found")
        elif len(results) == 0:
            st.info("No similar anime found at this threshold, try lowering it")
        else:
            result_df = pd.DataFrame(results, columns=["Anime Name", "Similarity Score"])
            st.table(result_df)
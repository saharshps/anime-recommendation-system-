import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

df= pd.read_csv(r"C:\Users\sahar\OneDrive\Desktop\ds and ml files\anime.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df["genre"] = df["genre"].fillna("")
df["rating"] = df["rating"].fillna(df["rating"].mean())

df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
df["episodes"] = df["episodes"].fillna(df["episodes"].median())

df["members"] = df["members"].fillna(df["members"].median())

tfidf = TfidfVectorizer()

genre_matrix = tfidf.fit_transform(df["genre"])

scaler = MinMaxScaler()

num_features = scaler.fit_transform(
    df[["rating", "episodes", "members"]]
)

features = np.hstack((genre_matrix.toarray(), num_features))

similarity = cosine_similarity(features)

def recommend_anime(anime_name, threshold=0.5):
    
    match = df[df["name"].str.lower() == anime_name.lower()]
    
    if len(match) == 0:
        print("Anime not found")
        return
    
    idx = match.index[0]
    
    scores = list(enumerate(similarity[idx]))
    
    recommendations = []
    
    for i, score in scores:
        if score >= threshold and i != idx:
            recommendations.append((df.iloc[i]["name"], score))
    
    recommendations = sorted(
        recommendations,
        key=lambda x: x[1],
        reverse=True
    )
    
    return recommendations[:5]

print("\nRecommendations:\n")
print(recommend_anime(input("Enter Anime Name: "), 0.5))

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

y_true = [1, 1, 1, 0, 0]
y_pred = [1, 1, 0, 0, 1]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
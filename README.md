# source.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Path to the u.item file (update this if your path is different)
DATA_PATH = "/content/u.item"

# Movie genres as per MovieLens 100K documentation
GENRES = [
    "Unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def load_movies(data_path):
    # u.item is pipe-separated, with the first 5 columns as:
    # movie id | movie title | release date | video release date | IMDb URL
    # followed by 19 binary genre flags
    columns = [
        "movie_id", "title", "release_date", "video_release_date", "imdb_url"
    ] + GENRES
    movies = pd.read_csv(
        data_path, sep='|', encoding='latin-1', header=None, names=columns
    )
    # Combine genres into a string for each movie
    def get_genres(row):
        return ' '.join([genre for genre in GENRES if row[genre] == 1])
    movies["genres"] = movies.apply(get_genres, axis=1)
    # Use only title and genres for content-based filtering
    movies["content"] = movies["title"] + " " + movies["genres"]
    return movies[["movie_id", "title", "genres", "content"]]

def build_feature_matrix(movies):
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_matrix = vectorizer.fit_transform(movies["content"])
    return feature_matrix, vectorizer

def recommend_movies(user_preferences, movies, feature_matrix, vectorizer, top_n=10):
    user_vec = vectorizer.transform([user_preferences])
    cosine_similarities = linear_kernel(user_vec, feature_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    recommendations = [(movies.iloc[i]["title"], movies.iloc[i]["genres"], cosine_similarities[i]) for i in top_indices]
    return recommendations

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Please download MovieLens 100K and set the correct path.")
        return

    print("Loading movies dataset...")
    movies = load_movies(DATA_PATH)
    feature_matrix, vectorizer = build_feature_matrix(movies)

    print("Welcome to the MovieLens AI-Driven Movie Recommender!")
    print("Type your favorite genres, keywords, or themes (e.g., 'action sci-fi space'):")
    user_input = input("Your preferences: ")

    recommendations = recommend_movies(user_input, movies, feature_matrix, vectorizer)
    print("\nTop movie recommendations for you:")
    for idx, (title, genres, score) in enumerate(recommendations, 1):
        print(f"{idx}. {title} [{genres}] (match score: {score:.2f})")

if __name__ == "__main__":
    main()


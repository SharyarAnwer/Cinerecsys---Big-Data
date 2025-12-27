import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import col
from models.actor_similarity_utils import get_top_actors
from pyspark.ml.linalg import DenseVector

# -------------------------------------------------------------------
# Spark Session
# -------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("CineRecSys-CLI") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# -------------------------------------------------------------------
# HDFS Paths
# -------------------------------------------------------------------
ALS_MODEL_PATH = "hdfs://localhost:9000/cinerecsys/models/als_model"
MOVIES_PATH = "hdfs://localhost:9000/cinerecsys/processed/movies"

# -------------------------------------------------------------------
# Load ALS Model
# -------------------------------------------------------------------
als_model = ALSModel.load(ALS_MODEL_PATH)

# -------------------------------------------------------------------
# Load Movies Metadata
# -------------------------------------------------------------------
movies_df = spark.read.parquet(MOVIES_PATH)
movies_dict = {row.movieId: row.title for row in movies_df.collect()}

# -------------------------------------------------------------------
# Helper function for reading input in Windows + Python
# -------------------------------------------------------------------
def read_input(prompt):
    print(prompt, end='', flush=True)
    return sys.stdin.readline().strip()

# -------------------------------------------------------------------
# Option 1: Predict Rating
# -------------------------------------------------------------------
def predict_rating():
    try:
        user_id = int(read_input("Enter userId: "))
        movie_id = int(read_input("Enter movieId: "))
    except ValueError:
        print("Invalid input. Please enter numeric IDs.\n")
        return

    test_df = spark.createDataFrame([(user_id, movie_id)], ["userId", "movieId"])
    prediction_row = als_model.transform(test_df).collect()[0]
    prediction = prediction_row.prediction
    movie_title = movies_dict.get(movie_id, f"Movie {movie_id}")
    print(f"\nPredicted rating for user {user_id} on '{movie_title}': {prediction:.2f}\n")

# -------------------------------------------------------------------
# Option 2: Recommend Top-N Movies for a User
# -------------------------------------------------------------------
def recommend_movies():
    try:
        user_id = int(read_input("Enter userId: "))
        top_n = int(read_input("Enter number of recommendations (e.g., 10): "))
    except ValueError:
        print("Invalid input. Please enter numeric IDs.\n")
        return

    # Generate top-N recommendations using ALS model
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    recs_df = als_model.recommendForUserSubset(user_df, top_n)

    if recs_df.count() == 0:
        print(f"No recommendations found for user {user_id}\n")
        return

    recs_list = recs_df.collect()[0].recommendations
    print(f"\nTop {top_n} Recommended Movies for User {user_id}:\n")
    for idx, row in enumerate(recs_list, 1):
        movie_title = movies_dict.get(row.movieId, f"Movie {row.movieId}")
        print(f"{idx}. {movie_title} ({row.rating:.2f})")
    print("")

def recommend_actors():
    # HDFS paths for index tables
    GENRE_INDEX_PATH = "hdfs://localhost:9000/cinerecsys/processed/genre_index_mapping"
    DIRECTOR_INDEX_PATH = "hdfs://localhost:9000/cinerecsys/processed/director_index_mapping"

    # Load as DataFrames
    genre_index_df = spark.read.parquet(GENRE_INDEX_PATH)
    director_index_df = spark.read.parquet(DIRECTOR_INDEX_PATH)

    # Convert to Python dict for fast lookup
    genre_index_dict = {row.genre: row.genre_index for row in genre_index_df.collect()}
    director_index_dict = {row.director: row.director_index for row in director_index_df.collect()}

    # Ask user for movie info
    genre = read_input("Enter main genre of the movie: ").strip()
    director = read_input("Enter director of the movie: ").strip()

    if genre not in genre_index_dict:
        print(f"Genre '{genre}' not found in indices.\n")
        return
    if director not in director_index_dict:
        print(f"Director '{director}' not found in indices.\n")
        return

    # Build movie feature vector
    movie_vector = DenseVector([
        0.0,                          # avg_rating unknown
        float(genre_index_dict[genre]),
        1.0,                          # genre_count (can be 1)
        float(director_index_dict[director]),
        1.0                           # director_count (can be 1)
    ])

    # Get top-5 actors
    top_actors = get_top_actors(movie_vector, top_k=5, spark=spark)
    print(f"\nTop 5 Recommended Actors for movie '{genre}' by '{director}':\n")
    top_actors.show(truncate=False)


# -------------------------------------------------------------------
# CLI Main Menu
# -------------------------------------------------------------------
def main_menu():
    while True:
        print("CineRecSys CLI")
        print("1) Predict user rating")
        print("2) Recommend movies for user")
        print("3) Recommend actors for a movie")
        print("4) Exit")
        choice = read_input("Enter choice: ")

        if choice == "1":
            predict_rating()
        elif choice == "2":
            recommend_movies()
        elif choice == "3":
            recommend_actors()
        elif choice == "4":
            print("Exiting CineRecSys CLI...")
            break
        else:
            print("Invalid choice. Try again.\n")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    main_menu()
    spark.stop()
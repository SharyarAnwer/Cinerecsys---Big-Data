from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, explode, split


def main():
    spark = SparkSession.builder \
        .appName("CineRecSys - Actor Feature Engineering") \
        .getOrCreate()

    # ------------------------------------------------
    # HDFS Paths
    # ------------------------------------------------
    movies_path = "hdfs://localhost:9000/cinerecsys/processed/movies"
    ratings_path = "hdfs://localhost:9000/cinerecsys/processed/ratings"
    cast_path = "hdfs://localhost:9000/cinerecsys/processed/cast"
    directors_path = "hdfs://localhost:9000/cinerecsys/processed/directors"

    output_path = "hdfs://localhost:9000/cinerecsys/processed/actor_features"

    # ------------------------------------------------
    # Load datasets
    # ------------------------------------------------
    movies_df = spark.read.parquet(movies_path)        # movieId, title, genres (STRING)
    ratings_df = spark.read.parquet(ratings_path)      # userId, movieId, rating
    cast_df = spark.read.parquet(cast_path)            # movieId, actorId, actorName
    directors_df = spark.read.parquet(directors_path)  # movieId, directorId

    # ------------------------------------------------
    # Prepare movies: convert genres STRING → ARRAY
    # ------------------------------------------------
    movies_df = movies_df.withColumn(
        "genres_array",
        split(col("genres"), "\\|")
    )

    # ------------------------------------------------
    # Join actor → movie → ratings
    # ------------------------------------------------
    actor_movies_df = cast_df \
        .join(movies_df, on="movieId", how="left") \
        .join(ratings_df.select("movieId", "rating"), on="movieId", how="left")

    # ------------------------------------------------
    # 1️⃣ Average movie rating per actor
    # ------------------------------------------------
    actor_avg_rating_df = actor_movies_df.groupBy("actorId") \
        .agg(avg("rating").alias("avg_movie_rating"))

    # ------------------------------------------------
    # 2️⃣ Genre frequency per actor
    # ------------------------------------------------
    actor_genres_df = actor_movies_df \
        .withColumn("genre", explode(col("genres_array"))) \
        .groupBy("actorId", "genre") \
        .agg(count("*").alias("genre_count"))

    actor_genres_pivot_df = actor_genres_df.groupBy("actorId") \
        .pivot("genre") \
        .sum("genre_count") \
        .fillna(0)

    # ------------------------------------------------
    # 3️⃣ Director collaboration frequency per actor
    # ------------------------------------------------
    actor_directors_df = cast_df \
        .join(directors_df, on="movieId", how="left") \
        .groupBy("actorId", "directorId") \
        .agg(count("*").alias("collaboration_count"))

    actor_directors_pivot_df = actor_directors_df.groupBy("actorId") \
        .pivot("directorId") \
        .sum("collaboration_count") \
        .fillna(0)

    # ------------------------------------------------
    # Combine all actor features
    # ------------------------------------------------
    actor_features_df = actor_avg_rating_df \
        .join(actor_genres_pivot_df, on="actorId", how="left") \
        .join(actor_directors_pivot_df, on="actorId", how="left") \
        .fillna(0)

    # ------------------------------------------------
    # Save actor feature dataset
    # ------------------------------------------------
    actor_features_df.write \
        .mode("overwrite") \
        .parquet(output_path)

    print("\nActor feature dataset successfully saved to HDFS:")
    print(output_path)

    spark.stop()


if __name__ == "__main__":
    main()

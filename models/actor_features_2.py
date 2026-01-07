from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, explode, split, row_number
)
from pyspark.sql.window import Window


def main():
    spark = SparkSession.builder \
        .appName("CineRecSys - Actor Feature Engineering") \
        .config("spark.sql.codegen.wholeStage", "false") \
        .getOrCreate()

    # ------------------------------------------------
    # HDFS Paths
    # ------------------------------------------------
    movies_path = "hdfs://localhost:9000/cinerecsys/processed/movies"
    ratings_path = "hdfs://localhost:9000/cinerecsys/processed/ratings"
    cast_path = "hdfs://localhost:9000/cinerecsys/processed/cast2"
    directors_path = "hdfs://localhost:9000/cinerecsys/processed/directors"

    temp_actor_ratings = "hdfs://localhost:9000/cinerecsys/temp/actor_ratings"
    temp_actor_genres = "hdfs://localhost:9000/cinerecsys/temp/actor_genres"
    temp_actor_directors = "hdfs://localhost:9000/cinerecsys/temp/actor_directors"

    output_path = "hdfs://localhost:9000/cinerecsys/processed/actor_features_2"

    TOP_K_DIRECTORS = 5

    # ------------------------------------------------
    # Load datasets
    # ------------------------------------------------
    movies_df = spark.read.parquet(movies_path)
    ratings_df = spark.read.parquet(ratings_path)
    cast_df = spark.read.parquet(cast_path)
    directors_df = spark.read.parquet(directors_path)

    # ------------------------------------------------
    # Prepare movies: genres STRING → ARRAY
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

    # ============================================================
    # STAGE 1️⃣ Average movie rating per actor
    # ============================================================
    actor_avg_rating_df = actor_movies_df.groupBy("actorId") \
        .agg(avg("rating").alias("avg_movie_rating"))

    actor_avg_rating_df \
        .repartition(200) \
        .write.mode("overwrite") \
        .parquet(temp_actor_ratings)

    # ============================================================
    # STAGE 2️⃣ Genre frequency per actor
    # ============================================================
    actor_genres_df = actor_movies_df \
        .withColumn("genre", explode(col("genres_array"))) \
        .groupBy("actorId", "genre") \
        .agg(count("*").alias("genre_count"))

    actor_genres_pivot_df = actor_genres_df.groupBy("actorId") \
        .pivot("genre") \
        .sum("genre_count") \
        .fillna(0)

    actor_genres_pivot_df \
        .repartition(200) \
        .write.mode("overwrite") \
        .parquet(temp_actor_genres)

    # ============================================================
    # STAGE 3️⃣ Top-K Director Collaborations per Actor (SAFE)
    # ============================================================
    actor_directors_df = cast_df \
        .join(directors_df, on="movieId", how="left") \
        .groupBy("actorId", "directorId") \
        .agg(count("*").alias("collaboration_count"))

    # Rank directors per actor
    window = Window.partitionBy("actorId") \
        .orderBy(col("collaboration_count").desc())

    top_directors_df = actor_directors_df \
        .withColumn("rank", row_number().over(window)) \
        .filter(col("rank") <= TOP_K_DIRECTORS)

    # Pivot on rank (bounded width!)
    actor_directors_pivot_df = top_directors_df.groupBy("actorId") \
        .pivot("rank", list(range(1, TOP_K_DIRECTORS + 1))) \
        .agg(
            avg("directorId").alias("directorId"),
            avg("collaboration_count").alias("collab_count")
        ) \
        .fillna(0)

    actor_directors_pivot_df \
        .repartition(200) \
        .write.mode("overwrite") \
        .parquet(temp_actor_directors)

    # ============================================================
    # FINAL STAGE 🔗 Join materialized features
    # ============================================================
    actor_features_df = spark.read.parquet(temp_actor_ratings) \
        .join(spark.read.parquet(temp_actor_genres), on="actorId", how="left") \
        .join(spark.read.parquet(temp_actor_directors), on="actorId", how="left") \
        .fillna(0)

    actor_features_df \
        .repartition(200) \
        .write.mode("overwrite") \
        .parquet(output_path)

    print("\nActor feature dataset successfully saved to HDFS:")
    print(output_path)

    spark.stop()


if __name__ == "__main__":
    main()

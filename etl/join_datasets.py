from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def main():
    spark = SparkSession.builder \
        .appName("CineRecSys - Join Datasets") \
        .getOrCreate()

    # HDFS paths
    ratings_path = "hdfs://localhost:9000/cinerecsys/processed/ratings"
    movies_path = "hdfs://localhost:9000/cinerecsys/processed/movies"
    tmdb_path = "hdfs://localhost:9000/cinerecsys/processed/tmdb"
    output_path = "hdfs://localhost:9000/cinerecsys/processed/joined"

    # ----------------------------
    # Load datasets
    # ----------------------------
    ratings_df = spark.read.parquet(ratings_path)
    movies_df = spark.read.parquet(movies_path)
    tmdb_df = spark.read.parquet(tmdb_path)

    # ----------------------------
    # Prepare movies dataframe
    # ----------------------------
    # Cast movieId to integer
    movies_df = movies_df.withColumn("movieId", col("movieId").cast("int"))

    # Drop MovieLens genres to avoid column conflicts
    movies_df = movies_df.drop("genres")

    # ----------------------------
    # Merge MovieLens movies with TMDB metadata
    # ----------------------------
    enriched_movies = movies_df.join(
        tmdb_df.select("movieId", "genres", "director", "actors"),
        on="movieId",
        how="left"
    )

    # ----------------------------
    # Join ratings with enriched movie metadata
    # ----------------------------
    joined_df = ratings_df.join(
        enriched_movies,
        on="movieId",
        how="left"
    )

    # ----------------------------
    # Save joined dataset
    # ----------------------------
    joined_df.write \
        .mode("overwrite") \
        .parquet(output_path)

    # Optional validation (Windows-safe)
    joined_df.select("userId", "movieId", "rating", "title", "genres", "director", "actors").show(5, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()

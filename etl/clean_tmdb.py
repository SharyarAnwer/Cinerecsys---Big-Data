from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, explode, collect_list, when
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, ArrayType
)

def main():
    spark = SparkSession.builder \
        .appName("CineRecSys - Clean TMDB Metadata") \
        .getOrCreate()

    # HDFS paths
    movies_path = "hdfs://localhost:9000/cinerecsys/raw/tmdb_5000_movies.csv"
    credits_path = "hdfs://localhost:9000/cinerecsys/raw/tmdb_5000_credits.csv"
    output_path = "hdfs://localhost:9000/cinerecsys/processed/tmdb"

    # Load CSVs with proper quoting for JSON fields
    movies_df = spark.read \
        .option("header", "true") \
        .option("quote", '"') \
        .option("escape", '"') \
        .option("multiLine", "true") \
        .csv(movies_path)

    credits_df = spark.read \
        .option("header", "true") \
        .option("quote", '"') \
        .option("escape", '"') \
        .option("multiLine", "true") \
        .csv(credits_path)

    # ----------------------------
    # Define JSON schemas
    # ----------------------------
    genre_schema = ArrayType(
        StructType([
            StructField("id", IntegerType()),
            StructField("name", StringType())
        ])
    )

    cast_schema = ArrayType(
        StructType([
            StructField("cast_id", IntegerType()),
            StructField("name", StringType()),
            StructField("character", StringType())
        ])
    )

    crew_schema = ArrayType(
        StructType([
            StructField("job", StringType()),
            StructField("name", StringType())
        ])
    )

    # ----------------------------
    # Parse JSON columns
    # ----------------------------
    movies_parsed = movies_df.withColumn(
        "genres_parsed",
        from_json(col("genres"), genre_schema)
    )

    credits_parsed = credits_df \
        .withColumn("cast_parsed", from_json(col("cast"), cast_schema)) \
        .withColumn("crew_parsed", from_json(col("crew"), crew_schema))

    # ----------------------------
    # Extract genres
    # ----------------------------
    genres_df = movies_parsed \
        .select(col("id").alias("movieId"), explode("genres_parsed").alias("g")) \
        .groupBy("movieId") \
        .agg(collect_list("g.name").alias("genres"))

    # ----------------------------
    # Extract director
    # ----------------------------
    director_df = credits_parsed \
        .select(col("movie_id").alias("movieId"), explode("crew_parsed").alias("c")) \
        .where(col("c.job") == "Director") \
        .select("movieId", col("c.name").alias("director"))

    # ----------------------------
    # Extract top 3 actors
    # ----------------------------
    actors_df = credits_parsed \
        .select(col("movie_id").alias("movieId"), explode("cast_parsed").alias("a")) \
        .groupBy("movieId") \
        .agg(collect_list("a.name").alias("actors"))

    # ----------------------------
    # Join everything
    # ----------------------------
    tmdb_clean = movies_df \
        .select(col("id").alias("movieId"), "title") \
        .join(genres_df, "movieId", "left") \
        .join(director_df, "movieId", "left") \
        .join(actors_df, "movieId", "left")

    # ----------------------------
    # Cast movieId to integer for compatibility with MovieLens
    # ----------------------------
    tmdb_clean = tmdb_clean.withColumn("movieId", col("movieId").cast(IntegerType()))

    # ----------------------------
    # Save as Parquet
    # ----------------------------
    tmdb_clean.write \
        .mode("overwrite") \
        .parquet(output_path)

    # Validation output (Unicode-safe)
    tmdb_clean.select("movieId", "title").show(5)

    spark.stop()

if __name__ == "__main__":
    main()

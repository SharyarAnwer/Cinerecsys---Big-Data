from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col

def main():
    spark = SparkSession.builder \
        .appName("CineRecSys - Clean Ratings ETL") \
        .getOrCreate()

    # HDFS paths
    input_path = "hdfs://localhost:9000/cinerecsys/raw/ratings.dat"
    output_path = "hdfs://localhost:9000/cinerecsys/processed/ratings"

    # Read raw ratings data
    ratings_raw = spark.read.text(input_path)

    # Parse and clean
    ratings_clean = ratings_raw.select(
        split(col("value"), "::").getItem(0).cast("int").alias("userId"),
        split(col("value"), "::").getItem(1).cast("int").alias("movieId"),
        split(col("value"), "::").getItem(2).cast("float").alias("rating")
    )

    # Write as Parquet
    ratings_clean.write \
        .mode("overwrite") \
        .parquet(output_path)

    # Show sample output (for validation)
    ratings_clean.show(5)

    spark.stop()

if __name__ == "__main__":
    main()

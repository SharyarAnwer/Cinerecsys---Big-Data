from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col

def main():
    spark = SparkSession.builder \
        .appName("CineRecSys - Clean Movies ETL") \
        .getOrCreate()

    # HDFS paths
    input_path = "hdfs://localhost:9000/cinerecsys/raw/movies.dat"
    output_path = "hdfs://localhost:9000/cinerecsys/processed/movies"

    # Read raw movies data
    movies_raw = spark.read.text(input_path)

    # Parse and clean
    movies_clean = movies_raw.select(
        split(col("value"), "::").getItem(0).cast("int").alias("movieId"),
        split(col("value"), "::").getItem(1).alias("title"),
        split(col("value"), "::").getItem(2).alias("genres")
    )

    # Write as Parquet
    movies_clean.write \
        .mode("overwrite") \
        .parquet(output_path)

    # Show sample output for validation
    movies_clean.show(5, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()

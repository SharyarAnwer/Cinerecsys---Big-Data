from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_json
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType


def main():
    spark = SparkSession.builder \
        .appName("Extract Cast Data") \
        .getOrCreate()

    # -----------------------------
    # Input & Output Paths
    # -----------------------------
    input_path = "data/raw/tmdb_5000_credits.csv"
    output_path = "hdfs://localhost:9000/cinerecsys/processed/cast"

    # -----------------------------
    # Read TMDB credits
    # -----------------------------
    credits_df = spark.read.option("header", "true").csv(input_path)

    # -----------------------------
    # Define cast JSON schema
    # -----------------------------
    cast_schema = ArrayType(
        StructType([
            StructField("cast_id", IntegerType(), True),
            StructField("character", StringType(), True),
            StructField("credit_id", StringType(), True),
            StructField("gender", IntegerType(), True),
            StructField("id", IntegerType(), True),      # actorId
            StructField("name", StringType(), True),     # actorName
            StructField("order", IntegerType(), True)
        ])
    )

    # -----------------------------
    # Parse & explode cast array
    # -----------------------------
    cast_df = credits_df \
        .withColumn("cast_array", from_json(col("cast"), cast_schema)) \
        .select(
            col("movie_id").cast("int").alias("movieId"),
            explode(col("cast_array")).alias("actor")
        ) \
        .select(
            col("movieId"),
            col("actor.id").alias("actorId"),
            col("actor.name").alias("actorName")
        )

    # -----------------------------
    # Write to HDFS
    # -----------------------------
    cast_df.write.mode("overwrite").parquet(output_path)

    print("Cast dataset successfully written to HDFS:")
    print(output_path)

    spark.stop()


if __name__ == "__main__":
    main()
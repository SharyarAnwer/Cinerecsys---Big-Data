from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, explode
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType

def main():
    spark = SparkSession.builder \
        .appName("CineRecSys - Extract Directors") \
        .getOrCreate()

    # ----------------------------
    # Input path (TMDB credits)
    # ----------------------------
    credits_path = "data/raw/tmdb_5000_credits.csv"

    credits_df = spark.read \
        .option("header", True) \
        .option("multiLine", True) \
        .option("escape", "\"") \
        .csv(credits_path)

    # ----------------------------
    # Define crew schema
    # ----------------------------
    crew_schema = ArrayType(
        StructType([
            StructField("credit_id", StringType(), True),
            StructField("department", StringType(), True),
            StructField("gender", IntegerType(), True),
            StructField("id", IntegerType(), True),
            StructField("job", StringType(), True),
            StructField("name", StringType(), True)
        ])
    )

    # ----------------------------
    # Parse crew JSON
    # ----------------------------
    crew_df = credits_df \
        .withColumn("crew_parsed", from_json(col("crew"), crew_schema)) \
        .withColumn("crew_member", explode(col("crew_parsed")))

    # ----------------------------
    # Filter directors
    # ----------------------------
    directors_df = crew_df \
        .filter(col("crew_member.job") == "Director") \
        .select(
            col("movie_id").cast("int").alias("movieId"),
            col("crew_member.id").alias("directorId"),
            col("crew_member.name").alias("directorName")
        ) \
        .dropDuplicates()

    # ----------------------------
    # Write to HDFS
    # ----------------------------
    output_path = "hdfs://localhost:9000/cinerecsys/processed/directors"

    directors_df.write \
        .mode("overwrite") \
        .parquet(output_path)

    print("\nDirectors dataset saved to:", output_path)
    spark.stop()

if __name__ == "__main__":
    main()

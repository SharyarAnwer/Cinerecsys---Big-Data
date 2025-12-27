import sys
import io

# Fix Windows console encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_json, regexp_replace, expr
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType

def main():
    spark = SparkSession.builder \
        .appName("Extract Cast Data") \
        .getOrCreate()

    input_path = "data/raw/tmdb_5000_credits.csv"
    output_path = "hdfs://localhost:9000/cinerecsys/processed/cast2"

    # Read CSV
    credits_df = spark.read.option("header", "true") \
                           .option("escape", "\"") \
                           .option("multiLine", "true") \
                           .csv(input_path)

    print("Original data (first row, full cast column):")
    credits_df.select("movie_id", "cast").show(1, truncate=100)  # Use truncate

    # More aggressive JSON cleaning
    credits_df = credits_df.withColumn(
        "cast_clean",
        regexp_replace(
            regexp_replace(
                regexp_replace(col("cast"), r'^"|"$', ''),
                '""', '"'
            ),
            r"'", "'"
        )
    )

    # DEBUG: Check cleaned JSON
    print("\n=== DEBUGGING: Cleaned JSON (first 2 rows) ===")
    credits_df.select("movie_id", "cast_clean").show(2, truncate=100)

    # Define schema
    cast_schema = ArrayType(
        StructType([
            StructField("cast_id", IntegerType(), True),
            StructField("character", StringType(), True),
            StructField("credit_id", StringType(), True),
            StructField("gender", IntegerType(), True),
            StructField("id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("order", IntegerType(), True)
        ])
    )

    # Parse JSON
    cast_df = credits_df \
        .withColumn("cast_array", from_json(col("cast_clean"), cast_schema))
    
    # DEBUG: Check if parsing worked (with truncate to avoid encoding issues)
    print("\n=== DEBUGGING: After from_json ===")
    cast_df.select("movie_id", "cast_array").show(5, truncate=100)
    
    # Count non-null arrays
    non_null_count = cast_df.filter(col("cast_array").isNotNull()).count()
    print(f"\nRows with non-null cast_array: {non_null_count}")
    
    # Now apply filters and explode
    cast_df = cast_df \
        .filter(col("cast_array").isNotNull()) \
        .select(
            col("movie_id").cast("int").alias("movieId"),
            explode(col("cast_array")).alias("actor")
        ) \
        .select(
            col("movieId"),
            col("actor.id").alias("actorId"),
            col("actor.name").alias("actorName")
        )

    print("\n=== Final Result ===")
    cast_df.show(10, truncate=50)  # Use truncate
    
    total_count = cast_df.count()
    print(f"\nTotal cast records: {total_count}")

    # Write to HDFS
    if total_count > 0:
        cast_df.write.mode("overwrite").parquet(output_path)
        print(f"✅ Cast dataset written to: {output_path}")
    else:
        print("⚠️ No records to write!")

    spark.stop()

if __name__ == "__main__":
    main()

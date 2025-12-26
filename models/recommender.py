from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import col, explode

def main():
    spark = SparkSession.builder \
        .appName("CineRecSys - Personalized Recommendations with Precision@10") \
        .getOrCreate()

    # ----------------------------
    # Load test ratings dataset
    # ----------------------------
    test_path = "hdfs://localhost:9000/cinerecsys/processed/ratings"
    test_df = spark.read.parquet(test_path)

    # ----------------------------
    # Load trained ALS model
    # ----------------------------
    model_path = "hdfs://localhost:9000/cinerecsys/models/als_model"
    als_model = ALSModel.load(model_path)

    # ----------------------------
    # Generate Top-10 recommendations for all users in test set
    # ----------------------------
    users_df = test_df.select("userId").distinct()
    top_n = 10
    recs_df = als_model.recommendForUserSubset(users_df, top_n)

    # ----------------------------
    # Explode recommendations to (userId, movieId) pairs
    # ----------------------------
    recs_exp = recs_df.select(
        col("userId"),
        explode(col("recommendations")).alias("rec")
    ).select(
        col("userId"),
        col("rec.movieId").alias("movieId")
    )

    # ----------------------------
    # Ground truth: movies actually rated by users in test set
    # ----------------------------
    test_gt = test_df.select("userId", "movieId").distinct()

    # ----------------------------
    # Compute hits: recommended & actually rated
    # ----------------------------
    hits_df = recs_exp.join(test_gt, ["userId", "movieId"])

    # ----------------------------
    # Precision@10 per user
    # ----------------------------
    from pyspark.sql import functions as F

    user_precision = hits_df.groupBy("userId") \
        .agg(F.count("movieId").alias("hits")) \
        .join(users_df, "userId") \
        .withColumn("precision_at_10", col("hits") / top_n)

    # ----------------------------
    # Average Precision@10 across all users
    # ----------------------------
    avg_precision = user_precision.agg(F.avg("precision_at_10")).collect()[0][0]
    print(f"\nAverage Precision@10 for ALS model: {avg_precision:.4f}\n")

    # ----------------------------
    # Show top-10 recommended movies + Precision@10 for a single user
    # ----------------------------
    single_user_id = 1  # change to desired user ID
    top_n_user = recs_df.filter(col("userId") == single_user_id).collect()[0].recommendations

    # Compute hits for this user
    user_test_movies = test_gt.filter(col("userId") == single_user_id).select("movieId")
    user_hits = [r for r in top_n_user if r["movieId"] in [row.movieId for row in user_test_movies.collect()]]
    precision_user = len(user_hits) / top_n

    print(f"Top {top_n} recommendations for user {single_user_id}:")
    for row in top_n_user:
        print(f"Movie ID: {row['movieId']}, Predicted Rating: {row['rating']:.4f}")
    print(f"\nPrecision@10 for user {single_user_id}: {precision_user:.4f}\n")

    spark.stop()


if __name__ == "__main__":
    main()

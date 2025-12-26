from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

def main():
    spark = SparkSession.builder \
        .appName("CineRecSys - ALS Training") \
        .getOrCreate()

    # ----------------------------
    # Load ratings dataset
    # ----------------------------
    ratings_path = "hdfs://localhost:9000/cinerecsys/processed/ratings"
    ratings_df = spark.read.parquet(ratings_path)

    # ----------------------------
    # Train/test split
    # ----------------------------
    train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=42)

    # ----------------------------
    # Build ALS model
    # ----------------------------
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",  # ignore NaN predictions
        nonnegative=True,
        implicitPrefs=False,
        rank=10,
        maxIter=15,
        regParam=0.1
    )

    # Train model
    als_model = als.fit(train_df)

    # ----------------------------
    # Evaluate model
    # ----------------------------
    predictions = als_model.transform(test_df)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)

    # ----------------------------
    # Save ALS model
    # ----------------------------
    output_model_path = "hdfs://localhost:9000/cinerecsys/models/als_model"
    als_model.write().overwrite().save(output_model_path)

    # ----------------------------
    # Save RMSE as single text file in HDFS (Spark-native)
    # ----------------------------
    rmse_str = f"RMSE on test set: {rmse:.4f}"
    rmse_rdd = spark.sparkContext.parallelize([rmse_str])
    rmse_rdd.coalesce(1).saveAsTextFile(f"{output_model_path}/als_rmse.txt")

    # ----------------------------
    # Print results (Windows-safe)
    # ----------------------------
    print("\nALS model saved to HDFS at", output_model_path)
    print("RMSE saved to HDFS at", f"{output_model_path}/als_rmse.txt")
    print(f"RMSE on test set: {rmse:.4f}\n")

    spark.stop()

if __name__ == "__main__":
    main()

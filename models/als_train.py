from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
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
    # Save RMSE as DataFrame to HDFS
    # ----------------------------
    rmse_df = spark.createDataFrame([(f"RMSE on test set: {rmse:.4f}",)], ["rmse"])
    output_model_path = "hdfs://localhost:9000/cinerecsys/models/als_model"
    rmse_df.write.mode("overwrite").text(f"{output_model_path}/als_rmse.txt")

    # ----------------------------
    # Save ALS model
    # ----------------------------
    als_model.write().overwrite().save(output_model_path)

    # ----------------------------
    # Print results (Windows-safe)
    # ----------------------------
    print("\nALS model saved to HDFS at", output_model_path)
    print("RMSE saved to HDFS at", f"{output_model_path}/als_rmse.txt")
    print(f"RMSE on test set: {rmse:.4f}\n")

    spark.stop()

if __name__ == "__main__":
    main()

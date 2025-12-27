# actor_similarity_utils.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.ml.linalg import Vector, DenseVector
import math

def get_top_actors(movie_vector: DenseVector, top_k=5, spark=None, actor_features_path=None):
    """
    Returns top-k recommended actors for a given movie vector.
    """
    if spark is None:
        spark = SparkSession.builder \
            .appName("CineRecSys-Actor-Similarity-Recommender") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

    if actor_features_path is None:
        actor_features_path = "hdfs://localhost:9000/cinerecsys/processed/actor_features"

    actors_df = spark.read.parquet(actor_features_path)
    movie_vector_bc = spark.sparkContext.broadcast(movie_vector)

    def cosine_similarity(v1: Vector) -> float:
        v2 = movie_vector_bc.value
        dot = float(v1.dot(v2))
        norm1 = math.sqrt(float(v1.dot(v1)))
        norm2 = math.sqrt(float(v2.dot(v2)))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    cosine_udf = udf(cosine_similarity)

    similarity_df = actors_df.withColumn("similarity", cosine_udf(col("features")))
    top_actors = similarity_df.orderBy(col("similarity").desc()) \
        .select("actor", "similarity") \
        .limit(top_k)

    return top_actors

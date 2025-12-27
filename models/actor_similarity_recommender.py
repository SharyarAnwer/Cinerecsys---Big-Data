from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.ml.linalg import Vector, DenseVector
import math

# ---------------------------------------------------------
# Spark Session
# ---------------------------------------------------------
spark = SparkSession.builder \
    .appName("CineRecSys-Actor-Similarity-Recommender") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------
# HDFS Paths
# ---------------------------------------------------------
ACTOR_FEATURES_PATH = "hdfs://localhost:9000/cinerecsys/processed/actor_features"

# ---------------------------------------------------------
# Load Actor Feature Vectors
# ---------------------------------------------------------
actors_df = spark.read.parquet(ACTOR_FEATURES_PATH)

# ---------------------------------------------------------
# INPUT: Target Movie Profile
# ---------------------------------------------------------
# NOTE:
# The vector length MUST match actor feature vectors
# From your data: length = 5
#
# Structure:
# [avg_rating (0), genre_index, genre_count, director_index, director_count]

movie_vector = DenseVector([
    0.0,   # avg_rating (unknown for new movie)
    13.0,   # genre_index (example)
    1.0,   # genre_count
    43.0,  # director_index (example)
    1.0    # director_count
])

# Broadcast movie vector
movie_vector_bc = spark.sparkContext.broadcast(movie_vector)


# ---------------------------------------------------------
# Cosine Similarity Function
# ---------------------------------------------------------
def cosine_similarity(v1: Vector) -> float:
    v2 = movie_vector_bc.value

    dot = float(v1.dot(v2))
    norm1 = math.sqrt(float(v1.dot(v1)))
    norm2 = math.sqrt(float(v2.dot(v2)))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)

cosine_udf = udf(cosine_similarity)

# ---------------------------------------------------------
# Compute Similarity
# ---------------------------------------------------------
similarity_df = actors_df \
    .withColumn("similarity", cosine_udf(col("features")))

# ---------------------------------------------------------
# Top-K Actor Recommendation
# ---------------------------------------------------------
TOP_K = 5

top_actors = similarity_df \
    .orderBy(col("similarity").desc()) \
    .select("actor", "similarity") \
    .limit(TOP_K)

# ---------------------------------------------------------
# Output
# ---------------------------------------------------------
print("\nTop Recommended Actors:\n")
top_actors.show(truncate=False)

# ---------------------------------------------------------
# Done
# ---------------------------------------------------------
spark.stop()

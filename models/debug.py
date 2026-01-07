from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
import math


# ----------------------------
# Cosine similarity
# ----------------------------
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def main():
    spark = SparkSession.builder \
        .appName("CineRecSys - Actor Similarity Recommender") \
        .getOrCreate()
    
    actor_features_df = spark.read.parquet(
    "hdfs://localhost:9000/cinerecsys/processed/actor_features_2"
    )

    actor_features_df.printSchema()

    actor_features_df.show(5, truncate=False)

    print(actor_features_df.columns)

    print("Number of rows:", actor_features_df.count())

    spark.stop()


if __name__ == "__main__":
    main()

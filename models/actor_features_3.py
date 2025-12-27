from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    explode, col, avg, count, lit
)
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

# -------------------------------------------------------------------
# Spark Session
# -------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("CineRecSys-Actor-Feature-Engineering") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# -------------------------------------------------------------------
# HDFS Paths
# -------------------------------------------------------------------
JOINED_PATH = "hdfs://localhost:9000/cinerecsys/processed/joined"
TMDB_PATH = "hdfs://localhost:9000/cinerecsys/processed/tmdb"
OUTPUT_PATH = "hdfs://localhost:9000/cinerecsys/processed/actor_features"

# -------------------------------------------------------------------
# Load Data
# -------------------------------------------------------------------
joined_df = spark.read.parquet(JOINED_PATH)
tmdb_df = spark.read.parquet(TMDB_PATH)

"""
Expected columns (minimum):
joined_df:
- movieId
- rating
- genres

tmdb_df:
- movieId
- cast (array<string>)
- director
"""

# -------------------------------------------------------------------
# Explode Actors
# -------------------------------------------------------------------
actor_movies = tmdb_df \
    .select(
        col("movieId"),
        explode(col("actors")).alias("actor"),
        col("director")
    )

# Join ratings
actor_ratings = actor_movies.join(
    joined_df.select("movieId", "rating", "genres"),
    on="movieId",
    how="inner"
)

# -------------------------------------------------------------------
# Feature 1: Average Movie Rating per Actor
# -------------------------------------------------------------------
actor_avg_rating = actor_ratings \
    .groupBy("actor") \
    .agg(avg("rating").alias("avg_rating"))

# -------------------------------------------------------------------
# Feature 2: Genre Frequency
# -------------------------------------------------------------------
actor_genres = actor_ratings \
    .select(
        col("actor"),
        explode(col("genres")).alias("genre")
    )

actor_genre_counts = actor_genres \
    .groupBy("actor", "genre") \
    .agg(count("*").alias("genre_count"))

# Index genres
genre_indexer = StringIndexer(
    inputCol="genre",
    outputCol="genre_index",
    handleInvalid="skip"
)

# -------------------------------------------------------------------
# Feature 3: Director Collaboration Frequency
# -------------------------------------------------------------------
actor_director_counts = actor_ratings \
    .groupBy("actor", "director") \
    .agg(count("*").alias("director_count"))

director_indexer = StringIndexer(
    inputCol="director",
    outputCol="director_index",
    handleInvalid="skip"
)

# -------------------------------------------------------------------
# Assemble Feature Data
# -------------------------------------------------------------------
# Join all actor features
actor_features_raw = actor_avg_rating \
    .join(actor_genre_counts, on="actor", how="left") \
    .join(actor_director_counts, on="actor", how="left") \
    .fillna(0)

# -------------------------------------------------------------------
# ML Pipeline: Indexing + Vector Assembly
# -------------------------------------------------------------------
assembler = VectorAssembler(
    inputCols=[
        "avg_rating",
        "genre_index",
        "genre_count",
        "director_index",
        "director_count"
    ],
    outputCol="features"
)

pipeline = Pipeline(stages=[
    genre_indexer,
    director_indexer,
    assembler
])

model = pipeline.fit(actor_features_raw)
actor_features_final = model.transform(actor_features_raw) \
    .select("actor", "features") \
    .dropDuplicates(["actor"])

# -------------------------------------------------------------------
# Save to HDFS
# -------------------------------------------------------------------
actor_features_final.write \
    .mode("overwrite") \
    .parquet(OUTPUT_PATH)

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------
print("Actor feature vectors successfully written to HDFS:")
print(OUTPUT_PATH)

# -------------------------------------------------------------------
# Save StringIndexer Mappings for Genre and Director
# -------------------------------------------------------------------
GENRE_INDEX_PATH = "hdfs://localhost:9000/cinerecsys/processed/genre_index_mapping"
DIRECTOR_INDEX_PATH = "hdfs://localhost:9000/cinerecsys/processed/director_index_mapping"

# --- Genre Mapping ---
genre_labels = model.stages[0].labels  # genre_indexer is first stage
genre_mapping_df = spark.createDataFrame(
    [(i, genre) for i, genre in enumerate(genre_labels)],
    ["genre_index", "genre"]
)
genre_mapping_df.write.mode("overwrite").parquet(GENRE_INDEX_PATH)

# --- Director Mapping ---
director_labels = model.stages[1].labels  # director_indexer is second stage
director_mapping_df = spark.createDataFrame(
    [(i, director) for i, director in enumerate(director_labels)],
    ["director_index", "director"]
)
director_mapping_df.write.mode("overwrite").parquet(DIRECTOR_INDEX_PATH)

print("\nGenre and Director index mappings written to HDFS:")
print("Genre ->", GENRE_INDEX_PATH)
print("Director ->", DIRECTOR_INDEX_PATH)

spark.stop()

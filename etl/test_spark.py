from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("VSCodeSparkTest") \
    .master("local[*]") \
    .getOrCreate()

print("Spark Version:", spark.version)

spark.read.text("hdfs://localhost:9000/test.txt").show(5)

spark.stop()

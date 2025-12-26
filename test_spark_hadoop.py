from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CheckHadoop").getOrCreate()
sc = spark.sparkContext

print("Hadoop Configuration:")
for k, v in sc._jsc.hadoopConfiguration().iterator():
    print(k, v)

spark.stop()

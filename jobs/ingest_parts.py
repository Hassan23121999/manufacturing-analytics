from pyspark.sql import SparkSession

def main():
    spark = (SparkSession.builder
             .appName("ingest_parts")
             .getOrCreate())

    # Read raw CSV
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv("data/parts.csv"))

    # Write to bronze as parquet
    (df.write
       .mode("overwrite")
       .parquet("bronze/parts/"))

    spark.stop()

if __name__ == "__main__":
    main()

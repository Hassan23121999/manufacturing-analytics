from pyspark.sql import SparkSession

def main():
    spark = (SparkSession.builder
             .appName("ingest_maintenance")
             .getOrCreate())

    # Read raw CSV (same contract as the generator)
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv("data/maintenance.csv"))

    # Land to bronze as Parquet (raw-but-columnar)
    (df.write
       .mode("overwrite")
       .parquet("bronze/maintenance/"))

    spark.stop()

if __name__ == "__main__":
    main()

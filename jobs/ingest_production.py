from pyspark.sql import SparkSession

def main():
    spark = (SparkSession.builder
             .appName("ingest_production")
             .getOrCreate())

    prod = (spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv("data/production.csv"))

    qual = (spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv("data/quality.csv"))

    prod.write.mode("overwrite").parquet("bronze/production/")
    qual.write.mode("overwrite").parquet("bronze/quality/")

    spark.stop()

if __name__ == "__main__":
    main()

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, when, col, mean, sum as s

def main():
    spark = (SparkSession.builder
             .appName("gold_features")
             .getOrCreate())

    parts = spark.read.parquet("silver/parts/")
    prod  = spark.read.parquet("silver/production/")
    qual  = spark.read.parquet("silver/quality/")
    maint = spark.read.parquet("silver/maintenance/")

    # ---- Machine KPIs ----
    kpi_machine = prod.groupBy("machine_id").agg(
        avg("cycle_time_sec").alias("avg_cycle_time_sec"),
        (s(when(col("scrap_flag") == True, 1).otherwise(0)) / count("*")).alias("scrap_rate")
    )

    # ---- Per-Part Features ----
    per_part = prod.groupBy("part_id").agg(
        avg("cycle_time_sec").alias("avg_cycle_time_sec"),
        avg("energy_kwh").alias("avg_energy_kwh"),
        (s(when(col("scrap_flag") == True, 1).otherwise(0)) / count("*")).alias("scrap_rate")
    ).join(parts, "part_id", "left").join(
        qual.groupBy("part_id").agg(mean("dim_dev_um").alias("mean_dim_dev_um")),
        "part_id", "left"
    )

    # ---- Maintenance KPIs (optional) ----
    kpi_maint = maint.groupBy("machine_id").agg(
        avg("downtime_min").alias("avg_downtime_min"),
        s("repair_cost").alias("total_repair_cost"),
        count("*").alias("failure_count")
    )

    # Write outputs
    kpi_machine.write.mode("overwrite").parquet("gold/kpi_machine/")
    per_part.write.mode("overwrite").parquet("gold/features_per_part/")
    kpi_maint.write.mode("overwrite").parquet("gold/kpi_maintenance/")

    spark.stop()

if __name__ == "__main__":
    main()

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, when, lit, lower, upper, regexp_replace, expr, pow
)

def clean_parts(df):
    # Types
    df = (df
          .withColumn("created_at", to_timestamp("created_at"))
          .withColumn("volume_mm3", col("volume_mm3").cast("double"))
          .withColumn("surface_area_mm2", col("surface_area_mm2").cast("double"))
          .withColumn("tolerance_um", col("tolerance_um").cast("double"))
          .withColumn("complexity_idx", col("complexity_idx").cast("double"))
          .withColumn("rev", col("rev").cast("int"))
         )
    # Nulls / defaults
    df = df.fillna({"material": "UNKNOWN"})
    # Impute SA from volume if missing (approx ~ volume^(2/3))
    df = df.withColumn(
        "surface_area_mm2",
        when(col("surface_area_mm2").isNull() & col("volume_mm3").isNotNull(),
             pow(col("volume_mm3"), lit(2.0/3.0))
        ).otherwise(col("surface_area_mm2"))
    )
    # Validity filters
    df = (df
          .filter(col("part_id").isNotNull())
          .filter(col("volume_mm3") > 0)
          .filter(col("tolerance_um") > 0)
         )
    return df

def clean_production(df, spark):
    # Casts
    df = (df
          .withColumn("event_ts", to_timestamp("event_ts"))
          .withColumn("cycle_time_sec", col("cycle_time_sec").cast("double"))
          .withColumn("energy_kwh", col("energy_kwh").cast("double"))
          .withColumn("scrap_flag", col("scrap_flag").cast("boolean"))
         )
    # Defaults
    df = df.fillna({"scrap_flag": False, "operator_id": "UNKNOWN"})
    df = df.withColumn(
        "scrap_reason",
        when(col("scrap_flag") & (col("scrap_reason").isNull() | (col("scrap_reason") == "")),
             lit("Unspecified")
        ).otherwise(col("scrap_reason"))
    )
    # Drop impossible cycle times; clamp negatives
    df = df.filter((col("cycle_time_sec") >= 1.0) & (col("cycle_time_sec") <= 10000.0))

    # Median impute energy per machine (percentile_approx)
    median_per_machine = (df.groupBy("machine_id")
                          .agg(expr("percentile_approx(energy_kwh, 0.5) as med_energy")))
    df = (df.join(median_per_machine, "machine_id", "left")
            .withColumn("energy_kwh",
                        when(col("energy_kwh").isNull(), col("med_energy"))
                        .otherwise(col("energy_kwh")))
            .drop("med_energy"))
    return df

def clean_quality(df):
    df = (df
          .withColumn("inspection_ts", to_timestamp("inspection_ts"))
          .withColumn("dim_dev_um", col("dim_dev_um").cast("double"))
          .withColumn("status", upper(col("status")))
         )
    # Drop rows without measurement; normalize status to PASS/FAIL/UNKNOWN
    df = df.filter(col("dim_dev_um").isNotNull())
    df = df.withColumn(
        "status",
        when(col("status").isin("PASS", "FAIL"), col("status")).otherwise(lit("UNKNOWN"))
    )
    return df

def clean_maintenance(df):
    df = (df
          .withColumn("event_ts", to_timestamp("event_ts"))
          .withColumn("downtime_min", col("downtime_min").cast("double"))
          .withColumn("repair_cost", col("repair_cost").cast("double"))
         )
    df = df.fillna({"failure_mode": "Unknown"})
    # Non-negatives
    df = df.withColumn("downtime_min", when(col("downtime_min") < 0, lit(0)).otherwise(col("downtime_min")))
    df = df.withColumn("repair_cost", when(col("repair_cost") < 0, lit(0)).otherwise(col("repair_cost")))
    # Keep essential keys
    df = df.filter(col("machine_id").isNotNull())
    return df

def main():
    spark = (SparkSession.builder
             .appName("silver_transform")
             .getOrCreate())

    parts_bronze = spark.read.parquet("bronze/parts/")
    prod_bronze  = spark.read.parquet("bronze/production/")
    qual_bronze  = spark.read.parquet("bronze/quality/")
    maint_bronze = spark.read.parquet("bronze/maintenance/")

    parts_s = clean_parts(parts_bronze)
    prod_s  = clean_production(prod_bronze, spark)
    qual_s  = clean_quality(qual_bronze)
    maint_s = clean_maintenance(maint_bronze)

    parts_s.write.mode("overwrite").parquet("silver/parts/")
    prod_s.write.mode("overwrite").parquet("silver/production/")
    qual_s.write.mode("overwrite").parquet("silver/quality/")
    maint_s.write.mode("overwrite").parquet("silver/maintenance/")

    spark.stop()

if __name__ == "__main__":
    main()

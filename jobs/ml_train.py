from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

def main():
    spark = (SparkSession.builder
             .appName("ml_train")
             .getOrCreate())

    df = spark.read.parquet("gold/features_per_part/").fillna(0)

    # --- Fake label for now: simulate manufacturing cost ---
    df = df.withColumn(
        "cost_per_part",
        0.0005*df.volume_mm3 + 0.2*df.avg_cycle_time_sec + 5*df.scrap_rate + 1.0
    )

    # --- ML pipeline ---
    mat_idx = StringIndexer(inputCol="material", outputCol="material_idx", handleInvalid="keep")
    mat_oh  = OneHotEncoder(inputCols=["material_idx"], outputCols=["material_oh"])
    vec = VectorAssembler(
        inputCols=["material_oh", "volume_mm3", "surface_area_mm2", "complexity_idx",
                   "avg_cycle_time_sec", "avg_energy_kwh", "scrap_rate", "mean_dim_dev_um"],
        outputCol="features"
    )
    scaler = StandardScaler(inputCol="features", outputCol="scaled")
    lr = LinearRegression(featuresCol="scaled", labelCol="cost_per_part")

    pipeline = Pipeline(stages=[mat_idx, mat_oh, vec, scaler, lr])
    model = pipeline.fit(df)

    # Save pipeline model
    model.write().overwrite().save("models/cost_model")

    print("Training complete. Model saved to models/cost_model")

    spark.stop()

if __name__ == "__main__":
    main()

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import current_timestamp, lit

def main():
    spark = (SparkSession.builder
             .appName("ml_score")
             .getOrCreate())

    df = spark.read.parquet("gold/features_per_part/").fillna(0)
    model = PipelineModel.load("models/cost_model")

    pred = model.transform(df).select("part_id", "prediction")
    pred = pred.withColumnRenamed("prediction", "pred_cost") \
               .withColumn("scored_at", current_timestamp()) \
               .withColumn("model_version", lit("v1"))

    pred.write.mode("overwrite").parquet("gold/predictions/")

    print("Scoring complete. Predictions saved to gold/predictions/")

    spark.stop()

if __name__ == "__main__":
    main()

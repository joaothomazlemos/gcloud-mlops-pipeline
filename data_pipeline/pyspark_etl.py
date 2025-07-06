import logging
import sys

from pyspark.sql import SparkSession  # type: ignore
from pyspark.sql.functions import col  # type: ignore
from pythonjsonlogger import jsonlogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = jsonlogger.JsonFormatter(  # type: ignore
    "%(asctime)s %(name)s %(levelname)s %(message)s",
    rename_fields={"levelname": "level", "asctime": "time"},
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def run_etl(spark, input_gcs_path, output_bq_table):
    logger.info(f"Reading raw data from: {input_gcs_path}")

    raw_df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(input_gcs_path)
    )

    # Clean column names by replacing spaces with underscores
    # and ensuring consistency with the feature list in train.py and predict.py
    original_columns = raw_df.columns

    # Map original (space-separated) names to cleaned (underscore-separated) names
    # This assumes a direct mapping based on the order from inspection.
    # If the CSV columns ever change order or name significantly, this mapping needs adjustment.
    column_mapping = {
        "Call  Failure": "call_failure",
        "Complains": "complains",
        "Subscription  Length": "subscription_length",
        "Charge  Amount": "charge_amount",
        "Seconds of Use": "seconds_of_use",
        "Frequency of use": "frequency_of_use",
        "Frequency of SMS": "frequency_of_sms",
        "Distinct Called Numbers": "distinct_called_numbers",
        "Age Group": "age_group",
        "Tariff Plan": "tariff_plan",
        "Status": "status",
        "Age": "age",
        "Customer Value": "customer_value",
        "Churn": "churn",
    }

    # Verify all expected original columns are in the DataFrame
    missing_original_cols = [
        col for col in column_mapping.keys() if col not in original_columns
    ]
    if missing_original_cols:
        logger.error(
            f"Missing expected original columns in input data: {missing_original_cols}"
        )
        raise ValueError(
            f"Input data is missing expected columns: {missing_original_cols}"
        )

    # Select and rename columns
    select_expressions = [
        col(original).alias(cleaned) for original, cleaned in column_mapping.items()
    ]
    processed_df = raw_df.select(*select_expressions)

    logger.info("Processed schema:")
    processed_df.printSchema()  # printSchema uses print internally, which is fine for schema output
    logger.info(f"Number of rows processed: {processed_df.count()}")

    logger.info(f"Writing processed data to BigQuery table: {output_bq_table}")
    processed_df.write.format("bigquery").option("table", output_bq_table).mode(
        "overwrite"
    ).save()
    logger.info("Data written to BigQuery successfully.")


if __name__ == "__main__":
    # Initialize Spark Session
    # For Dataproc, spark-bigquery-with-dependencies is typically provided,
    # but explicitly adding it here ensures it's configured.
    spark = (
        SparkSession.builder.appName("ChurnFeatureEngineering")
        .config(
            "spark.jars.packages",
            "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.29.1",
        )
        .getOrCreate()
    )

    if len(sys.argv) != 3:
        logger.error("Usage: pyspark_etl.py <input_gcs_path> <output_bq_table>")
        logger.info(
            "Arguments received: input_gcs_path = {}, output_bq_table = {}".format(
                sys.argv[1], sys.argv[2]
            )
        )
        sys.exit(1)

    input_gcs_path = sys.argv[1]
    output_bq_table = sys.argv[2]

    try:
        run_etl(spark, input_gcs_path, output_bq_table)
    except Exception:
        logger.exception("An error occurred during PySpark ETL.")
        sys.exit(1)
    finally:
        spark.stop()
        logger.info("Spark session stopped.")

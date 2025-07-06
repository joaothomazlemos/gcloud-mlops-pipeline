# trainer/train.py
import argparse
import logging
import os
import sys

import joblib
from google.cloud import bigquery  # type: ignore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(bq_table_id, model_output_dir):
    logger.info(f"Loading data from BigQuery table: {bq_table_id}...")
    client = bigquery.Client()

    # Use the CLEANED feature names from the PySpark ETL
    features = [
        "Call_Failure",
        "Complains",
        "Subscription_Length",
        "Charge_Amount",
        "Seconds_of_Use",
        "Frequency_of_use",
        "Frequency_of_SMS",
        "Distinct_Called_Numbers",
        "Age_Group",
        "Tariff_Plan",
        "Status",
        "Age",
        "Customer_Value",
    ]
    target = "Churn"

    query = f"""
    SELECT
        `{"`, `".join(features)}`,
        `{target}`
    FROM
        `{bq_table_id}`
    """

    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        logger.error(f"Error loading data from BigQuery: {e}")
        raise

    logger.info(f"Loaded {len(df)} rows from BigQuery.")

    if df.empty:
        logger.error("No data loaded from BigQuery. Cannot train model.")
        raise ValueError("Empty DataFrame loaded from BigQuery.")

    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(solver="liblinear", random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_val, y_val)
    logger.info(f"Model trained. Validation Accuracy: {accuracy:.4f}")

    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bq-table-id",
        type=str,
        required=True,
        help="Full BigQuery table ID (e.g., project.dataset.table).",
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        required=True,
        help="Directory to save the trained model.",
    )
    args = parser.parse_args()

    try:
        train_model(args.bq_table_id, args.model_output_dir)
    except Exception:
        logger.exception("An error occurred during model training.")
        sys.exit(1)

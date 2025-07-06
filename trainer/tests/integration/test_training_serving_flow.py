# trainer/tests/integration/test_training_serving_flow.py
import json
import logging
import os
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest

from trainer.train import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define temporary paths for integration test artifacts
IT_MODEL_DIR = "it_model_output"
IT_MODEL_FILE = os.path.join(IT_MODEL_DIR, "model.joblib")

# Define features - MUST match cleaned feature names from PySpark ETL
FEATURES = [
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

# Define a dummy BigQuery table ID for the integration test
IT_BQ_TABLE_ID = "test_project.test_dataset.integration_test_features"


@pytest.fixture(scope="module", autouse=True)
def setup_integration_test_environment():
    """Mocks BigQuery client and sets up a dummy model for integration tests."""

    # Create a small, simple DataFrame for integration test training
    # This data simulates the output of the PySpark ETL, with cleaned names.
    it_df = pd.DataFrame(
        [
            [10, 1, 10, 50, 100, 10, 5, 3, 3, 1, 1, 40, 50.0, 1],  # Churn
            [0, 0, 30, 200, 10, 1, 1, 1, 1, 1, 1, 25, 150.0, 0],  # No churn
        ],
        columns=FEATURES + ["Churn"],
    )

    # Mock the BigQuery client to return our in-memory DataFrame
    with patch("google.cloud.bigquery.Client") as mock_bq_client:
        mock_instance = mock_bq_client.return_value
        mock_instance.query.return_value.to_dataframe.return_value = it_df
        logger.info("Mocked BigQuery client for integration test.")

        # Train a minimal model using the mocked BigQuery data
        logger.info(
            f"Training dummy model for integration test from mocked BQ table: {IT_BQ_TABLE_ID}"
        )
        train_model(IT_BQ_TABLE_ID, IT_MODEL_DIR)
        logger.info("Dummy model training complete.")

        yield  # This runs the tests

    # Teardown: Clean up generated files and directories
    if os.path.exists(IT_MODEL_DIR):
        import shutil

        shutil.rmtree(IT_MODEL_DIR)
        logger.info(f"Cleaned up integration test model directory: {IT_MODEL_DIR}")


def test_training_script_runs_and_saves_model():
    """Verifies that train.py executed and saved a model artifact."""
    logger.info("Running test: training_script_runs_and_saves_model")
    assert os.path.exists(IT_MODEL_FILE)
    model = joblib.load(IT_MODEL_FILE)
    assert model is not None
    assert hasattr(model, "predict")
    assert len(model.feature_names_in_) == len(FEATURES)
    logger.info("Test passed: training_script_runs_and_saves_model")


def test_model_can_be_loaded_and_predicts():
    """Verifies that the saved model can be loaded by the serving logic and makes predictions."""
    logger.info("Running test: model_can_be_loaded_and_predicts")
    with patch.dict(os.environ, {"AIP_MODEL_DIR": IT_MODEL_DIR}):
        # Import `app` and `_load_model` from server.predict *after* patching env var
        from server.predict import _load_model, app

        _load_model()  # Manually call the loading function to load the model for testing
        logger.info("Server model loaded for integration test prediction.")

        # Create a dummy Flask request payload matching the FEATURES
        mock_request_payload = {
            "instances": [
                {
                    "Call_Failure": 5,
                    "Complains": 1,
                    "Subscription_Length": 10,
                    "Charge_Amount": 50,
                    "Seconds_of_Use": 100,
                    "Frequency_of_use": 10,
                    "Frequency_of_SMS": 5,
                    "Distinct_Called_Numbers": 3,
                    "Age_Group": 3,
                    "Tariff_Plan": 1,
                    "Status": 1,
                    "Age": 40,
                    "Customer_Value": 50.0,
                },
                {
                    "Call_Failure": 0,
                    "Complains": 0,
                    "Subscription_Length": 30,
                    "Charge_Amount": 200,
                    "Seconds_of_Use": 10,
                    "Frequency_of_use": 1,
                    "Frequency_of_SMS": 1,
                    "Distinct_Called_Numbers": 1,
                    "Age_Group": 1,
                    "Tariff_Plan": 1,
                    "Status": 1,
                    "Age": 25,
                    "Customer_Value": 150.0,
                },
            ]
        }

        with app.test_client() as client:
            response = client.post("/predict", json=mock_request_payload)

        assert response.status_code == 200
        predictions = json.loads(response.data)["predictions"]
        assert isinstance(predictions, list)
        assert len(predictions) == 2
        assert "prediction" in predictions[0]
        assert "probability_churn" in predictions[0]
        assert predictions[0]["prediction"] in [0, 1]
        assert 0 <= predictions[0]["probability_churn"] <= 1
        logger.info(
            "Test passed: model_can_be_loaded_and_predicts - predictions are valid."
        )

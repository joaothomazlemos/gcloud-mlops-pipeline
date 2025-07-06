# server/predict.py
import logging
import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("AIP_MODEL_DIR")

model = None
app = Flask(__name__)

# Define features - MUST match training features and order
# These are the CLEANED feature names from the PySpark ETL
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


@app.before_first_request
def _load_model():
    global model
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    logger.info(f"Attempting to load model from {model_path}...")
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        # In a real serving environment, you might want to return a 500 error or similar
        # before the first request if the model fails to load.
        raise


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        logger.error("Prediction requested but model is not loaded.")
        return jsonify({"error": "Model not loaded"}), 500

    try:
        json_payload = request.get_json()
        if not json_payload or "instances" not in json_payload:
            logger.warning(f"Invalid request payload: {json_payload}")
            return (
                jsonify(
                    {"error": 'Invalid request format. Expecting {"instances": [...]}'}
                ),
                400,
            )

        instances = json_payload["instances"]
        logger.info(f"Received prediction request for {len(instances)} instances.")

        input_df = pd.DataFrame(instances)
        # Ensure column order matches FEATURES
        X_predict = input_df[FEATURES]

        predictions = model.predict(X_predict).tolist()
        probabilities = model.predict_proba(X_predict).tolist()

        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({"prediction": pred, "probability_churn": prob[1]})

        logger.info("Prediction successful.")
        return jsonify({"predictions": results})

    except KeyError as ke:
        logger.error(f"Missing expected feature in request: {ke}")
        return (
            jsonify(
                {
                    "error": f"Missing expected feature: {ke}. Ensure all features are present: {FEATURES}"
                }
            ),
            400,
        )
    except Exception as e:
        logger.exception("Error during prediction.")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if MODEL_DIR is None:
        logger.warning(
            "AIP_MODEL_DIR environment variable not set. Running in local test mode."
        )
        local_test_model_dir = "local_model_test"
        if not os.path.exists(local_test_model_dir):
            os.makedirs(local_test_model_dir)
            logger.info(f"Created local test model directory: {local_test_model_dir}")

        local_model_path = os.path.join(local_test_model_dir, "model.joblib")
        if not os.path.exists(local_model_path):
            from sklearn.linear_model import LogisticRegression

            dummy_model = LogisticRegression()
            # Fit on dummy data with the correct number of features and names
            dummy_data = pd.DataFrame([[0] * len(FEATURES)], columns=FEATURES)
            dummy_model.fit(dummy_data, [0])
            joblib.dump(dummy_model, local_model_path)
            logger.info(f"Created dummy model for local testing at {local_model_path}.")
        os.environ["AIP_MODEL_DIR"] = local_test_model_dir

    try:
        _load_model()
        app.run(debug=True, host="0.0.0.0", port=os.environ.get("AIP_HTTP_PORT", 8080))
    except Exception as e:
        logger.exception("Failed to start Flask app.")
        sys.exit(1)

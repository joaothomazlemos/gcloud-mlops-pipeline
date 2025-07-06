# MLOps Churn Prediction Pipeline on Google Cloud
This repository contains an end-to-end MLOps CI/CD pipeline for a churn prediction model, orchestrated by Google Cloud Build. I build this template for future use as well as for me to understand deeper concepts of GCP when going for MLOps best pratices. It demonstrates key MLOps practices including:

### Key Features

- **Raw Data Ingestion Simulation:**
    - Assumes raw `Customer Churn.csv` data is stored in a GCS "data lake".

- **PySpark ETL:**
    - Processes raw data using a PySpark job running on Dataproc.
    - Cleans column names and transforms data into features.
    - Stores processed features in a BigQuery table, which serves as the feature source.

- **Model Training:**
    - Trains a Logistic Regression model using features from the BigQuery table.

- **Unit & Integration Testing:**
    - Ensures code quality and system integrity.
    - **Integration Tests:** Validate the training and serving components on a small, in-memory dataset, including mocked BigQuery access.

- **Model Deployment:**
    - Uploads the trained model to Vertex AI Model Registry.
    - Deploys the model to a Vertex AI Endpoint.

- **Model Monitoring (Data Drift):**
    - Configures Vertex AI Model Monitoring to detect data drift.
    - Uses the BigQuery feature table as a baseline for monitoring.

- **Load Generation for Monitoring:**
    - Deploys and executes a K6 load testing script on a Compute Engine VM.
    - Simulates real-world prediction requests to feed data into the Vertex AI Model Monitoring system.

### Project Structure

```plaintext
mlops-churn-prediction/
├── data_pipeline/
│   ├── pyspark_etl.py              # PySpark script for ETL (reads GCS, writes BQ)
│   └── requirements.txt            # Python dependencies for PySpark (minimal)
├── trainer/
│   ├── Dockerfile                  # Dockerfile for training environment
│   ├── train.py                    # Main training script (reads BQ, saves model)
│   ├── requirements.txt            # Python dependencies for training
│   └── tests/
│       └── integration/
│           └── test_training_serving_flow.py # Integration tests
├── server/
│   ├── predict.py                  # Main serving logic (used by Vertex AI pre-built container)
│   └── requirements.txt            # Python dependencies for serving
├── k6/
│   └── churn_load_test.js          # K6 script for load generation
└── cloudbuild.yaml                 # Google Cloud Build configuration
```

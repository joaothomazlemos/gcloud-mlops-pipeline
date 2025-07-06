// k6/churn_load_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter } from'k6/metrics';

// This needs to be passed as an environment variable to k6, e.g., using --env ENDPOINT_URL
const ENDPOINT_URL = __ENV.ENDPOINT_URL || 'YOUR_VERTEX_AI_ENDPOINT_URL';
if (!ENDPOINT_URL || ENDPOINT_URL === 'YOUR_VERTEX_AI_ENDPOINT_URL') {
  console.error('ENDPOINT_URL is not set or is default. Please set it via --env ENDPOINT_URL.');
  throw new Error('ENDPOINT_URL not configured');
}

// Custom metrics to track prediction outcomes
const successfulPredictions = new Counter('successful_predictions');
const failedPredictions = new Counter('failed_predictions');

// Define the shape of your input data for prediction (matching cleaned features)
const TEST_INSTANCE = {
    "Call_Failure": 1, "Complains": 0, "Subscription_Length": 30,
    "Charge_Amount": 100, "Seconds_of_Use": 500, "Frequency_of_use": 10,
    "Frequency_of_SMS": 2, "Distinct_Called_Numbers": 5, "Age_Group": 2,
    "Tariff_Plan": 1, "Status": 1, "Age": 35, "Customer_Value": 100.0
};

export const options = {
  vus: 10,    // 10 virtual users
  duration: '30s', // For 30 seconds (adjust for longer simulations)
  // Define custom thresholds for pass/fail criteria for the overall test
  thresholds: {
    http_req_failed: ['rate<0.01'], // http errors should be less than 1%
    http_req_duration: ['p(95)<2000'], // 95% of requests should be below 2s
    successful_predictions: ['count>10'], // At least 10 successful predictions overall
  },
  ext: {
    cloud: {
      projectID: 'YOUR_GCP_PROJECT_ID', // Replace with your GCP Project ID
      name: 'churn-prediction-load-test',
    },
  },
};

export default function () {
  const payload = JSON.stringify({
    instances: [TEST_INSTANCE],
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const res = http.post(ENDPOINT_URL, payload, params);

  const checkResult = check(res, {
    'is status 200': (r) => r.status === 200,
    'has predictions array': (r) => {
        try {
            const body = JSON.parse(r.body);
            return body.predictions !== undefined && Array.isArray(body.predictions);
        } catch (e) {
            console.error(`Failed to parse response body: ${r.body}`);
            return false;
        }
    },
    'prediction is 0 or 1': (r) => {
        try {
            const body = JSON.parse(r.body);
            return body.predictions && body.predictions.every(p => p.prediction === 0 || p.prediction === 1);
        } catch (e) {
            return false;
        }
    },
    'probability is valid': (r) => {
        try {
            const body = JSON.parse(r.body);
            return body.predictions && body.predictions.every(p => typeof p.probability_churn === 'number' && p.probability_churn >= 0 && p.probability_churn <= 1);
        } catch (e) {
            return false;
        }
    },
  });

  if (checkResult) {
    successfulPredictions.add(1);
  } else {
    failedPredictions.add(1);
    console.error(`Request failed: Status=${res.status}, Body=${res.body}`);
  }

  sleep(1); // Wait 1 second between requests (per VU)
}

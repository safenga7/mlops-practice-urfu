#!/usr/bin/env bash
set -e

# Microservice test client script
PORT="${1:-8000}"
BASE_URL="http://127.0.0.1:$PORT"

echo "Testing service at $BASE_URL..."

echo -e "\n1. Root Info:"
curl -s "$BASE_URL/" | python3 -m json.tool

echo -e "\n2. Healthcheck:"
curl -s "$BASE_URL/health" | python3 -m json.tool

echo -e "\n3. Single Prediction (Setosa):"
curl -s -X POST "$BASE_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' | python3 -m json.tool

echo -e "\n4. Batch Prediction (Setosa + Virginica):"
curl -s -X POST "$BASE_URL/predict/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "samples": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.4, "petal_width": 2.1}
        ]
    }' | python3 -m json.tool

echo -e "\n5. Telemetry Metrics:"
curl -s "$BASE_URL/metrics" | python3 -m json.tool

echo -e "\nTesting complete."

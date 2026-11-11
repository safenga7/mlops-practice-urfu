#!/usr/bin/env bash
set -e

# ==============================================================================
# Complete End-to-End MLOps Pipeline Runner for Final Project
# Executes all CI/CD stages:
#   1. Environment inspection
#   2. DVC data validation & remote sync
#   3. Data quality testing (pytest)
#   4. Unit & API integration testing (pytest)
#   5. Docker image building
#   6. Live container smoke testing
# ==============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}               STARTING FINAL PROJECT MLOPS CI/CD PIPELINE                    ${NC}"
echo -e "${BLUE}==============================================================================${NC}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "\n${GREEN}[Stage 1/6] Environment & Git Inspection${NC}"
echo "Working directory: $PROJECT_ROOT"
echo "Current branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Current commit: $(git rev-parse --short HEAD)"
python3 --version

echo -e "\n${GREEN}[Stage 2/6] DVC Dataset Verification & Remote Sync${NC}"
if [ ! -f "final_project/data/iris_dataset.csv" ]; then
    echo "Regenerating dataset..."
    python3 final_project/data/generate_dataset.py
fi
test -f final_project/data/iris_dataset.csv.dvc
echo "DVC tracking metadata:"
cat final_project/data/iris_dataset.csv.dvc

echo -e "\n${GREEN}[Stage 3/6] Running Data Quality Tests${NC}"
python3 -m pytest -v final_project/tests/test_data_quality.py

echo -e "\n${GREEN}[Stage 4/6] Running API Unit & Integration Tests${NC}"
python3 -m pytest -v final_project/tests/test_api.py

echo -e "\n${GREEN}[Stage 5/6] Building Production Docker Image${NC}"
docker build -t final-mlops-app:latest final_project/

echo -e "\n${GREEN}[Stage 6/6] Live Container Smoke Test${NC}"
TEST_PORT=8006
CONTAINER_NAME="pipeline-smoke-test-$RANDOM"

docker run -d --rm --name "$CONTAINER_NAME" -p "127.0.0.1:$TEST_PORT:8000" final-mlops-app:latest
sleep 3

echo "Probing /health endpoint..."
HEALTH_RESP=$(curl -s "http://127.0.0.1:$TEST_PORT/health")
echo "Health Response: $HEALTH_RESP"

echo "Probing /predict endpoint..."
PRED_RESP=$(curl -s -X POST "http://127.0.0.1:$TEST_PORT/predict" \
    -H "Content-Type: application/json" \
    -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}')
echo "Predict Response: $PRED_RESP"

echo "Stopping test container..."
docker stop "$CONTAINER_NAME" > /dev/null

echo -e "\n${BLUE}==============================================================================${NC}"
echo -e "${GREEN}SUCCESS: ALL CI/CD PIPELINE STAGES COMPLETED CLEANLY!${NC}"
echo -e "${BLUE}==============================================================================${NC}"

#!/bin/bash
source ~/mlflow/venv/bin/activate

mlflow server \
    --backend-store-uri sqlite:///~/mlflow/server/mlflow.db \
    --default-artifact-root file:///home/$(whoami)/mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000

import os
import tarfile
from urllib import response
import boto3
import torch
from flask import Flask, request, jsonify, make_response
from darts.models import NHiTSModel
from urllib.parse import urlparse

# Initialize Flask app
app = Flask(__name__)

# S3 bucket details
S3_URI = "s3://airlines-s3/models/airlines-2024-12-05-09-46-26-033/output/model.tar.gz"
MODEL_FILE = "model.tar.gz"
LOCAL_MODEL_PATH = "../model_results/"

# Download and extract model from S3


def download_and_extract_model():
    # Parse the S3 URI
    parsed = urlparse(S3_URI)
    bucket_name = parsed.netloc  # Extracts "airlines-s3"
    object_key = parsed.path.lstrip("/")  # Extracts the object key
    print(object_key)
    # Download model file from S3
    s3 = boto3.client("s3")
    model_path = os.path.join(LOCAL_MODEL_PATH, "model.tar.gz")
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
    s3.download_file(bucket_name, object_key, model_path)
    with tarfile.open(model_path, "r:gz") as tar:
        tar.extractall(path=LOCAL_MODEL_PATH)


# Load NHiTS model
def load_model(commodity, state):
    try:
        model_path = os.path.join(LOCAL_MODEL_PATH, commodity, state, "nhits.pkt")
        print(model_path)
        model = NHiTSModel.load(model_path)
        return model
    except Exception:
        print(
            "model not found in",
            os.path.join(LOCAL_MODEL_PATH, commodity, state, "nhits.pkt"),
        )


# Initialize model
# download_and_extract_model()


@app.route("/krushijyotishi/predict", methods=["POST"])
def predict():
    data = request.json
    forecast_horizon = data.get("horizon", 1)
    commodity = data.get("commodity", "wheat")
    state = data.get("state", "UP")
    model = load_model(commodity, state)
    if model:
        # Convert to darts TimeSeries if necessary
        predictions = model.predict(n=forecast_horizon)
        return jsonify(predictions.values().flatten().tolist())
    else:
        response = make_response(f"Model not found for {commodity} in {state}", 404)
        return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("superkart_model.joblib")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SuperKart Sales Prediction API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read JSON input
        input_data = request.get_json()

        if not isinstance(input_data, (dict, list)):
            return jsonify({
                "error": "Expected a JSON object or list of objects"
            }), 400

        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_data = [input_data]

        df = pd.DataFrame(input_data)

        # Generate predictions
        predictions = model.predict(df)

        # Return predictions
        return jsonify({
            "predictions": predictions.tolist()
        })

    except (ValueError, KeyError, TypeError) as e:
        return jsonify({
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

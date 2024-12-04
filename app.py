from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    power = data.get('power')
    scan_speed = data.get('scanSpeed')
    if power is None or scan_speed is None:
        return jsonify({"error": "Invalid input"}), 400

    # Predict RGB value
    prediction = model.predict([[power, scan_speed]])
    rgb = prediction[0].astype(int).tolist()
    return jsonify({"rgb": rgb})

if __name__ == '__main__':
        # Bind to 0.0.0.0 and use the PORT environment variable
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)

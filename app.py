# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import pandas as pd
# import joblib
# import tensorflow as tf
# import os

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
# MODEL_PATH = os.path.join(BASE_DIR, "kidney_cancer_ann_model.h5")
# SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
# ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
# DATASET_PATH = os.path.join(BASE_DIR, "kidney_cancer_filtered_dataset.csv")

# # Initialize Flask app
# app = Flask(__name__)

# # Load model & preprocessors
# try:
#     if not all(os.path.exists(path) for path in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, DATASET_PATH]):
#         raise FileNotFoundError("One or more required files (model, scaler, encoders, dataset) are missing.")

#     # Load trained model
#     model = tf.keras.models.load_model(MODEL_PATH)
    
#     # Load preprocessing tools
#     scaler = joblib.load(SCALER_PATH)
#     label_encoders = joblib.load(ENCODER_PATH)
    
#     # Load dataset to extract feature names
#     df = pd.read_csv(DATASET_PATH)
#     if 'htn' not in df.columns:
#         raise ValueError("Dataset is missing the target column 'htn'.")

#     # Ensure all expected features are included
#     X_columns = df.drop(columns=['htn']).columns.tolist()
    
#     print("✅ Model and preprocessors loaded successfully!")

# except Exception as e:
#     print(f"❌ Error loading model or preprocessors: {e}")
#     model, scaler, label_encoders, X_columns = None, None, None, None

# # Route Definitions
# @app.route('/')
# def index():
#     return render_template('data.html')

# @app.route('/annpredict')
# def annpredict():
#     return render_template('annpredict.html')

# @app.route('/cnnprediction')
# def cnnprediction():
#     return render_template('cnnprediction.html')

# @app.route('/upload')
# def upload():
#     return render_template('upload.html')

# @app.route('/open')  # Renamed to avoid conflict with built-in `open()`
# def open():
#     return render_template('open.html')

# @app.route('/signup')
# def signup():
#     return render_template('signup.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

# @app.route('/data')
# def data():
#     return render_template('data.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handles predictions for kidney cancer detection using the trained ANN model."""
#     if not model or not scaler or not label_encoders:
#         return jsonify({"error": "Model or preprocessors not loaded properly."}), 500

#     try:
#         # Parse incoming JSON data
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "No input data provided"}), 400

#         print(f"🔹 Received Data: {data}")  # Debugging log

#         # Check for missing features
#         missing_features = [feature for feature in X_columns if feature not in data]
#         if missing_features:
#             print(f"⚠️ Missing features: {missing_features}")  # Debugging log
#             return jsonify({"error": f"Missing features: {missing_features}"}), 400

#         # Convert categorical values (Yes/No) before encoding
#         categorical_mappings = {"Yes": 1, "No": 0}
#         categorical_cols = ['htn', 'dm']  # Add other categorical fields if needed

#         for col in categorical_cols:
#             if col in data and data[col] in categorical_mappings:
#                 data[col] = categorical_mappings[data[col]]

#         # Apply label encoding if necessary
#         for col, encoder in label_encoders.items():
#             if col in data and isinstance(data[col], str):  # Encode only if it's a string
#                 try:
#                     data[col] = encoder.transform([data[col]])[0]
#                 except ValueError:
#                     return jsonify({"error": f"Invalid value '{data[col]}' for feature '{col}'"}), 400

#         # Convert input data to numpy array
#         input_data = np.array([data[feature] for feature in X_columns]).reshape(1, -1)

#         # Apply feature scaling
#         input_data = scaler.transform(input_data)

#         # Make prediction
#         prediction = model.predict(input_data)[0][0]

#         # Convert output to meaningful class label
#         result = "Abnormal" if prediction > 0.5 else "Normal"

#         print(f"✅ Prediction: {result}, Probability: {prediction}")  # Debugging log

#         return jsonify({"prediction": result, "probability": float(prediction)})

#     except Exception as e:
#         print(f"❌ Error: {e}")  # Debugging log
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
MODEL_PATH = os.path.join(BASE_DIR, "kidney_cancer_ann_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "kidney_cancer_filtered_dataset.csv")

# Initialize Flask app
app = Flask(__name__)

# Load model & preprocessors
try:
    if not all(os.path.exists(path) for path in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, DATASET_PATH]):
        raise FileNotFoundError("One or more required files (model, scaler, encoders, dataset) are missing.")

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load preprocessing tools
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    
    # Load dataset to extract feature names
    df = pd.read_csv(DATASET_PATH)
    if 'htn' not in df.columns:
        raise ValueError("Dataset is missing the target column 'htn'.")

    # Ensure all expected features are included
    X_columns = df.drop(columns=['htn']).columns.tolist()
    
    print("✅ Model and preprocessors loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model or preprocessors: {e}")
    model, scaler, label_encoders, X_columns = None, None, None, None

# Route Definitions
@app.route('/')
def index():
    return render_template('data.html')

@app.route('/annpredict')
def annpredict():
    return render_template('annpredict.html')

@app.route('/cnnprediction')
def cnnprediction():
    return render_template('cnnprediction.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/open')  # Renamed to avoid conflict with built-in `open()`
def open():
    return render_template('open.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles predictions for kidney cancer detection using the trained ANN model."""
    if not model or not scaler or not label_encoders:
        return jsonify({"error": "Model or preprocessors not loaded properly."}), 500

    try:
        # Check Content-Type
        if request.content_type != "application/json":
            return jsonify({"error": "Invalid content type. Use application/json"}), 415

        # Parse incoming JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        print(f"🔹 Received Data: {data}")  # Debugging log

        # Check for missing features
        missing_features = [feature for feature in X_columns if feature not in data]
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")  # Debugging log
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Convert categorical values (Yes/No) before encoding
        categorical_mappings = {"Yes": 1, "No": 0}
        categorical_cols = ['htn', 'dm']  # Add other categorical fields if needed

        for col in categorical_cols:
            if col in data and data[col] in categorical_mappings:
                data[col] = categorical_mappings[data[col]]

        # Apply label encoding if necessary
        for col, encoder in label_encoders.items():
            if col in data and isinstance(data[col], str):  # Encode only if it's a string
                try:
                    data[col] = encoder.transform([data[col]])[0]
                except ValueError:
                    return jsonify({"error": f"Invalid value '{data[col]}' for feature '{col}'"}), 400

        # Convert input data to numpy array
        input_data = np.array([data[feature] for feature in X_columns]).reshape(1, -1)

        # Apply feature scaling
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)[0][0]

        # Convert output to meaningful class label
        result = "Abnormal" if prediction > 0.5 else "Normal"

        print(f"✅ Prediction: {result}, Probability: {prediction}")  # Debugging log

        return jsonify({"prediction": result, "probability": float(prediction)})

    except Exception as e:
        print(f"❌ Error: {e}")  # Debugging log
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

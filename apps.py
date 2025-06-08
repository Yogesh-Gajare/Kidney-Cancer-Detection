# from flask import Flask, render_template, request, jsonify, session, redirect, url_for
# from flask_pymongo import PyMongo
# from werkzeug.security import generate_password_hash, check_password_hash

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Change this in production!

# # MongoDB connection
# app.config["MONGO_URI"] = "mongodb://localhost:27017/kidneyDB"
# mongo = PyMongo(app)

# # Home route
# @app.route('/')
# @app.route('/open')
# def open():
#     return render_template('open.html')

# # Signup route
# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         data = request.get_json()
#         fullname = data.get('fullname')
#         email = data.get('email')
#         username = data.get('username')
#         password = data.get('password')
#         confirm_password = data.get('confirmPassword')

#         if password != confirm_password:
#             return jsonify({'success': False, 'message': 'Passwords do not match'})

#         if mongo.db.users.find_one({'username': username}):
#             return jsonify({'success': False, 'message': 'Username already exists'})

#         hashed_password = generate_password_hash(password)

#         mongo.db.users.insert_one({
#             'fullname': fullname,
#             'email': email,
#             'username': username,
#             'password': hashed_password
#         })

#         return jsonify({'success': True})
    
#     return render_template('signup.html')

# # Login route
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         data = request.get_json()
#         username = data.get('username')
#         password = data.get('password')

#         user = mongo.db.users.find_one({'username': username})
#         if user and check_password_hash(user['password'], password):
#             session['username'] = username
#             return jsonify({'success': True})
#         return jsonify({'success': False, 'message': 'Invalid username or password'})

#     return render_template('login.html')

# # Upload route (protected)
# @app.route('/upload')
# def upload():
#     if 'username' not in session:
#         return redirect(url_for('login'))
#     return render_template('upload.html')

# # Logout route
# @app.route('/logout')
# def logout():
#     session.pop('username', None)
#     return redirect(url_for('login'))

# if __name__ == '__main__':
#     app.run(debug=True)





import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Flask app setup
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = 'your_secret_key'

# MongoDB connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/kidneyDB"
mongo = PyMongo(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ANN_MODEL_PATH = os.path.join(BASE_DIR, "kidney_cancer_ann_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "kidney_cancer_filtered_dataset.csv")

CNN_MODEL_PATH = os.path.join(BASE_DIR, "kidney_cancer_model.h5")
CNN_CSV_PATH = os.path.join(BASE_DIR, "kidneyData_filtered.csv")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ANN model and preprocessors
try:
    if not all(os.path.exists(path) for path in [ANN_MODEL_PATH, SCALER_PATH, ENCODER_PATH, DATASET_PATH]):
        raise FileNotFoundError("One or more required ANN files are missing.")

    ann_model = tf.keras.models.load_model(ANN_MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    ann_df = pd.read_csv(DATASET_PATH)

    if 'htn' not in ann_df.columns:
        raise ValueError("ANN dataset missing target column 'htn'.")

    ann_features = ann_df.drop(columns=['htn']).columns.tolist()
    print("✅ ANN model and preprocessors loaded successfully!")
except Exception as e:
    print(f"❌ Error loading ANN model or preprocessors: {e}")
    ann_model, scaler, label_encoders, ann_features = None, None, None, None

# Load CNN model and dataset
try:
    if not all(os.path.exists(path) for path in [CNN_MODEL_PATH, CNN_CSV_PATH]):
        raise FileNotFoundError("One or more required CNN files are missing.")

    cnn_model = load_model(CNN_MODEL_PATH)
    cnn_df = pd.read_csv(CNN_CSV_PATH)
    print("✅ CNN model and dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading CNN model or dataset: {e}")
    cnn_model, cnn_df = None, None

# --- Routes ---

@app.route('/')
@app.route('/open')
def open():
    return render_template('open.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        fullname = data.get('fullname')
        email = data.get('email')
        username = data.get('username')
        password = data.get('password')
        confirm_password = data.get('confirmPassword')

        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match'})

        if mongo.db.users.find_one({'username': username}):
            return jsonify({'success': False, 'message': 'Username already exists'})

        hashed_password = generate_password_hash(password)
        mongo.db.users.insert_one({
            'fullname': fullname,
            'email': email,
            'username': username,
            'password': hashed_password
        })

        return jsonify({'success': True})

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        user = mongo.db.users.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Invalid username or password'})

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload')
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/annpredict')
def annpredict():
    return render_template('annpredict.html')

@app.route('/cnnprediction')
def cnnprediction():
    return render_template('cnnprediction.html')

# --- ANN Prediction Route ---
@app.route('/annpredict', methods=['POST'])
def ann_predict():
    if not ann_model or not scaler or not label_encoders:
        return jsonify({"error": "ANN model or preprocessors not loaded properly."}), 500

    try:
        if request.content_type != "application/json":
            return jsonify({"error": "Invalid content type. Use application/json"}), 415

        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        missing_features = [feature for feature in ann_features if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        categorical_mappings = {"Yes": 1, "No": 0}
        categorical_cols = ['htn', 'dm']

        for col in categorical_cols:
            if col in data and data[col] in categorical_mappings:
                data[col] = categorical_mappings[data[col]]

        for col, encoder in label_encoders.items():
            if col in data and isinstance(data[col], str):
                try:
                    data[col] = encoder.transform([data[col]])[0]
                except ValueError:
                    return jsonify({"error": f"Invalid value '{data[col]}' for feature '{col}'"}), 400

        input_data = np.array([data[feature] for feature in ann_features]).reshape(1, -1)
        input_data = scaler.transform(input_data)

        prediction = ann_model.predict(input_data)[0][0]
        result = "Abnormal" if prediction > 0.5 else "Normal"

        return jsonify({"prediction": result, "probability": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- CNN Prediction Route ---
def predict_cnn_image(image):
    img_size = (128, 128)
    img = image.resize(img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = cnn_model.predict(img_array)
    class_names = ["Normal", "Tumor"]
    predicted_class = class_names[np.argmax(prediction)]

    if predicted_class in cnn_df['Class'].values:
        info = cnn_df[cnn_df['Class'] == predicted_class].iloc[0].to_dict()
        for field in ['Unnamed: 0', 'full_path', 'image_id', 'path', 'target']:
            info.pop(field, None)
    else:
        info = {"Note": "No extra info available in database"}

    return {"class": predicted_class, "database_info": info}

@app.route('/predict_cnn', methods=['POST'])
def predict_cnn():
    if not cnn_model or cnn_df is None:
        return jsonify({"error": "CNN model or dataset not loaded properly."}), 500

    base64_image = request.form.get('image')
    if not base64_image:
        return jsonify({"error": "No image data provided"}), 400

    try:
        image_data = base64.b64decode(base64_image.split(',')[1])
        image = Image.open(BytesIO(image_data))

        result = predict_cnn_image(image)
        return jsonify({
            "prediction": result["class"],
            "database_info": result["database_info"]
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# --- Main ---
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

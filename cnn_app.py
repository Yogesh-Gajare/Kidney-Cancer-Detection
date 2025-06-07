
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Constants
IMG_SIZE = (128, 128)
MODEL_PATH = "kidney_cancer_model.h5"
UPLOAD_FOLDER = "uploads/"
CSV_PATH = "kidneyData_filtered.csv"  # Path to CSV dataset

# Load Model & Dataset
model = load_model(MODEL_PATH)
df = pd.read_csv(CSV_PATH)  # Load the database

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def predict_image(image):
    # Resize and process the image for the model
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class using the model
    prediction = model.predict(img_array)
    class_names = ["Normal", "Tumor",]  # Define your classes here
    predicted_class = class_names[np.argmax(prediction)]  # Get the predicted class

    # Fetch additional information from the CSV dataset
    if predicted_class in df['Class'].values:
        # Get the row that matches the predicted class
        info = df[df['Class'] == predicted_class].iloc[0].to_dict()

        # Remove unwanted fields from the info dictionary
        unwanted_fields = ['Unnamed: 0', 'full_path', 'image_id', 'path', 'target']
        for field in unwanted_fields:
            if field in info:
                del info[field]  # Delete unwanted field from the dictionary

    else:
        info = {"Note": "No extra info available in database"}

    return {"class": predicted_class, "database_info": info}

# Web Interface
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/cnnprediction')
def cnnprediction():
    return render_template('cnnprediction.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')  # Ensure this matches your actual upload page file name

@app.route('/open')
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

@app.route('/annpredict')
def annpredict():
    return render_template('annpredict.html')



@app.route('/predict', methods=['POST'])
def predict():
    # Get the base64 image from the form data
    base64_image = request.form.get('image')
    
    if not base64_image:
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        # Decode the base64 string to image
        image_data = base64.b64decode(base64_image.split(',')[1])  # Skip the "data:image/png;base64,"
        image = Image.open(BytesIO(image_data))

        # Process the image using the prediction logic
        result = predict_image(image)

        # Return the prediction and database information
        return jsonify({
            "prediction": result["class"],
            "database_info": result["database_info"]
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": "Prediction failed"}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)







# import os
# import base64
# from io import BytesIO
# from PIL import Image
# import numpy as np
# import pandas as pd
# import joblib
# import tensorflow as tf
# from flask import Flask, request, jsonify, render_template
# from tensorflow.keras.preprocessing.image import img_to_array
# from werkzeug.utils import secure_filename

# # Constants
# IMG_SIZE = (128, 128)
# CNN_MODEL_PATH = "kidney_cancer_model.h5"
# ANN_MODEL_PATH = "kidney_cancer_ann_model.h5"
# SCALER_PATH = "scaler.pkl"
# ENCODER_PATH = "label_encoders.pkl"
# CNN_DATASET_PATH = "kidneyData_filtered.csv"
# ANN_DATASET_PATH = "kidney_cancer_filtered_dataset.csv"
# UPLOAD_FOLDER = "uploads/"

# # Initialize Flask app
# app = Flask(__name__, template_folder="templates", static_folder="static")
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # Load CNN model and dataset
# cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
# cnn_df = pd.read_csv(CNN_DATASET_PATH)

# # Load ANN model, scaler, encoder, and dataset
# try:
#     if not all(os.path.exists(p) for p in [ANN_MODEL_PATH, SCALER_PATH, ENCODER_PATH, ANN_DATASET_PATH]):
#         raise FileNotFoundError("Missing one or more ANN model components.")

#     ann_model = tf.keras.models.load_model(ANN_MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)
#     label_encoders = joblib.load(ENCODER_PATH)
#     ann_df = pd.read_csv(ANN_DATASET_PATH)
#     X_columns = ann_df.drop(columns=['htn']).columns.tolist()

# except Exception as e:
#     print(f"❌ Error loading ANN components: {e}")
#     ann_model, scaler, label_encoders, X_columns = None, None, None, []


# def predict_cnn_image(image):
#     image = image.convert("RGB")
#     img = image.resize(IMG_SIZE)
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = cnn_model.predict(img_array)
#     class_names = ["Normal", "Tumor"]
#     predicted_class = class_names[np.argmax(prediction)]

#     if predicted_class in cnn_df['Class'].values:
#         info = cnn_df[cnn_df['Class'] == predicted_class].iloc[0].to_dict()
#         for field in ['Unnamed: 0', 'full_path', 'image_id', 'path', 'target']:
#             info.pop(field, None)
#     else:
#         info = {"Note": "No extra info available in database"}

#     return {"class": predicted_class, "database_info": info}


# @app.route('/')
# def index():
#     return render_template('upload.html')


# @app.route('/upload')
# def upload():
#     return render_template('upload.html')


# @app.route('/open')
# def open():
#     return render_template('open.html')  # Important: matches route name for url_for()


# @app.route('/signup')
# def signup():
#     return render_template('signup.html')


# @app.route('/login')
# def login():
#     return render_template('login.html')


# @app.route('/data')
# def data():
#     return render_template('data.html')


# @app.route('/annpredict')
# def annpredict():
#     return render_template('annpredict.html')


# @app.route('/cnnprediction')
# def cnnprediction():
#     return render_template('cnnprediction.html')


# @app.route('/predict_cnn', methods=['POST'])
# def predict_cnn():
#     base64_image = request.form.get('image')
#     if not base64_image:
#         print("❌ No image data provided")
#         return jsonify({"error": "No image data provided"}), 400

#     try:
#         if ',' not in base64_image:
#             raise ValueError("Invalid image format")

#         header, encoded = base64_image.split(',', 1)
#         image_data = base64.b64decode(encoded)
#         image = Image.open(BytesIO(image_data))

#         result = predict_cnn_image(image)
#         return jsonify({
#             "prediction": result["class"],
#             "database_info": result["database_info"]
#         })

#     except Exception as e:
#         print(f"❌ CNN Prediction Error: {e}")
#         return jsonify({"error": f"CNN Prediction failed: {str(e)}"}), 500


# @app.route('/predict_ann', methods=['POST'])
# def predict_ann():
#     if not ann_model or not scaler or not label_encoders:
#         return jsonify({"error": "ANN model or preprocessors not loaded."}), 500

#     try:
#         if request.content_type != "application/json":
#             return jsonify({"error": "Content-Type must be application/json"}), 415

#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "No input data provided"}), 400

#         missing = [f for f in X_columns if f not in data]
#         if missing:
#             return jsonify({"error": f"Missing features: {missing}"}), 400

#         categorical_map = {"Yes": 1, "No": 0}
#         for col in ['htn', 'dm']:
#             if col in data and data[col] in categorical_map:
#                 data[col] = categorical_map[data[col]]

#         for col, encoder in label_encoders.items():
#             if col in data and isinstance(data[col], str):
#                 try:
#                     data[col] = encoder.transform([data[col]])[0]
#                 except ValueError:
#                     return jsonify({"error": f"Invalid value for {col}"}), 400

#         input_array = np.array([data[col] for col in X_columns]).reshape(1, -1)
#         input_array = scaler.transform(input_array)
#         prediction = ann_model.predict(input_array)[0][0]
#         result = "Abnormal" if prediction > 0.5 else "Normal"

#         return jsonify({"prediction": result, "probability": float(prediction)})

#     except Exception as e:
#         print(f"❌ ANN Prediction Error: {e}")
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     app.run(debug=True)

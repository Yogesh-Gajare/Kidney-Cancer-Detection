import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Constants
IMG_SIZE = (128, 128)  
BATCH_SIZE = 32
EPOCHS = 10
DATASET_CSV = "kidneyData_filtered.csv"  
IMAGE_FOLDER = "C:/Users/Yogesh/OneDrive/Desktop/Kidney cancer detection/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

# Load dataset
df = pd.read_csv(DATASET_CSV)
df['full_path'] = df['path'].apply(lambda x: os.path.join(IMAGE_FOLDER, str(x).strip()))

# Function to load images
def load_images(df):
    images, labels = [], []
    label_map = {"normal": 0, "tumor": 1,}

    for _, row in df.iterrows():
        img_path = row['full_path']
        label = label_map.get(row['Class'].strip().lower(), -1)

        if label == -1:
            print(f"Warning: Unknown label '{row['Class']}' for {img_path}")
            continue
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0  
            images.append(img_array)
            labels.append(label)
        else:
            print(f"Warning: File not found - {img_path}")

    return np.array(images), to_categorical(labels, num_classes=4)

# Load data
X, y = load_images(df)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  
])

# Compile & Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Save model
model.save("kidney_cancer_model.h5")
print("✅ Model saved as 'kidney_cancer_model.h5'.")


# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.metrics.pairwise import cosine_similarity

# # Constants
# IMG_SIZE = (128, 128)  
# DATASET_CSV = "kidneyData_filtered.csv"  
# IMAGE_FOLDER = "C:/Users/Yogesh/OneDrive/Desktop/Kidney cancer detection/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

# # Load dataset
# df = pd.read_csv(DATASET_CSV)
# df['full_path'] = df['path'].apply(lambda x: os.path.join(IMAGE_FOLDER, str(x).strip()))

# # Label Mapping
# label_map = {"normal": 0, "tumor": 1}  
# reverse_label_map = {v: k for k, v in label_map.items()}

# # Load trained model
# model = load_model("kidney_cancer_model.h5")

# # Function to load and preprocess an image
# def preprocess_image(image_path):
#     img = load_img(image_path, target_size=IMG_SIZE)
#     img_array = img_to_array(img) / 255.0  # Normalize pixel values
#     return np.expand_dims(img_array, axis=0)  # Add batch dimension

# # Function to load dataset images for comparison
# def load_dataset_images(df):
#     images, labels = [], []
#     for _, row in df.iterrows():
#         img_path = row['full_path']
#         label = label_map.get(row['Class'].strip().lower(), -1)

#         if label == -1:
#             continue
#         if os.path.exists(img_path):
#             img = load_img(img_path, target_size=IMG_SIZE)
#             img_array = img_to_array(img) / 255.0  
#             images.append(img_array)
#             labels.append(label)

#     return np.array(images), np.array(labels)

# # Load dataset images for similarity comparison
# X_dataset, y_dataset = load_dataset_images(df)

# # Function to predict an external image
# def predict_external_image(image_path):
#     if not os.path.exists(image_path):
#         return "Error: Image not found!"

#     img_array = preprocess_image(image_path)
#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction)  # Get highest probability class
#     predicted_label = reverse_label_map.get(predicted_class, "Unknown")

#     # Find closest dataset match
#     closest_match = find_closest_dataset_image(img_array, X_dataset, y_dataset)

#     return f"Predicted: {predicted_label}, Closest Match in Dataset: {closest_match}"

# # Function to find the closest matching dataset image
# def find_closest_dataset_image(external_img_array, dataset_images, dataset_labels):
#     if dataset_images.size == 0:
#         return "No dataset images available for comparison."

#     external_img_flat = external_img_array.flatten().reshape(1, -1)
#     dataset_flat = dataset_images.reshape(dataset_images.shape[0], -1)

#     similarities = cosine_similarity(external_img_flat, dataset_flat)
#     best_match_index = np.argmax(similarities)
    
#     return reverse_label_map.get(dataset_labels[best_match_index], "Unknown")

# # Example usage
# external_image_path = "C:/path/to/external/image.jpg"
# result = predict_external_image(external_image_path)
# print(result)




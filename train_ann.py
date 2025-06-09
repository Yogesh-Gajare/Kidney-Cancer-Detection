
# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import tensorflow as tf
# from tensorflow import keras
# import joblib

# # Constants
# DATASET_PATH = "kidney_cancer_large_dataset.csv"
# MODEL_SAVE_PATH = "kidney_cancer_ann_model.h5"
# SCALER_SAVE_PATH = "scaler.pkl"
# ENCODER_SAVE_PATH = "label_encoders.pkl"

# # Load dataset
# df = pd.read_csv(DATASET_PATH)

# # Encode categorical variables
# label_encoders = {}
# categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# for col in categorical_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le

# # Split dataset
# X = df.drop(columns=['htn'])  # Assuming 'htn' is the target variable
# y = df['htn']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Normalize numerical data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Build ANN model
# model = keras.Sequential([
#     keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
#     keras.layers.Dense(16, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

# # Save the trained model and preprocessors
# model.save(MODEL_SAVE_PATH)
# joblib.dump(scaler, SCALER_SAVE_PATH)
# joblib.dump(label_encoders, ENCODER_SAVE_PATH)

# print(f"✅ ANN Model saved as '{MODEL_SAVE_PATH}'.")
# print(f"✅ Scaler saved as '{SCALER_SAVE_PATH}'.")
# print(f"✅ Label Encoders saved as '{ENCODER_SAVE_PATH}'.")




import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
import joblib

# Constants
DATASET_PATH = "kidney_cancer_filtered_dataset.csv"
MODEL_SAVE_PATH = "kidney_cancer_ann_model.h5"
SCALER_SAVE_PATH = "scaler.pkl"
ENCODER_SAVE_PATH = "label_encoders.pkl"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Drop rows with missing values to ensure data integrity
df.dropna(inplace=True)

# Identify categorical and numerical columns
categorical_cols = ['htn', 'dm']  # Categorical features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Ensure target column is numeric
target_col = 'htn'  # Change if the target is different
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' is missing from dataset.")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Ensure all expected features are included
expected_features = [col for col in df.columns if col != target_col]

# Split dataset
X = df[expected_features]  # Features
y = df[target_col]  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize numerical data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN model
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    #keras.Input(shape=(X_train.shape[1],)),
    #keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

# Save the trained model and preprocessors
model.save(MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)
joblib.dump(label_encoders, ENCODER_SAVE_PATH)

print(f"✅ ANN Model saved as '{MODEL_SAVE_PATH}'.")
print(f"✅ Scaler saved as '{SCALER_SAVE_PATH}'.")
print(f"✅ Label Encoders saved as '{ENCODER_SAVE_PATH}'.")

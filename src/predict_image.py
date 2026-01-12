import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# =====================
# CONFIG
# =====================
MODEL_PATH = "deepfake_final.h5"   # change if needed
IMG_SIZE = 224

# =====================
# LOAD MODEL
# =====================
model = load_model(MODEL_PATH)

# =====================
# PREDICT FUNCTION
# =====================
def predict_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Image not found:", image_path)
        return

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # IMPORTANT: same preprocessing as training
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        label = "Fake"
        confidence = pred
    else:
        label = "Real"
        confidence = 1 - pred

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

# =====================
# EXAMPLE USAGE
# =====================
if __name__ == "__main__":
    image_path = "/Users/nikhilshivhare/Downloads/real image 1.jpeg"  # 👈 change this path
    predict_image(image_path)

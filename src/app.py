import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="🧠 AI Deepfake Detector",
    page_icon="🧠",
    layout="wide"
)

IMG_SIZE = 224
LAST_CONV_LAYER = "top_conv"

# ----------------------------
# PATH HANDLING (VERY IMPORTANT)
# ----------------------------



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "deepfake_final.h5")
BG_IMAGE_PATH = os.path.join(BASE_DIR, "bg.jpg")


# ----------------------------
# BACKGROUND IMAGE
# ----------------------------
def set_background(image_path):
    if os.path.exists(image_path):
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/jpg;base64,{open(image_path,'rb').read().decode('latin1')}") no-repeat center center fixed;
                background-size: cover;
            }}
            .content-box {{
                background-color: rgba(0, 0, 0, 0.6);
                padding: 20px;
                border-radius: 15px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

set_background(BG_IMAGE_PATH)


# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# ----------------------------
# HEADER
# ----------------------------
st.markdown(
    """
    <div class="content-box">
        <h1 style='text-align:center;color:#f0fdf4;'>🧠 AI Deepfake Detector</h1>
        <h3 style='text-align:center;color:#f0fdf4;'>Image & Video Deepfake Detection</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# UPLOAD SECTION
# ----------------------------
st.subheader("📤 Upload Images")
uploaded_images = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.subheader("🎥 Upload Video")
uploaded_video = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi"],
    accept_multiple_files=False
)

# ----------------------------
# IMAGE PREPROCESS
# ----------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ----------------------------
# PREDICTION
# ----------------------------
def predict(img_array):
    pred = model.predict(img_array, verbose=0)[0][0]
    if pred < 0.5:
        return "REAL", (1 - pred) * 100
    else:
        return "FAKE", pred * 100

# ----------------------------
# GRAD-CAM
# ----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap

# ----------------------------
# HEATMAP OVERLAY
# ----------------------------
def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, original_img.size)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    original_np = np.array(original_img)
    return cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

# ----------------------------
# IMAGE DETECTION
# ----------------------------
if uploaded_images:
    st.subheader("🖼 Image Detection Results")

    for uploaded_file in uploaded_images:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = preprocess_image(image)

        label, confidence = predict(img_array)
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
        overlay = overlay_heatmap(image, heatmap)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", width=400)
        with col2:
            st.image(overlay, caption="AI Attention (Grad-CAM)", width=400)

        if label == "REAL":
            st.success(f"✅ REAL — Confidence: {confidence:.2f}%")
        else:
            st.error(f"🚨 FAKE — Confidence: {confidence:.2f}%")

        st.progress(int(confidence))
        st.markdown("---")

# ----------------------------
# VIDEO DETECTION (NO GRAD-CAM)
# ----------------------------
if uploaded_video:
    st.subheader("🎥 Video Detection Result")
    st.info("Analyzing video frames...")

    temp_video = "temp_video.mp4"
    with open(temp_video, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_video)
    frame_skip = 15
    preds = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            preds.append(model.predict(img_array, verbose=0)[0][0])

        count += 1

    cap.release()
    os.remove(temp_video)

    avg_pred = np.mean(preds)

    if avg_pred < 0.5:
        st.success(f"✅ VIDEO REAL — Confidence: {(1 - avg_pred) * 100:.2f}%")
        st.info("Most frames appear natural with no strong AI artifacts.")
    else:
        st.error(f"🚨 VIDEO FAKE — Confidence: {avg_pred * 100:.2f}%")
        st.info("Multiple frames show AI-generated inconsistencies.")

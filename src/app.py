import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="🧠 AI Deepfake Detector",
    page_icon="🧠",
    layout="wide"
)

IMG_SIZE = 224
MODEL_PATH = "/Users/nikhilshivhare/PycharmProjects/deepfake_detector/deepfake_final.h5"
LAST_CONV_LAYER = "top_conv"  # Change according to your model

# ----------------------------
# BACKGROUND IMAGE
# ----------------------------
def set_background(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_file}") no-repeat center center fixed;
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

set_background("bg.jpg")  # relative path inside src folder

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
        <h1 style='text-align: center; color: #f0fdf4;'>🧠 AI Deepfake Detector</h1>
        <h3 style='text-align: center; color: #f0fdf4;'>Images & Videos Detection with AI explanation</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# UPLOAD
# ----------------------------
st.subheader("Upload Images")
uploaded_images = st.file_uploader(
    "Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

st.subheader("Upload Video")
uploaded_video = st.file_uploader(
    "Upload a video (.mp4, .avi)", type=["mp4", "avi"], accept_multiple_files=False
)

# ----------------------------
# IMAGE PREPROCESS
# ----------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)/255.0
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
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap

# ----------------------------
# OVERLAY HEATMAP
# ----------------------------
def overlay_heatmap(original_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
    original_np = np.array(original_img)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)
    return overlay

# ----------------------------
# EXPLANATION
# ----------------------------
def generate_explanation(heatmap, label):
    mean_activation = np.mean(heatmap)
    max_activation = np.max(heatmap)
    if label == "FAKE":
        if max_activation > 0.7:
            return ("🧠 Explanation: Model focused on facial regions indicating AI-generated artifacts.")
        else:
            return ("🧠 Explanation: Subtle inconsistencies suggest manipulation.")
    else:
        if mean_activation < 0.35:
            return "🧠 Explanation: Low attention; image appears natural."
        else:
            return "🧠 Explanation: Some attention observed but consistent with real images."

# ----------------------------
# IMAGE DETECTION
# ----------------------------
if uploaded_images:
    st.subheader("Image Detection Results")
    for uploaded_file in uploaded_images:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert("RGB")
        img_array = preprocess_image(image)
        label, confidence = predict(img_array)
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
        overlay = overlay_heatmap(image, heatmap)
        explanation = generate_explanation(heatmap, label)

        with col1:
            st.image(image, caption="Original Image", width=400)
        with col2:
            st.image(overlay, caption="Grad-CAM Heatmap", width=400)

        if label == "REAL":
            st.success(f"✅ REAL IMAGE — Confidence: {confidence:.2f}%")
        else:
            st.error(f"🚨 FAKE IMAGE — Confidence: {confidence:.2f}%")

        st.progress(int(confidence))
        st.info(explanation)
        st.markdown("---")

# ----------------------------
# VIDEO DETECTION (No Grad-CAM)
# ----------------------------
if uploaded_video:
    st.subheader("Video Detection Results")
    st.info("Processing video frames ...")

    # Save uploaded video temporarily
    video_path = f"temp_{uploaded_video.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(video_path)
    frame_rate = 15  # analyze one frame every 15 frames (~0.5s for 30fps)
    count = 0
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(np.array(img_pil)/255.0, axis=0)
            pred = model.predict(img_array, verbose=0)[0][0]
            predictions.append(pred)
        count += 1
    cap.release()

    # Aggregate results
    avg_pred = np.mean(predictions)
    if avg_pred < 0.5:
        video_label = "REAL"
        video_conf = (1 - avg_pred) * 100
        st.success(f"✅ VIDEO PREDICTION: REAL — Confidence: {video_conf:.2f}%")
        st.info("🧠 Explanation: Most frames appear natural, no significant AI artifacts detected.")
    else:
        video_label = "FAKE"
        video_conf = avg_pred * 100
        st.error(f"🚨 VIDEO PREDICTION: FAKE — Confidence: {video_conf:.2f}%")
        st.info("🧠 Explanation: Several frames show AI-generated inconsistencies, indicating a manipulated video.")

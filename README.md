Detect Deepfake Images and Videos using AI with Grad-CAM visualization
Overview
This project is a Streamlit web application that detects real vs. AI-generated (deepfake) content in both images and videos. It leverages a Convolutional Neural Network (CNN) model trained on real and fake faces. For images, it also provides Grad-CAM heatmaps to show which regions of the image influenced the prediction, along with AI-generated explanations.
The video detection uses the image-trained model to analyze sampled frames, providing a confidence score and a textual explanation without overlaying Grad-CAM heatmaps for faster and cleaner results.
Features
Image Detection
Upload one or more images (jpg, jpeg, png).
Get predictions: REAL or FAKE with confidence %.
View Grad-CAM heatmaps to visualize model focus areas.
Receive AI-generated explanations of the prediction.
Video Detection
Upload a video (.mp4, .avi).
The app samples frames and predicts each frame using the trained model.
Aggregates results to provide a final video prediction.
Provides textual explanation of possible manipulations.
No heatmap overlay on video frames for cleaner visualization.
Tech Stack
Python 3.10+
TensorFlow / Keras
OpenCV (video processing)
Streamlit (web interface)
NumPy, Pillow, h5py

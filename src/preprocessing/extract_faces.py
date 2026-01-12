import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

detector = MTCNN()

RAW_DIR = "/Users/nikhilshivhare/PycharmProjects/deepfake_detector/data/raw/images"
OUT_DIR = "/Users/nikhilshivhare/PycharmProjects/deepfake_detector/data/processed/images"

os.makedirs(OUT_DIR, exist_ok=True)

def process_folder(label):
    input_path = os.path.join(RAW_DIR, label)
    output_path = os.path.join(OUT_DIR, label)
    os.makedirs(output_path, exist_ok=True)

    for img_name in tqdm(os.listdir(input_path), desc=f"Processing {label}"):
        img_path = os.path.join(input_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = detector.detect_faces(img)

        for i, face in enumerate(faces):
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)
            face_img = img[y:y+h, x:x+w]

            if face_img.size == 0:
                continue

            face_img = cv2.resize(face_img, (224, 224))
            save_name = f"{os.path.splitext(img_name)[0]}_{i}.jpg"
            cv2.imwrite(os.path.join(output_path, save_name), face_img)

for label in ["real", "fake"]:
    process_folder(label)

print("✅ Face extraction completed")

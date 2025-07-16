import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from insightface.app import FaceAnalysis
import os

IMG_SIZE = 128
label_map = {"japanese": 0, "korean": 1, "chinese": 2}
rev_label_map = {v: k for k, v in label_map.items()}

# insightfaceセットアップ
face_app = FaceAnalysis(name="buffalo_sc")
face_app.prepare(ctx_id=0, det_size=(640, 640))

def crop_face(img, save_path=None):
    faces = face_app.get(img)
    if len(faces) == 0:
        st.error("顔が検出されません")
        return None
    kps = faces[0]['kps']
    left_eye, right_eye, nose, mouth_left, mouth_right = kps
    eye_center = (left_eye + right_eye) / 2
    mouth_center = (mouth_left + mouth_right) / 2
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    center = (int(eye_center[0]), int(eye_center[1]))
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    img_rot = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    def rotate(point, center, angle_rad):
        x, y = point - center
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        return np.array([x_new, y_new]) + center
    landmarks_rot = np.array([rotate(pt, eye_center, -angle_rad) for pt in kps])
    left_eye_rot, right_eye_rot, nose_rot, mouth_left_rot, mouth_right_rot = landmarks_rot
    mouth_center_rot = (mouth_left_rot + mouth_right_rot) / 2
    eye_center_rot = (left_eye_rot + right_eye_rot) / 2
    crop_center = (eye_center_rot + mouth_center_rot) / 2
    eye_dist = np.linalg.norm(right_eye_rot - left_eye_rot)
    mouth_dist = np.linalg.norm(mouth_right_rot - mouth_left_rot)
    crop_size = int(max(eye_dist, mouth_dist) * 2.5)
    x1 = int(crop_center[0] - crop_size // 2)
    y1 = int(crop_center[1] - crop_size // 2)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)
    img_crop = img_rot[y1:y2, x1:x2]
    if img_crop.size == 0:
        st.error("クロップ範囲が不正")
        return None
    img_crop = cv2.resize(img_crop, (IMG_SIZE, IMG_SIZE))
    if save_path is not None:
        cv2.imwrite(save_path, img_crop)
    return img_crop

def predict_nationality(img):
    model = load_model("face_nationality_vgg16.h5")
    os.makedirs("input_image", exist_ok=True)
    save_crop_path = os.path.join("input_image", "streamlit_input.jpg")
    img_face = crop_face(img, save_path=save_crop_path)
    if img_face is None:
        return None, None, None
    img_face_norm = img_face.astype('float32') / 255.0
    pred = model.predict(np.expand_dims(img_face_norm, axis=0))[0]
    idx = np.argmax(pred)
    return rev_label_map[idx], pred[idx], img_face  # ← 切り抜き画像も返す

# --- Streamlit UI ---
st.title("国籍判定AI（カメラ画像対応）")

img_file = st.camera_input("カメラで顔写真を撮影してください")

if img_file is not None:
    # 画像をOpenCV形式に変換
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="入力画像", channels="BGR")
    label, score, face_img = predict_nationality(img)
    if label is not None:
        st.success(f"判定: {label}（確信度: {score:.2f}）")
        # 切り抜き顔画像も表示
        st.image(face_img, caption="切り抜き顔画像", channels="BGR")
    else:
        st.error("判定できませんでした")
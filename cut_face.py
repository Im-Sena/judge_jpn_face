from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os
import glob

img_dir = "D:/Univ/judge_jpn_face/image japanese/"
output_dir = "D:/Univ/judge_jpn_face/cut_face/"
output_size = 256

os.makedirs(output_dir, exist_ok=True)

InsightFaceLandmarksDetector = FaceAnalysis(name="buffalo_sc")
InsightFaceLandmarksDetector.prepare(ctx_id=0, det_size=(640, 640))

def rotate(point, center, angle_rad):
    """点をcenter中心にangle_rad回転"""
    x, y = point - center
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a
    return np.array([x_new, y_new]) + center

img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

for img_path in img_files:
    print(f"処理開始: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"画像が読み込めません: {img_path}")
        continue

    faces = InsightFaceLandmarksDetector.get(img)
    if len(faces) == 0:
        print(f"顔が検出されません: {img_path}")
        continue

    landmarks = faces[0]['kps']
    if landmarks is None:
        print(f"ランドマークが取得できません: {img_path}")
        continue

    # 5点: [左目, 右目, 鼻先, 口左, 口右]
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    mouth_left = landmarks[3]
    mouth_right = landmarks[4]

    # 目の中心・口の中心
    eye_center = (left_eye + right_eye) / 2
    mouth_center = (mouth_left + mouth_right) / 2

    # 顔の回転角度
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    # 回転
    #center = tuple(eye_center.astype(int))
    center = (int(eye_center[0]), int(eye_center[1]))
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    img_rot = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    # ランドマークも回転
    landmarks_rot = np.array([rotate(pt, eye_center, -angle_rad) for pt in landmarks])
    left_eye_rot, right_eye_rot, nose_rot, mouth_left_rot, mouth_right_rot = landmarks_rot
    mouth_center_rot = (mouth_left_rot + mouth_right_rot) / 2
    eye_center_rot = (left_eye_rot + right_eye_rot) / 2

    # クロップ中心とサイズ
    crop_center = (eye_center_rot + mouth_center_rot) / 2
    eye_dist = np.linalg.norm(right_eye_rot - left_eye_rot)
    mouth_dist = np.linalg.norm(mouth_right_rot - mouth_left_rot)
    crop_size = int(max(eye_dist, mouth_dist) * 2.5)  # 顔全体が入るように調整

    x1 = int(crop_center[0] - crop_size // 2)
    y1 = int(crop_center[1] - crop_size // 2)
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # 範囲外補正
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    img_crop = img_rot[y1:y2, x1:x2]
    if img_crop.size == 0:
        print(f"クロップ範囲が不正: {img_path}")
        continue

    img_crop = cv2.resize(img_crop, (output_size, output_size))
    base_name = os.path.basename(img_path)
    dst_file = os.path.join(output_dir, base_name)
    cv2.imwrite(dst_file, img_crop)
    print(f"保存しました: {dst_file}")
import os
import cv2
import numpy as np

# 入力フォルダと出力フォルダ
input_dirs = ["true_image_japanese", "true_image_korean", "true_image_chinese"]
output_dirs = ["expand_japanese_face", "expand_korean_face", "expand_chinese_face"]

def random_rotate(img, angle_range=20):
    angle = np.random.uniform(-angle_range, angle_range)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

for in_dir, out_dir in zip(input_dirs, output_dirs):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(in_dir):
        in_path = os.path.join(in_dir, fname)
        img = cv2.imread(in_path)
        if img is None:
            continue
        # 元画像を保存
        cv2.imwrite(os.path.join(out_dir, fname), img)
        # 反転画像を保存
        img_flip = cv2.flip(img, 1)
        name, ext = os.path.splitext(fname)
        flip_name = f"{name}_flip{ext}"
        cv2.imwrite(os.path.join(out_dir, flip_name), img_flip)
        # 明度変更画像を保存
        img_bright = cv2.convertScaleAbs(img, alpha=1.0, beta=40)# betaを大きくすると明るく
        bright_name = f"{name}_bright{ext}"
        cv2.imwrite(os.path.join(out_dir, bright_name), img_bright)
        # 明度変更＋反転画像
        img_bright_flip = cv2.flip(img_bright, 1)
        bright_flip_name = f"{name}_bright_flip{ext}"
        cv2.imwrite(os.path.join(out_dir, bright_flip_name), img_bright_flip)
        # ランダム回転画像
        img_rot = random_rotate(img, angle_range=20)
        rot_name = f"{name}_rot{ext}"
        cv2.imwrite(os.path.join(out_dir, rot_name), img_rot)
        # ランダム回転＋反転画像
        img_rot_flip = cv2.flip(img_rot, 1)
        rot_flip_name = f"{name}_rot_flip{ext}"
import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# --- 設定 ---
IMG_SIZE = 128
DATASET = {
    "japanese": "true_image_japanese",
    "korean": "true_image_korean",
    "chinese": "true_image_chinese"
}

def load_images():
    images = []
    labels = []
    label_map = {k: i for i, k in enumerate(DATASET.keys())}
    for label, folder in DATASET.items():
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label_map[label])
    return np.array(images), np.array(labels), label_map

# --- データ読み込み ---
X, y, label_map = load_images()
X = X.astype('float32') / 255.0
y_cat = to_categorical(y, num_classes=len(DATASET))

# --- 学習 ---
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# VGG16ベースモデル（全結合層なし、入力サイズ指定）
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False  # 転移学習なので畳み込み層は固定

# 新しい全結合層を追加
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(DATASET), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

model.save("face_nationality_vgg16.h5")
print("モデル保存完了")

# --- 推論関数 ---
def predict_nationality(img_path):
    model = load_model("face_nationality_vgg16.h5")
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    pred = model.predict(np.expand_dims(img, axis=0))[0]
    idx = np.argmax(pred)
    rev_label_map = {v: k for k, v in label_map.items()}
    print(f"判定: {rev_label_map[idx]} (確信度: {pred[idx]:.2f})")
    return rev_label_map[idx], pred[idx]

# --- 使い方例 ---
# predict_nationality("test.jpg")
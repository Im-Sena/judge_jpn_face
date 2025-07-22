import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras.layers import MaxPooling2D, Conv2D

# --- 画像の読み込み ---
img_path = "kju.jpg"  # 可視化したい画像ファイル名
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# --- モデルのロード ---
model = VGG16(weights="imagenet")

# --- 可視化したい層の出力を取得 ---
layers = model.layers[1:19]  # VGG16の畳み込み・プーリング層
layer_outputs = [layer.output for layer in layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img)

# --- 畳み込み層・プーリング層だけ抽出 ---
conv_and_pool_activations = []
for layer, activation in zip(layers, activations):
    is_pooling_layer = isinstance(layer, MaxPooling2D)
    is_convolution_layer = isinstance(layer, Conv2D)
    if is_pooling_layer or is_convolution_layer:
        conv_and_pool_activations.append([layer.name, activation])

# --- ヒートマップとして保存 ---
os.makedirs("heatmap_output", exist_ok=True)
for name, activation in conv_and_pool_activations:
    n_imgs = activation.shape[-1]
    n_cols = math.ceil(math.sqrt(n_imgs))
    n_rows = math.ceil(n_imgs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    for i in range(n_imgs):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        featuremap_img = activation[0, :, :, i]
        sns.heatmap(featuremap_img, ax=ax, cbar=False, xticklabels=False, yticklabels=False)
        ax.axis('off')
    # 空きサブプロットを非表示
    for j in range(n_imgs, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join("heatmap_output", f"{name}.png")
    plt.savefig(save_path)
    plt.close()
print("特徴マップのヒートマップ画像を 'heatmap_output' フォルダに保存しました。")
import cv2
import numpy as np
import os
from tqdm import tqdm
import joblib

# 画像フォルダから画像を読み込む関数
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# HOG特徴量を計算する関数
def compute_hog_features(images, target_size=(64, 128)):
    hog = cv2.HOGDescriptor()
    features = []
    for img in tqdm(images, desc="Calculating HOG features"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, target_size)
        h = hog.compute(resized)
        features.append(h.flatten())  # フラットにして統一
    return np.array(features)

# データセットのパスを設定
positive_folder = 'data/positive'
negative_folder = 'data/negative'

# データセットをロード
positive_images = load_images_from_folder(positive_folder)
negative_images = load_images_from_folder(negative_folder)

# HOG特徴量を計算
print("Calculating HOG features for positive images...")
positive_features = compute_hog_features(positive_images)

print("Calculating HOG features for negative images...")
negative_features = compute_hog_features(negative_images)

# ラベルを作成
positive_labels = np.ones(len(positive_features))
negative_labels = np.zeros(len(negative_features))

# データとラベルを結合
X = np.vstack((positive_features, negative_features))
y = np.hstack((positive_labels, negative_labels))

# データとラベルを保存
joblib.dump((X, y), 'hog_features_labels.pkl')
print("HOG features and labels saved to 'hog_features_labels.pkl'")

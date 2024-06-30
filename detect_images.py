import cv2
import joblib
import numpy as np
import sys
import os

# コマンドライン引数から入力ディレクトリと出力ディレクトリを取得
if len(sys.argv) != 3:
    print("Usage: python detect_images.py <input_directory> <output_directory>")
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

# 入力ディレクトリが存在するか確認
if not os.path.exists(input_dir):
    print(f"Error: Input directory {input_dir} does not exist.")
    sys.exit(1)

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存されたモデルをロード
svm = joblib.load('svm_model.pkl')

# HOG Descriptorを初期化
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 画像サイズを学習時と一致させる
target_size = (64, 128)  # 学習時に使用したサイズ

# 入力ディレクトリ内のすべての画像ファイルを処理
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 画像を読み込む
        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: Could not read image {input_path}")
            continue
        
        # フレームをグレースケールに変換する
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # フレームを学習時と同じサイズにリサイズ
        resized = cv2.resize(gray, target_size)
        
        # HOG特徴量を計算
        h = hog.compute(resized).reshape(1, -1)
        
        # SVMモデルを使用して人を検出
        pred = svm.predict(h)
        confidence = svm.decision_function(h)  # 信頼度スコアを取得
        
        # 人が検出された場合にバウンディングボックスを描画
        if pred == 1 and confidence > 0:  # 信頼度スコアを考慮
            # HOG特徴量を使用して人を検出
            boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
            for (x, y, w, h), weight in zip(boxes, weights):
                # if weight > 0.5:  # 信頼度スコアが0.5以上の場合に描画
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 検出結果を出力ディレクトリに保存
        cv2.imwrite(output_path, img)
        print(f"Processed {input_path}, saved to {output_path}")

print("Detection finished for all images in the directory.")

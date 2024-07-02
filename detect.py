import cv2
import joblib
import numpy as np
import sys
import os

# コマンドライン引数から入力ファイルを取得
if len(sys.argv) != 2:
    print("Usage: python detect.py <input_video.mp4>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = os.path.splitext(input_file)[0] + "_output.mp4"

# 保存されたモデルをロード
svm = joblib.load('svm_model.pkl')

# HOG Descriptorを初期化
hog = cv2.HOGDescriptor()

# 動画ファイルをキャプチャする
cap = cv2.VideoCapture(input_file)
if not cap.isOpened():
    print(f"Error: Cannot open video file {input_file}")
    sys.exit(1)

# 動画のプロパティを取得
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力ファイルのフォーマット（MP4）
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力動画ファイルを作成
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while True:
    # フレームをキャプチャする
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # フレームをグレースケールに変換する
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 画像サイズを64x128のパッチに分割
    for y in range(0, gray.shape[0] - 128, 64):
        for x in range(0, gray.shape[1] - 64, 32):
            patch = gray[y:y + 128, x:x + 64]
            h = hog.compute(patch).reshape(1, -1)
            pred = svm.predict(h)
            if pred == 1:
                if weight > 0.5:  # 信頼度スコアが0.5以上の場合に描画
                    cv2.rectangle(frame, (x, y), (x + 64, y + 128), (0, 255, 0), 2)
    
    # フレームを出力動画ファイルに書き込み
    out.write(frame)

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
out.release()
print(f"Detection finished. Output saved as {output_file}")

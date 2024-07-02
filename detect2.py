import cv2
import joblib
import numpy as np
import sys
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def non_max_suppression(boxes, scores, overlapThresh):
    if len(boxes) == 0:
        return []
    
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")

def detect_patches(patches, svm, hog, scale, confidence_threshold=0.7):
    boxes = []
    scores = []
    for (x, y, patch) in patches:
        h = hog.compute(patch).reshape(1, -1)
        pred = svm.predict(h)
        confidence = svm.decision_function(h)[0]
        if pred == 1 and confidence > confidence_threshold:
            x1 = int(x * scale)
            y1 = int(y * scale)
            x2 = int((x + patch.shape[1]) * scale)
            y2 = int((y + patch.shape[0]) * scale)
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
    return boxes, scores

# コマンドライン引数から入力ファイルを取得
if len(sys.argv) != 2:
    print("Usage: python detect.py <input_video.mp4>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = os.path.splitext(input_file)[0] + "_output2.mp4"

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

# 総フレーム数を取得
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 画像サイズを64x128のパッチに分割
patch_width, patch_height = 64, 128
step_size = 8  # パッチのスライドステップ
scales = [0.5, 1.0, 1.5, 2.0]  # マルチスケール検出用のスケールリスト

def process_frame(frame):
    all_boxes = []
    all_scores = []

    for scale in scales:
        resized_frame = cv2.resize(frame, (int(width / scale), int(height / scale)))
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        patches = []
        for y in range(0, gray.shape[0] - patch_height + 1, step_size):
            for x in range(0, gray.shape[1] - patch_width + 1, step_size):
                patch = gray[y:y + patch_height, x:x + patch_width]
                patches.append((x, y, patch))

        boxes, scores = detect_patches(patches, svm, hog, scale)
        all_boxes.extend(boxes)
        all_scores.extend(scores)

    # 非最大抑制を適用
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    final_boxes = non_max_suppression(all_boxes, all_scores, overlapThresh=0.3)

    for (x1, y1, x2, y2) in final_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame

# フレームごとに人を検出
with ThreadPoolExecutor() as executor:
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        # フレームをキャプチャする
        ret, frame = cap.read()
        
        if not ret:
            break

        frame = executor.submit(process_frame, frame).result()

        # フレームを出力動画ファイルに書き込み
        out.write(frame)

# キャプチャを解放し、ウィンドウを閉じる
cap.release()
out.release()
print(f"Detection finished. Output saved as {output_file}")

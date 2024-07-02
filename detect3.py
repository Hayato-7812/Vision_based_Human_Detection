import cv2
import joblib
import numpy as np
import sys
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def pyramid(image, scale=1.5, min_size=(64, 128)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        yield image

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    return hog.compute(image)

def detect_human_multiscale(image, model, window_size=(64, 128), step_size=32, scale=1.2, confidence_threshold=0.9):
    detections = []
    confidences = []
    for resized in pyramid(image, scale=scale):
        scale_factor = image.shape[0] / float(resized.shape[0])
        with ThreadPoolExecutor() as executor:
            futures = []
            for (x, y, window) in sliding_window(resized, step_size=step_size, window_size=window_size):
                if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                    continue
                futures.append(executor.submit(extract_hog_features, window))
            
            for i, future in enumerate(futures):
                features = future.result().reshape(1, -1)
                prediction = model.predict(features)
                confidence = model.decision_function(features)[0]
                if prediction == 1 and confidence > confidence_threshold:
                    x, y = i % (resized.shape[1] // step_size) * step_size, i // (resized.shape[1] // step_size) * step_size
                    x1 = int(x * scale_factor)
                    y1 = int(y * scale_factor)
                    x2 = int((x + window_size[0]) * scale_factor)
                    y2 = int((y + window_size[1]) * scale_factor)
                    detections.append((x1, y1, x2, y2))
                    confidences.append(confidence)
    return np.array(detections), np.array(confidences)

def non_max_suppression(boxes, scores, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

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

def filter_boxes_by_size(boxes, confidences, image_shape, min_ratio=(0.1, 0.3)):
    min_width = image_shape[1] * min_ratio[0]
    min_height = image_shape[0] * min_ratio[1]
    filtered_boxes = []
    filtered_confidences = []
    for (x1, y1, x2, y2), confidence in zip(boxes, confidences):
        if (x2 - x1) >= min_width and (y2 - y1) >= min_height:
            filtered_boxes.append((x1, y1, x2, y2))
            filtered_confidences.append(confidence)
    return np.array(filtered_boxes), np.array(filtered_confidences)

def process_video(input_file, output_file, model):
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_file}")
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections, confidences = detect_human_multiscale(gray, model)
        detections, confidences = filter_boxes_by_size(detections, confidences, gray.shape, min_ratio=(0.1, 0.3))
        final_detections = non_max_suppression(detections, confidences, overlapThresh=0.05)

        for (x1, y1, x2, y2) in final_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Detection finished. Output saved as {output_file}")

# コマンドライン引数から入力ファイルを取得
if len(sys.argv) != 2:
    print("Usage: python detect_video.py <input_video.mp4>")
    sys.exit(1)

input_file = sys.argv[1]

# model_name = "hog_svm_model_notebook2.pkl"
model_name = "svm_model.pkl"
# 保存されたモデルをロード
model = joblib.load(model_name)

output_file = os.path.splitext(input_file)[0] + f"_output_{model_name}.mp4"

# ビデオを処理して出力
process_video(input_file, output_file, model)

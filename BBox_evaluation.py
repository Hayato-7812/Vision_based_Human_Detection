import json
import numpy as np

def load_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        coco = json.load(f)
    annotations = coco['annotations']
    images = {img['id']: img for img in coco['images']}
    return annotations, images

def load_detection_results(json_path):
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def evaluate_detections(detections, ground_truths, iou_threshold=0.3):
    total_iou = 0
    count_iou = 0
    false_positives = 0

    matched_detections = []

    for gt in ground_truths:
        gt_box = gt['bbox']
        gt_box = [gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]
        
        max_iou = 0
        best_det = None
        for det in detections:
            det_box = det['bbox']
            iou = compute_iou(gt_box, det_box)
            
            if iou > max_iou:
                max_iou = iou
                best_det = det
        
        if max_iou >= iou_threshold:
            total_iou += max_iou
            count_iou += 1
            matched_detections.append(best_det)
    
    # 人を認識できたものに限定した場合の精度の算出
    precision_iou = total_iou / count_iou if count_iou > 0 else 0
    
    
    # 全ての検出結果のうち存在しない物体を誤って検出した割合 
    false_positive_rate = 1.0 - (len(matched_detections)/len(detections)) if len(detections) > 0 else 0
    
    return precision_iou, false_positive_rate

# COCOアノテーションファイルと検出結果のパスを指定
coco_annotations_path = '/Users/shimizuhayato/Desktop/Waseda/M1_spring/コンピュータービジョン/Sitting_Prediction/coco/annotations/instances_val2017.json'
detection_results_path = 'outputs/detection_results.json'

# アノテーションと検出結果をロード
annotations, images = load_coco_annotations(coco_annotations_path)
detection_results = load_detection_results(detection_results_path)

# 画像ごとに評価を実行
image_id_to_annotations = {}
for ann in annotations:
    image_id = ann['image_id']
    if image_id not in image_id_to_annotations:
        image_id_to_annotations[image_id] = []
    image_id_to_annotations[image_id].append(ann)

all_precision_ious = []
all_false_positive_rates = []

for result in detection_results:
    image_name = result['image']
    image_info = next((img for img in images.values() if img['file_name'] == image_name), None)
    if image_info is None:
        print(f"Warning: No image info found for {image_name}")
        continue

    image_id = image_info['id']
    detections = result['detections']
    ground_truths = image_id_to_annotations.get(image_id, [])
    
    precision_iou, false_positive_rate = evaluate_detections(detections, ground_truths)
    
    all_precision_ious.append(precision_iou)
    all_false_positive_rates.append(false_positive_rate)

# 総合的な評価結果のサマリーを出力
mean_precision_iou = np.mean(all_precision_ious)
mean_false_positive_rate = np.mean(all_false_positive_rates)

print("\nSummary of Evaluation Results:")
print("==============================")
print(f"Mean Precision IoU: {mean_precision_iou}")
print(f"Mean False Positive Rate: {mean_false_positive_rate}")

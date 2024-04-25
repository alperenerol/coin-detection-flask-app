from roboflow import Roboflow
import supervision as sv
import numpy as np
import json
from utils import iou

rf = Roboflow(api_key="fGmdOlbOib3mVhdoJluB")
project = rf.workspace().project("coin-detection-for-new-dataset")
model = project.version(2).model

# Evaluation Loop mean IoU calculation
# Open the JSON file and load its content
gt_json_file_path = "coin-dataset/combined_gt_annotations_xywh_xyxy.json"
with open(gt_json_file_path, 'r') as file:
    gt_json_data = json.load(file)  # Using json.load to read file directly

# Okay
file_name_list = []
gt_images = gt_json_data["gt_images"]
for image_item in gt_images:
    file_name = image_item["file_name"]
    file_name_list.append(file_name)

mean_iou = []
for i in range(len(file_name_list)):
    img_path = file_name_list[i]
    image_objs = []
    image_iou = []
    result = model.predict(img_path, confidence=40, overlap=30).json()
    detections = sv.Detections.from_inference(result).xyxy
    for idx, pred in enumerate(detections):
        # Convert bbox format (x, y, width, height) to (x1, y1, x2, y2)
        pred_bbox_xyxy = [pred[0], pred[1], pred[2], pred[3]] # imajdaki objeleri aldik xyxy cinsinden okey, sortlasin, digerini de sortlasin, skorlasin, kenara yazsin
        image_objs.append(pred_bbox_xyxy)
    
    image_objs.sort(key=lambda x: x[0])
   
    # simdi skorlama kismi, ilgili imaji bul pathe gore, objeleri sec sortla, iki obje listesni al skorla, yaz kenara
    gt_bbox_objs = []
    gt_objects = gt_images[i]['objects']
    for object in gt_objects:
        gt_bbox_xyxy = object["bbox_xyxy"]
        gt_bbox_objs.append(gt_bbox_xyxy)
    gt_bbox_objs.sort(key=lambda x: x[0])
    

    # skorla
    # Calculate IoU
    for k in range(len(gt_bbox_objs)):
        try:
            object_iou = iou(image_objs[k], gt_bbox_objs[k])
            image_iou.append(object_iou)
        except:
            image_iou.append(0)
            continue
    
    image_iou = np.mean(image_iou)
    print(f"Image {i} IoU Score: {image_iou}")
    mean_iou.append(image_iou)
    
mean_iou = np.mean(mean_iou)
print(f"Model IoU Score: {mean_iou}")


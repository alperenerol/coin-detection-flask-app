from roboflow import Roboflow
import supervision as sv
import cv2
from utils import radii_mm
import argparse

rf = Roboflow(api_key="fGmdOlbOib3mVhdoJluB")
project = rf.workspace().project("coin-detection-for-new-dataset")
model = project.version(2).model

parser = argparse.ArgumentParser(description="Process an image with a machine learning model.")
parser.add_argument('image_path', type=str, help="Path to the image file")
args = parser.parse_args()

# Upload an image
# img_path = 'coin-dataset/175_1479423456_jpg.rf.0723ceef6a241da65f4f36db2132002b.jpg'
img_path = args.image_path

result = model.predict(img_path, confidence=40, overlap=30).json()
predictions = result["predictions"] 

labels = [f"object_{idx+1}" for idx, item in enumerate(predictions)]

# Calculate centroids and radii
centroids = [(int(item['x']), int(item['y'])) for item in predictions]
radii = [max(item['width'], item['height']) // 2 for item in predictions]
radii = radii_mm(radii)

bbox = [(int(item['x']-item['width']/2), int(item['y']-item['height']/2), int(item['width']), int(item['height'])) for item in predictions]

# Output results
for idx, label in enumerate(labels):
    print(f"{label}: Centroid at {centroids[idx]}, Radius {radii[idx]}, Bbox {bbox[idx]}")

detections = sv.Detections.from_inference(result)

# Annotate predictions
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoundingBoxAnnotator()

image = cv2.imread(img_path)

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(image=annotated_image, size=(8, 8))
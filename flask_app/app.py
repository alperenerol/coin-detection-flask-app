from roboflow import Roboflow
import supervision as sv
import cv2
from utils import radii_mm
from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import uuid

rf = Roboflow(api_key="fGmdOlbOib3mVhdoJluB")
project = rf.workspace().project("coin-detection-for-new-dataset")
model = project.version(2).model

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploaded_images'
GENERATE_FOLDER = 'masked_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATE_FOLDER'] = GENERATE_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATE_FOLDER, exist_ok=True)


def allowed_file(filename): # Returns True or False
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully", "filename": 'uploaded_images/' + filename}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/retrieve', methods=['GET'])
def retrieve_image():
    filename = request.form.get('filename')
    if filename is None or filename.strip() == "":
        return jsonify({"error": "Filename is required"}), 400
    filename = secure_filename(filename)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        result = model.predict(file_path, confidence=40, overlap=30).json()
        predictions = result["predictions"] 

        # Generate the objects list with bounding box and label for each object
        objects = [
            {
                "id": f"object_{idx+1}",
                "bbox": (int(item['x']-item['width']/2), int(item['y']-item['height']/2), int(item['width']), int(item['height']))
            }
            for idx, item in enumerate(predictions)
        ]
        # Encapsulate in a parent dictionary if needed
        result_dict = {"objects": objects}

        return jsonify(result_dict)
    else:
        return jsonify({"error": "File not found"}), 404


@app.route('/details', methods=['GET'])
def object_details():
    filename = request.form.get('filename')
    object_id = request.form.get('object_id')
    if filename is None or filename.strip() == "":
        return jsonify({"error": "Filename is required"}), 400
    filename = secure_filename(filename)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        result = model.predict(file_path, confidence=40, overlap=30).json()
        predictions = result["predictions"] 

        labels = [f"object_{idx+1}" for idx, item in enumerate(predictions)]
        # Calculate centroids and radii
        centroids = [(int(item['x']), int(item['y'])) for item in predictions]
      
        radii = [max(item['width'], item['height']) // 2 for item in predictions]
        radii = radii_mm(radii)
     
        bbox = [(int(item['x']-item['width']/2), int(item['y']-item['height']/2), int(item['width']), int(item['height'])) for item in predictions]
        
        # Output results
        for idx, label in enumerate(labels):
            if label == object_id:
                centroids =centroids[idx]
                radius = radii[idx]
                bbox = bbox[idx]
                break
            else:
                continue
        
        # Generate the objects list with bounding box and label for each object
        objects = [
            {
                "id": object_id,
                "bbox": bbox,
                "centroid": centroids,
                "radius": radius
            }
        ]
        # Encapsulate in a parent dictionary if needed
        result_dict = {"objects": objects}

        return jsonify(result_dict)
    
    else:
        return jsonify({"error": "File not found"}), 404
    

@app.route('/save', methods=['POST'])
def save_image():
    filename = request.form.get('filename')
    if filename is None or filename.strip() == "":
        return jsonify({"error": "Filename is required"}), 400
    filename = secure_filename(filename)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        result = model.predict(file_path, confidence=40, overlap=30).json()
        predictions = result["predictions"] 

        labels = [f"object_{idx+1}" for idx, item in enumerate(predictions)]

        detections = sv.Detections.from_inference(result)

        # Annotate predictions
        label_annotator = sv.LabelAnnotator()
        bounding_box_annotator = sv.BoundingBoxAnnotator()

        image = cv2.imread(file_path)

        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)
        
        # Save annotated image to a temporary file
        temp_filename = str(uuid.uuid4()) + '.png'
        temp_filepath = os.path.join(app.config['GENERATE_FOLDER'], temp_filename)
        cv2.imwrite(temp_filepath, annotated_image)

        return send_from_directory(app.config['GENERATE_FOLDER'], temp_filename)
    
    else:
        return jsonify({"error": "File not found"}), 404

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

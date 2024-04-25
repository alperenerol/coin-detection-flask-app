import json

# Open the JSON file and load its content
json_file_path = "coin-dataset/_annotations.coco.json"
with open(json_file_path, 'r') as file:
    json_data = json.load(file)  # Using json.load to read file directly

# Convert strings to JSON objects
images = json_data["images"]
annotations = json_data["annotations"]

# Dictionary to hold the combined results
combined = {"gt_images": []}

# Map each image with its annotations
for image in images:
    # Prepare a dictionary for each image including its annotations
    image_info = {
        "image_id": image["id"],
        "file_name": "coin-dataset/" + image["file_name"],
        "objects": []
    }
    
    # Loop through annotations to find those that match the current image_id
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        x1, y1 = x, y
        x2, y2 = x + width, y + height
        annotation["bbox_xyxy"] = [x1, y1, x2, y2]
        if annotation["image_id"] == image["id"]:
            object_info = {
                "object_id": annotation["id"],
                "bbox": annotation["bbox"],
                "bbox_xyxy": annotation["bbox_xyxy"]
            }
            image_info["objects"].append(object_info)
    
    # Add the prepared image information to the combined list
    combined["gt_images"].append(image_info)

# Convert the combined dictionary to JSON string
combined_json = json.dumps(combined, indent=4)

# Print the JSON or write it to a file
# print(combined_json)
# Define the output file path
output_file_path = "coin-dataset/combined_gt_annotations_xywh_xyxy.json"

# Save the combined JSON to a file
with open(output_file_path, 'w') as output_file:
    json.dump(combined, output_file, indent=4)

print(f"Combined JSON has been saved to {output_file_path}")
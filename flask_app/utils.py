import numpy as np

def radii_mm(radii):
    # Calculate the current average of the radii in an image
    target_average = 11.81 # average size of coins of the pound sterling
    # Calculate the scale factor needed to adjust the average to the target average
    radii_average = sum(radii) / len(radii)
    scale_factor = target_average / radii_average
    # Apply the scale factor to each radius to get the calibrated radii in millimeters
    radii_in_milimeters = [round(radius * scale_factor, 2) for radius in radii]
    
    return radii_in_milimeters

def convert_bboxes_to_xywh(bboxes):
    """
    Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height].
    
    Parameters:
    - bboxes (numpy.array): Array of bounding boxes in [x1, y1, x2, y2] format.
    
    Returns:
    - numpy.array: Array of bounding boxes in [x, y, width, height] format.
    """
    # Ensure the input is a numpy array
    bboxes = np.array(bboxes)
    # Calculate x, y, width, and height
    x = bboxes[:, 0]
    y = bboxes[:, 1]
    width = bboxes[:, 2] - bboxes[:, 0]
    height = bboxes[:, 3] - bboxes[:, 1]
    return np.column_stack((x, y, width, height))

def iou(boxA, boxB):
    """Compute the Intersection over Union (IoU) between two bounding box arrays."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the union area
    unionArea = boxAArea + boxBArea - interArea

    # Compute the IoU
    iou = interArea / unionArea
    return iou
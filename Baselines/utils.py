import cv2
from PIL import Image, ImageFont
import numpy as np
import json

def visualize_bounding_boxes(image, xyxy, phrases, confidences):
    """
    Visualize bounding boxes on an image with phrases and confidence scores.

    Args:
        image (PIL.Image.Image): The input image.
        xyxy (numpy.ndarray): Bounding boxes in xyxy format (x_min, y_min, x_max, y_max).
        phrases (list): List of phrases corresponding to each bounding box.
        confidences (list): List of confidence scores corresponding to each bounding box.

    Returns:
        PIL.Image.Image: The image with bounding boxes, phrases, and confidence scores drawn.
    """
    # Convert PIL image to NumPy array (OpenCV format)
    image_np = np.array(image)

    for box, phrase, confidence in zip(xyxy, phrases, confidences):
        x_min, y_min, x_max, y_max = map(int, box)
        # Draw the bounding box
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Prepare the label with phrase and confidence
        label = f"{phrase} ({confidence:.2f})"
        # Put the label text
        cv2.putText(image_np, label, (x_min, y_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert NumPy array back to PIL image
    return Image.fromarray(image_np)

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    while lines[-1] == '':
        lines = lines[:-1]
    return lines

def save_to_json(filepath, dict):
    if not filepath.endswith('.json'):
        raise RuntimeError("filepath must end with .json")

    with open(filepath, 'w') as f:
        json.dump(dict, f)
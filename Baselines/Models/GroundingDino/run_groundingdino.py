# Use env grdino on pda 
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from PIL import Image 
from typing import Literal

model_weight_path = "/home/ksmehrab/GroundingModels/GroundingDINO/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
model_config_path = "/home/ksmehrab/GroundingModels/GroundingDINO/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"

def load_grdino_model():
    model = load_model(model_config_path, model_weight_path)
    model.to(device="cuda")
    return model

# Write code to run grdino on image and phrase 
# In another file, loop through the pascalpart val files and run it through grdino 

def run_groundingdino(
        model,
        image_path,
        text_prompt,
        box_threshold,
        text_threshold
):
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    xyxy, _ = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    return xyxy.tolist(), logits.tolist(), phrases

def filter_boxes_with_phrase_check(
        xyxy_boxes,
        conf_scores,
        pred_phrases,
        phrase_to_check,
        choose_boxes: Literal["all", "best"],
        conf_threshold=0.25
):
    filtered_boxes = []
    filtered_conf_scores = []
    filtered_phrases = []
    for box, conf_score, pred_phrase in zip(xyxy_boxes, conf_scores, pred_phrases):
        if conf_score < conf_threshold:
            continue
        """
        The following piece of code assumes that the main "thing" we want to detect would be the last word of the phrase_to_check.
        For example, if we want to check "left ear" the main thing to detect would be the ear,
        so any predicted phrase with "ear is relevant"
        """
        if len(phrase_to_check.split()) > 1:
            word_to_check = phrase_to_check.split()[-1]
        else:
            word_to_check = phrase_to_check
        if word_to_check not in pred_phrase.split():
            continue
        filtered_boxes.append(box)
        filtered_conf_scores.append(conf_score)
        filtered_phrases.append(pred_phrase)

    if len(filtered_boxes) > 0 and choose_boxes == 'best':
        best_idx = filtered_conf_scores.index(max(filtered_conf_scores))
        return [filtered_boxes[best_idx]], [filtered_conf_scores[best_idx]], [filtered_phrases[best_idx]]
    
    return filtered_boxes, filtered_conf_scores, filtered_phrases

def filter_boxes(
        xyxy_boxes,
        conf_scores,
        pred_phrases,
        choose_boxes: Literal["all", "best"],
        conf_threshold=0.25
):
    filtered_boxes = []
    filtered_conf_scores = []
    filtered_phrases = []
    for box, conf_score, pred_phrase in zip(xyxy_boxes, conf_scores, pred_phrases):
        if conf_score < conf_threshold:
            continue
        filtered_boxes.append(box)
        filtered_conf_scores.append(conf_score)
        filtered_phrases.append(pred_phrase)

    if len(filtered_boxes) > 0 and choose_boxes == 'best':
        best_idx = filtered_conf_scores.index(max(filtered_conf_scores))
        return [filtered_boxes[best_idx]], [filtered_conf_scores[best_idx]], [filtered_phrases[best_idx]]
    
    return filtered_boxes, filtered_conf_scores, filtered_phrases

def remove_overlapping_boxes(xyxy_boxes, intersection_threshold=0.9):
    """
    Remove overlapping boxes based on intersection area threshold.
    If intersection area is over threshold% of a box's area, remove that box.
    Keeps larger boxes when overlap is detected.
    
    Args:
        xyxy_boxes: List of boxes in [x1, y1, x2, y2] format
        intersection_threshold: Threshold (0-1) for intersection ratio
    
    Returns:
        List of non-overlapping boxes
    """
    if len(xyxy_boxes) == 0:
        return []
    
    # Calculate area for each box
    areas = []
    for box in xyxy_boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)
    
    # Sort boxes by area (descending) - keep larger boxes
    sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
    
    keep_indices = []
    
    for i in sorted_indices:
        should_keep = True
        box_i = xyxy_boxes[i]
        area_i = areas[i]
        
        # Check against all boxes we've decided to keep
        for j in keep_indices:
            box_j = xyxy_boxes[j]
            
            # Calculate intersection area
            inter_area = calculate_intersection_area(box_i, box_j)
            
            # Check if intersection is over threshold% of box_i's area
            if inter_area / area_i > intersection_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep_indices.append(i)
    
    # Return filtered results in original order
    keep_indices.sort()
    filtered_boxes = [xyxy_boxes[i] for i in keep_indices]
    
    return filtered_boxes


def calculate_intersection_area(box1, box2):
    """
    Calculate intersection area between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        Intersection area
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Calculate intersection area
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    return inter_area
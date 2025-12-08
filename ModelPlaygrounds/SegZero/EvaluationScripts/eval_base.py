import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/GitRepoLatest/Seg-Zero')

import json
import re

import numpy as np
import matplotlib.pyplot as plt

def extract_bbox_points_think(output_text, x_factor, y_factor):
    """
        Extract bounding boxes, points, and think text from model output.
        Parses original segzero output format 
    """
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(1))
        pred_bboxes = [[
            int(item['bbox_2d'][0] * x_factor + 0.5),
            int(item['bbox_2d'][1] * y_factor + 0.5),
            int(item['bbox_2d'][2] * x_factor + 0.5),
            int(item['bbox_2d'][3] * y_factor + 0.5)
        ] for item in data]
        pred_points = [[
            int(item['point_2d'][0] * x_factor + 0.5),
            int(item['point_2d'][1] * y_factor + 0.5)
        ] for item in data]
    
    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    think_text = ""
    if think_match:
        think_text = think_match.group(1)
    
    return pred_bboxes, pred_points, think_text

def extract_information_vrpart(output_text, x_factor, y_factor):
    """
        extract bounding boxes, points, and various text fields from model output.
        parses VRPart1 output format.
        <think> ... </think> <decide> ... </decide> <first_answer> ... </first_answer> <criticism> ... </criticism> <final_answer> ... </final_answer>
    """
    # Extract think tag
    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    think_text = think_match.group(1).strip() if think_match else ""
    
    # Extract decide tag
    decide_pattern = r'<decide>([^<]+)</decide>'
    decide_match = re.search(decide_pattern, output_text)
    decide_text = decide_match.group(1).strip() if decide_match else ""
    
    # Extract first_answer tag
    first_answer_pattern = r'<first_answer>\s*(.*?)\s*</first_answer>'
    first_answer_match = re.search(first_answer_pattern, output_text, re.DOTALL)
    first_answer_text = first_answer_match.group(1).strip() if first_answer_match else ""
    
    # Extract criticism tag
    criticism_pattern = r'<criticism>([^<]+)</criticism>'
    criticism_match = re.search(criticism_pattern, output_text)
    criticism_text = criticism_match.group(1).strip() if criticism_match else ""
    
    # Extract final_answer and parse bbox/points
    final_answer_pattern = r'<final_answer>\s*(.*?)\s*</final_answer>'
    final_answer_match = re.search(final_answer_pattern, output_text, re.DOTALL) 
    final_answer_text = final_answer_match.group(1).strip() if final_answer_match else ""

    output_text_parsed = {
        "think": think_text,
        "decide": decide_text,
        "first_answer": first_answer_text,
        "criticism": criticism_text,
        "final_answer": final_answer_text
    }
    
    pred_bboxes = []
    pred_points = []
    
    if final_answer_match:
        data = json.loads(final_answer_match.group(1))
        pred_bboxes = [[
            int(item['bbox_2d'][0] * x_factor + 0.5),
            int(item['bbox_2d'][1] * y_factor + 0.5),
            int(item['bbox_2d'][2] * x_factor + 0.5),
            int(item['bbox_2d'][3] * y_factor + 0.5)
        ] for item in data]
        pred_points = [[
            int(item['point_2d'][0] * x_factor + 0.5),
            int(item['point_2d'][1] * y_factor + 0.5)
        ] for item in data]

    return pred_bboxes, pred_points, output_text_parsed

def extract_information_vrpart2(output_text, x_factor, y_factor):
    """
        extract bounding boxes, points, and various text fields from model output.
        parses VRPart2 output format.
        <think> ... </think> <target> ... </target> <first_answer> ... </first_answer> <criticism> ... </criticism> <answer> ... </answer>
        this returns empty strings and lists if it does not find the expected tags in the output.
    """
    # Extract think tag
    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    think_text = think_match.group(1).strip() if think_match else ""
    
    # Extract decide tag
    decide_pattern = r'<target>([^<]+)</target>'
    decide_match = re.search(decide_pattern, output_text)
    decide_text = decide_match.group(1).strip() if decide_match else ""
    
    # Extract first_answer tag
    first_answer_pattern = r'<first_answer>\s*(.*?)\s*</first_answer>'
    first_answer_match = re.search(first_answer_pattern, output_text, re.DOTALL)
    first_answer_text = first_answer_match.group(1).strip() if first_answer_match else ""
    
    # Extract criticism tag
    criticism_pattern = r'<criticism>([^<]+)</criticism>'
    criticism_match = re.search(criticism_pattern, output_text)
    criticism_text = criticism_match.group(1).strip() if criticism_match else ""
    
    # Extract final_answer and parse bbox/points
    final_answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    final_answer_match = re.search(final_answer_pattern, output_text, re.DOTALL) 
    final_answer_text = final_answer_match.group(1).strip() if final_answer_match else ""

    output_text_parsed = {
        "think": think_text,
        "decide": decide_text,
        "first_answer": first_answer_text,
        "criticism": criticism_text,
        "final_answer": final_answer_text
    }
    
    pred_bboxes = []
    pred_points = []
    
    if final_answer_match:
        data = json.loads(final_answer_match.group(1))
        pred_bboxes = [[
            int(item['bbox_2d'][0] * x_factor + 0.5),
            int(item['bbox_2d'][1] * y_factor + 0.5),
            int(item['bbox_2d'][2] * x_factor + 0.5),
            int(item['bbox_2d'][3] * y_factor + 0.5)
        ] for item in data]
        pred_points = [[
            int(item['point_2d'][0] * x_factor + 0.5),
            int(item['point_2d'][1] * y_factor + 0.5)
        ] for item in data]

    return pred_bboxes, pred_points, output_text_parsed

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0, 0
    return intersection, union

def combine_masks(masks):
    if len(masks) == 0:
        print("Error: No masks to combine in combine_masks function.")
        return None
    combined_mask = masks[0]
    for i in range(1, len(masks)):
        combined_mask = combined_mask | masks[i]
    return combined_mask

# visualization code from chatgpt
def visualize_first_and_final_bbox(image, first_bboxes, final_bboxes, query_text, save_path=None):
    """
    Visualize side-by-side comparison of first_answer and final_answer bounding boxes.
    
    Args:
        image: PIL Image object
        first_bboxes: List of bboxes from first_answer [[x1,y1,x2,y2], ...]
        final_bboxes: List of bboxes from final_answer [[x1,y1,x2,y2], ...]
        query_text: Query text for the image
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    print("debugfigure")
    # First answer boxes
    axes[0].imshow(image)
    axes[0].set_title(f'First Answer Boxes\n{query_text}', fontsize=14, fontweight='bold')
    for idx, box in enumerate(first_bboxes):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=3, edgecolor='blue', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, y1-5, f"First_{idx}", color='blue', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[0].axis('off')
    axes[0].text(0.5, -0.05, f'{len(first_bboxes)} boxes', 
                transform=axes[0].transAxes, ha='center', fontsize=12)
    
    # Final answer boxes
    axes[1].imshow(image)
    axes[1].set_title(f'Final Answer Boxes\n{query_text}', fontsize=14, fontweight='bold')
    for idx, box in enumerate(final_bboxes):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=3, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x1, y1-5, f"Final_{idx}", color='red', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1].axis('off')
    axes[1].text(0.5, -0.05, f'{len(final_bboxes)} boxes', 
                transform=axes[1].transAxes, ha='center', fontsize=12)
    
    plt.tight_layout()
    # plt.show()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    # print(f"Saved box comparison to {save_path}")

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

        if area_i == 0:
            continue
        
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

def boxes_intersect(box1, box2):
    """
    Check if two boxes intersect (have any overlap).
    
    Args:
        box1, box2: Bounding boxes in format [x_min, y_min, x_max, y_max]
    
    Returns:
        True if boxes intersect, False otherwise
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    
    # Boxes intersect if they overlap in both x and y dimensions
    x_overlap = x_min1 < x_max2 and x_max1 > x_min2
    y_overlap = y_min1 < y_max2 and y_max1 > y_min2
    
    return x_overlap and y_overlap


def merge_intersecting_boxes(bboxes):
    """
    Merge intersecting bounding boxes into groups and create merged boxes.
    
    Args:
        bboxes: List of bounding boxes in format [x_min, y_min, x_max, y_max]
    
    Returns:
        List of merged bounding boxes
    """
    if not bboxes:
        return []
    
    n = len(bboxes)
    # Track which group each box belongs to (-1 means not assigned yet)
    box_to_group = [-1] * n
    groups = {}
    group_id = 0
    
    # Assign boxes to groups
    for i in range(n):
        if box_to_group[i] == -1:
            # Start a new group
            groups[group_id] = [i]
            box_to_group[i] = group_id
            
            # Find all boxes that intersect with any box in this group
            changed = True
            while changed:
                changed = False
                for j in range(n):
                    if box_to_group[j] == -1:
                        # Check if box j intersects with any box in current group
                        for box_idx in groups[group_id]:
                            if boxes_intersect(bboxes[j], bboxes[box_idx]):
                                groups[group_id].append(j)
                                box_to_group[j] = group_id
                                changed = True
                                break
            
            group_id += 1
    
    # Merge boxes in each group
    merged_boxes = []
    for group_indices in groups.values():
        # Get all boxes in this group
        group_boxes = [bboxes[i] for i in group_indices]
        
        # Find min x_min, y_min and max x_max, y_max
        x_min = min(box[0] for box in group_boxes)
        y_min = min(box[1] for box in group_boxes)
        x_max = max(box[2] for box in group_boxes)
        y_max = max(box[3] for box in group_boxes)
        
        merged_boxes.append([x_min, y_min, x_max, y_max])
    
    return merged_boxes
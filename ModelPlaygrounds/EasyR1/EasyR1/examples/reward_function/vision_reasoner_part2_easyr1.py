import re
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Any

# Utility functions for geometry calculations (same as in original code)
def batch_iou(boxes1, boxes2):
    """Compute IoU between each box in boxes1 and each box in boxes2."""
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # (M,1) each
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # (N,1)
    # Intersection coords
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    # Areas of boxes
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)
    # Union
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / np.clip(unionArea, a_min=1e-9, a_max=None)  # avoid division by zero
    return iou

def batch_l1_distance(boxes1, boxes2):
    """Compute mean L1 distance between each predicted box and each ground truth box."""
    # L1 distance on bounding box coordinates (average absolute difference per coordinate)
    boxes1 = boxes1[:, np.newaxis, :]  # (M,1,4)
    boxes2 = boxes2[np.newaxis, :, :]  # (1,N,4)
    return np.mean(np.abs(boxes1 - boxes2), axis=2)  # (M,N)

def batch_points_distance(points1, points2):
    """Compute Euclidean distance between each predicted point and each ground truth point."""
    points1 = points1[:, np.newaxis, :]  # (M,1,2)
    points2 = points2[np.newaxis, :, :]  # (1,N,2)
    dist = np.sqrt(np.sum((points1 - points2) ** 2, axis=2))
    return dist  # (M,N)

def batch_points_in_box(points, boxes):
    """Check if each point lies inside the corresponding box (by index)."""
    # points: (M,2), boxes: (M,4)
    if len(points) == 0 or len(boxes) == 0:
        return np.array([], dtype=bool)
    try:
        x_check = (points[:, 0] >= boxes[:, 0]) & (points[:, 0] <= boxes[:, 2])
        y_check = (points[:, 1] >= boxes[:, 1]) & (points[:, 1] <= boxes[:, 3])
        return x_check & y_check  # (M,) boolean array
    except Exception as e:
        print("Error in batch_points_in_box:", e)
        return np.array([], dtype=bool)

# Reward components:

# modified from part 1 to handle revised output format 
def vision_reasoner_format_reward(predict_str: str) -> float:
    """Reward for producing the correct output format (reasoning and answer tags with JSON)."""
    # Updated pattern for new format
    pattern = (
        r"^<think>.*?</think>\s*"
        r"<target>(object|part)</target>\s*"
        r"<object_hint>\s*(\[.*?\])\s*</object_hint>\s*"
        r"<first_answer>\s*\[.*?\]\s*</first_answer>\s*"
        r"<criticism>.*?ADJUSTMENT:\s*(YES|NO)\s*</criticism>\s*"
        r"<answer>\s*\[.*?\]\s*</answer>$"
    )
    format_correct = 1.0 if re.fullmatch(pattern, predict_str.strip(), re.DOTALL | re.IGNORECASE) else 0.0
    # print(f"Format correct: {format_correct}")
    # Check JSON content format for object_hint, first_answer and answer
    content_reward = 0.0
    try:
        # Extract <target> to check if it's "part"
        target_match = re.search(r'<target>\s*(object|part)\s*</target>', predict_str, re.IGNORECASE)
        is_part = target_match and target_match.group(1).lower() == 'part' if target_match else False
        
        # Extract and validate object_hint
        hint_match = re.search(r'<object_hint>\s*(\[.*?\])\s*</object_hint>', predict_str, re.DOTALL)
        if hint_match:
            hint_data = json.loads(hint_match.group(1))
            if isinstance(hint_data, list):
                # If target is "part", object_hint should have objects; if "object", should be empty
                if is_part and len(hint_data) > 0:
                    # Validate structure of hint objects
                    per_item_scores = []
                    for item in hint_data:
                        score = 0.0
                        if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                            score += 0.5
                        if 'point_2d' in item and isinstance(item['point_2d'], list) and len(item['point_2d']) == 2:
                            score += 0.5
                        per_item_scores.append(score)
                    if per_item_scores:
                        content_reward += (sum(per_item_scores) / len(per_item_scores)) * 0.5
                        # print(f"Object hint score for parts: {(sum(per_item_scores) / len(per_item_scores)) * 0.5}")
                elif not is_part and len(hint_data) == 0:
                    content_reward += 0.5  # Correct: empty list for object target
                elif is_part and len(hint_data) == 0:
                    content_reward += 0.0  # Incorrect: should have hints for parts
                elif not is_part and len(hint_data) > 0:
                    content_reward += 0.0  # Incorrect: shouldn't have hints for objects

        # Extract and validate first_answer
        first_match = re.search(r'<first_answer>\s*(\[.*?\])\s*</first_answer>', predict_str, re.DOTALL)
        if first_match:
            first_data = json.loads(first_match.group(1))
            if isinstance(first_data, list) and len(first_data) > 0:
                per_item_scores = []
                for item in first_data:
                    score = 0.0
                    if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                        score += 0.5
                    if 'point_2d' in item and isinstance(item['point_2d'], list) and len(item['point_2d']) == 2:
                        score += 0.5
                    per_item_scores.append(score)
                if per_item_scores:
                    content_reward += (sum(per_item_scores) / len(per_item_scores)) * 0.75
                    # print(f"First answer score: {(sum(per_item_scores) / len(per_item_scores)) * 0.75}")

        # Extract and validate answer (final answer)
        final_match = re.search(r'<answer>\s*(\[.*?\])\s*</answer>', predict_str, re.DOTALL)
        if final_match:
            final_data = json.loads(final_match.group(1))
            if isinstance(final_data, list) and len(final_data) > 0:
                per_item_scores = []
                for item in final_data:
                    score = 0.0
                    if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                        score += 0.5
                    if 'point_2d' in item and isinstance(item['point_2d'], list) and len(item['point_2d']) == 2:
                        score += 0.5
                    per_item_scores.append(score)
                if per_item_scores:
                    content_reward += (sum(per_item_scores) / len(per_item_scores)) * 0.75
                    # print(f"Final answer score: {(sum(per_item_scores) / len(per_item_scores)) * 0.75}")
        
        # Check for ADJUSTMENT flag in criticism
        criticism_match = re.search(r'<criticism>.*?ADJUSTMENT:\s*(YES|NO)', predict_str, re.DOTALL | re.IGNORECASE)
        if criticism_match:
            content_reward += 0.5  # Bonus for including the adjustment flag
            # print("Criticism adjustment flag found, adding 0.5 to content reward.")

    except Exception as e:
        # JSON parsing failure -> format content likely incorrect
        content_reward = 0.0
        print("Caught error in format reward calculation:", e)

    # Total: format_correct (1.0) + content_reward (up to ~3.0)
    return format_correct + content_reward

def vision_reasoner_decision_reward(predict_str: str, ground_truth_type: bool = None) -> float:
    """R1: Reward if the model correctly decides 'object' vs 'part'."""
    # ground_truth_type: True if query is asking for a part, False if asking for a whole object.
    try:
        target_match = re.search(r'<target>\s*(.*?)\s*</target>', predict_str, re.IGNORECASE)
        if not target_match:
            return 0.0

        target_text = target_match.group(1).strip().lower()
        # Determine model's decision: assume it mentions "part" if it's an object part query.
        predicted_is_part = "part" in target_text  # (this works if output contains "object part" or similar phrasing)
        actual_is_part = bool(ground_truth_type)
        return 1.0 if predicted_is_part == actual_is_part else 0.0
    except Exception as e:
        print("Error in vision_reasoner_decision_reward:", e)
        return 0.0

def vision_reasoner_initial_reward(predict_str: str, gt_bboxes: np.ndarray) -> float:
    """R2: Although it's called initial reward, this method was modified to only compute the initial average IoU, which is used by itself for reward computation later, and is used to compute the improvement reward."""
    try:
        match = re.search(r'<first_answer>\s*(.*?)\s*</first_answer>', predict_str, re.DOTALL)
        if not match:
            return 0.0
        pred_data = json.loads(match.group(1))
        if not isinstance(pred_data, list):
            return 0.0
        pred_bboxes = np.array([item['bbox_2d'] for item in pred_data], dtype=np.float64) if len(pred_data) > 0 else np.empty((0,4))
        # If no prediction or no ground truth, handle edge cases
        M, N = len(pred_bboxes), len(gt_bboxes)
        if M == 0 or N == 0:
            # if both empty, full reward
            if M == 0 and N == 0:
                return 1.0  # no objects quried and none predicted, so its correct.
            return 0.0  # no initial prediction or no target, no iou
        
        iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # (M,N)
        
        # l1 might be unused for the intitial reward, we'll see
        l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)

        cost_matrix = 1.0 - iou_matrix  # hungarian match on iou
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
        # Computer average IoU over matched pairs 
        initial_avg_iou = 0.0
        if len(row_ind) > 0:
            initial_ious = [iou_matrix[i, j] for i, j in zip(row_ind, col_ind)]
            initial_avg_iou = np.mean(initial_ious)

        return initial_avg_iou
    except Exception as e:
        print("Caught error in vision_reasoner_initial_reward:", e)
        return 0.0

def vision_reasoner_final_bbox_reward(pred_bboxes: np.ndarray, 
                                    gt_bboxes: np.ndarray,
                                    final_points,
                                    gt_points) -> float:
    """R4 (part 1): Reward for final bounding box localization and count (IoU criteria).

    Args:
        pred_bboxes: np.ndarray of shape (M,4) predicted final bounding boxes
        gt_bboxes: np.ndarray of shape (N,4) ground truth bounding boxes
        final_points: np.ndarray of shape (M,2) predicted final points
        gt_points: np.ndarray of shape (N,2) ground truth points
    """
    # try boxes reward and iou 
    try:
        M, N = len(pred_bboxes), len(gt_bboxes)
        if M == 0 or N == 0:
            # if both empty, full reward
            if M == 0 and N == 0:
                return 1.0, 1.0, 1.0  # no objects asked and none predicted, so its correct.
            # otherwise no reward
            return 0.0, 0.0, 0.0

        iou_matrix = batch_iou(pred_bboxes, gt_bboxes)
        l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)

        # cost matrix and hungarian matching based on iou only
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        total_iou_reward = 0.0
        total_l1_reward = 0.0
        matched_ious = []

        for i, j in zip(row_ind, col_ind):
            iou_val = iou_matrix[i, j]
            l1_val = l1_matrix[i, j]

            if iou_val > 0.3:
                total_iou_reward += iou_val # -> [0,1]

            total_l1_reward += max(0, 1.0 - (l1_val / 10.0))  # scaled -> [0,1]. no reward if l1 > 10

            matched_ious.append(iou_val)

        max_count = max(M, N)
        final_box_reward = (1.5 * total_iou_reward + 0.5 * total_l1_reward) / max_count  # more wieight to iou 

        average_iou = np.mean(matched_ious) if matched_ious else 0.0
        # return float(final_box_reward), average_iou
    except Exception as e:
        print("Error in vision_reasoner_final_bbox_reward:", e)
        # return 0.0, 0.0
        final_box_reward = 0.0
        average_iou = 0.0

    # try points reward
    try:
        point_reward = 0.0
        success_count = 0
        if len(pred_bboxes) > 0 and len(gt_points) > 0 :
            # Ensure predicted points are inside their boxes for matched pairs
            points_inside = batch_points_in_box(final_points, pred_bboxes)
            for i, j in zip(row_ind, col_ind):
                if i < len(final_points) and j < len(gt_points):
                    if not points_inside[i]:
                        continue  # skip if point not in its box
                    dist = np.linalg.norm(final_points[i] - gt_points[j])
                    if dist < 30.0:  # within threshold. setting relatively high threshold to account for the limitations with points 
                        success_count += 1
            point_reward = success_count / max(len(final_points), len(gt_points))
    except Exception as e:
        print("Error in vision_reasoner_final_point_reward:", e)
        point_reward = 0.0
    
    return float(final_box_reward), float(average_iou), float(point_reward)


def vision_reasoner_compactness_reward(pred_bboxes: np.ndarray, gt_bboxes: np.ndarray, ground_truth_type: bool, alpha: float = 1.0) -> float:
    """Additional reward for tight bounding boxes (precision of coverage)."""
    try:
        M, N = len(pred_bboxes), len(gt_bboxes)
        if M == 0 or N == 0:
            return 0.0
        # Compute IoU and intersection areas
        iou_matrix = batch_iou(pred_bboxes, gt_bboxes)
        # Intersection and areas:
        x11, y11, x12, y12 = np.split(pred_bboxes, 4, axis=1)
        x21, y21, x22, y22 = np.split(gt_bboxes, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        inter = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)  # intersection area matrix
        pred_areas = ((x12 - x11 + 1) * (y12 - y11 + 1)).reshape(-1)  # area of each pred box
        gt_areas = ((x22 - x21 + 1) * (y22 - y21 + 1)).reshape(-1)    # area of each gt box
        # Solve assignment maximizing IoU (like before)
        cost = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        precisions = []
        for i, j in zip(row_ind, col_ind):
            if pred_areas[i] <= 0:
                continue
            if iou_matrix[i, j] > 0.5:
                # Only consider pairs that have decent IoU (object correctly identified)
                prec = float(inter[i, j] / pred_areas[i])  # fraction of pred box that intersects ground truth
                # Optionally, one could also check recall: inter/gt_area to avoid tiny boxes. (omitted here or could gate if needed)
                precisions.append(prec)
        if len(precisions) == 0:
            return 0.0
        # Average precision of boxes * alpha
        # TODO: Hard coding alpha = 1.5 for parts to boost part compactness reward
        if ground_truth_type:
            alpha = 1.5
        return alpha * float(np.mean(precisions))
    except Exception as e:
        print("Error in vision_reasoner_compactness_reward:", e)
        return 0.0

def vision_reasoner_non_repeat_reward(predict_str: str) -> float:
    """Reward for non-repetitive reasoning (penalize repeated sentences in chain-of-thought)."""
    non_repeat_reward = 1.0
    try:
        # Split reasoning text into sentences by period.
        sentences = [s.strip() for s in predict_str.split('.') if s.strip()]
        seen = set()
        repeats = 0
        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >= 2:  # allow one repetition maybe, but >=2 repeated sentences -> fail
                non_repeat_reward = 0.0
                break
            seen.add(sentence)
    except Exception as e:
        print("caught error in vision_reasoner_non_repeat_reward:", e)
    return non_repeat_reward

def vision_reasoner_adjustment_consistency_reward(predict_str: str) -> float:
    """
    Penalty for inconsistency between ADJUSTMENT flag and actual changes made.
    - ADJUSTMENT: YES should result in different first_answer vs answer (penalize heavily if same)
    - ADJUSTMENT: NO should result in same first_answer vs answer (penalize lightly if different)

    Returns: reward (penalty) value (negative value to indicate penalty)
    """
    try:
        # Extract ADJUSTMENT flag
        criticism_match = re.search(r'<criticism>.*?ADJUSTMENT:\s*(YES|NO)', predict_str, re.DOTALL | re.IGNORECASE)
        if not criticism_match:
            return 0.0  # No adjustment flag found, no reward/penalty. This will already be rewarded less in the format reward.
        
        adjustment_flag = criticism_match.group(1).strip().upper()
        
        # Extract first_answer and final answer
        first_match = re.search(r'<first_answer>\s*(\[.*?\])\s*</first_answer>', predict_str, re.DOTALL)
        final_match = re.search(r'<answer>\s*(\[.*?\])\s*</answer>', predict_str, re.DOTALL)
        
        if not first_match or not final_match:
            return 0.0  # Missing answers, can't evaluate
        
        first_data = json.loads(first_match.group(1))
        final_data = json.loads(final_match.group(1))
        
        # Check if answers are identical (simple equality check)
        answers_identical = (first_data == final_data)
        
        # Compute reward/penalty based on consistency
        if adjustment_flag == "YES":
            if answers_identical:
                # Strong penalty: model said it would adjust but didn't
                return -2.0
               
        elif adjustment_flag == "NO":
            if not answers_identical:
                return -0.5
        
        return 0.0

    except Exception as e:
        print("Caught error in adjustment consistency reward calculation:", e)
        return 0.0

def vision_reasoner_part_containment_reward(predict_str: str, ground_truth_type: bool = None) -> float:
    """
    Reward for ensuring predicted part bboxes are contained within object_hint bboxes.
    Only applies when target is "part" (ground_truth_type is True).
    
    Constraints:
    - Part bbox must be fully contained within at least one object bbox
    - Part bbox must not be the same as the object bbox
    - Part bbox area must be smaller than the object bbox area
    
    Args:
        predict_str: Full prediction string
        ground_truth_type: True if query is asking for a part, False if asking for a whole object
    
    Returns:
        Containment reward (0.0 to 1.0):
        - 1.0 if all part boxes satisfy containment constraints
        - Proportional score based on how many parts are properly contained
        - 0.0 if not a part query or no proper containment
    """
    try:
        target_match = re.search(r'<target>\s*(object|part)\s*</target>', predict_str, re.IGNORECASE)
        if not target_match or target_match.group(1).strip().lower() != 'part':
            return 0.0
        
        hint_match = re.search(r'<object_hint>\s*(\[.*?\])\s*</object_hint>', predict_str, re.DOTALL)
        if not hint_match:
            return 0.0
        
        hint_data = json.loads(hint_match.group(1))
        if not isinstance(hint_data, list) or len(hint_data) == 0:
            return 0.0  # No object hints provided for part query
        
        object_boxes = np.array([item['bbox_2d'] for item in hint_data], dtype=np.float64)
        
        answer_match = re.search(r'<answer>\s*(\[.*?\])\s*</answer>', predict_str, re.DOTALL)
        if not answer_match:
            return 0.0
        
        answer_data = json.loads(answer_match.group(1))
        if not isinstance(answer_data, list) or len(answer_data) == 0:
            return 0.0
        
        part_boxes = np.array([item['bbox_2d'] for item in answer_data], dtype=np.float64)

        # Check containment: each part box should be fully inside at least one object box
        # with additional constraints: not identical and smaller area
        num_properly_contained = 0
        
        for part_box in part_boxes:
            px1, py1, px2, py2 = part_box
            part_area = (px2 - px1) * (py2 - py1)
            
            # Check if this part is properly contained in any object box
            is_properly_contained = False
            
            for obj_box in object_boxes:
                ox1, oy1, ox2, oy2 = obj_box
                obj_area = (ox2 - ox1) * (oy2 - oy1)
                
                is_spatially_contained = (px1 >= ox1 and py1 >= oy1 and px2 <= ox2 and py2 <= oy2)
                
                if not is_spatially_contained:
                    continue
                
                # Not identical boxes 
                is_identical = (
                    abs(px1 - ox1) < 1e-6 and 
                    abs(py1 - oy1) < 1e-6 and 
                    abs(px2 - ox2) < 1e-6 and 
                    abs(py2 - oy2) < 1e-6
                )
                
                if is_identical:
                    continue
                
                # Part area should be smaller than object area
                if part_area >= obj_area:
                    continue
                
                # All constraints satisfied
                is_properly_contained = True
                break
            
            if is_properly_contained:
                num_properly_contained += 1
        
        # reward = proportion of parts that are properly contained
        containment_reward = num_properly_contained / len(part_boxes)
        return float(containment_reward)
        
    except Exception as e:
        print("Caught error in part containment reward calculation:", e)
        return 0.0

# def compute_score(predict_str: str, ground_truth: str, ground_truth_type: bool = None) -> float:
def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """
    Compute the total reward for a given model output (predict_str) against the ground truth annotations.
    ground_truth: JSON string of list of objects with 'bbox_2d' and 'point_2d'.
    ground_truth_type: Boolean indicating if the query was about an object part (True) or whole object (False).
    """
    print("Reward Function Check: computing vision_reasoner_part2_compute_score...")
    # Set score weights for each component
    weights = {
        'format_reward': 0.5, # uw 3.5
        'decision_reward': 0.25, # uw 1
        'final_bbox_reward': 2.0, # uw 2
        'point_reward': 0.5, # uw 1 
        'non_repeat_reward': 1.0, # uw 1
        'adjustment_consistency_reward': 1.0, # uw -2
        'initial_iou_reward': 1.0, # uw 1
        'improvement_reward': 1.0, # uw max 1 linear 
        'part_containment_reward': 1.0, # uw 1
        'compactness_reward': 0.5, # uw 1 
    }

    scores = []
    for reward_input in reward_inputs:
        predict_str = reward_input['response']
        ground_truth = reward_input['ground_truth']
        ground_truth_type = reward_input.get('ground_truth_type', None)

        # Parse ground truth data
        try:
            gt_data = json.loads(ground_truth)
            # ground truth could be a list of dicts or a dict with keys
            if isinstance(gt_data, dict):
                # ground_truth provided as {"bboxes": [...], "points": [...], "is_part": ...} for example
                if 'is_part' in gt_data:
                    ground_truth_type = bool(gt_data['is_part'])
                if 'objects' in gt_data:
                    # e.g {"objects": [{"bbox_2d": [...], "point_2d": [...]}, ...], ...}
                    gt_list = gt_data['objects']
                elif 'bboxes' in gt_data:
                    # e.g {"bboxes": [...], "points": [...]} separate lists
                    bboxes = gt_data['bboxes']
                    points = gt_data.get('points', [])
                    gt_list = []
                    for bb, pt in zip(bboxes, points):
                        gt_list.append({"bbox_2d": bb, "point_2d": pt})
                else:
                    # If dict but doesn't match expected format, try to interpret it as list
                    gt_list = gt_data
            else:
                # for us, the gt is a list of dicts. this branch should always be taken in our setting 
                gt_list = gt_data  
        except Exception as e:
            print("Error parsing ground truth JSON, assuming empty ground truth:", e)
            gt_list = []

        # Convert ground truth list to numpy arrays for bboxes and points
        # ideally, gt should always have 'bbox_2d' and 'point_2d' keys for each item

        try:
            gt_bboxes = np.array([item['bbox_2d'] for item in gt_list], dtype=np.float64) if len(gt_list) > 0 else np.empty((0,4))
        except Exception as e:
            print("Error extracting ground truth bounding boxes, assuming empty:", e)
            gt_bboxes = np.empty((0,4))
        
        try:
            gt_points = np.array([item['point_2d'] for item in gt_list], dtype=np.float64) if len(gt_list) > 0 else np.empty((0,2))
        except Exception as e:
            print("Error extracting ground truth points, assuming empty:", e)
            gt_points = np.empty((0,2))

        # Compute each reward component
        format_reward = vision_reasoner_format_reward(predict_str)
        decision_reward = vision_reasoner_decision_reward(predict_str, ground_truth_type)
        initial_average_iou = vision_reasoner_initial_reward(predict_str, gt_bboxes)
        # Final predictions parsing
        final_boxes = []
        final_points = []
    
        try:
            final_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
            if final_match:
                final_data = json.loads(final_match.group(1))
                if isinstance(final_data, list):
                    if len(final_data) > 0:
                        # loop through final data and only add to final boxes/points if both bbox and point exist
                        for item in final_data:
                            if 'bbox_2d' in item and 'point_2d' in item:
                                final_boxes.append(np.array(item['bbox_2d'], dtype=np.float64))
                                final_points.append(np.array(item['point_2d'], dtype=np.float64))
        except Exception as e:
            print("Error parsing final predictions into boxes and points:", e)

        final_boxes = np.array(final_boxes, dtype=np.float64) if len(final_boxes) > 0 else np.empty((0,4))
        final_points = np.array(final_points, dtype=np.float64) if len(final_points) > 0 else np.empty((0,2))

        # Use final boxes and ground truth to compute final box and point rewards
        final_bbox_reward = 0.0
        point_reward = 0.0
        if len(gt_bboxes) == 0 and len(final_boxes) == 0:
            # If there were no objects to find and none predicted, consider it a correct scenario.
            final_bbox_reward = 1.0
            point_reward = 1.0  # give full credit to avoid zero-sum
        else:
            # Compute final bounding box reward (IoU/L1-based) and get assignment for point eval
            # We reuse the assignment from IoU matching for point reward to maintain consistency
            final_bbox_reward, final_average_iou, point_reward = vision_reasoner_final_bbox_reward(final_boxes, gt_bboxes, final_points, gt_points)


        # Improvement reward: based on final vs initial bounding box, measured by average iou improvement
        try:
            if initial_average_iou > 0.7:
                # if initial is already very good, no improvement reward necessary
                initial_iou_reward = 1.0
                # check that iou gain is not negative. if it did, penalize
                iou_gain = final_average_iou - initial_average_iou
                if iou_gain < 0.0:
                    improvement_reward = iou_gain * 10.0  # penalize
                else:
                    improvement_reward = 0.0  # no reward if already good
            else:
                iou_gain = final_average_iou - initial_average_iou

                improvement_reward = 0.0
                if iou_gain >= 0.05:
                    improvement_reward = min(1.0, iou_gain * 10.0)
                else:
                    # smaller reward for smaller gain
                    # if iou_gain is negative, penalize
                    improvement_reward = iou_gain * 10.0

                # just set initial iou reward to be the initial average iou. max can be 1. The idea is (improvement reward + initial iou) together reflect the overall reward for the "initial prediction". 
                # Low initial iou + high improvement = good
                # High initial iou + low improvement = also good
                initial_iou_reward = initial_average_iou 
        except Exception as e:
            print("Caught error computing improvement reward:", e)
            initial_iou_reward = 0.0
            improvement_reward = 0.0

        # Compactness reward for final boxes 
        compact_reward = 0.0
        try:
            if len(final_boxes) > 0 and len(gt_bboxes) > 0:
                compact_reward = vision_reasoner_compactness_reward(final_boxes, gt_bboxes, ground_truth_type, alpha=1.0)
        except Exception as e:
            print("Caught error computing compactness reward:", e)
            compact_reward = 0.0

        # adjustment consistency reward/penalty
        adjustment_reward = vision_reasoner_adjustment_consistency_reward(predict_str)

        # part containment reward (only for part queries)
        containment_reward = vision_reasoner_part_containment_reward(predict_str, ground_truth_type)

        non_repeat = vision_reasoner_non_repeat_reward(predict_str)

        # weigh the rewards according to the predefined weights at the start of this method 
        weighed_format_reward = weights['format_reward'] * format_reward
        weighed_decision_reward = weights['decision_reward'] * decision_reward
        weighed_initial_iou_reward = weights['initial_iou_reward'] * initial_iou_reward
        weighed_improvement_reward = weights['improvement_reward'] * improvement_reward
        weighed_final_bbox_reward = weights['final_bbox_reward'] * final_bbox_reward
        weighed_point_reward = weights['point_reward'] * point_reward
        weighed_compact_reward = weights['compactness_reward'] * compact_reward
        weighed_adjustment_reward = weights['adjustment_consistency_reward'] * adjustment_reward
        weighed_containment_reward = weights['part_containment_reward'] * containment_reward
        weighed_non_repeat = weights['non_repeat_reward'] * non_repeat

        # sum all rewards
        total_reward = (
            weighed_format_reward +
            weighed_decision_reward +
            weighed_initial_iou_reward +
            weighed_improvement_reward +
            weighed_final_bbox_reward +
            weighed_point_reward +
            weighed_compact_reward +
            weighed_adjustment_reward +
            weighed_containment_reward +
            weighed_non_repeat
        )


        scores.append(
            {
                'overall': float(total_reward),
                'format': float(weighed_format_reward),
                'decision': float(weighed_decision_reward),
                'initial_iou': float(weighed_initial_iou_reward),
                'improvement': float(weighed_improvement_reward),
                'final_bbox': float(weighed_final_bbox_reward),
                'point': float(weighed_point_reward),
                'compactness': float(weighed_compact_reward),
                'adjustment_consistency': float(weighed_adjustment_reward),
                'part_containment': float(weighed_containment_reward),
                'non_repetition': float(weighed_non_repeat)
            }
        )


        # print each component for debugging
        # print(f"Format Reward: {weights['format_reward'] * format_reward:.4f}")
        # print(f"Decision Reward: {weights['decision_reward'] * decision_reward:.4f}")
        # print(f"Initial IoU Reward: {weights['initial_iou_reward'] * initial_iou_reward:.4f}")
        # print(f"Improvement Reward: {weights['improvement_reward'] * improvement_reward:.4f}")
        # print(f"Final BBox Reward: {weights['final_bbox_reward'] * final_bbox_reward:.4f}")
        # print(f"Point Reward: {weights['point_reward'] * point_reward:.4f}")
        # print(f"Compactness Reward: {weights['compactness_reward'] * compact_reward:.4f}")
        # print(f"Adjustment Consistency Penalty: {weights['adjustment_consistency_reward'] * adjustment_reward:.4f}")
        # print(f"Containment Reward: {weights['part_containment_reward'] * containment_reward:.4f}")
        # print(f"Non-Repetition Reward: {weights['non_repeat_reward'] * non_repeat:.4f}")

    return scores
import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/EvaluationScripts')

from eval_base import compute_iou, remove_overlapping_boxes, merge_intersecting_boxes

from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='Process reasoning model and save directory')
parser.add_argument('--chunk_id', type=int, required=False, help='Chunk ID for distributed evaluation')
parser.add_argument('--save_dir', type=str, help='Directory to save results')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
parser.add_argument('--total_chunks', type=int, default=4, help='Total number of chunks for distributed evaluation')

args = parser.parse_args()

save_dir = args.save_dir
batch_size = args.batch_size

device = "cuda" if torch.cuda.is_available() else "cpu"

# SAM3 util
def combine_masks_sam3(masks):
    """
    Combine multiple binary masks into a single mask.
    
    Args:
        masks: Tensor of shape (n, w, h) containing n binary masks
    
    Returns:
        combined_mask: Single mask of shape (w, h) in numpy format
    """
    combined_mask = masks.any(dim=0)    
    return combined_mask.cpu().numpy()

# Get bboxes from from segments
from scipy import ndimage

def get_bboxes_from_mask(mask):
    """
    Extract bounding boxes for each connected component (segment) in the mask.
    
    Args:
        mask: Binary mask as numpy array
    
    Returns:
        bboxes: List of bounding boxes in format [x_min, y_min, x_max, y_max]
    """
    # connected components
    labeled_mask, num_segments = ndimage.label(mask > 0)
    
    bboxes = []
    
    # bounding box for each segment
    for segment_id in range(1, num_segments + 1):
        coords = np.argwhere(labeled_mask == segment_id)
        
        if len(coords) > 0:
            # min and max coords
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    
    return bboxes

##### Dataset-specific functions #####
def load_mask_and_parse_name(image_name, masks_dir):
    """
    Load mask and extract object and part names from filename for the InstructPart dataset.
    
    Args:
        image_name: Name of the image file (with or without extension)
        masks_dir: Directory containing the mask files
    
    Returns:
        mask: Loaded mask as numpy array
        object_name: Name of the object
        part_name: Name of the part
    """
    #get basename
    basename = os.path.splitext(image_name)[0]
    
    # mask 
    mask_path = os.path.join(masks_dir, f"{basename}.png")
    mask = np.array(Image.open(mask_path))
    
    # parse the filename: image_id-object_name-part_name
    filename_parts = basename.split('-')
    part_name = filename_parts[-1]
    object_name = filename_parts[-2]

    return mask, object_name, part_name
#######################################

############# Dataset-specific settings #############
# this needs to be changed to the train dataset 
instructpart_test_dir = "/data/VLMGroundingProject/Datasets/InstructPart/train1800"
masks_dir = os.path.join(instructpart_test_dir, "masks")
images_dir = os.path.join(instructpart_test_dir, "images")
val_filenames = os.listdir(images_dir)
###################################################### 

# chunk into 4 parts for distributed evaluation if chunk_id is provided
if args.chunk_id is not None:
    total_chunks = args.total_chunks
    chunk_size = len(val_filenames) // total_chunks
    start_idx = args.chunk_id * chunk_size
    if args.chunk_id == total_chunks - 1:
        end_idx = len(val_filenames)
    else:
        end_idx = (args.chunk_id + 1) * chunk_size
    val_filenames = val_filenames[start_idx:end_idx]
    print(f"Processing chunk {args.chunk_id}: images {start_idx} to {end_idx}")

# set objects and parts results file paths
if args.chunk_id is not None:
    # object_results_filepath = os.path.join(save_dir, f"objects_results_{args.chunk_id}.json")
    parts_results_filepath = os.path.join(save_dir, f"sam3_bboxes_{args.chunk_id}.json")
else:
    # object_results_filepath = os.path.join(save_dir, "objects_results.json")
    parts_results_filepath = os.path.join(save_dir, "sam3_bboxes.json")


## Load SAM3 model and processor
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Use all validation filenames as required 
req_filenames = val_filenames
all_results = []
for batch_start in tqdm(range(0, len(req_filenames), batch_size)):
    batch_filenames = req_filenames[batch_start: batch_start + batch_size]

    # prepare batch data
    batch_images = []
    batch_text_prompts = []
    batch_gt_masks = []
    
    for image_name in batch_filenames:
        # Load image
        img_filepath = os.path.join(images_dir, image_name)
        image = Image.open(img_filepath).convert("RGB")
        batch_images.append(image)
        
        # Load ground truth mask and parse names
        gt_mask, object_name, part_name = load_mask_and_parse_name(image_name, masks_dir)
        
        # Create query text (part within object)
        query_text = f"{object_name}'s {part_name}"
        batch_text_prompts.append(query_text)
        batch_gt_masks.append(gt_mask)
    
    inputs = processor(images=batch_images, text=batch_text_prompts, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results for batch images
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )

    # print(len(results))

    assert len(results) == len(batch_gt_masks), "Mismatch in number of results and GT masks"
    
    for i, res in enumerate(results):
        if len(res['masks']) == 0:
            # No masks predicted, skip this sample
            all_results.append({
                "image_name": batch_filenames[i],
                "text_prompt": batch_text_prompts[i],
                "intersection": 0,
                "union": int(np.sum(batch_gt_masks[i]) ),
                "pred_bboxes": [0, 0, 0, 0]
            })
            continue

        combined_mask = combine_masks_sam3(res['masks'])
        gt_mask = batch_gt_masks[i]

        assert combined_mask.shape == gt_mask.shape, "Mismatch in mask shapes"

        # Compute IoU
        intersection, union = compute_iou(combined_mask, gt_mask)

        # Get bounding boxes from predicted mask
        pred_bboxes = get_bboxes_from_mask(combined_mask)
        # Remove overlapping boxes
        if len(pred_bboxes) > 1:
            pred_bboxes = remove_overlapping_boxes(pred_bboxes)
        if len(pred_bboxes) > 1:
            pred_bboxes = merge_intersecting_boxes(pred_bboxes)

        all_results.append({
            "image_name": batch_filenames[i],
            "text_prompt": batch_text_prompts[i],
            "intersection": int(intersection),
            "union": int(union),
            "pred_bboxes": pred_bboxes
        })

        # print(combined_mask.shape, gt_mask.shape)

        # # visualize the results
        # # Convert masks to numpy for visualization
        # pred_mask_np = combined_mask
        
        # # Visualize the results
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # # Original image
        # axes[0].imshow(batch_images[i])
        # axes[0].set_title(f'Original Image: {batch_text_prompts[i]}')
        # axes[0].axis('off')
        
        # # Ground truth mask overlay
        # axes[1].imshow(batch_images[i])
        # axes[1].imshow(gt_mask, alpha=0.5, cmap='Reds')
        # axes[1].set_title('Ground Truth Mask')
        # axes[1].axis('off')
        
        # # Predicted mask overlay
        # axes[2].imshow(batch_images[i])
        # axes[2].imshow(pred_mask_np, alpha=0.5, cmap='Blues')
        # axes[2].set_title('Predicted Mask')
        # axes[2].axis('off')
        
        # plt.tight_layout()
        # plt.show()

    # break # debug. remove this later

with open(parts_results_filepath, 'w') as f:
    json.dump(all_results, f)


# - REVISE: SAM3 expects small text phrases (32 tokens or less), and the VisionReasoner dataset has multi objects, appended with 'and', resulting in longer text prompts.
# -- We deal with this by setting truncation=True in the tokenizer call within the Sam3Processor (inside /home/ksmehrab/miniconda/envs/sam3/lib/python3.12/site-packages/transformers/models/sam3/processing_sam3.py)
# -- The baseline box iou obtained using this would likely not be very high due to this. A better fix might be to run sam3 separately for each of the phrases, get the boxes for each phrase, and then average the box iou as the baseline iou 
# -- Alternate approach could be to look at the original refcoco dataset and prompts 

import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/EvaluationScripts')

from eval_base import remove_overlapping_boxes, merge_intersecting_boxes

from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='sam3')
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
from datasets import load_dataset
data_path = "/data/VLMGroundingProject/Datasets/SegZeroVisualReasoner/VisionReasoner_multi_object_7k_840"
dataset = load_dataset(data_path)
dataset = dataset['train']
###################################################### 

# chunk into 4 parts for distributed evaluation if chunk_id is provided
if args.chunk_id is not None:
    total_chunks = args.total_chunks
    chunk_size = len(dataset) // total_chunks
    start_idx = args.chunk_id * chunk_size
    if args.chunk_id == total_chunks - 1:
        end_idx = len(dataset)
    else:
        end_idx = (args.chunk_id + 1) * chunk_size
    data_subset = []
    for i in range(start_idx, end_idx):
        data_subset.append(dataset[i])
    print(f"Processing chunk {args.chunk_id}: images {start_idx} to {end_idx}")
else:
    data_subset = dataset

# set objects and parts results file paths
if args.chunk_id is not None:
    # object_results_filepath = os.path.join(save_dir, f"objects_results_{args.chunk_id}.json")
    results_filepath = os.path.join(save_dir, f"sam3_bboxes_{args.chunk_id}.json")
else:
    # object_results_filepath = os.path.join(save_dir, "objects_results.json")
    results_filepath = os.path.join(save_dir, "sam3_bboxes.json")


## Load SAM3 model and processor
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Use all validation filenames as required 

all_results = []
for batch_start in tqdm(range(0, len(data_subset), batch_size)):
    batch_data = data_subset[batch_start: batch_start + batch_size]

    # prepare batch data
    batch_images = []
    batch_text_prompts = []

    for data in batch_data:
        # Load image
        image = data['image']
        batch_images.append(image)

        # Create query text (part within object)
        query_text = data['problem']

        # # Truncate to first 15 words to fit SAM3 text input constraints
        # query_text = ' '.join(query_text.split()[:15])

        batch_text_prompts.append(query_text)

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

    assert len(results) == len(batch_images), "Mismatch in number of results and inputs"
    
    for i, res in enumerate(results):
        if len(res['masks']) == 0:
            # No masks predicted, skip this sample
            all_results.append({
                "image_name": batch_data[i]['id'],
                "text_prompt": batch_text_prompts[i],
                "pred_bboxes": [0, 0, 0, 0]
            })
            continue

        combined_mask = combine_masks_sam3(res['masks'])

        # Get bounding boxes from predicted mask
        pred_bboxes = get_bboxes_from_mask(combined_mask)

        # Remove overlapping boxes
        if len(pred_bboxes) > 1:
            pred_bboxes = remove_overlapping_boxes(pred_bboxes)
        if len(pred_bboxes) > 1:
            pred_bboxes = merge_intersecting_boxes(pred_bboxes)

        all_results.append({
            "image_name": batch_data[i]['id'],
            "text_prompt": batch_text_prompts[i],
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

with open(results_filepath, 'w') as f:
    json.dump(all_results, f)


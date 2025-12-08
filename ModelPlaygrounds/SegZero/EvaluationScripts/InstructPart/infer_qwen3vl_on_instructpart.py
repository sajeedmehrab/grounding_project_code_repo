import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/GitRepoLatest/Seg-Zero')
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/EvaluationScripts')

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sam2.sam2_image_predictor import SAM2ImagePredictor
import re

import argparse

from eval_base import (
    extract_bbox_points_think,
    extract_information_vrpart,
    extract_information_vrpart2,
    compute_iou,
    combine_masks,
    visualize_first_and_final_bbox
)

parser = argparse.ArgumentParser(description='Process reasoning model and save directory')
parser.add_argument('--reasoning_model_path', type=str, required=True,
                    help='Path to the reasoning model')
parser.add_argument('--save_dir', type=str, required=True,
                    help='Directory to save outputs')
parser.add_argument('--segmentation_model_path', type=str, default="facebook/sam2-hiera-large",
                    help='Path to the segmentation model')
parser.add_argument('--save_masks', action='store_true',
                    help='Whether to save predicted masks. for quantitative evaluation, we can probably save time by not saving masks')
parser.add_argument('--prompt_type', type=str, choices=['segzero', 'vrpart', 'vrpart2'])
parser.add_argument('--skip_already_done', action='store_true',
                    help='Whether to skip images that have already been done')
parser.add_argument('--chunk_id', type=int, required=False, help='Chunk ID for distributed evaluation')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode with visualizations')

args = parser.parse_args()

save_masks = args.save_masks
reasoning_model_path = args.reasoning_model_path
segmentation_model_path = args.segmentation_model_path
debug = args.debug
batch_size = args.batch_size

save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

max_response_length = 2000

############# Dataset-specific settings #############
instructpart_test_dir = "/data/VLMGroundingProject/Datasets/InstructPart/test"
masks_dir = os.path.join(instructpart_test_dir, "masks")
images_dir = os.path.join(instructpart_test_dir, "images")
val_filenames = os.listdir(images_dir)
######################################################

# chunk into 4 parts for distributed evaluation if chunk_id is provided
if args.chunk_id is not None:
    total_chunks = 4
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
    parts_results_filepath = os.path.join(save_dir, f"parts_results_{args.chunk_id}.json")
else:
    # object_results_filepath = os.path.join(save_dir, "objects_results.json")
    parts_results_filepath = os.path.join(save_dir, "parts_results.json")

# skip already done images if mentioned 
if args.skip_already_done:
    # read json files in save_dir, get a set if the image_id
    already_done = set()
    if os.path.exists(parts_results_filepath):
        with open(parts_results_filepath, 'r') as f:
            existing_parts_outputs = json.load(f)
        for item in existing_parts_outputs:
            already_done.add(item['image_id'])
else:
    already_done = set()  

req_filenames = [img_name for img_name in val_filenames if img_name not in already_done]
print(f'Already done: {len(already_done)}. Processing remaining: {len(req_filenames)}...')  

# TODO: CHECK
# Load prompt template
# The LLM output parsing depends on the prompt format, set the output extraction function accordingly
if args.prompt_type == 'segzero':
    raise NotImplementedError("SegZero prompt not implemented in this script.")
elif args.prompt_type == 'vrpart':
    prompt_filepath = '../Prompts/vrpart_prompt.txt'
    with open(prompt_filepath, 'r') as f:
        QUESTION_TEMPLATE = f.read()
    llm_output_parser = extract_information_vrpart
elif args.prompt_type == 'vrpart2':
    prompt_filepath = '../Prompts/vrpart2_prompt.txt'
    with open(prompt_filepath, 'r') as f:
        QUESTION_TEMPLATE = f.read()
    llm_output_parser = extract_information_vrpart2
else:
    raise ValueError("Invalid prompt type specified.")

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

# load models
print("Loading reasoning model...")
reasoning_model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.reasoning_model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    local_files_only=True
)
reasoning_model.eval()

print("Loading segmentation model...")
segmentation_model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")

all_results = []
resize_size = 1024
prediction_grid_size = 1000

if debug:
    req_filenames = req_filenames[:5]  # do first 5 images in debug mode

for batch_start in tqdm(range(0, len(req_filenames), batch_size)):
    batch_filenames = req_filenames[batch_start: batch_start + batch_size]

    # prepare batch data
    batch_messages = []
    batch_metadata = []
    
    for image_name in batch_filenames:
        # Load image
        img_filepath = os.path.join(images_dir, image_name)
        image = Image.open(img_filepath).convert("RGB")
        original_width, original_height = image.size
        x_factor, y_factor = original_width/prediction_grid_size, original_height/prediction_grid_size
        
        # Load ground truth mask and parse names
        gt_mask, object_name, part_name = load_mask_and_parse_name(image_name, masks_dir)
        
        # Create query text (part within object)
        query_text = f"{object_name}'s {part_name}"
        
        # Prepare message for model
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image.resize((resize_size, resize_size), Image.BILINEAR)
                },
                {   
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=query_text.lower().strip("."))
                }
            ]
        }]

        batch_messages.append(message)
        batch_metadata.append({
            "image_name": image_name,
            "image": image,
            "original_width": original_width,
            "original_height": original_height,
            "x_factor": x_factor,
            "y_factor": y_factor,
            "gt_mask": gt_mask,
            "object_name": object_name,
            "part_name": part_name,
            "query_text": query_text
        })

    # Prepare inputs
    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate reasoning output
    with torch.inference_mode():
        generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=max_response_length, do_sample=False)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # process each item in the batch
    for output_text, metadata in zip(output_texts, batch_metadata):
        if debug:
            print(f"Output for {metadata['image_name']}:\n{output_text}\n{'-'*50}\n")
        
        try:
            bboxes, points, parsed_output = llm_output_parser(
                output_text, metadata['x_factor'], metadata['y_factor']
            )

            # visualize the first and final boxes in debug
            if debug and parsed_output['first_answer']:
                try:
                    # print("debugdebug")
                    first_answer_data = json.loads(parsed_output['first_answer'])
                    first_bboxes = [[
                        int(item['bbox_2d'][0] * metadata['x_factor'] + 0.5),
                        int(item['bbox_2d'][1] * metadata['y_factor'] + 0.5),
                        int(item['bbox_2d'][2] * metadata['x_factor'] + 0.5),
                        int(item['bbox_2d'][3] * metadata['y_factor'] + 0.5)
                    ] for item in first_answer_data]

                    viz_save_path = os.path.join(save_dir, f"{os.path.splitext(metadata['image_name'])[0]}_boxes_comparison.png")
                    visualize_first_and_final_bbox(metadata['image'], first_bboxes, bboxes, metadata['query_text'], viz_save_path) # viz_save_path
                    
                except Exception as viz_error:
                    print(f"Visualization error for {metadata['image_name']}: {viz_error}")

            # Generate segmentation masks
            segmentation_model.set_image(metadata['image'])
            mask_all = np.zeros((metadata['original_height'], metadata['original_width']), dtype=bool)
            
            for bbox, point in zip(bboxes, points):
                masks, scores, _ = segmentation_model.predict(
                    point_coords=[point],
                    point_labels=[1],
                    box=bbox
                )
                sorted_ind = np.argsort(scores)[::-1]
                mask = masks[sorted_ind][0].astype(bool)
                mask_all = np.logical_or(mask_all, mask)
            
            # Compute IoU
            intersection, union = compute_iou(mask_all, metadata['gt_mask'])
            
            # Save predicted mask only if mentioned
            if save_masks:
                mask_save_dir = os.path.join(save_dir, os.path.splitext(metadata['image_name'])[0])
                os.makedirs(mask_save_dir, exist_ok=True)
                np.save(os.path.join(mask_save_dir, "predicted_mask.npy"), mask_all)
            
            all_results.append({
                "image_id": metadata['image_name'],
                "object_name": metadata['object_name'],
                "part_name": metadata['part_name'],
                "query": metadata['query_text'],
                "think": parsed_output,
                "intersection": int(intersection),
                "union": int(union),
                "iou": float(intersection/union) if union > 0 else 0.0
            })
        
        except Exception as e:
            print(f"Error processing {metadata['image_name']}: {e}")
            all_results.append({
                "image_id": metadata['image_name'],
                "object_name": metadata['object_name'],
                "part_name": metadata['part_name'],
                "query": metadata['query_text'],
                "error": str(e),
                "intersection": 0,
                "union": int(metadata['gt_mask'].sum()),
                "iou": 0.0
            })
    
        # after every 100 predictions, save all_object_outputs and all_parts_outputs to json by appending to existing files
        # if files do not exist, create new files 
        # then clear the lists
        if (len(all_results)) % 100 == 0:
            print(f'Saving intermediate results after {len(all_results)} images...')
            # we only have parts in InstructPart
            if os.path.exists(parts_results_filepath):
                with open(parts_results_filepath, 'r') as f:
                    existing_parts_outputs = json.load(f)
                existing_parts_outputs.extend(all_results)
                with open(parts_results_filepath, 'w') as f:
                    json.dump(existing_parts_outputs, f)
            else:
                with open(parts_results_filepath, 'w') as f:
                    json.dump(all_results, f)

            all_results = []

    # Clean GPU memory
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

# Save results any remaining results
print(f'Saving any remaining results after loop')
if os.path.exists(parts_results_filepath):
    with open(parts_results_filepath, 'r') as f:
        existing_parts_outputs = json.load(f)
    existing_parts_outputs.extend(all_results)
    with open(parts_results_filepath, 'w') as f:
        json.dump(existing_parts_outputs, f)
else:
    with open(parts_results_filepath, 'w') as f:
        json.dump(all_results, f)

print(f"results saved to {save_dir}")


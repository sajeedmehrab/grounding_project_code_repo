import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/GitRepoLatest/Seg-Zero')
sys.path.append('/home/ksmehrab/AttentionGrounding/Baselines/Datasets') # this is where the PartImageNetDataset is
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/EvaluationScripts') # this is for eval_base


import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sam2.sam2_image_predictor import SAM2ImagePredictor
import re

from partimagenet_dataset import PartImageNetDataset

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
chunk_id = args.chunk_id

save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

max_response_length = 2000

############# Dataset-specific settings #############
test_images_dir = "/data/VLMGroundingProject/Datasets/PartImageNet/test"
test_json_path = "/data/VLMGroundingProject/Datasets/PartImageNet/test.json"

partimagenet_dataset = PartImageNetDataset(
    json_path=test_json_path,
    images_dir=test_images_dir
)

num_annotations = partimagenet_dataset.num_annotations
######################################################

# chunk into 4 parts for distributed evaluation if chunk_id is provided
if args.chunk_id is not None:
    total_chunks = 4
    chunk_size = num_annotations // total_chunks
    start_idx = args.chunk_id * chunk_size
    if args.chunk_id == total_chunks - 1:
        end_idx = num_annotations
    else:
        end_idx = (args.chunk_id + 1) * chunk_size

    print(f"Processing chunk {args.chunk_id}: annotations {start_idx} to {end_idx} out of {num_annotations}")

# set objects and parts results file paths
if args.chunk_id is not None:
    # object_results_filepath = os.path.join(save_dir, f"objects_results_{args.chunk_id}.json")
    parts_results_filepath = os.path.join(save_dir, f"parts_results_{args.chunk_id}.json")
else:
    # object_results_filepath = os.path.join(save_dir, "objects_results.json")
    parts_results_filepath = os.path.join(save_dir, "parts_results.json")

if os.path.exists(parts_results_filepath):
    raise ValueError(f"Results file {parts_results_filepath} already exists. Risk of overwriting or using previous undesired results. Delete it or choose a different save directory.")

# skip already done images (if mentioned)
# for partimagenet, skip by annotation id. the saved json will have image_id so we can skip based on that 
if args.skip_already_done:
    # read json files in save_dir
    already_done = set()
    if os.path.exists(parts_results_filepath):
        with open(parts_results_filepath, 'r') as f:
            existing_parts_outputs = json.load(f)
        for item in existing_parts_outputs:
            already_done.add(item['annotation_idx'])
else:
    already_done = set()  

if args.chunk_id is not None:
    req_annotation_ids = [i for i in range(start_idx, end_idx) if i not in already_done]
else:
    req_annotation_ids = [i for i in range(num_annotations) if i not in already_done]

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

# load models
print("Loading reasoning model...")
reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    reasoning_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
reasoning_model.eval()

print("Loading segmentation model...")
segmentation_model = SAM2ImagePredictor.from_pretrained(segmentation_model_path)

processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")

all_results = []
resize_size = 840

if debug:
    debug_end_index = 5  # do first 5 annotations in debug mode

for j in tqdm(range(0, len(req_annotation_ids), batch_size)):
    # prepare batch data
    batch_messages = []
    batch_metadata = []
    
    batch_annotation_ids = req_annotation_ids[j: j + batch_size]
    for i in batch_annotation_ids:
        try:
            annotation = partimagenet_dataset.get_annotation(i)
        except Exception as e:
            print(f"Error loading annotation {i}: {e}")
            continue

        # Load ground truth mask and parse names
        gt_mask = annotation['mask']
        if gt_mask is None:
            print(f"Skipping annotation {annotation['image_filename']}, index: {i} due to missing GT mask.")
            continue

         # query text (part within object)
        query_text = annotation['class_name']
        if query_text is None:
            print(f"Skipping annotation {annotation['image_filename']}, index: {i} due to missing class name.")
            continue

        image_name = annotation['image_filename']
        image_path = annotation['image_filepath']

        # Load image
        image = annotation['image']
        original_width, original_height = image.size
        x_factor, y_factor = original_width/resize_size, original_height/resize_size

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
            "query_text": query_text,
            "annotation_idx": i
        })

    # this is probably not needed, this is from old code 
    if len(batch_messages) == 0:
        print("All images in this batch already processed, skipping batch.")
        continue

    # Prepare inputs
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in batch_messages]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
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
                    
                    # viz_save_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_boxes_comparison.png")
                    visualize_first_and_final_bbox(metadata['image'], first_bboxes, bboxes, metadata['query_text']) # viz_save_path
                    
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
            
            # Save predicted mask
            if save_masks:
                mask_save_dir = os.path.join(save_dir, os.path.splitext(metadata['image_name'])[0])
                os.makedirs(mask_save_dir, exist_ok=True)
                mask_save_name = metadata['query_text'].replace("'s", "").replace(" ", "_") # cow's head -> cow_head
                np.save(os.path.join(mask_save_dir, f"{mask_save_name}.npy"), mask_all)

            all_results.append({
                "image_id": metadata['image_name'],
                "query": metadata['query_text'],
                "think": parsed_output,
                "intersection": int(intersection),
                "union": int(union),
                "iou": float(intersection/union) if union > 0 else 0.0,
                "annotation_idx": metadata['annotation_idx']
            })
        
        except Exception as e:
            print(f"Error processing {metadata['image_name']}: {e}")
            all_results.append({
                "image_id": metadata['image_name'],
                "query": metadata['query_text'],
                "error": str(e),
                "intersection": 0,
                "union": int(metadata['gt_mask'].sum()),
                "iou": 0.0,
                "annotation_idx": metadata['annotation_idx']
            })

        # if debug mode, break after first few
        if debug and len(all_results) >= debug_end_index:
            break

        # after every 100 predictions, save all_object_outputs and all_parts_outputs to json by appending to existing files
        # if files do not exist, create new files 
        # then clear the lists
        if (len(all_results)) % 100 == 0:
            print(f'Saving intermediate results after {len(all_results)} images...')
            # we only have parts in partimagenet
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

    if debug and len(all_results) >= debug_end_index:
        break

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

from multiprocessing.util import debug
import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/GitRepoLatest/Seg-Zero')
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/EvaluationScripts')

import os
import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import pdb

import cv2
from PIL import Image as PILImage
import re
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt

import argparse

from eval_base import (
    extract_bbox_points_think,
    extract_information_vrpart,
    extract_information_vrpart2,
    compute_iou,
    combine_masks
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

args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

debug = False
debug_examine = 3

save_masks = args.save_masks
reasoning_model_path = args.reasoning_model_path
segmentation_model_path = args.segmentation_model_path

save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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
    

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.reasoning_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

reasoning_model.eval()

segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

# default processer
processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

############## Dataset-specific code: pascalpart images and object/object parts  ######################
sys.path.append('/home/ksmehrab/AttentionGrounding/Baselines/Datasets')
sys.path.append('/home/ksmehrab/AttentionGrounding/Baselines')

from pascalpart import get_pascalpart_classes, get_pascalpart_masks
from utils import read_txt_file, save_to_json
from tqdm import tqdm
import os

pascal_image_dir = "/data/Pascal_VOC_2012/VOCdevkit/VOC2012/JPEGImages" # 17125 images
annotations_path= "/data/PartSegmentationDatasets/PascalPart/Annotations_Part"
val_filepath = "/data/PartSegmentationDatasets/PascalPart/val.txt" # 925 images. File contains just the file prefix. Add .jpg extension for images, and .mat extension for annotations
val_filenames = read_txt_file(val_filepath)

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
    object_results_filepath = os.path.join(save_dir, f"objects_results_{args.chunk_id}.json")
    parts_results_filepath = os.path.join(save_dir, f"parts_results_{args.chunk_id}.json")
else:
    object_results_filepath = os.path.join(save_dir, "objects_results.json")
    parts_results_filepath = os.path.join(save_dir, "parts_results.json")

# skip already done images if mentioned 
if args.skip_already_done:
    # read json files in save_dir, get a set if the image_id
    already_done = set()

    if os.path.exists(object_results_filepath):
        with open(object_results_filepath, 'r') as f:
            existing_object_outputs = json.load(f)
        for item in existing_object_outputs:
            already_done.add(item['image_id'])
    if os.path.exists(parts_results_filepath):
        with open(parts_results_filepath, 'r') as f:
            existing_parts_outputs = json.load(f)
        for item in existing_parts_outputs:
            already_done.add(item['image_id'])
else:
    already_done = set()    

req_filenames = [f for f in val_filenames if f not in already_done]

print(f'Already done on {len(already_done)} images. Running on remaining {len(req_filenames)}...')

all_object_outputs = []
all_parts_outputs = []

if debug:
    debug_count = 0 

for i, filename in tqdm(enumerate(req_filenames)):
    img_filepath = os.path.join(pascal_image_dir, filename+'.jpg')

    image = PILImage.open(img_filepath)
    image = image.convert("RGB")
    original_width, original_height = image.size
    resize_size = 840
    x_factor, y_factor = original_width/resize_size, original_height/resize_size

    annot_filename = filename + '.mat'
    anno_dict = get_pascalpart_masks(annot_filename, annotations_path, images_path=pascal_image_dir)
    
    # classes_to_detect = get_pascalpart_classes(annot_filename, annotations_path) # This is a dictionary in this format {object: [list of parts]}
    # print(classes_to_detect)
    for obj_name, anno in anno_dict.items():
        obj_masks = anno['object_maps']
        obj_mask = combine_masks(obj_masks)

        gt_mask_list = [obj_mask]
        seg_list = [obj_name]

        parts_masks = anno['parts']
        # print(parts_masks.keys())
        for part_name, masks in parts_masks.items():
            part_mask = combine_masks(masks)
            gt_mask_list.append(part_mask)
            part_full_name = obj_name + "'s " + part_name
            seg_list.append(part_full_name)

        messages = []
        for args_text in seg_list:
            message = [{
                "role": "user",
                "content": [
                {
                    "type": "image", 
                    "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
                },
                {   
                    "type": "text",
                    # TODO: CHANGE
                    "text": QUESTION_TEMPLATE.format(
                        Question=args_text.lower().strip(".")
                    )    
                }
            ]
            }]
            messages.append(message)

        # print debug to see a few message texts
        # if debug:
        #     print(f"Printing first {debug_examine} messages for debugging...")
        #     for msg in messages[:debug_examine]:
        #         print('====================')
        #         print(msg[0]['content'][1]['text'])
            
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    
        image_inputs, video_inputs = process_vision_info(messages)
 
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        assert len(batch_output_text) == len(gt_mask_list) == len(seg_list)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for id_idx in range(len(batch_output_text)):
                if id_idx == 0:
                    all_outputs = all_object_outputs
                else:
                    all_outputs = all_parts_outputs
                try:
                    bboxes, points, think = llm_output_parser(
                                            batch_output_text[id_idx], 
                                            original_width/resize_size, 
                                            original_height/resize_size
                                        )
                except Exception as e:
                    # add penalty in this situation
                    print("Reasoning error: ", e, "Text: ", batch_output_text[id_idx], "Image: ", filename, "Seg: ", seg_list[id_idx])
                    think = ""
                    intersection = 0
                    union = np.array(gt_mask_list[id_idx]).sum()
                    bbox_iou = 0.0
                    all_outputs.append({
                        "image_id": filename,
                        "ann_id": filename,
                        "seg_id": seg_list[id_idx],
                        "think": think,
                        "intersection": int(intersection),
                        "union": int(union),
                        "bbox_iou": bbox_iou,
                        "visualization_path": None
                    })
                    continue
                try:
                    segmentation_model.set_image(image)
                    mask_all = np.zeros((original_height, original_width), dtype=bool)
                except Exception as e:
                    print("Set image error: ", e, filename)
                    # skip this because the image or mask is not correct
                    continue
                try:
                    for bbox, point in zip(bboxes, points):
                        masks, scores, _ = segmentation_model.predict(
                            point_coords=[point],
                            point_labels=[1],
                            box=bbox
                        )
                        sorted_ind = np.argsort(scores)[::-1]
                        masks = masks[sorted_ind]
                        mask = masks[0].astype(bool)
                        mask_all = np.logical_or(mask_all, mask)
                    gt_mask = np.array(gt_mask_list[id_idx])
                except Exception as e:
                    print("Segmentation error: ", e, filename)
                    # skip this because the image or mask is not correct
                    continue
                try:
                    intersection, union = compute_iou(mask_all, gt_mask)
                except Exception as e:
                    print("Image error: ", e)
                    # skip this because the image or mask is not correct
                    continue 
                
                # save mask_all in save_dir/filename/mask_save_filename 
                if save_masks:
                    mask_save_dir = os.path.join(save_dir, filename)
                    os.makedirs(mask_save_dir, exist_ok=True)
                    mask_save_filename = seg_list[id_idx].replace("'s", "").replace(" ", "_") + ".npy"
                    mask_save_filepath = os.path.join(mask_save_dir, mask_save_filename)
                    np.save(mask_save_filepath, mask_all)

                # Create visualization if you are in debug mode 
                if debug:
                    import matplotlib.pyplot as plt

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    # Original image
                    axes[0].imshow(image)
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')

                    # Predicted mask overlay
                    axes[1].imshow(image)
                    predicted_overlay = np.zeros((mask_all.shape[0], mask_all.shape[1], 4))
                    predicted_overlay[mask_all, 0] = 1.0  # Red color for prediction
                    predicted_overlay[mask_all, 3] = 0.5  # Alpha transparency
                    axes[1].imshow(predicted_overlay)
                    axes[1].set_title(f'Predicted: {seg_list[id_idx]}')
                    axes[1].axis('off')

                    # Ground truth mask overlay
                    axes[2].imshow(image)
                    gt_overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 4))
                    gt_overlay[gt_mask.astype(bool), 1] = 1.0  # Green color for ground truth
                    gt_overlay[gt_mask.astype(bool), 3] = 0.5  # Alpha transparency
                    axes[2].imshow(gt_overlay)
                    axes[2].set_title(f'Ground Truth: {seg_list[id_idx]}')
                    axes[2].axis('off')

                    plt.tight_layout()
                    visualization_path = os.path.join(mask_save_dir, f"{seg_list[id_idx].replace('\'s', '').replace(' ', '_')}_visualization.png")
                    plt.savefig(visualization_path)
                    # plt.show()
                    plt.close()
                else:
                    visualization_path = None
                
                all_outputs.append({
                    "image_id": filename,
                    "ann_id": filename,
                    "seg_id": seg_list[id_idx],
                    "think": think,
                    "intersection": int(intersection),
                    "union": int(union),
                    "bbox_iou": 0,
                    "visualization_path": visualization_path
                })
        
        # clean GPU memory
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    # after every 100 images, save all_object_outputs and all_parts_outputs to json by appending to existing files
    # if files do not exist, create new files 
    # then clear the lists
    if (i+1) % 100 == 0:
        print(f'Saving intermediate results after {i+1} images...')

        if os.path.exists(object_results_filepath):
            with open(object_results_filepath, 'r') as f:
                existing_object_outputs = json.load(f)
            existing_object_outputs.extend(all_object_outputs)
            with open(object_results_filepath, 'w') as f:
                json.dump(existing_object_outputs, f)
        else:
            with open(object_results_filepath, 'w') as f:
                json.dump(all_object_outputs, f)

        if os.path.exists(parts_results_filepath):
            with open(parts_results_filepath, 'r') as f:
                existing_parts_outputs = json.load(f)
            existing_parts_outputs.extend(all_parts_outputs)
            with open(parts_results_filepath, 'w') as f:
                json.dump(existing_parts_outputs, f)
        else:
            with open(parts_results_filepath, 'w') as f:
                json.dump(all_parts_outputs, f)

        all_object_outputs = []
        all_parts_outputs = []
    
    if debug:
        debug_count += 1
        if debug_count >= debug_examine:
            print(f"Debug mode: breaking after {debug_examine} images.")
            break

# Finally, save remaining results

print(f'Saving any remaining results after loop...')


if os.path.exists(object_results_filepath):
    with open(object_results_filepath, 'r') as f:
        existing_object_outputs = json.load(f)
    existing_object_outputs.extend(all_object_outputs)
    with open(object_results_filepath, 'w') as f:
        json.dump(existing_object_outputs, f)
else:
    with open(object_results_filepath, 'w') as f:
        json.dump(all_object_outputs, f)

if os.path.exists(parts_results_filepath):
    with open(parts_results_filepath, 'r') as f:
        existing_parts_outputs = json.load(f)
    existing_parts_outputs.extend(all_parts_outputs)
    with open(parts_results_filepath, 'w') as f:
        json.dump(existing_parts_outputs, f)
else:
    with open(parts_results_filepath, 'w') as f:
        json.dump(all_parts_outputs, f)

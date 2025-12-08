import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/GitRepoLatest/Seg-Zero')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

## variables to CHANGE 
reasoning_model_path = "/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/GitRepoLatest/Seg-Zero/pretrained_models/VisionReasoner-7B"

save_dir = "/data/VLMGroundingProject/BaselineResults/PascalPart/VRInstructPart"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

debug = False
debug_examine = 3

# configuration object, to replace args
class Config:
    def __init__(self):
        self.reasoning_model_path = reasoning_model_path
        self.segmentation_model_path = "facebook/sam2-hiera-large"

args = Config()

def extract_bbox_points_think(output_text, x_factor, y_factor):
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

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection, union

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

QUESTION_TEMPLATE = \
    "Please find \"{Question}\" with bboxs and points." \
    "Compare the difference between object(s) and find the most closely matched object(s)." \
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
    "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
    "i.e., <think> thinking process here </think>" \
    "<answer>{Answer}</answer>"

## Code for getting pascalpart images and object/object parts 
sys.path.append('/home/ksmehrab/AttentionGrounding/Baselines/Models/GLAMM')
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

## Change 
save_dir = "/data/VLMGroundingProject/BaselineResults/PascalPart/SegZero"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

already_done = [os.path.splitext(f)[0] for f in os.listdir(save_dir)]

req_filenames = [f for f in val_filenames if f not in already_done]

print(f'Already done on {len(already_done)} images. Running on remaining {len(req_filenames)}...')

def combine_masks(masks):
    combined_mask = masks[0]
    for i in range(1, len(masks)):
        combined_mask = combined_mask | masks[i]
    return combined_mask

all_object_outputs = []
all_parts_outputs = []
for filename in tqdm(req_filenames):
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
                    "text": QUESTION_TEMPLATE.format(
                        Question=args_text.lower().strip("."),
                        Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                    )    
                }
            ]
            }]
            messages.append(message)
        
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
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for id_idx in range(len(batch_output_text)):
                if id_idx == 0:
                    all_outputs = all_object_outputs
                else:
                    all_outputs = all_parts_outputs
                try:
                    bboxes, points, think = extract_bbox_points_think(
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
                mask_save_dir = os.path.join(save_dir, filename)
                os.makedirs(mask_save_dir, exist_ok=True)
                mask_save_filename = seg_list[id_idx].replace("'s", "").replace(" ", "_") + ".npy"
                mask_save_filepath = os.path.join(mask_save_dir, mask_save_filename)
                np.save(mask_save_filepath, mask_all)


                # Create visualization
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

save_to_json(os.path.join("objects_results.json"), all_object_outputs)
save_to_json(os.path.join("parts_results.json"), all_parts_outputs)
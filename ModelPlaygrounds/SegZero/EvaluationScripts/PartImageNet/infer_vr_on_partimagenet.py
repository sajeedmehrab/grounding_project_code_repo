import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/GitRepoLatest/Seg-Zero')
sys.path.append('/home/ksmehrab/AttentionGrounding/Baselines/Datasets')

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sam2.sam2_image_predictor import SAM2ImagePredictor
import re

from partimagenet_dataset import PartImageNetDataset

# Configuration

reasoning_model_path = "/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/GitRepoLatest/Seg-Zero/pretrained_models/VisionReasoner-7B"

save_dir = "/data/VLMGroundingProject/BaselineResults/PartImageNet/VR"
os.makedirs(save_dir, exist_ok=True)

debug = False  # Set to True to save visualizations and run on 5 samples only
skip_existing = False  # Set to True to skip already processed images, mainly for debugging

chunk_id = 1  # Change this to process different chunks

max_response_length = 2000

batch_size = 32

test_images_dir = "/data/VLMGroundingProject/Datasets/PartImageNet/test"
test_json_path = "/data/VLMGroundingProject/Datasets/PartImageNet/test.json"

partimagenet_dataset = PartImageNetDataset(
    json_path=test_json_path,
    images_dir=test_images_dir
)

num_chunks = 2
num_annotations = partimagenet_dataset.num_annotations
chunk_size = num_annotations // num_chunks

chunk_start_index = chunk_id * chunk_size
chunk_end_index = (chunk_id + 1) * chunk_size if chunk_id < num_chunks - 1 else num_annotations


QUESTION_TEMPLATE = \
   "Please find \"{Question}\" with bboxs and points." \
    "Compare the difference between object(s) and find the most closely matched object(s)." \
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
    "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
    "i.e., <think> thinking process here </think>" \
    "<answer>{Answer}</answer>"


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
    """Compute intersection and union between two masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0, 0
    return intersection, union

# visualization code from chatgpt
def visualize_bbox_comparison(image, first_bboxes, final_bboxes, query_text):
    """
    Visualize side-by-side comparison of first_answer and final_answer bounding boxes.
    
    Args:
        image: PIL Image object
        first_bboxes: List of bboxes from first_answer [[x1,y1,x2,y2], ...]
        final_bboxes: List of bboxes from final_answer [[x1,y1,x2,y2], ...]
        query_text: Query text for the image
        save_path: Path to save the visualization
    """
    plt.figure()
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
    plt.show()
    # plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    # print(f"Saved box comparison to {save_path}")

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
segmentation_model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")

all_results = []
resize_size = 840

if debug:
    chunk_end_index = 5  # do first 5 annotations in debug mode

for batch_start_index in tqdm(range(chunk_start_index, chunk_end_index, batch_size)):
    # prepare batch data
    batch_messages = []
    batch_metadata = []

    batch_end_index = min(batch_start_index + batch_size, chunk_end_index)
    for i in range(batch_start_index, batch_end_index):
        annotation = partimagenet_dataset.get_annotation(i)

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

        if skip_existing:
            # skip ones we have already saved 
            image_name_prefix = os.path.splitext(image_name)[0]
            if os.path.exists(os.path.join(save_dir, image_name_prefix)):
                print(f"Skipping {image_name} in debug mode")
                continue

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
                    "text": QUESTION_TEMPLATE.format(
                        Question=query_text.lower().strip("."),
                        Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                    )
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
            "query_text": query_text
        })

    if skip_existing and len(batch_messages) == 0:
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
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    
    # process each item in the batch
    for output_text, metadata in zip(output_texts, batch_metadata):
        if debug:
            print(f"Output for {metadata['image_name']}:\n{output_text}\n{'-'*50}\n")
        
        try:
            bboxes, points, parsed_output = extract_bbox_points_think(
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
                    visualize_bbox_comparison(metadata['image'], first_bboxes, bboxes, metadata['query_text']) # viz_save_path
                    
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
                "iou": float(intersection/union) if union > 0 else 0.0
            })
        
        except Exception as e:
            print(f"Error processing {metadata['image_name']}: {e}")
            all_results.append({
                "image_id": metadata['image_name'],
                "query": metadata['query_text'],
                "error": str(e),
                "intersection": 0,
                "union": int(gt_mask.sum()),
                "iou": 0.0
            })
    
    # Clean GPU memory
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

# Save results
with open(os.path.join(save_dir, f"part_results_{chunk_id}.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print(f"results saved to {save_dir}")

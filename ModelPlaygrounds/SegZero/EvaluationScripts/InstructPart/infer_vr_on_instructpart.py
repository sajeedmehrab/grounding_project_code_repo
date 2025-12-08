import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/GitRepoLatest/Seg-Zero')

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sam2.sam2_image_predictor import SAM2ImagePredictor
import re

# Configuration

reasoning_model_path = "/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/GitRepoLatest/Seg-Zero/pretrained_models/VisionReasoner-7B"

save_dir = "/data/VLMGroundingProject/BaselineResults/InstructPart/VR"
os.makedirs(save_dir, exist_ok=True)

max_response_length = 2000

batch_size = 16

debug = False  # Set to True to save visualizations

instructpart_test_dir = "/data/VLMGroundingProject/Datasets/InstructPart/test"
masks_dir = os.path.join(instructpart_test_dir, "masks")
images_dir = os.path.join(instructpart_test_dir, "images")
image_names = os.listdir(images_dir)

QUESTION_TEMPLATE = \
    "Please find \"{Question}\" with bboxs and points." \
    "Compare the difference between object(s) and find the most closely matched object(s)." \
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
    "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
    "i.e., <think> thinking process here </think>" \
    "<answer>{Answer}</answer>"

def load_mask_and_parse_name(image_name, masks_dir):
    """
    Load mask and extract object and part names from filename.
    
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
def visualize_bbox_comparison(image, final_bboxes, query_text):
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
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    # print("debugfigure")
    
    # Final answer boxes
    axes.imshow(image)
    axes.set_title(f'Final Answer Boxes\n{query_text}', fontsize=14, fontweight='bold')
    for idx, box in enumerate(final_bboxes):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=3, edgecolor='red', facecolor='none')
        axes.add_patch(rect)
        axes.text(x1, y1-5, f"Final_{idx}", color='red', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes.axis('off')
    axes.text(0.5, -0.05, f'{len(final_bboxes)} boxes', 
                transform=axes.transAxes, ha='center', fontsize=12)
    
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

# Get already processed images
already_done = set(os.listdir(save_dir))
req_filenames = [img_name for img_name in image_names if img_name not in already_done]

print(f'Already done: {len(already_done)}. Processing remaining: {len(req_filenames)}...')

all_results = []
resize_size = 840

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
        x_factor, y_factor = original_width/resize_size, original_height/resize_size
        
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
            "object_name": object_name,
            "part_name": part_name,
            "query_text": query_text
        })

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
                    # viz_save_path = os.path.join(save_dir, f"{os.path.splitext(image_name)[0]}_boxes_comparison.png")
                    visualize_bbox_comparison(metadata['image'], bboxes, metadata['query_text']) # viz_save_path
                    
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
                "union": int(gt_mask.sum()),
                "iou": 0.0
            })
    
    # Clean GPU memory
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

# Save results
with open(os.path.join(save_dir, "part_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print(f"results saved to {save_dir}")


import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/Baselines/Models/GroundingDino')
sys.path.append('/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/EvaluationScripts')

from run_groundingdino import load_grdino_model, run_groundingdino, filter_boxes
from eval_base import remove_overlapping_boxes

from tqdm import tqdm
import os
import numpy as np
from PIL import Image   

instructpart_test_dir = "/data/VLMGroundingProject/Datasets/InstructPart/train1800"
masks_dir = os.path.join(instructpart_test_dir, "masks")
images_dir = os.path.join(instructpart_test_dir, "images")
val_filenames = os.listdir(images_dir)

box_threshold = 0.35
text_threshold = 0.25

# load model
model = load_grdino_model()

results = []

# run loop 
for filename in tqdm(val_filenames):
    img_filepath = os.path.join(images_dir, filename)

    #get basename
    basename = os.path.splitext(filename)[0]
    
    # parse the filename: image_id-object_name-part_name
    filename_parts = basename.split('-')
    object_name = filename_parts[-2]
   
    boxes, confs, phrases = run_groundingdino(
        model=model,
        image_path=img_filepath,
        text_prompt=object_name,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    if len(boxes) > 1:
        remove_overlapping_boxes(boxes)
    
    if len(boxes) == 0:
        print(f'No boxes found for {filename} with prompt {object_name}')
        # continue
        boxes = [[0,0,0,0]]
        confs = [0.0]
        phrases = ["no_box_found"]

    # one image has only one object, so we can save one result per image
    result = {
        'image_filename': filename,
        'object_name': object_name,
        'boxes': boxes,
        'conf_scores': confs,
        'pred_phrases': phrases
    }
    results.append(result)

# save all results to a json file
import json
save_path = "/data/VLMGroundingProject/Datasets/InstructPart/train1800/groundingdino_object_boxes_on_instrucpart_train.json"

with open(save_path, 'w') as f:
    json.dump(results, f)
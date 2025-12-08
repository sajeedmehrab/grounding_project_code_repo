import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/Baselines/Models/GroundingDino')
sys.path.append('/home/ksmehrab/AttentionGrounding/Baselines/Datasets')

from run_groundingdino import load_grdino_model, run_groundingdino, filter_boxes
from pascalpart import get_pascalpart_classes
from utils import read_txt_file, save_to_json
from tqdm import tqdm
import os

pascal_image_dir = "/data/Pascal_VOC_2012/VOCdevkit/VOC2012/JPEGImages" # 17125 images
annotations_path= "/data/PartSegmentationDatasets/PascalPart/Annotations_Part"
val_filepath = "/data/PartSegmentationDatasets/PascalPart/val.txt" # 925 images. File contains just the file prefix. Add .jpg extension for images, and .mat extension for annotations
val_filenames = read_txt_file(val_filepath)

model_name = 'grounding_dino'
box_threshold = 0.35
text_threshold = 0.25

save_dir = "/data/VLMGroundingProject/BaselineResults/PascalPart/GroundingDino"
already_done = [os.path.splitext(f)[0] for f in os.listdir(save_dir)]
req_filenames = [f for f in val_filenames if f not in already_done]

print(f'Already done on {len(already_done)} images. Running on remaining {len(req_filenames)}...')

def run_pascapart_on_groundingdino(model):
    for filename in tqdm(req_filenames):
        img_filepath = os.path.join(pascal_image_dir, filename+'.jpg')

        annot_filename = filename + '.mat'
        classes_to_detect = get_pascalpart_classes(annot_filename, annotations_path) # This is a dictionary in this format {object: [list of parts]}
        
        file_result = {}
        for object, parts in classes_to_detect.items():
            text_prompts = [object] + [f'{object} {part}' for part in parts]
            for i, text_prompt in enumerate(text_prompts):
                boxes, confs, phrases = run_groundingdino(
                    model=model,
                    image_path=img_filepath,
                    text_prompt=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )

                boxes, confs, phrases = filter_boxes(
                    xyxy_boxes=boxes,
                    conf_scores=confs,
                    pred_phrases=phrases,
                    choose_boxes="all", # we choose all boxes because some objects and parts can be visible multiple times, and we want all instances 
                    conf_threshold=0.25
                )

                if i == 0: # text_prompt is object name, like "person"
                    file_result[object] = {}
                    file_result[object]['parts'] = {}
                    file_result[object]['object_boxes'] = (boxes, confs, phrases)
                else: # text prompt is part name, like "person left hand"
                    file_result[object]['parts'][parts[i-1]] = (boxes, confs, phrases) # parts[i-1] would give the part name, without the object attached to it 
        save_to_json(os.path.join(save_dir, f'{filename}.json'), file_result)
        
if model_name == 'grounding_dino':
    model = load_grdino_model()
    run_pascapart_on_groundingdino(model)

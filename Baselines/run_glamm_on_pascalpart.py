import sys
sys.path.append('/home/ksmehrab/AttentionGrounding/Baselines/Models/GLAMM')
sys.path.append('/home/ksmehrab/AttentionGrounding/Baselines/Datasets')

from run_glamm import segment_image_using_glamm
from pascalpart import get_pascalpart_classes
from utils import read_txt_file, save_to_json
from tqdm import tqdm
import os

pascal_image_dir = "/data/Pascal_VOC_2012/VOCdevkit/VOC2012/JPEGImages" # 17125 images
annotations_path= "/data/PartSegmentationDatasets/PascalPart/Annotations_Part"
val_filepath = "/data/PartSegmentationDatasets/PascalPart/val.txt" # 925 images. File contains just the file prefix. Add .jpg extension for images, and .mat extension for annotations
val_filenames = read_txt_file(val_filepath)

model_name = 'glamm'

save_dir = "/data/VLMGroundingProject/BaselineResults/PascalPart/GLAMM"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

already_done = [os.path.splitext(f)[0] for f in os.listdir(save_dir)]

req_filenames = [f for f in val_filenames if f not in already_done]

print(f'Already done on {len(already_done)} images. Running on remaining {len(req_filenames)}...')

def run_pascapart_on_glamm():
    PROMPT_TEMPLATE = "Can you segment the <obj> in this image?"
    for filename in tqdm(req_filenames):
        img_filepath = os.path.join(pascal_image_dir, filename+'.jpg')

        annot_filename = filename + '.mat'
        classes_to_detect = get_pascalpart_classes(annot_filename, annotations_path) # This is a dictionary in this format {object: [list of parts]}
        
        file_result = {}
        for object, parts in classes_to_detect.items():
            # TODO: need to add a prompt template here 
            seg_list = [object] + [f'{object} {part}' for part in parts]
            text_prompts = [PROMPT_TEMPLATE.replace('<obj>', seg) for seg in seg_list]

            for i, text_prompt in enumerate(text_prompts):
                seg_mask = segment_image_using_glamm(
                    input_image_path=img_filepath,
                    prompt_text=text_prompt
                )

                save_dir_for_example = os.path.join(save_dir, filename)
                os.makedirs(save_dir_for_example, exist_ok=True)

                if i == 0: # text_prompt is object name, like "person"
                    object_underscore = object.replace(' ', '_')
                    seg_mask.save(os.path.join(save_dir_for_example, f'{object_underscore}.png'))
                else: # text prompt is part name, like "person left hand"
                    part_underscore = parts[i-1].replace(' ', '_')
                    seg_mask.save(os.path.join(save_dir_for_example, f'{object_underscore}_{part_underscore}.png'))
            
if model_name == 'glamm':
    run_pascapart_on_glamm()

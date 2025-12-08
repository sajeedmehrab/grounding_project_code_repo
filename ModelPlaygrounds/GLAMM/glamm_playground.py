# %%
# use env glamm
import sys
sys.path.append('/home/ksmehrab/GLAMM/groundingLMM')

# %%
import cv2
import random
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.generate_utils import center_crop, create_feathered_mask
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from tools.markdown_utils import (markdown_default, examples, title, description, article, process_markdown, colors,
                                  draw_bbox, ImageSketcher)

# %%
from app_modified import(
    inference
)

# %%

# ------------------------------------
# Hard‑code your inputs here:
# ------------------------------------
input_image = "/data/PartSegmentationDatasets/PascalPart/examples/2008_000652.jpg"
prompt_text = "Please segment the baby in the picture."
all_inputs = {"boxes": [], "image": input_image}
follow_up = False
generate = False

# ------------------------------------
# Run inference & grab outputs
# ------------------------------------
output_image, markdown_out, seg_mask, attentions = inference(
     prompt_text, all_inputs, follow_up, generate, return_attentions=True
)

# # %%
# import matplotlib.pyplot as plt
# plt.imshow(seg_mask)

# # %%
# # ------------------------------------
# # Save segmentation / overlay to disk
# # ------------------------------------
# from PIL import Image
# import numpy as np

# if isinstance(output_image, np.ndarray):
#         result_img = Image.fromarray(output_image)
# else:
#       result_img = output_image  # PIL.Image

# out_path = "segmentation_result.png"
# result_img.save(out_path)
# print(f"Saved segmentation result to: {out_path}")
# # %%

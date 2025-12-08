# %%
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
def segment_image_using_glamm(input_image_path, prompt_text):
    """
    Segments an image based on the given prompt.

    Args:
        input_image_path (str): Path to the input image.
        prompt_text (str): Prompt describing the segmentation task.

    Returns:
        np.ndarray: Segmentation mask.
    """
    all_inputs = {"boxes": [], "image": input_image_path}
    follow_up = False
    generate = False

    # Run inference
    output_image, markdown_out, seg_mask = inference(
        prompt_text, all_inputs, follow_up, generate
    )

    return seg_mask

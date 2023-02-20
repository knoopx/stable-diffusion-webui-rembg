from enum import Enum

import cv2
from modules.processing import StableDiffusionProcessingImg2Img, process_images
from PIL import Image, ImageChops
from rembg import remove, new_session
from typing import Optional
import gc
import gradio as gr
import modules.scripts as scripts
import numpy as np
import torch


RED_COLOR = "#FF0000"
GREEN_COLOR = "#00FF00"


class BrushMaskMode(Enum):
    Add = 0
    Substract = 1
    Discard = 2


def apply_brush_mask(mask: Image.Image, brush_mask: Optional[Image.Image], brush_mask_mode: BrushMaskMode):
    if brush_mask:
        if brush_mask_mode == BrushMaskMode.Add:
            return ImageChops.add(mask, brush_mask)
        elif brush_mask_mode == BrushMaskMode.Substract:
            return ImageChops.subtract(mask, brush_mask)

    return mask


def dilate_mask(mask, dilation_factor, iter=1):
    kernel = np.ones((dilation_factor, dilation_factor), np.uint8)
    return Image.fromarray(cv2.dilate(np.array(mask), kernel, iterations=iter))


CLOTH_SEG_CHOICES = ['Top', 'Bottom', "Combined"]


class Script(scripts.Script):
    def title(self):
        return "rembg"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        with gr.Row():
            model_type = gr.Dropdown(
                label="Model",
                choices=[
                    'u2net',
                    'u2netp',
                    'u2net_cloth_seg',
                    'u2net_human_seg',
                    'silueta'
                ],
                value='u2net_human_seg',
                type="value",
            )

        with gr.Row():
            brush_mask_mode = gr.Radio(
                label="Brush Mask",
                choices=['Add', 'Substract', 'Discard'],
                value='Add',
                type="index"
            )

            dilate = gr.Slider(
                label="Dilate",
                minimum=0,
                maximum=255,
                value=0,
                step=1
            )

            cloth_seg = gr.CheckboxGroup(
                label="Cloth Seg",
                choices=CLOTH_SEG_CHOICES,
                value=CLOTH_SEG_CHOICES,
            )

            debug = gr.Checkbox(
                label="Debug",
                value=False,
            )

        return model_type, dilate, cloth_seg, brush_mask_mode, debug

    def run(self, pipeline: StableDiffusionProcessingImg2Img, model, dilate, cloth_seg, brush_mask_mode, debug):
        session = new_session(model)
        image = pipeline.init_images[0]

        mask = remove(
            image,
            session=session,
            alpha_matting=True,
            only_mask=True
        )

        debug_masks = []

        if model == 'u2net_cloth_seg':
            cloth_seg_masks = np.split(np.array(mask), 3, axis=0)
            if debug:
                debug_masks.extend([Image.fromarray(m)
                                   for m in cloth_seg_masks])

            cloth_seg_masks = [cloth_seg_masks[i]
                               for i in range(3) if CLOTH_SEG_CHOICES[i] in cloth_seg]

            mask = Image.fromarray(
                np.sum(cloth_seg_masks, axis=0).astype(np.uint8)).convert("L")

        brush_mask = pipeline.image_mask
        if debug:
            debug_masks.append(brush_mask.convert('L'))

        final_mask = apply_brush_mask(mask, brush_mask, BrushMaskMode(brush_mask_mode))

        if dilate > 0:
            final_mask = dilate_mask(final_mask, dilate)

        pipeline.image_mask = final_mask
        pipeline.mask_for_overlay = final_mask
        pipeline.latent_mask = None

        processed = process_images(pipeline)

        if debug:
            processed.images.extend(debug_masks)
            processed.images.append(final_mask)

        del session
        gc.collect()
        torch.cuda.empty_cache()

        return processed

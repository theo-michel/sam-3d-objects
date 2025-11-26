import os
import fal_client
from fastapi import HTTPException
from transformers import pipeline
from PIL import Image
import torch
import numpy as np


def get_bbox_from_mask(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """
    Compute the tight bounding box around white (non-zero) pixels in the mask.
    Returns (x, y, height, width) in pixel coordinates.
    If the mask has no white pixels, returns (0, 0, 0, 0).
    """
    m = mask.detach().cpu().numpy()
    # Squeeze singleton channel if present (e.g., 1xHxW or HxWx1)
    if m.ndim == 3 and 1 in m.shape:
        m = np.squeeze(m)
    m = m.astype(np.uint8)
    # Consider any non-zero as white
    ys, xs = np.where(m > 0)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)
    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())
    x = x_min
    y = y_min
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    return (x, y, height, width)


def segment_image(image_url: str, prompt: str = "Pickable object") -> dict:
    """
    Segments an image using SAM3 via fal.ai with text prompt.
    """
    if not os.environ.get("FAL_KEY"):
        raise HTTPException(
            status_code=500, detail="FAL_KEY environment variable not set"
        )

    print(f"Running SAM3 segmentation with prompt '{prompt}'...")
    try:
        segmentation_result = fal_client.subscribe(
            "fal-ai/sam-3/image-rle",
            arguments={
                "image_url": image_url,
                "text_prompt": prompt,
                "include_scores": True,
                "include_boxes": True,
                "return_multiple_masks": True,
                "max_masks": 10,
            },
        )
        return segmentation_result
    except Exception as e:
        print(f"Error during segmentation: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

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

generator = pipeline("mask-generation", model="facebook/sam2-hiera-large", device=0)
def segment_img_local_sam2(image: Image.Image) -> tuple[list[torch.Tensor], list[tuple[int, int, int, int]]]:
    """
    Segments an image using SAM2 locally.
    Returns a tuple of lists: (masks, bounding boxes) : [torch.Tensor, tuple[int, int, int, int]]
    
    """
    outputs = generator(image, points_per_batch=64)
    return (outputs["masks"], [get_bbox_from_mask(mask) for mask in outputs["masks"]])

def segment_image(image_url: str):
    """
    Segments an image using SAM2 via fal.ai.
    """
    if not os.environ.get("FAL_KEY"):
        raise HTTPException(status_code=500, detail="FAL_KEY environment variable not set")

    print("Running SAM2 segmentation...")
    try:
        segmentation_result = fal_client.subscribe(
            "fal-ai/sam2/auto-segment",
            arguments={
                "image_url": image_url,
                "points_per_side": 64,
                "pred_iou_thresh": 0.8,
                "stability_score_thresh": 0.9,
                "min_mask_region_area": 120,
            }
        )
        return segmentation_result
    except Exception as e:
        print(f"Error during segmentation: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

def upload_image_to_fal(image_path: str) -> str:
    """
    Uploads an image to fal.ai and returns the URL.
    """
    print("Uploading image to fal.ai...")
    try:
        return fal_client.upload_file(image_path)
    except Exception as e:
        print(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

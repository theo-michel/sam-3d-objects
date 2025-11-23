import os
import fal_client
from fastapi import HTTPException

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

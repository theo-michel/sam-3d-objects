import os
import fal_client
from fastapi import HTTPException
from utils import download_image

def segment_image_with_moondream(image_url: str, prompt: str = "Pickable objects") -> list[dict]:
    """
    Extracts objects using Moondream (detect) and then segments them using SAM3.
    """
    if not os.environ.get("FAL_KEY"):
        raise HTTPException(status_code=500, detail="FAL_KEY environment variable not set")

    print(f"Detecting objects with Moondream using prompt '{prompt}'...")
    try:
        moondream_result = fal_client.subscribe(
            "fal-ai/moondream3-preview/detect",
            arguments={
                "image_url": image_url,
                "prompt": prompt
            }
        )
    except Exception as e:
        print(f"Error during Moondream detection: {e}")
        raise HTTPException(status_code=500, detail=f"Moondream detection failed: {str(e)}")

    objects = moondream_result.get("objects", [])
    if not objects:
        print("No objects found by Moondream.")
        return []

    print(f"Found {len(objects)} objects. Fetching image dimensions...")
    
    # We need image dimensions to convert normalized coordinates to pixels
    image = download_image(image_url)
    if not image:
        raise HTTPException(status_code=500, detail="Failed to download image for dimension check")
    
    width, height = image.size
    
    results = []
    print(f"Segmenting {len(objects)} objects with SAM3...")
    
    for i, obj in enumerate(objects):
        # Moondream returns normalized coordinates
        # x_min, y_min, x_max, y_max
        
        x_min_px = int(obj["x_min"] * width)
        y_min_px = int(obj["y_min"] * height)
        x_max_px = int(obj["x_max"] * width)
        y_max_px = int(obj["y_max"] * height)
        
        # Ensure coordinates are within bounds
        x_min_px = max(0, min(x_min_px, width - 1))
        y_min_px = max(0, min(y_min_px, height - 1))
        x_max_px = max(0, min(x_max_px, width - 1))
        y_max_px = max(0, min(y_max_px, height - 1))
        
        # Ensure min < max
        if x_min_px >= x_max_px: x_max_px = x_min_px + 1
        if y_min_px >= y_max_px: y_max_px = y_min_px + 1
        
        box_prompt = {
            "x_min": x_min_px,
            "y_min": y_min_px,
            "x_max": x_max_px,
            "y_max": y_max_px
        }
        
        print(f"Processing object {i+1}/{len(objects)}: {box_prompt}")
        
        try:
            sam_result = fal_client.subscribe(
                "fal-ai/sam-3/image-rle",
                arguments={
                    "image_url": image_url,
                    "box_prompts": [box_prompt],
                    "include_scores": True,
                    "include_boxes": True,
                    "return_multiple_masks": False 
                }
            )
            # Add metadata about which object generated this
            sam_result["source_object"] = obj
            results.append(sam_result)
        except Exception as e:
            print(f"Error segmenting object {i}: {e}")
            # Continue with other objects instead of failing everything
            continue
            
    return results

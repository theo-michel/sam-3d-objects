import os
import fal_client
from fastapi import HTTPException
from utils import download_image
from PIL import Image
import numpy as np

# Try to import ultralytics for local YOLO
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

def segment_image_with_yolo(image_url: str, prompt: str = "Pickable object", local_image_path: str = None) -> list[dict]:
    """
    Extracts objects using YOLOv12 (locally) and then segments them using SAM3.
    
    Args:
        image_url: URL of the image (for SAM3)
        prompt: Not used by YOLO (detects all COCO classes)
        local_image_path: Optional local path to image for YOLO detection
    """
    if not os.environ.get("FAL_KEY"):
        raise HTTPException(status_code=500, detail="FAL_KEY environment variable not set")

    if not HAS_ULTRALYTICS:
        raise HTTPException(status_code=500, detail="ultralytics package not installed. Run: pip install ultralytics")

    print(f"Detecting objects with YOLOv12 (local)...")
    
    # Download image for local YOLO processing if not provided
    if local_image_path is None:
        image = download_image(image_url)
        if not image:
            raise HTTPException(status_code=500, detail="Failed to download image")
        # Save temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            local_image_path = tmp.name
    else:
        image = Image.open(local_image_path)
    
    width, height = image.size
    
    try:
        # Load YOLOv12 model
        # YOLOv12 models: yolo12n.pt, yolo12s.pt, yolo12m.pt, yolo12l.pt, yolo12x.pt
        # Using nano version for speed
        model = YOLO('yolo12n.pt')
        
        # Run inference
        results = model(local_image_path, verbose=False)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "score": conf,
                    "class": cls,
                    "class_name": model.names[cls]
                })
        
        print(f"Found {len(detections)} objects: {[d['class_name'] for d in detections]}")
        
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        raise HTTPException(status_code=500, detail=f"YOLO detection failed: {str(e)}")
    
    if not detections:
        print("No objects found by YOLO.")
        return []
    
    sam_results = []
    print(f"Segmenting {len(detections)} objects with SAM3...")
    
    for i, det in enumerate(detections):
        box = det["box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        
        # Ensure bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        if x1 >= x2: x2 = x1 + 1
        if y1 >= y2: y2 = y1 + 1
        
        box_prompt = {
            "x_min": x1,
            "y_min": y1,
            "x_max": x2,
            "y_max": y2
        }
        
        print(f"Processing {det['class_name']} {i+1}/{len(detections)}: {box_prompt}")
        
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
            
            # Add metadata
            norm_box = {
                "x_min": x1 / width,
                "y_min": y1 / height,
                "x_max": x2 / width,
                "y_max": y2 / height
            }
            sam_result["source_object"] = norm_box
            sam_result["yolo_class"] = det["class_name"]
            sam_result["yolo_confidence"] = det["score"]
            sam_results.append(sam_result)
        except Exception as e:
            print(f"Error segmenting object {i}: {e}")
            continue
    
    # Clean up temp file if created
    if local_image_path and local_image_path.startswith('/tmp'):
        try:
            os.unlink(local_image_path)
        except:
            pass
            
    return sam_results

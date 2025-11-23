import shutil
import tempfile

import numpy as np
import asyncio
import concurrent.futures
from pathlib import Path
from PIL import Image
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse

from services.segmentation import segment_image, upload_image_to_fal
from services.reconstruction import reconstruct_object, get_model
from utils import download_image

app = FastAPI(title="SAM 3D Objects API")

# ThreadPoolExecutor for blocking IO/CPU tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@app.on_event("startup")
async def startup_event():
    print("Warming up model...")
    # Run in executor to avoid blocking startup
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(executor, get_model)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")

def resize_image_if_needed(image_path: str, max_size: int = 1024) -> str:
    """
    Resizes image if larger than max_size. Overwrites the file.
    Returns the path.
    """
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if w > max_size or h > max_size:
                # Calculate new size maintaining aspect ratio
                ratio = min(max_size / w, max_size / h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                img.save(image_path)
                print(f"Resized image from {w}x{h} to {new_size}")
    except Exception as e:
        print(f"Error resizing image: {e}")
    return image_path

@app.post("/image-to-3d")
async def image_to_3d(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    # quality: str = Query("balanced", description="Quality level: 'fast' (fewer steps), 'balanced' (default), 'high' (more steps)"),
):
    """
    Takes an image, segments it using SAM2 (via fal.ai), and reconstructs a 3D model for the first detected object.
    Returns a PLY file (Gaussian Splat).
    """
    # # Determine inference steps based on quality
    # if quality == "fast":
    #     stage1_steps = 30
    #     stage2_steps = 30
    # elif quality == "high":
    #     stage1_steps = 75
    #     stage2_steps = 75
    # else:  # balanced
    #     stage1_steps = 50
    #     stage2_steps = 50
    print(f"Image: {image}")
    print(f"Image type: {type(image)}")

    # Create temp dir for this request
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Schedule cleanup
    background_tasks.add_task(shutil.rmtree, temp_dir)
    
    try:
        # Save input image
        input_image_path = temp_path / f"input_{image.filename}"
        with open(input_image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
            
        pil_image = Image.open(input_image_path)
        
            
        # Resize image if needed (runs in thread pool)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, resize_image_if_needed, str(input_image_path))
            
        # Upload to fal.ai
        image_url = await loop.run_in_executor(executor, upload_image_to_fal, str(input_image_path))
        print(f"Image URL: {image_url}")
        
        # Run SAM2 segmentation
        segmentation_result = await loop.run_in_executor(executor, segment_image, image_url)
        print(f"Segmentation result: {segmentation_result}")
        
        masks, bboxes = await loop.run_in_executor(executor, segment_img_local_sam2, pil_image)
        print(f"Masks: {masks}")
        print(f"Bboxes: {bboxes}")
        
        # individual_masks = masks
        # individual_masks = segmentation_result.get("individual_masks", [])
        # if not individual_masks:
        #     raise HTTPException(status_code=400, detail="No objects detected in the image")
            
        # Process only the first object
        # mask_info = individual_masks[0]
        # mask_url = mask_info.get("url")
        # if not mask_url:
        #      raise HTTPException(status_code=500, detail="Invalid mask URL")
            
        # Download mask
        # mask_pil = await loop.run_in_executor(executor, download_image, mask_url)
        # if mask_pil is None:
        #      raise HTTPException(status_code=500, detail="Failed to download mask")
            
        # mask_np = np.array(mask_pil)
        mask_np = np.array(masks[0])
        
        # Run inference
        # Only generating PLY, so no mesh postprocess needed
        output = await loop.run_in_executor(
            executor,
            reconstruct_object,
            str(input_image_path), 
            mask_np, 
            42, # seed
            False, # with_mesh_postprocess
            False, # with_texture_baking
        )
        
        # Save PLY (Gaussian Splat)
        gs = output.get("gs")
        rotation = output.get("rotation")
        translation = output.get("translation")
        scale = output.get("scale")
        print(f"Rotation: {rotation}")
        print(f"Translation: {translation}")
        print(f"Scale: {scale}")
        
        if gs is None:
            raise HTTPException(status_code=500, detail="Failed to generate Gaussian Splat")
        
        try:
            print(f"Output all keys: {output.keys()}")
            print(f"Output: {output}")
            print(f"Output type: {type(output)}")
            
            
        except Exception as e:
            print(f"Error printing output keys: {e}")
        
        output_filename = "reconstruction.ply"
        output_path = temp_path / output_filename
        gs.save_ply(str(output_path))

        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type="application/octet-stream"
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in image-to-3d: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7310)

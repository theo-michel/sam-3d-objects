import shutil
import tempfile
import zipfile
import numpy as np
from pathlib import Path
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse

from services.segmentation import segment_image, upload_image_to_fal
from services.reconstruction import reconstruct_object
from utils import download_image

app = FastAPI(title="SAM 3D Objects API")

@app.post("/image-to-3d")
async def image_to_3d(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    max_objects: int = Query(5, description="Maximum number of objects to reconstruct to avoid timeouts"),
):
    """
    Takes an image, segments it using SAM2 (via fal.ai), and reconstructs 3D meshes for detected objects.
    Returns a ZIP file containing the 3D meshes (GLB format).
    """
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
            
        # Upload to fal.ai
        image_url = upload_image_to_fal(str(input_image_path))
        
        # Run SAM2 segmentation
        segmentation_result = segment_image(image_url)
        
        individual_masks = segmentation_result.get("individual_masks", [])
        if not individual_masks:
            raise HTTPException(status_code=400, detail="No objects detected in the image")
            
        # Limit number of objects
        masks_to_process = individual_masks[:max_objects]
        print(f"Processing {len(masks_to_process)} objects (limit: {max_objects})...")
        
        results_dir = temp_path / "results"
        results_dir.mkdir()
        
        for i, mask_info in enumerate(masks_to_process):
            mask_url = mask_info.get("url")
            if not mask_url:
                continue
                
            print(f"Processing object {i+1}/{len(masks_to_process)}...")
            
            # Download mask
            mask_pil = download_image(mask_url)
            if mask_pil is None:
                print(f"Failed to download mask for object {i}")
                continue
                
            mask_np = np.array(mask_pil)
            
            # Run inference
            try:
                output = reconstruct_object(str(input_image_path), mask_np)
                
                # Save GLB
                glb = output.get("glb")
                if glb is not None:
                    output_filename = f"object_{i}.glb"
                    output_path = results_dir / output_filename
                    glb.export(str(output_path))
                else:
                    print(f"Warning: No GLB generated for object {i}")
                    
            except Exception as e:
                print(f"Error processing object {i}: {e}")
                continue

        # Zip results
        zip_filename = "reconstruction_results.zip"
        zip_path = temp_path / zip_filename
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in results_dir.glob("*.glb"):
                zipf.write(file, arcname=file.name)
                
        if not zip_path.exists() or zip_path.stat().st_size == 0:
             raise HTTPException(status_code=500, detail="Failed to generate any 3D models")

        return FileResponse(
            path=zip_path,
            filename=zip_filename,
            media_type="application/zip"
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in image-to-3d: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7310)

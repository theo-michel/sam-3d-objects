import shutil
import tempfile
import asyncio
import concurrent.futures
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from services.reconstruction import get_model
from services.image_to_3d import image_to_3d
from utils import resize_image_if_needed, zip_files, save_reconstructed_objects


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


@app.post("/image-to-3d")
async def image_to_3d_endpoint(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
):
    """
    Takes an image, segments it using YOLOv12 + SAM3, and reconstructs 3D models.
    Returns a ZIP file containing PLY files (Gaussian Splats) and transformation metadata.
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    background_tasks.add_task(shutil.rmtree, temp_dir)

    # Save input image
    input_image_path = temp_path / f"input_{image.filename}"
    with open(input_image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Resize if needed
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, resize_image_if_needed, str(input_image_path))

    # Run full image-to-3D pipeline (segmentation + reconstruction)
    result = await loop.run_in_executor(
        executor,
        image_to_3d,
        str(input_image_path),
        None,  # image_url will be uploaded inside
        42,  # seed
        False,  # with_mesh_postprocess
        False,  # with_texture_baking
    )

    objects = result["objects"]
    if not objects:
        raise HTTPException(
            status_code=400, detail="No objects detected or reconstructed in the image"
        )

    print(f"Reconstructed {len(objects)} objects")

    # Save PLY files and generate transformation metadata
    saved_files, transformations = save_reconstructed_objects(objects, str(temp_path))

    if not saved_files:
        raise HTTPException(
            status_code=500, detail="Failed to generate any 3D reconstructions"
        )

    # Save transformations JSON
    transformations_path = temp_path / "transformations.json"
    with open(str(transformations_path), "w") as f:
        json.dump(transformations, f, indent=2)
    saved_files.append(str(transformations_path))

    # Zip all files
    zip_path = temp_path / "reconstruction.zip"
    await loop.run_in_executor(executor, zip_files, saved_files, str(zip_path))

    return FileResponse(
        path=zip_path, filename="reconstruction.zip", media_type="application/zip"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7310)

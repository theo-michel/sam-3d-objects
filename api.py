import shutil
import tempfile
import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from services.reconstruction import get_model
from services.image_to_3d import image_to_3d
from utils import resize_image_if_needed, build_scene_glb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ThreadPoolExecutor for blocking IO/CPU tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


# Global flag to track if model should be loaded (for hot-reload support)
_model_loaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_loaded
    
    # Only load model if not already loaded (helps with hot-reload)
    # During uvicorn reload, this will be False, so model will load once per process
    if not _model_loaded:
        logger.info("Warming up model...")
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(executor, get_model)
            _model_loaded = True
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model on startup: {e}", exc_info=True)
    else:
        logger.info("Model already loaded, skipping warmup (hot-reload mode).")

    yield

    # Shutdown (if needed in the future)
    pass


app = FastAPI(title="SAM 3D Objects API", lifespan=lifespan)

# Simple health check endpoint
@app.get("/health")
async def health() -> dict:
    """Return a basic health status for the service."""
    return {"status": "ok"}


# Global exception handler to log all unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "error_type": type(exc).__name__,
        }
    )


@app.post("/image-to-3d")
async def image_to_3d_endpoint(
    image: UploadFile = File(...),
):
    """
    Takes an image, segments it using YOLOv12 + SAM3, and reconstructs 3D models.
    Returns a single zstd-compressed GLB file (scene.glb.zst) with all objects
    positioned correctly in the scene.
    
    Response headers:
        Content-Type: model/gltf-binary
        Content-Encoding: zstd
        Content-Disposition: attachment; filename="scene.glb.zst"
    """
    logger.info(f"Received image: {image.filename}, content_type: {image.content_type}")
    
    try:
        # Use TemporaryDirectory for automatic cleanup
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_image_path = temp_path / f"input_{image.filename}"
            
            # Save input image
            with open(input_image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
            
            # Resize if needed
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, resize_image_if_needed, str(input_image_path))

            # Run full image-to-3D pipeline (segmentation + reconstruction)
            logger.info("Starting image-to-3D pipeline...")
            result = await loop.run_in_executor(
                executor,
                image_to_3d,
                str(input_image_path),
                None,  # image_url will be uploaded inside
                42,  # seed
                False,  # with_mesh_postprocess
                False,  # with_texture_baking
            )

            objects = result.get("objects", []) if isinstance(result, dict) else []
            
            if not objects:
                logger.warning("No objects detected or reconstructed")
                raise HTTPException(
                    status_code=400, detail="No objects detected or reconstructed in the image"
                )

            logger.info(f"Reconstructed {len(objects)} objects")

            # Build scene GLB with all objects positioned correctly
            glb_data = await loop.run_in_executor(
                executor, build_scene_glb, objects
            )

            if not glb_data:
                logger.error("Failed to generate GLB")
                raise HTTPException(
                    status_code=500, detail="Failed to generate scene GLB"
                )

            return Response(
                content=glb_data,
                media_type="model/gltf-binary",
                headers={
                    "Content-Encoding": "zstd",
                    "Content-Disposition": 'attachment; filename="scene.glb.zst"'
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image-to-3D request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import os
    
    # For reload to work, uvicorn needs the app as an import string, not the object
    # This assumes the script is run from the sam-3d-objects directory
    # If run from workspace root, set APP_MODULE env var (e.g., "api:app")
    app_module = os.getenv("APP_MODULE", "api:app")
    

    uvicorn.run(
        app_module,
        host="0.0.0.0",
        port=7311,
        log_level="info",
        access_log=True,
        reload=True,
        # Only reload on Python file changes, exclude common non-code directories
        reload_includes=["*.py"],
        reload_excludes=[
            "*/__pycache__/*",
            "*/.*",
            "*/checkpoints/*",
            "*/tmp/*",
            "*/temp/*",
            "*.pyc",
            "*.pyo",
        ],
    )
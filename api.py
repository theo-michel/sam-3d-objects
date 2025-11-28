import shutil
import tempfile
import asyncio
import concurrent.futures
import json
import logging
import traceback
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
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
        print("Warming up model...")
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(executor, get_model)
            _model_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model on startup: {e}")
            traceback.print_exc()
    else:
        print("Model already loaded, skipping warmup (hot-reload mode).")

    yield

    # Shutdown (if needed in the future)
    pass


app = FastAPI(title="SAM 3D Objects API", lifespan=lifespan)


# Global exception handler to log all unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_traceback = traceback.format_exc()
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback:\n{error_traceback}")
    # Also print to stderr so it's visible in terminal
    print(f"\n{'='*80}", file=sys.stderr, flush=True)
    print(f"GLOBAL EXCEPTION HANDLER - Unhandled exception: {exc}", file=sys.stderr, flush=True)
    print(f"Error type: {type(exc).__name__}", file=sys.stderr, flush=True)
    print(f"{'='*80}", file=sys.stderr, flush=True)
    print(error_traceback, file=sys.stderr, flush=True)
    print(f"{'='*80}\n", file=sys.stderr, flush=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "error_type": type(exc).__name__,
            "traceback": error_traceback
        }
    )


@app.post("/image-to-3d")
async def image_to_3d_endpoint(
    background_tasks: BackgroundTasks,
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
    temp_dir = None
    debug_dir = None
    try:
        logger.info(f"Received image: {image.filename}, content_type: {image.content_type}")
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Save to debug directory instead of deleting
        debug_base = Path(__file__).parent / "debug_outputs"
        debug_base.mkdir(exist_ok=True)
        import time
        debug_dir = debug_base / f"run_{int(time.time())}_{image.filename}"
        debug_dir.mkdir(exist_ok=True)
        logger.info(f"Saving debug output to: {debug_dir}")
        
        # Don't delete temp dir - we'll copy it to debug location

        # Save input image
        input_image_path = temp_path / f"input_{image.filename}"
        logger.info(f"Saving input image to: {input_image_path}")
        with open(input_image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        logger.info(f"Image saved successfully, size: {input_image_path.stat().st_size} bytes")

        # Resize if needed
        loop = asyncio.get_event_loop()
        logger.info("Resizing image if needed...")
        await loop.run_in_executor(executor, resize_image_if_needed, str(input_image_path))

        # Run full image-to-3D pipeline (segmentation + reconstruction)
        logger.info("Starting image-to-3D pipeline...")
        try:
            result = await loop.run_in_executor(
                executor,
                image_to_3d,
                str(input_image_path),
                None,  # image_url will be uploaded inside
                42,  # seed
                False,  # with_mesh_postprocess
                False,  # with_texture_baking
            )
        except Exception as pipeline_error:
            error_traceback = traceback.format_exc()
            logger.error(f"Pipeline error: {pipeline_error}")
            logger.error(f"Pipeline traceback:\n{error_traceback}")
            # Also print to stderr so it's visible in terminal
            print(f"\n{'='*80}", file=sys.stderr, flush=True)
            print(f"ERROR in image-to-3D pipeline: {pipeline_error}", file=sys.stderr, flush=True)
            print(f"{'='*80}", file=sys.stderr, flush=True)
            print(error_traceback, file=sys.stderr, flush=True)
            print(f"{'='*80}\n", file=sys.stderr, flush=True)
            raise

        logger.info(f"Pipeline completed. Result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
        objects = result.get("objects", []) if isinstance(result, dict) else []
        
        if not objects:
            logger.warning("No objects detected or reconstructed")
            raise HTTPException(
                status_code=400, detail="No objects detected or reconstructed in the image"
            )

        logger.info(f"Reconstructed {len(objects)} objects")

        # Build scene GLB with all objects positioned correctly
        logger.info("Building scene GLB...")
        glb_data = await loop.run_in_executor(
            executor, build_scene_glb, objects
        )
        logger.info(f"GLB created, size: {len(glb_data)} bytes (zstd compressed)")

        if not glb_data:
            logger.error("Failed to generate GLB")
            raise HTTPException(
                status_code=500, detail="Failed to generate scene GLB"
            )

        # Save GLB to file for debugging
        glb_path = temp_path / "scene.glb.zst"
        with open(str(glb_path), "wb") as f:
            f.write(glb_data)
        logger.info(f"GLB saved: {glb_path}, size: {glb_path.stat().st_size} bytes")

        # Copy all files to debug directory for debugging
        if debug_dir and temp_path.exists():
            try:
                logger.info(f"Copying files to debug directory: {debug_dir}")
                for item in temp_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, debug_dir / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, debug_dir / item.name, dirs_exist_ok=True)
                logger.info(f"Debug files saved to: {debug_dir}")
            except Exception as copy_error:
                logger.warning(f"Failed to copy files to debug directory: {copy_error}")
        
        return Response(
            content=glb_data,
            media_type="model/gltf-binary",
            headers={
                "Content-Encoding": "zstd",
                "Content-Disposition": 'attachment; filename="scene.glb.zst"'
            }
        )
    except HTTPException:
        # Re-raise HTTP exceptions (these are intentional)
        raise
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing image-to-3D request: {e}")
        logger.error(f"Traceback:\n{error_traceback}")
        # Also print to stderr so it's visible in terminal
        print(f"\n{'='*80}", file=sys.stderr, flush=True)
        print(f"ERROR in endpoint: {e}", file=sys.stderr, flush=True)
        print(f"Error type: {type(e).__name__}", file=sys.stderr, flush=True)
        print(f"{'='*80}", file=sys.stderr, flush=True)
        print(error_traceback, file=sys.stderr, flush=True)
        print(f"{'='*80}\n", file=sys.stderr, flush=True)
        # Copy to debug directory even on error
        if debug_dir and temp_dir and Path(temp_dir).exists():
            try:
                logger.info(f"Copying error files to debug directory: {debug_dir}")
                for item in Path(temp_dir).iterdir():
                    if item.is_file():
                        shutil.copy2(item, debug_dir / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, debug_dir / item.name, dirs_exist_ok=True)
            except Exception as copy_error:
                logger.warning(f"Failed to copy error files to debug directory: {copy_error}")
        # Don't delete temp directory - it's saved in debug location
        # Re-raise to trigger the global exception handler
        raise


if __name__ == "__main__":
    import os
    
    # Enable reload for development (set RELOAD=false to disable)
    # When reload is enabled, uvicorn watches for file changes and restarts
    # The model will reload only if the process restarts (not on code changes in this file)
    enable_reload = os.getenv("RELOAD", "true").lower() == "true"
    
    # For reload to work, uvicorn needs the app as an import string, not the object
    # This assumes the script is run from the sam-3d-objects directory
    # If run from workspace root, set APP_MODULE env var (e.g., "api:app")
    app_module = os.getenv("APP_MODULE", "api:app")
    
    if enable_reload:
        uvicorn.run(
            app_module,
            host="0.0.0.0",
            port=7310,
            log_level="info",
            access_log=True,
            reload=enable_reload,
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
    else:
        # Without reload, we can pass the app object directly
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=7310,
            log_level="info",
            access_log=True,
        )

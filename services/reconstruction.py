import sys
import os
import numpy as np
from fastapi import HTTPException
from typing import Optional

# Add notebook to sys.path to import inference
# We assume this file is in services/ and notebook is in ../notebook
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "notebook")
if NOTEBOOK_DIR not in sys.path:
    sys.path.append(NOTEBOOK_DIR)

# Global variables for lazy loading
Inference = None
load_image = None
load_mask = None
inference_model = None

def init_modules():
    global Inference, load_image, load_mask
    if Inference is None:
        try:
            from inference import Inference as Inf, load_image as li, load_mask as lm
            Inference = Inf
            load_image = li
            load_mask = lm
        except ImportError as e:
            print(f"Warning: Could not import inference modules: {e}")

def get_model():
    global inference_model
    init_modules()
    if Inference is None:
        raise HTTPException(status_code=500, detail="Inference module not available")
        
    if inference_model is None:
        tag = os.getenv("MODEL_TAG", "hf")
        config_path = os.path.join(PROJECT_ROOT, f"checkpoints/{tag}/pipeline.yaml")
        
        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found.")
            
        try:
            # We need to change cwd temporarily because Inference might rely on relative paths
            # Or we ensure config_path is absolute. 
            # The original code used relative path "checkpoints/{tag}/pipeline.yaml"
            # and Inference class does: config.workspace_dir = os.path.dirname(config_file)
            
            inference_model = Inference(config_path, compile=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {e}")
            
    return inference_model

def reconstruct_object(image_path: str, mask_np: np.ndarray, seed: int = 42):
    """
    Runs the 3D reconstruction on a single object.
    """
    model = get_model()
    
    # Load image using the model's utility or just standard PIL/numpy
    # The original code used load_image from inference.py
    if load_image is None:
         init_modules()
         
    loaded_image = load_image(image_path)
    
    try:
        output = model(
            loaded_image, 
            mask_np, 
            seed=seed, 
            with_mesh_postprocess=True, 
            with_texture_baking=True
        )
        return output
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        raise e

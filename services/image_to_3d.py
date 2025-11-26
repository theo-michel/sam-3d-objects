"""
Image to 3D service using YOLOv12 + SAM3 pipeline.
"""

import os
from typing import Optional
import numpy as np
from PIL import Image
from services.segmentation import upload_image_to_fal
from services.segmentation_yolo import segment_image_with_yolo
from services.reconstruction import reconstruct_object


def decode_rle_to_mask(rle_string: str, height: int, width: int) -> np.ndarray:
    """
    Decode RLE string to binary mask.

    Args:
        rle_string: RLE encoded string
        height: Image height
        width: Image width

    Returns:
        Binary mask as numpy array
    """
    # Try pycocotools first
    try:
        from pycocotools import mask as mask_util

        rle_obj = {
            "counts": rle_string
            if isinstance(rle_string, bytes)
            else rle_string.encode("utf-8"),
            "size": [height, width],
        }
        return mask_util.decode(rle_obj)
    except ImportError:
        pass

    # Fallback to manual decoding for space-separated format
    if not isinstance(rle_string, str) or " " not in rle_string:
        return np.zeros((height, width), dtype=np.uint8)

    counts = [int(x) for x in rle_string.split()]
    mask = np.zeros(height * width, dtype=np.uint8)

    for i in range(0, len(counts), 2):
        if i + 1 < len(counts):
            start_pos = counts[i]
            length = counts[i + 1]
            if start_pos < len(mask):
                end_pos = min(start_pos + length, len(mask))
                mask[start_pos:end_pos] = 1

    return mask.reshape((width, height), order="F").T


def image_to_3d(
    image_path: str,
    image_url: Optional[str] = None,
    seed: int = 42,
    with_mesh_postprocess: bool = False,
    with_texture_baking: bool = False,
) -> dict:
    """
    Process an image to extract and reconstruct 3D objects using YOLO + SAM3 + 3D reconstruction.

    Args:
        image_path: Path to the local image file
        image_url: Optional pre-uploaded image URL (if None, will upload)
        seed: Random seed for reconstruction
        with_mesh_postprocess: Whether to apply mesh postprocessing
        with_texture_baking: Whether to bake textures

    Returns:
        Dictionary containing:
            - objects: List of reconstructed 3D objects with their metadata
            - image_url: URL of the uploaded image
            - image_path: Local path to the image
            - num_objects: Number of detected objects
    """
    # Upload image if URL not provided
    if image_url is None:
        image_url = upload_image_to_fal(image_path)

    # Get image dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # Run YOLO + SAM3 segmentation
    segmentation_results = segment_image_with_yolo(
        image_url, local_image_path=image_path
    )

    if not segmentation_results:
        return {
            "objects": [],
            "image_url": image_url,
            "image_path": image_path,
            "num_objects": 0,
        }

    print(f"Found {len(segmentation_results)} objects, running 3D reconstruction...")

    # Reconstruct each detected object
    objects = []
    for i, result in enumerate(segmentation_results):
        # Get RLE mask
        rle_data = result.get("rle")
        rle_string = rle_data[0] if isinstance(rle_data, list) else rle_data

        if not rle_string:
            continue

        # Decode RLE to mask
        mask_np = decode_rle_to_mask(rle_string, img_height, img_width)

        # Get metadata
        boxes = result.get("boxes", [])
        bbox = boxes[0] if boxes else [0, 0, 0, 0]
        class_name = result.get("yolo_class", f"object_{i}")

        # Run 3D reconstruction
        try:
            output = reconstruct_object(
                image_path=image_path,
                mask_np=mask_np,
                seed=seed,
                with_mesh_postprocess=with_mesh_postprocess,
                with_texture_baking=with_texture_baking,
            )
        except Exception as e:
            print(f"Failed to reconstruct object {i} ({class_name}): {e}")
            continue

        gs = output.get("gs")
        if gs is None:
            continue

        # Build object result
        obj_result = {
            "index": i,
            "class_name": class_name,
            "bbox": bbox,
            "gs": gs,
            "rotation": output["rotation"].detach().cpu().view(-1).tolist(),
            "translation": output["translation"].detach().cpu().view(-1).tolist(),
            "scale": output["scale"].detach().cpu().view(-1).tolist(),
        }
        objects.append(obj_result)
        print(f"  Reconstructed object {i}: {class_name}")

    return {
        "objects": objects,
        "image_url": image_url,
        "image_path": image_path,
        "num_objects": len(objects),
    }

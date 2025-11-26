#!/usr/bin/env python3
"""
Test script for SAM3 segmentation API.
Uploads an image to fal.ai, runs segmentation with text prompt, and displays the resulting masks.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import dotenv

dotenv.load_dotenv()

# Add parent directory to path to import services
sys.path.insert(0, str(Path(__file__).parent))

from utils import upload_image_to_fal, segment_image
from utils import download_image

# Try to import pycocotools for RLE decoding
try:
    from pycocotools import mask as mask_util

    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("Warning: pycocotools not installed. Install with: pip install pycocotools")


def decode_rle_string(rle_string: str, height: int, width: int) -> np.ndarray:
    """
    Decode RLE string from SAM3 API.

    Args:
        rle_string: RLE encoded string from SAM3 (space-separated integers)
        height: Image height
        width: Image width

    Returns:
        Binary mask as numpy array
    """
    try:
        # SAM3 returns space-separated integers for uncompressed RLE
        if isinstance(rle_string, str) and " " in rle_string:
            counts = [int(x) for x in rle_string.split()]
            return decode_uncompressed_rle(counts, height, width)

        # Try COCO compressed RLE format (for other cases)
        if HAS_PYCOCOTOOLS:
            rle_obj = {
                "counts": rle_string
                if isinstance(rle_string, bytes)
                else rle_string.encode("utf-8"),
                "size": [height, width],
            }
            mask = mask_util.decode(rle_obj)
            return mask
        else:
            print("Warning: pycocotools not available and unknown RLE format")
            return np.zeros((height, width), dtype=np.uint8)

    except Exception as e:
        print(f"  Error decoding RLE: {e}")
        return np.zeros((height, width), dtype=np.uint8)


def decode_uncompressed_rle(counts: list, height: int, width: int) -> np.ndarray:
    """
    Decode uncompressed RLE format from SAM3.
    Format is [start_pos, length, start_pos, length, ...] for runs of 1s.
    Positions are in column-major (Fortran) order.

    Args:
        counts: List of integers alternating between start positions and lengths
        height: Image height
        width: Image width

    Returns:
        Binary mask as numpy array
    """
    # Initialize mask with zeros
    mask = np.zeros(height * width, dtype=np.uint8)

    # Process pairs of (start_position, length)
    for i in range(0, len(counts), 2):
        if i + 1 < len(counts):
            start_pos = counts[i]
            length = counts[i + 1]

            # Clip to mask bounds
            if start_pos < len(mask):
                end_pos = min(start_pos + length, len(mask))
                mask[start_pos:end_pos] = 1

    # Reshape using Fortran order (column-major) and transpose
    # This is the standard COCO RLE format
    mask = mask.reshape((width, height), order="F").T
    return mask


def visualize_masks(
    image_path: str, segmentation_result: dict, save_dir: str = "output"
):
    """
    Visualize the segmentation results with RLE masks overlaid on the original image.

    Args:
        image_path: Path to the original image
        segmentation_result: Result from segment_image API call (SAM3 format)
        save_dir: Directory to save visualization outputs
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Load original image
    original_image = Image.open(image_path)
    img_width, img_height = original_image.size

    # Get RLE data from SAM3 response
    rle_data = segmentation_result.get("rle")
    metadata = segmentation_result.get("metadata", [])
    scores = segmentation_result.get("scores", [])
    boxes = segmentation_result.get("boxes", [])

    if not rle_data:
        print("No RLE mask data found!")
        return

    # Handle both single string and list of strings
    if isinstance(rle_data, str):
        rle_list = [rle_data]
    else:
        rle_list = rle_data

    num_masks = len(rle_list)
    print(f"\nFound {num_masks} mask(s)")

    # Create a figure with subplots
    cols = min(3, num_masks)
    rows = (num_masks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_masks == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_masks > 1 else [axes]

    # Process each mask
    for i, rle_string in enumerate(rle_list):
        score = scores[i] if i < len(scores) else None
        box = boxes[i] if i < len(boxes) else None

        score_str = f" (score: {score:.3f})" if score is not None else ""
        print(f"  Processing mask {i}{score_str}...")

        # Decode RLE mask
        try:
            mask_array = decode_rle_string(rle_string, img_height, img_width)

            # Save individual mask
            mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8))
            mask_pil.save(os.path.join(save_dir, f"mask_{i}.png"))

            # Create visualization
            ax = axes[i]
            ax.imshow(original_image)

            # Overlay mask with transparency
            mask_rgba = np.zeros((*mask_array.shape, 4))
            mask_rgba[:, :, 0] = 1.0  # Red channel
            mask_rgba[:, :, 3] = mask_array * 0.5  # Alpha channel
            ax.imshow(mask_rgba)

            # Draw bounding box if available (convert from normalized to pixel coords)
            if box is not None and len(box) == 4:
                cx, cy, w, h = box
                # Convert normalized coordinates to pixel coordinates
                x = int((cx - w / 2) * img_width)
                y = int((cy - h / 2) * img_height)
                width = int(w * img_width)
                height = int(h * img_height)

                rect = patches.Rectangle(
                    (x, y),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                )
                ax.add_patch(rect)

            title = f"Mask {i}"
            if score is not None:
                title += f" (score: {score:.3f})"
            ax.set_title(title)
            ax.axis("off")
        except Exception as e:
            print(f"  Error decoding mask {i}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Hide unused subplots
    for i in range(num_masks, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    output_path = os.path.join(save_dir, "segmentation_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.show()


def main():
    """Main test function."""
    # Check for FAL_KEY
    if not os.environ.get("FAL_KEY"):
        print("ERROR: FAL_KEY environment variable not set!")
        print("Please set it with: export FAL_KEY='your-key-here'")
        sys.exit(1)

    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for test images in current directory
        test_images = (
            list(Path(".").glob("test*.jpg"))
            + list(Path(".").glob("test*.png"))
            + list(Path(".").glob("test*.jpeg"))
        )
        if test_images:
            image_path = str(test_images[0])
        else:
            print("Usage: python test_segmentation.py <image_path>")
            print(
                "Or place a test image (test.jpg, test.png, etc.) in the current directory"
            )
            sys.exit(1)

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    print(f"Testing SAM3 segmentation with image: {image_path}")
    print("=" * 60)

    # Step 1: Upload image to fal.ai
    print("\n1. Uploading image to fal.ai...")
    try:
        image_url = upload_image_to_fal(image_path)
        print(f"✓ Image uploaded: {image_url}")
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        sys.exit(1)

    # Step 2: Run segmentation
    print("\n2. Running SAM3 segmentation with prompt 'Pickable object'...")
    try:
        segmentation_result = segment_image(image_url)
        print(f"✓ Segmentation complete")
        print(f"  Result keys: {list(segmentation_result.keys())}")
    except Exception as e:
        print(f"✗ Segmentation failed: {e}")
        sys.exit(1)

    # Step 3: Visualize results
    print("\n3. Visualizing results...")
    try:
        visualize_masks(image_path, segmentation_result)
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Test complete!")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# Add the project root to the python path so we can import services
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import upload_image_to_fal
from services.segmentation_yolo import segment_image_with_yolo

# Try to import pycocotools for RLE decoding
try:
    from pycocotools import mask as mask_util

    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("Warning: pycocotools not installed. Install with: pip install pycocotools")

# Load environment variables
load_dotenv()


def decode_rle_string(rle_string: str, height: int, width: int) -> np.ndarray:
    """
    Decode RLE string from SAM3 API.
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
    mask = mask.reshape((width, height), order="F").T
    return mask


def visualize_results(
    image_path: str, results: list[dict], save_dir: str = "output_yolo"
):
    """
    Visualize the segmentation results from YOLO + SAM3.
    """
    os.makedirs(save_dir, exist_ok=True)

    original_image = Image.open(image_path)
    img_width, img_height = original_image.size

    num_results = len(results)
    if num_results == 0:
        print("No results to visualize.")
        return

    print(f"\nVisualizing {num_results} results...")

    # Create a figure with subplots
    cols = min(3, num_results)
    rows = (num_results + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_results == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_results > 1 else [axes]

    for i, result in enumerate(results):
        ax = axes[i]

        # Get data
        rle_data = result.get("rle")
        scores = result.get("scores", [])
        boxes = result.get("boxes", [])
        source_object = result.get("source_object", {})
        yolo_class = result.get("yolo_class", "unknown")
        yolo_conf = result.get("yolo_confidence", 0.0)

        # SAM3 might return a list of RLEs if return_multiple_masks=True,
        # but we set it to False, so we expect a single RLE string or a list with 1 item.
        if isinstance(rle_data, list):
            rle_string = rle_data[0] if rle_data else None
        else:
            rle_string = rle_data

        score = scores[0] if scores else None
        box = boxes[0] if boxes else None

        # Decode mask
        if rle_string:
            mask_array = decode_rle_string(rle_string, img_height, img_width)

            # Save individual mask
            mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8))
            mask_pil.save(os.path.join(save_dir, f"mask_{yolo_class}_{i}.png"))

            # Show image
            ax.imshow(original_image)

            # Overlay mask
            mask_rgba = np.zeros((*mask_array.shape, 4))
            mask_rgba[:, :, 0] = 1.0  # Red
            mask_rgba[:, :, 3] = mask_array * 0.5  # Alpha
            ax.imshow(mask_rgba)
        else:
            ax.imshow(original_image)
            ax.text(
                0.5,
                0.5,
                "No Mask",
                ha="center",
                va="center",
                color="red",
                transform=ax.transAxes,
            )

        # Draw SAM3 Bounding Box (Green)
        if box is not None and len(box) == 4:
            cx, cy, w, h = box
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
                label="SAM3 Box",
            )
            ax.add_patch(rect)

        # Draw YOLO Source Box (Cyan)
        if source_object:
            # source_object has x_min, y_min, x_max, y_max in normalized coords
            mx_min = source_object.get("x_min", 0) * img_width
            my_min = source_object.get("y_min", 0) * img_height
            mx_max = source_object.get("x_max", 0) * img_width
            my_max = source_object.get("y_max", 0) * img_height
            mw = mx_max - mx_min
            mh = my_max - my_min

            rect_yolo = patches.Rectangle(
                (mx_min, my_min),
                mw,
                mh,
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
                linestyle="--",
                label="YOLO Box",
            )
            ax.add_patch(rect_yolo)

        title = f"{yolo_class}"
        if yolo_conf > 0:
            title += f" (YOLO: {yolo_conf:.2f})"
        if score is not None:
            title += f"\n(SAM: {score:.2f})"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

        # Add legend only to the first plot to avoid clutter
        if i == 0:
            ax.legend(loc="upper right", fontsize="small")

    # Hide unused subplots
    for i in range(num_results, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    output_path = os.path.join(save_dir, "yolo_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved visualization to: {output_path}")
    # plt.show() # Commented out for headless environments


def test_local_image(image_path: str, prompt: str):
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    print(f"Uploading {image_path} to fal.ai (for SAM3)...")
    image_url = upload_image_to_fal(image_path)
    print(f"Image uploaded successfully: {image_url}")

    print(f"Testing YOLOv12 + SAM3 pipeline...")
    try:
        results = segment_image_with_yolo(
            image_url, prompt=prompt, local_image_path=image_path
        )
        print(f"\nSuccessfully got {len(results)} results.")

        # Visualize
        visualize_results(image_path, results)

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test YOLOv12 + SAM3 segmentation with a local image."
    )
    parser.add_argument("image_path", type=str, help="Path to the local image file")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Pickable objects",
        help="Prompt (ignored by standard YOLO)",
    )

    args = parser.parse_args()

    test_local_image(args.image_path, args.prompt)

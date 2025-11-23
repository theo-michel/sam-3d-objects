import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import pipeline
from PIL import Image
import numpy as np
import os
import base64



def _encode_image_as_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # Guess a sane default; most of your images here are PNGs
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png" if ext in (".png",) else "image/jpeg"
    return f"data:{mime};base64,{b64}"

# ==================== SAM2 ====================

device = "cuda"

generator = pipeline("mask-generation", model="facebook/sam2-hiera-large", device=0)



def save_mask(mask: torch.Tensor, index: int):
    mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    mask = mask * 255
    mask = Image.fromarray(mask)
    mask.save(f"mask_{index}.png")

def save_masked_image(image: Image.Image, mask: torch.Tensor, bbox: tuple[int, int, int, int], index: int):
    # mask the image and save it as mask_index.png
    m = mask.detach().cpu().numpy()
    # Squeeze singleton channel if present (e.g., 1xHxW)
    if m.ndim == 3 and 1 in m.shape:
        m = np.squeeze(m)
    # Ensure binary mask 0/255 uint8
    m = (m > 0).astype(np.uint8) * 255
    alpha = Image.fromarray(m, mode="L")
    # Match alpha size to image size if needed
    if alpha.size != image.size:
        alpha = alpha.resize(image.size, resample=Image.NEAREST)
    # Ensure RGBA base
    base = image if image.mode == "RGBA" else image.convert("RGBA")
    masked_image = base.copy()
    masked_image.putalpha(alpha)
    # Convert bbox (x, y, height, width) -> PIL box (left, upper, right, lower)
    x, y, height, width = bbox
    left = int(max(0, x))
    upper = int(max(0, y))
    right = int(min(masked_image.width, left + int(width)))
    lower = int(min(masked_image.height, upper + int(height)))
    if right > left and lower > upper:
        masked_image = masked_image.crop((left, upper, right, lower))
    masked_image.save(f"masked_image_{index}.png")

def crop_image(image: Image.Image, bbox: tuple[int, int, int, int]) -> Optional[Image.Image]:
    # Convert bbox (x, y, height, width) -> PIL box (left, upper, right, lower)
    base = image if image.mode == "RGBA" else image.convert("RGBA")
    img = base.copy()
    x, y, height, width = bbox
    left = int(max(0, x))
    upper = int(max(0, y))
    right = int(min(img.width, left + int(width)))
    lower = int(min(img.height, upper + int(height)))
    if right > left and lower > upper:
        return img.crop((left, upper, right, lower))
    return None
    
def get_bbox_from_mask(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """
    Compute the tight bounding box around white (non-zero) pixels in the mask.
    Returns (x, y, height, width) in pixel coordinates.
    If the mask has no white pixels, returns (0, 0, 0, 0).
    """
    m = mask.detach().cpu().numpy()
    # Squeeze singleton channel if present (e.g., 1xHxW or HxWx1)
    if m.ndim == 3 and 1 in m.shape:
        m = np.squeeze(m)
    m = m.astype(np.uint8)
    # Consider any non-zero as white
    ys, xs = np.where(m > 0)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)
    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())
    x = x_min
    y = y_min
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    return (x, y, height, width)
    

if __name__ == "__main__":
    image = "test.jpeg"
    image = Image.open(image)
    image = image.convert("RGBA")

    outputs = generator(image, points_per_batch=64)

    for index, mask in enumerate(outputs["masks"]):
        save_mask(mask, index)
        bbox = get_bbox_from_mask(mask)
        save_masked_image(image, mask, bbox, index)

import os
import zipfile
import requests
import io
from PIL import Image
from fastapi import HTTPException
import fal_client
import dotenv

dotenv.load_dotenv()


def download_image(url: str) -> Image.Image:
    """
    Downloads an image from a URL and returns a PIL Image.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None


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


def zip_files(files: list[str], zip_path: str) -> str:
    """
    Zips the files into a single ZIP file.
    """
    # Ensure parent directory exists
    parent_dir = os.path.dirname(zip_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    # Write with compression and safe arcnames
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file in files or []:
            if not file or not os.path.isfile(file):
                continue
            arcname = os.path.basename(file)
            zipf.write(file, arcname=arcname)
    return zip_path


def save_reconstructed_objects(
    objects: list[dict], output_dir: str
) -> tuple[list[str], list[dict]]:
    """
    Save reconstructed 3D objects as PLY files and generate transformation metadata.

    Args:
        objects: List of reconstructed objects from image_to_3d
        output_dir: Directory to save PLY files

    Returns:
        Tuple of (saved_files, transformations)
    """
    saved_files = []
    transformations = []

    for obj in objects:
        gs = obj["gs"]
        class_name = obj["class_name"]
        idx = obj["index"]

        # Save PLY
        output_filename = f"reconstruction_{class_name}_{idx}.ply"
        output_path = os.path.join(output_dir, output_filename)
        gs.save_ply(output_path)
        saved_files.append(output_path)

        # Save transformation metadata
        transformations.append(
            {
                "filename": output_filename,
                "bbox": obj["bbox"],
                "rotation": obj["rotation"],
                "translation": obj["translation"],
                "scale": obj["scale"],
                "class": class_name,
            }
        )

    return saved_files, transformations


def upload_image_to_fal(image_path: str) -> str:
    """
    Uploads an image to fal.ai and returns the URL.
    """
    print("Uploading image to fal.ai...")
    try:
        return fal_client.upload_file(image_path)
    except Exception as e:
        print(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

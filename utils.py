import requests
import io
from PIL import Image

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

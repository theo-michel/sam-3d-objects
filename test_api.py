"""
Test script for the image-to-3d API endpoint.
"""
import requests
import sys

def test_image_to_3d(image_path: str):
    """
    Test the image-to-3d endpoint with a local image.
    
    Args:
        image_path: Path to the image file
    """
    url = "http://localhost:7310/image-to-3d"
    
    print(f"Testing image-to-3d API")
    print(f"  Image: {image_path}")
    print()
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        
        print("Sending request to API...")
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            output_filename = "reconstruction.zip"
            with open(output_filename, 'wb') as out:
                out.write(response.content)
            print(f"✓ Success! Saved to: {output_filename}")
            print(f"  File size: {len(response.content)} bytes")
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  {response.text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <image_path>")
        print()
        print("Example:")
        print("  python test_api.py images/truck.jpg")
        sys.exit(1)
    
    test_image_to_3d(sys.argv[1])

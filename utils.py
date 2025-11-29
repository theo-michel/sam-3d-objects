import os
import requests
import io
from PIL import Image
from fastapi import HTTPException
import fal_client
import dotenv
import numpy as np
from plyfile import PlyData

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


def build_scene_glb(objects: list[dict], compress: bool = True) -> bytes:
    """
    Build a single GLB file containing all objects with their relative positions.
    
    Args:
        objects: List of reconstructed objects from image_to_3d, each containing:
            - "gs": Gaussian splat object with save_ply method
            - "class_name": str
            - "index": int
            - "bbox": bounding box [x1, y1, x2, y2]
            - "rotation": quaternion [w, x, y, z]
            - "translation": [x, y, z] translation vector
            - "scale": [sx, sy, sz] scale factors
        compress: Whether to apply zstd compression (default: True)
    
    Returns:
        bytes: GLB binary (optionally zstd compressed)
    
    Usage:
        glb_data = build_scene_glb(objects, compress=True)
        
        # Serve via FastAPI:
        return Response(
            content=glb_data,
            media_type="model/gltf-binary",
            headers={
                "Content-Encoding": "zstd",
                "Content-Disposition": 'attachment; filename="scene.glb.zst"'
            }
        )
    """
    import tempfile
    import struct
    import json
    
    nodes = []
    meshes = []
    accessors = []
    buffer_views = []
    binary_blob = bytearray()

    for obj in objects:
        gs = obj["gs"]
        class_name = obj["class_name"]
        idx = obj["index"]
        node_name = f"{class_name}_{idx}"
        
        # Save PLY and extract point cloud data
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
            temp_ply = tmp_file.name
        
        try:
            gs.save_ply(temp_ply)
            plydata = PlyData.read(temp_ply)
            
            # Extract positions (in local object space)
            x = np.asarray(plydata['vertex']['x'], dtype=np.float32)
            y = np.asarray(plydata['vertex']['y'], dtype=np.float32)
            z = np.asarray(plydata['vertex']['z'], dtype=np.float32)
            positions_local = np.column_stack([x, y, z])
            
            # Extract colors if available (use f_dc_0, f_dc_1, f_dc_2 for RGB)
            try:
                r = np.clip((np.asarray(plydata['vertex']['f_dc_0']) + 0.5) * 255, 0, 255).astype(np.uint8)
                g = np.clip((np.asarray(plydata['vertex']['f_dc_1']) + 0.5) * 255, 0, 255).astype(np.uint8)
                b = np.clip((np.asarray(plydata['vertex']['f_dc_2']) + 0.5) * 255, 0, 255).astype(np.uint8)
                colors = np.column_stack([r, g, b, np.full_like(r, 255)])  # RGBA
                has_colors = True
            except (ValueError, KeyError):
                has_colors = False
                colors = None
                
        finally:
            if os.path.exists(temp_ply):
                os.remove(temp_ply)
        
        # Transform points from local to world/scene space
        # This matches the behavior of make_scene() in notebook/inference.py
        quat = np.array(obj["rotation"], dtype=np.float32)  # [w, x, y, z]
        translation = np.array(obj["translation"], dtype=np.float32)
        scale = np.array(obj["scale"], dtype=np.float32)
        
        # Convert quaternion to rotation matrix
        # Quaternion format is [w, x, y, z]
        if len(quat) == 4:
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            rotation_matrix = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ], dtype=np.float32)
        else:
            # Fallback: assume it's already a 3x3 matrix flattened
            rotation_matrix = np.array(quat, dtype=np.float32).reshape(3, 3)
        
        # Apply transformation: world_pos = scale * (R @ local_pos) + translation
        # First scale, then rotate, then translate
        scale_matrix = np.diag(scale)
        positions_world = (positions_local @ scale_matrix.T @ rotation_matrix.T) + translation
        
        # Use the transformed positions
        positions = positions_world
        
        # Calculate bounds
        pos_min = positions.min(axis=0).tolist()
        pos_max = positions.max(axis=0).tolist()
        
        # Add position data to buffer
        pos_offset = len(binary_blob)
        pos_bytes = positions.tobytes()
        binary_blob.extend(pos_bytes)
        # Pad to 4-byte alignment
        while len(binary_blob) % 4 != 0:
            binary_blob.append(0)
        
        # Position buffer view
        pos_buffer_view_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": pos_offset,
            "byteLength": len(pos_bytes),
            "target": 34962  # ARRAY_BUFFER
        })
        
        # Position accessor
        pos_accessor_idx = len(accessors)
        accessors.append({
            "bufferView": pos_buffer_view_idx,
            "componentType": 5126,  # FLOAT
            "count": len(positions),
            "type": "VEC3",
            "min": pos_min,
            "max": pos_max
        })
        
        # Mesh primitive attributes
        primitive_attributes = {"POSITION": pos_accessor_idx}
        
        # Add colors if available
        if has_colors:
            color_offset = len(binary_blob)
            color_bytes = colors.tobytes()
            binary_blob.extend(color_bytes)
            while len(binary_blob) % 4 != 0:
                binary_blob.append(0)
            
            color_buffer_view_idx = len(buffer_views)
            buffer_views.append({
                "buffer": 0,
                "byteOffset": color_offset,
                "byteLength": len(color_bytes),
                "target": 34962
            })
            
            color_accessor_idx = len(accessors)
            accessors.append({
                "bufferView": color_buffer_view_idx,
                "componentType": 5121,  # UNSIGNED_BYTE
                "count": len(colors),
                "type": "VEC4",
                "normalized": True
            })
            primitive_attributes["COLOR_0"] = color_accessor_idx
        
        # Create mesh
        mesh_idx = len(meshes)
        meshes.append({
            "name": node_name,
            "primitives": [{
                "attributes": primitive_attributes,
                "mode": 0  # POINTS
            }]
        })
        
        # Points are already in world space, so use identity matrix for the node
        nodes.append({
            "name": node_name,
            "mesh": mesh_idx
        })
    
    # Build GLB structure
    gltf = {
        "asset": {"version": "2.0", "generator": "sam-3d-objects"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary_blob)}]
    }
    
    # Serialize JSON
    gltf_json = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    while len(gltf_json) % 4 != 0:
        gltf_json += b' '
    
    # Pad binary to 4-byte alignment
    while len(binary_blob) % 4 != 0:
        binary_blob.append(0)
    
    # GLB structure:
    # Header: magic(4) + version(4) + length(4) = 12 bytes
    # JSON chunk: length(4) + type(4) + data
    # BIN chunk: length(4) + type(4) + data
    json_chunk_type = 0x4E4F534A  # "JSON"
    bin_chunk_type = 0x004E4942   # "BIN\0"
    total_length = 12 + 8 + len(gltf_json) + 8 + len(binary_blob)
    
    glb = bytearray()
    glb.extend(struct.pack('<I', 0x46546C67))   # "glTF" magic
    glb.extend(struct.pack('<I', 2))             # version
    glb.extend(struct.pack('<I', total_length))  # total length
    glb.extend(struct.pack('<I', len(gltf_json)))
    glb.extend(struct.pack('<I', json_chunk_type))
    glb.extend(gltf_json)
    glb.extend(struct.pack('<I', len(binary_blob)))
    glb.extend(struct.pack('<I', bin_chunk_type))
    glb.extend(binary_blob)
    
    glb_bytes = bytes(glb)
    
    # Compress with zstd
    if compress:
        try:
            import zstandard as zstd
            compressor = zstd.ZstdCompressor(level=6)
            return compressor.compress(glb_bytes)
        except ImportError:
            print("Warning: zstandard not available, returning uncompressed GLB")
            return glb_bytes
    
    return glb_bytes


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

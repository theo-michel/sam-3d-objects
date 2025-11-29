import numpy as np
import struct
import json
from utils import build_scene_glb
import os

class MockGS:
    def save_ply(self, path):
        # Create a simple PLY with 1 point at (1, 2, 3)
        with open(path, 'wb') as f:
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(b"element vertex 1\n")
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"property float f_dc_0\n")
            f.write(b"property float f_dc_1\n")
            f.write(b"property float f_dc_2\n")
            f.write(b"end_header\n")
            # x, y, z, r, g, b
            f.write(struct.pack('<ffffff', 1.0, 2.0, 3.0, 0.5, 0.5, 0.5))

def test_glb_transform():
    objects = [{
        "gs": MockGS(),
        "class_name": "test",
        "index": 0,
        "bbox": [0,0,0,0],
        "rotation": [1, 0, 0, 0], # Identity quaternion [w, x, y, z]
        "translation": [10, 20, 30],
        "scale": [1, 1, 1]
    }]
    
    glb_bytes = build_scene_glb(objects, compress=False)
    
    # Parse GLB to check matrix
    # Header 12 bytes
    # JSON chunk
    json_len = struct.unpack('<I', glb_bytes[12:16])[0]
    json_chunk_type = struct.unpack('<I', glb_bytes[16:20])[0]
    json_data = glb_bytes[20:20+json_len]
    gltf = json.loads(json_data)
    
    node = gltf['nodes'][0]
    print("Matrix:", node['matrix'])
    
    # Check binary data for vertex position
    # We need to find the bufferView for position
    mesh = gltf['meshes'][node['mesh']]
    primitive = mesh['primitives'][0]
    pos_accessor_idx = primitive['attributes']['POSITION']
    pos_accessor = gltf['accessors'][pos_accessor_idx]
    pos_buffer_view = gltf['bufferViews'][pos_accessor['bufferView']]
    
    bin_start = 20 + json_len + 8
    pos_offset = bin_start + pos_buffer_view['byteOffset']
    
    x, y, z = struct.unpack('<fff', glb_bytes[pos_offset:pos_offset+12])
    print(f"Vertex: {x}, {y}, {z}")

if __name__ == "__main__":
    test_glb_transform()

# 3D Object Orientation Fix - Analysis and Solution

## Problem Summary
The 3D objects in the scene were appearing with incorrect orientations in the visualization because the transformation from local object space to world/scene space was not being applied correctly.

## Root Cause Analysis

### How the Notebook Works (`notebook/inference.py`)
The `make_scene()` function in the notebook does the following:

1. Takes Gaussian splat objects with points in **local object space**
2. Applies transformation using `SceneVisualizer.object_pointcloud()`:
   - Scales the points
   - Rotates them using the quaternion
   - Translates them to their position in the scene
3. Updates the Gaussian object with **world-space coordinates**
4. Merges all objects into a single Gaussian splat with all points in world space

```python
# From notebook/inference.py, line 264-270
PC = SceneVisualizer.object_pointcloud(
    points_local=output["gaussian"][0].get_xyz.unsqueeze(0),
    quat_l2c=output["rotation"],
    trans_l2c=output["translation"],
    scale_l2c=output["scale"],
)
output["gaussian"][0].from_xyz(PC.points_list()[0])
```

### The Original Bug in `utils.py`
The `build_scene_glb()` function was:

1. Extracting points from the PLY file (which are in **local object space**)
2. Adding them to the GLB buffer as-is
3. Trying to apply transformation via the GLB node's transformation matrix

**This was incorrect** because:
- The transformation parameters (rotation, translation, scale) represent the transform from local-to-world space
- But we were treating them as if they should be applied by the GLB viewer
- The GLB node matrix was being applied to already-local points, not transforming them to world space

## The Fix

### What Changed
Modified `build_scene_glb()` in `utils.py` to:

1. Extract points in local object space from the PLY
2. **Apply the transformation to the points directly** (matching the notebook behavior):
   ```python
   # Transform points from local to world/scene space
   rotation_matrix = quaternion_to_rotation_matrix(quat)
   scale_matrix = np.diag(scale)
   positions_world = (positions_local @ scale_matrix.T @ rotation_matrix.T) + translation
   ```
3. Add the **world-space points** to the GLB buffer
4. Use an **identity transformation** for the GLB node (no matrix needed)

### Transformation Order
The transformation follows the standard order:
1. **Scale**: Apply object scale
2. **Rotate**: Apply rotation (from quaternion)
3. **Translate**: Move to position in scene

Formula: `world_pos = (local_pos × scale) × rotation + translation`

### Quaternion Convention
The code uses **[w, x, y, z]** quaternion format (scalar-first), which is converted to a 3×3 rotation matrix using the standard quaternion-to-matrix formula.

## Files Modified

### `/Users/theomichel/Robots/sam-3d-objects-1/utils.py`
- **Function**: `build_scene_glb()`
- **Lines**: ~119-147 (transformation code added)
- **Lines**: ~220-224 (removed duplicate transformation, use identity for node)

## Testing
Created `debug_glb_transform.py` to verify the transformation matrix structure in the generated GLB files.

## Result
Objects should now appear with correct orientations in the 3D visualization, matching the behavior of the notebook's `make_scene()` function.

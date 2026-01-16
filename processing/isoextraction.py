import torch 
import numpy as np
import pyvista as pv

from pytorch3d.ops.marching_cubes import marching_cubes

from processing.postprocess import postprocess_mesh 


def mesh_from_sdf(vol, min_bounds, spacing, device): 
    """
    Extract a mesh from an SDF volume using marching cubes and apply high-quality post-processing.
    
    This function performs the core mesh reconstruction:
    1. Run marching cubes to find the zero-crossing isosurface
    2. Transform vertices from grid-space to world-space
    3. Apply comprehensive post-processing for visual quality
    
    :param vol: SDF volume tensor (resolution, resolution, resolution) - should be normalized
    :param min_bounds: World-space coordinate form of grid.
    :param spacing: Spacing between each voxel
    :param device: Torch device
    :return: High-quality PyVista PolyData mesh, or None if extraction fails
    """

    # Unpack to form (1, D, H, W) for marching cubes - float32 is also used due to being optimised for GPU computation
    vol_batched = vol.unsqueeze(0).to(torch.float32)

    try:
        verts, faces = marching_cubes(vol_batched, isolevel = 0.0)
    except Exception as e:
        # Fallback to CPU
        verts, faces = marching_cubes(vol_batched.cpu(), isolevel = 0.0)
    
    # marching_cubes returns a list of tensors (one per batch item), extract first batch
    if isinstance(verts, (list, tuple)):
        verts = verts[0]
    if isinstance(faces, (list, tuple)):
        faces = faces[0]

    # Replace with func
    if _is_empty_tensor(verts) or _is_empty_tensor(faces):
        print("  No surface found at this interpolation step")
        return None

    # Convert to NumPy 
    verts = _to_numpy(verts)
    faces = _to_numpy(faces)

    if len(verts) == 0 or len(faces) == 0:
        print("  No surface found")
        return None 
    
    # Convert from grid-space to world-space
    min_bounds = _to_numpy(min_bounds)
    spacing = _to_numpy(spacing)


    # This is a simple affine transformation applied per-axis
    #   world_pos = grid_pos * voxel_spacing + grid_origin
    verts[:, 0] = verts[:, 0] * spacing[0] + min_bounds[0]
    verts[:, 1] = verts[:, 1] * spacing[1] + min_bounds[1]
    verts[:, 2] = verts[:, 2] * spacing[2] + min_bounds[2]


    # PyVista expects faces in a flat array format: [n_verts, v0, v1, v2, n_verts, v0, v1, v2, ...]
    # For triangles, n_verts is always 3
    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    mesh = pv.PolyData(verts, faces_pv)

    mesh = postprocess_mesh(mesh)

    return mesh 


def _is_empty_tensor(tensor):
    """
    Check if a tensor (Torch or NumPy) is empty (has zero elements).
    
    :param tensor: Input tensor (Torch Tensor or NumPy ndarray)
    :return: True if empty, False otherwise
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.numel() == 0
    elif isinstance(tensor, np.ndarray):
        return tensor.size == 0
    else:
        raise TypeError("Input must be a Torch Tensor or NumPy ndarray.")
    
    
def _to_numpy(data: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert a Torch Tensor to a NumPy ndarray if necessary.
    
    :param tensor: Input tensor (Torch Tensor or NumPy ndarray)
    :return: NumPy ndarray
    """
    if isinstance(data, torch.Tensor):
        # Check if GPU pipeline
        if data.is_cuda:
            data = data.cpu()
        return data.detach().numpy()
    
    elif isinstance(data, np.ndarray):
        return data
    
    elif isinstance(data, (list, tuple)):
        return np.array(data)

    else:
        raise TypeError(f"Expected Tensor/ndarray/list/tuple, got {type(data).__name__}")
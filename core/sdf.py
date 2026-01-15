import os 
import tempfile 
import torch

import numpy as np
import pytorch_volumetric as ptv 

from config import SDF_NORMALIZATION_SCALE

def save_temp_trimesh(trimesh, mesh_index = 0): 
    """
    Save a Trimesh to a temporary OBJ file for loading into PyTorch Volumetric. 
    
    :param trimesh: Trimesh object
    :param mesh_index: Index of the mesh (for unique naming)
    """
    temp_dir = tempfile.mkdtemp(prefix = "mm_temp_")
    temp_path = os.path.join(temp_dir, f"temp_mesh_{mesh_index}.obj")
    trimesh.export(temp_path)

    return temp_path, temp_dir 


def build_mesh_sdf(trimesh, device, mesh_index = 0):
    """
    Build an SDF representation using PyTorch Volumetric's MeshSDF which computes
    the exact signed distances using a GPU-accelerated algorithm with the following caveats:

    1. Outside points: distance to nearest triangle
    2. Inside points: negative distance to nearest triangle
    3. Sign determination uses ray casting (odd crossings = inside)
 
    :param trimesh: Trimesh object
    :param device: Torch device
    :param global_bounds: Bounding box for the SDF grid (unused now, kept for API compatibility)
    :param mesh_index: Index of the mesh (for temp file naming)
    :param resolution: Resolution parameter (unused now, kept for API compatibility)
    :return: MeshSDF object that computes exact signed distances
    """

    temp_path, temp_dir = save_temp_trimesh(trimesh, mesh_index=mesh_index)

    try:
        # Load mesh into PyTorch Volumetric
        mesh = ptv.MeshObjectFactory(temp_path, device=device)
        

        # MeshSDF computes exact signed distances using GPU-accelerated algorithms:
        # - For outside points: distance to nearest triangle
        # - For inside points: negative distance to nearest triangle
        # - Sign determination uses ray casting (odd crossings = inside)
        sdf = ptv.MeshSDF(mesh)

    finally:
        # Clean up temporary files
        os.remove(temp_path)
        os.rmdir(temp_dir)
    
    return sdf


def sdf_vol_from_mesh(mesh_sdf, query_points, resolution, device):
    """
    Query SDF values from a MeshSDF object and reshape into a 3D volume.
    
    This function queries the MeshSDF at each point in the query grid to get
    exact signed distance values. 

    :param mesh_sdf: MeshSDF object (from pytorch_volumetric)
    :param query_points: Tensor of shape (N, 3) with query coordinates
    :param resolution: Grid resolution (for reshaping output to 3D)
    :param device: Torch device for computation
    :return: SDF volume tensor of shape (resolution, resolution, resolution)
    """

    # Ensure query points are on the correct device and dtype
    # MeshSDF requires float32 tensors on the same device as the mesh
    query_points = torch.as_tensor(query_points, device=device, dtype=torch.float32)

    # Query the MeshSDF - this computes EXACT signed distances
    out = mesh_sdf(query_points) 
    
    # Handle different return formats from pytorch_volumetric
    # Some versions return (sdf, gradient), others just sdf
    sdf_vals = out[0] if isinstance(out, (tuple, list)) else out

    if sdf_vals.numel() != resolution**3:
        raise ValueError(f"Expected {resolution**3} SDF values, but got {sdf_vals.numel()} values.")

    # Reshape from flat (N,) array to 3D volume (resolution, resolution, resolution)
    # The query points were generated in 'ij' indexing order from meshgrid,
    # so we reshape in the same order to maintain spatial correspondence
    vol = sdf_vals.reshape((resolution, resolution, resolution))

    return vol


def normalize_sdf_volume(sdf_vol, voxel_spacing):
    """
    Normalize an SDF volume so values represent distances in voxel units, not world units.
    This reduces gradient steepness at the surface and therefore allows for less noisy marching cubes.
    Doing this also blends SDFs from different scales more cleanly, matching characteristics
    
    :param sdf_vol: Raw SDF volume tensor (resolution, resolution, resolution)
    :param voxel_spacing: Tuple of (dx, dy, dz) world-space distance between voxels
    :param device: Torch device for computation
    :return: Normalized SDF volume where values are in voxel-distance units
    """
    
    # We use the geometric mean to handle anisotropic grids fairly
    # Elementwise cube root of array ensures that representative scaling to calculate voxel size
    avg_voxel_size = float(
        np.cbrt(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])
        )
    
    sdf_normalized = (sdf_vol / avg_voxel_size) * SDF_NORMALIZATION_SCALE
    
    return sdf_normalized


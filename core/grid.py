import torch 
import numpy as np

def query_points_maker(min_bounds, max_bounds, resolution, device):
    """
    Generate the query points grid for SDF computation through Torch Tensors. 
    
    :param min_bounds: Bounding box minimum coordinates
    :param max_bounds: Bounding box maximum coordinates
    """

    # Ensures enters as a 0D Torch Tensor (i.e. a scalar) and accounts for NumPy inconsistency.  
    x = torch.linspace(float(min_bounds[0]), float(max_bounds[0]), resolution, device=device)
    y = torch.linspace(float(min_bounds[1]), float(max_bounds[1]), resolution, device=device)
    z = torch.linspace(float(min_bounds[2]), float(max_bounds[2]), resolution,  device=device)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

    # Ensures a contiguous tensor for better performance - Ensures the elements are in an uninterrupted block of memory
    # Crucial for certain functions like reshaping and view operations 
    # Ensures that RAM isn't used excessively on large grids and therefore decreases time.
    query_points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1)
    spacing = (
        float(max_bounds[0] - min_bounds[0]) / (resolution - 1), 
        float(max_bounds[1] - min_bounds[1]) / (resolution - 1), 
        float(max_bounds[2] - min_bounds[2]) / (resolution - 1)  
    ) # Interval spacing between points in each dimension i.e. only resolution - 1 intervals between the total extent therefore find voxel spacing

    return query_points, spacing


def calculate_bounds(trimeshes, padding_percent):
    """
    Docstring for calculate_bounds
    
    :param trimeshes: Description
    :param padding_percent: Description
    """

    # Use of NumPy instead of CuPy due to being more efficient for smaller arrays
    raw_min = np.min([mesh.bounds[0] for mesh in trimeshes], axis=0)
    raw_max = np.max([mesh.bounds[1] for mesh in trimeshes], axis=0)

    # Utilise isotropic padding 
    diag_length = np.linalg.norm(raw_max - raw_min)
    padding = diag_length * padding_percent

    min_bounds = raw_min - padding
    max_bounds = raw_max + padding

    return min_bounds, max_bounds


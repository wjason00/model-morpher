"""
Main pipeline for mesh morphing
"""

import torch
import numpy as np
from time import time

from core.device import get_device 
from core.grid import calculate_bounds, query_points_maker
from core.sdf import (
    build_mesh_sdf, batched_sdf_query, 
    normalise_sdf_volume, normalise_sdf_batched)    

from processing.isoextraction import mesh_from_sdf, mesh_from_sdf_batched
from config import PADDING_PERCENT, SDF_QUERY_BATCH_SIZE, FRAME_BATCH_SIZE



def morph_mesh_sequence_torch(trimeshes, resolution = 64, frames_per_transition = 20, device = "cuda"):
    """
    Backend for the Torch SDF sampling + marching cubes (all done in CUDA pipeline optimally)
    
    :param trimeshes: List of Trimesh objects to morph between
    :param resolution: Resolution of the SDF grid
    :param frames_per_transition: Number of frames per morph transition
    :param device: Device to use ("cuda" or "cpu")
    :return: List of morphed PyVista meshes
    """
    device = get_device(device)
    morph_frames = []

    # Global Bounding Box - Using NumPy as it's faster than CuPy for small arrays.
    # Inclusion of isotropic padding.

    min_bounds, max_bounds = calculate_bounds(trimeshes, PADDING_PERCENT)
    query_points, spacing = query_points_maker(min_bounds, max_bounds, resolution, device)

    print(f"""
        Grid resolution: {resolution}x{resolution}x{resolution}
        Total query points: {query_points.shape[0]}
        Global bounding box: min={min_bounds}, max={max_bounds}
        """)
    
    start_time = time()

    mesh_sdfs = [build_mesh_sdf(mesh, device, i) for i, mesh in enumerate(trimeshes)]
    print(f"Built MeshSDFs in {time() - start_time:.2f} seconds")
    #
    sdfs = batched_sdf_query(
        mesh_sdfs, query_points, resolution, device, batch_size = SDF_QUERY_BATCH_SIZE
    )

    print(f"Queried SDF volumes in {time() - start_time:.2f} seconds")


    # i = mesh index
    for i in range(len(sdfs) - 1):
        sdf_a = sdfs[i]
        sdf_b = sdfs[i + 1]

        # Interpolation weights 
        transition = np.linspace(0.0, 1.0, frames_per_transition)

        if i < len(trimeshes) - 2:
            # Prevent going from A->B B->C then D->D on the last frame 
            transition = transition[:-1] 
        
        interpolated_sdfs = generate_frames_batched(sdf_a, sdf_b, transition, device)
        normalised_sdfs = normalise_sdf_batched(interpolated_sdfs, spacing)

        num_frames = normalised_sdfs.shape[0]

        t0 = time()
        print(f"\nGenerating {num_frames} frames for transition {i} -> {i + 1}... in {time() - t0:.2f} seconds")

        for batch_start in range(0, num_frames, FRAME_BATCH_SIZE):
            batch_end = min(batch_start + FRAME_BATCH_SIZE, num_frames)
            batch_sdfs = normalised_sdfs[batch_start:batch_end]

            # Process Each Batch
            meshes = mesh_from_sdf_batched(
                batch_sdfs, min_bounds, spacing, device
                )
            
            morph_frames.extend([m for m in meshes if m])

        print(f"\n{'='*50}")
        print(f"Total frames generated: {len(morph_frames)}")
        print(f"{'='*50}")

        print("time taken: {:.2f} seconds".format(time() - start_time))

    return morph_frames


def generate_frames_batched(sdf_a, sdf_b, t_values, device): 
    """
    Generate morphed frames in batches. 
    """

    # Stack t values into a tensor to vectorize operations (stored operations on GPU)
    t_tensor = torch.tensor(t_values, device=device, dtype=torch.float32)

    # Reshape for broadcasting (num_frames, 1, 1, 1) and then add batch dimensions
    t_tensor = t_tensor.view(-1, 1, 1, 1) 

    sdf_a = sdf_a.unsqueeze(0)
    sdf_b = sdf_b.unsqueeze(0)

    interpolated = (1 - t_tensor) * sdf_a + t_tensor * sdf_b

    return interpolated

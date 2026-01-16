"""
Main pipeline for mesh morphing
"""

import torch
import numpy as np
from time import time

from core.device import get_device 
from core.grid import calculate_bounds, query_points_maker
from core.sdf import build_mesh_sdf, sdf_vol_from_mesh, normalize_sdf_volume, batched_sdf_query

from processing.isoextraction import mesh_from_sdf
from config import PADDING_PERCENT


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
    sdfs = batched_sdf_query(
        mesh_sdfs, query_points, resolution, device, batch_size = 100000
        )
    print(f"Queried SDF volumes in {time() - start_time:.2f} seconds")


    # Generate morph frames
    morph_frames = []

    # i = mesh index
    for i in range(len(sdfs) - 1):
        sdf_a = sdfs[i]
        sdf_b = sdfs[i + 1]

        # Interpolation weights 
        transition = np.linspace(0.0, 1.0, frames_per_transition)

        if i < len(trimeshes) - 2:
            # Prevent going from A->B B->C then D->D on the last frame 
            transition = transition[:-1] 

        # j = frame index
        for j, t in enumerate(transition):
            print(f"  Generating frame {j + 1}/{frames_per_transition} for transition {i + 1}/{len(trimeshes) - 1}...")
            
            # Keep interpolation within GPU
            t_torch = torch.tensor(float(t), device = device, dtype = torch.float32)
            sdf_interp = (1 - t_torch) * sdf_a + t_torch * sdf_b

            mesh = mesh_from_sdf(
                sdf_interp,
                min_bounds,
                spacing,
                device
            )

            if mesh:
                morph_frames.append(mesh) 
            elif morph_frames:
                morph_frames.append(morph_frames[-1])  # Repeat last valid frame
                print("repeating previous frame")
            else:
                print(f"no surface found: frame {j + 1} for transition {i + 1}")

        print(f"\n{'='*50}")
        print(f"Total frames generated: {len(morph_frames)}")
        print(f"{'='*50}")

    return morph_frames


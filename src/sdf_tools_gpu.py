import os 
import tempfile 
import torch

import numpy as np 
import pyvista as pv
import pytorch_volumetric as ptv

from config import (
    TOLERANCE, 
    PADDING_PERCENT,
    SDF_NORMALIZATION_SCALE,
    TAUBIN_ITERATIONS,
    TAUBIN_PASS_BAND,
    SUBDIVIDE_ITERATIONS,
)
from sdf_tools_cpu import extract_isosurface_cpu
from pytorch3d.ops.marching_cubes import marching_cubes 

def _get_device(device = "cuda"): 
    """
    Decide the device to use. 
    
    :param device: "cuda" or "cpu"
    """
    if device == "cuda" and torch.cuda.is_available(): 
        return torch.device("cuda") 
    else: 
        return torch.device("cpu")
    
# Same logic as the CPU pipeline 
def _query_points_maker(min_bounds, max_bounds, resolution, device):
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

def _trimesh_to_temp_obj(trimesh, name = "mesh.obj"): 
    """
    Save a Trimesh to a temporary OBJ file for loading into PyTorch Volumetric. 
    
    :param trimesh: Trimesh object
    :param name: Temporary file name
    """
    temp_dir = tempfile.mkdtemp(prefix = "mm_temp_")
    temp_path = os.path.join(temp_dir, name)
    trimesh.export(temp_path)

    return temp_path, temp_dir 


def _build_mesh_sdf(trimesh, device, global_bounds, mesh_index = 0, resolution = 256):
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

    try:
        file_name = f"mesh_{mesh_index}.obj"
        temp_path, temp_dir = _trimesh_to_temp_obj(trimesh, name=file_name)
        
        # Load mesh into PyTorch Volumetric
        mesh = ptv.MeshObjectFactory(temp_path, device=device)
        

        # MeshSDF computes exact signed distances using GPU-accelerated algorithms:
        # - For outside points: distance to nearest triangle
        # - For inside points: negative distance to nearest triangle
        # - Sign determination uses ray casting (odd crossings = inside)
        sdf = ptv.MeshSDF(mesh)

        print(f"    MeshSDF built on device: {device} (exact distances, no caching)")

    finally:
        # Clean up temporary files
        os.remove(temp_path)
        os.rmdir(temp_dir)
    
    return sdf


def _sdf_vol_from_mesh(mesh_sdf, query_points, resolution, device):
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

    print(f"  Query shape: {query_points.shape}, dtype: {query_points.dtype}")
    print(f"  Expected output: {resolution**3} values")

    # Query the MeshSDF - this computes EXACT signed distances
    # The SDF values represent:
    #   - Positive: point is outside the mesh, value = distance to nearest surface
    #   - Negative: point is inside the mesh, value = -distance to nearest surface
    #   - Zero: point is exactly on the mesh surface
    out = mesh_sdf(query_points) 
    
    # Handle different return formats from pytorch_volumetric
    # Some versions return (sdf, gradient), others just sdf
    sdf_vals = out[0] if isinstance(out, (tuple, list)) else out

    print(f"  Output shape: {sdf_vals.shape}, dtype: {sdf_vals.dtype}")

    if sdf_vals.numel() != resolution**3:
        raise ValueError(f"Expected {resolution**3} SDF values, but got {sdf_vals.numel()} values.")

    # Reshape from flat (N,) array to 3D volume (resolution, resolution, resolution)
    # The query points were generated in 'ij' indexing order from meshgrid,
    # so we reshape in the same order to maintain spatial correspondence
    vol = sdf_vals.reshape((resolution, resolution, resolution))

    return vol 


def _normalize_sdf_volume(sdf_vol, voxel_spacing):
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
    avg_voxel_size = float(np.cbrt(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]))
    
    # Divide SDF by voxel size to convert from world-units to voxel-units allowing for scaling 
    # of the SDF gradient.
    # This makes the gradient approximately 1.0 at the surface (ideal for marching cubes)
    sdf_normalized = sdf_vol / avg_voxel_size
    
    # Values > 1.0 make surfaces sharper (steeper gradient)
    # Values < 1.0 make surfaces smoother (gentler gradient)
    sdf_normalized = sdf_normalized * SDF_NORMALIZATION_SCALE
    
    return sdf_normalized


def _high_quality_mesh_postprocess(mesh):
    """
    Apply post-processing to an extracted mesh.,
    This includes: cleaning, hole filling, optional subdivision, Taubin smoothing, and normal computation.

    :param mesh: PyVista PolyData mesh from marching cubes
    :return: High-quality processed mesh, or None if processing fails
    """
    
    if mesh is None or mesh.n_points == 0:
        return None
    
   # Removal of degenerate faces and unreferenced points.
    mesh = mesh.clean(tolerance = TOLERANCE, inplace = False)
    
    if mesh.n_points == 0:
        return None
    
    # Ensure largest connected components are kept and therefore reduces blobbing.
    mesh = mesh.connectivity(extraction_mode = 'largest')
    
    if mesh.n_points == 0:
        return None
    
    # Despite marching cubes returning triangles, edge case. 
    mesh = mesh.triangulate()
    
    try:
        mesh = mesh.fill_holes(hole_size = 100000)
    except Exception:
        pass
    


    # We skip this if the mesh is already dense (>100k faces) to avoid
    # excessive memory usage 
    if SUBDIVIDE_ITERATIONS > 0 and mesh.n_cells < 100000:
        try:
            mesh = mesh.subdivide(SUBDIVIDE_ITERATIONS, subfilter = 'loop')
        except Exception:
            # Subdivision can fail on non-manifold meshes - continue without it
            pass

    # Taubin used over laplacian to preserve the volume of the mesh
    # Standard Laplacian smoothing shrinks the mesh over iterations because
    # it always moves vertices toward their neighbors' centroid.
    # This produces smooth surfaces WITHOUT the shrinkage problem that
    # makes Laplacian-smoothed meshes look "deflated"
    if TAUBIN_ITERATIONS > 0:
        try:
            mesh = mesh.smooth_taubin(
                n_iter = TAUBIN_ITERATIONS,
                pass_band = TAUBIN_PASS_BAND,
                normalize_coordinates = True  # Improves numerical stability
            )
        except Exception:
            # Fall back to basic smoothing if Taubin fails
            try:
                mesh = mesh.smooth(n_iter = 20, relaxation_factor = 0.1)
            except Exception:
                pass
    
    # Recompute normals for smooth shading
    mesh = mesh.compute_normals(
        cell_normals = False,  
        point_normals = True,  # Vertex normals for smooth interpolation
        split_vertices = False,  # Don't split - smooth across edges
        flip_normals = False,
        consistent_normals = True  # Ensure all normals point same direction
    )
    
    return mesh


def _mesh_from_volume_torch(vol, min_bounds, spacing, device): 
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

    # Early exit if marching cubes found no surface (empty tensors)
    verts_empty = (isinstance(verts, torch.Tensor) and verts.numel() == 0) or \
                  (isinstance(verts, np.ndarray) and verts.size == 0)
    faces_empty = (isinstance(faces, torch.Tensor) and faces.numel() == 0) or \
                  (isinstance(faces, np.ndarray) and faces.size == 0)
    
    if verts_empty or faces_empty:
        print("  No surface found at this interpolation step")
        return None

    # Converting back to NumPy for PyVista (if they aren't already) then mapping to world coordinates
    # Lots of checks cause loads of issues pop up - check later
    if isinstance(verts, torch.Tensor):
        if verts.is_cuda:
            verts = verts.cpu()

        verts = verts.detach().numpy()

    elif not isinstance(verts, np.ndarray):
        verts = np.array(verts)

    if isinstance(faces, torch.Tensor):
        if faces.is_cuda:
            faces = faces.cpu()

        faces = faces.detach().numpy()

    elif not isinstance(faces, np.ndarray):
        faces = np.array(faces)

    if len(verts) == 0 or len(faces) == 0:
        print("  No surface found")
        return None 
    
    # Ensure pure NumPy - handle CUDA tensors by moving to CPU first
    if isinstance(min_bounds, torch.Tensor):
        if min_bounds.is_cuda:
            min_bounds = min_bounds.cpu()
        min_bounds = min_bounds.detach().numpy()
    else:
        min_bounds = np.array(min_bounds)

    if isinstance(spacing, torch.Tensor):
        if spacing.is_cuda:
            spacing = spacing.cpu()
        spacing = spacing.detach().numpy()
    else:
        spacing = np.array(spacing) 


    # This is a simple affine transformation applied per-axis
    #   world_pos = grid_pos * voxel_spacing + grid_origin
    verts[:, 0] = verts[:, 0] * spacing[0] + min_bounds[0]
    verts[:, 1] = verts[:, 1] * spacing[1] + min_bounds[1]
    verts[:, 2] = verts[:, 2] * spacing[2] + min_bounds[2]


    # PyVista expects faces in a flat array format: [n_verts, v0, v1, v2, n_verts, v0, v1, v2, ...]
    # For triangles, n_verts is always 3
    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    morph_mesh = pv.PolyData(verts, faces_pv)

    morph_mesh = _high_quality_mesh_postprocess(morph_mesh)

    return morph_mesh 


def morph_mesh_sequence_torch(trimeshes, resolution = 64, frames_per_transition = 20, device = "cuda"):
    """
    Backend for the Torch SDF sampling + marching cubes (all done in CUDA pipeline optimally)
    
    :param trimeshes: List of Trimesh objects to morph between
    :param resolution: Resolution of the SDF grid
    :param frames_per_transition: Number of frames per morph transition
    :param device: Device to use ("cuda" or "cpu")
    :return: List of morphed PyVista meshes
    """
    device = _get_device(device)
    morph_frames = []

    # Global Bounding Box - Using NumPy as it's faster than CuPy for small arrays.
    # Inclusion of isotropic padding.

    raw_min_bounds = np.min([trimesh.bounds[0] for trimesh in trimeshes], axis = 0)
    raw_max_bounds = np.max([trimesh.bounds[1] for trimesh in trimeshes], axis = 0)

    # Extent times padding percentage
    padding = np.linalg.norm((raw_max_bounds - raw_min_bounds) * PADDING_PERCENT) 

    min_bounds = raw_min_bounds - padding
    max_bounds = raw_max_bounds + padding

    global_bounds = np.array([
        [min_bounds[0], max_bounds[0]],
        [min_bounds[1], max_bounds[1]],
        [min_bounds[2], max_bounds[2]]
    ])

    query_points, spacing = _query_points_maker(min_bounds, max_bounds, resolution, device)

    print(f"""
        Grid resolution: {resolution}x{resolution}x{resolution}
        Total query points: {query_points.shape[0]}
        Global bounding box: min={min_bounds}, max={max_bounds}
        """)
    
    sdfs = []

    for i, trimesh in enumerate(trimeshes): 
        print(f"  Building MeshSDF for mesh {i + 1}/{len(trimeshes)}...")
        
        # Build exact SDF representation (not cached - computes true distances)
        mesh_sdf = _build_mesh_sdf(trimesh, device, global_bounds, mesh_index = i, resolution = resolution)
        
        # Query SDF at all grid points - this computes EXACT distances, no interpolation
        sdf = _sdf_vol_from_mesh(mesh_sdf, query_points, resolution, device)
        

        sdf = _normalize_sdf_volume(sdf, spacing)

        sdfs.append(sdf)

        print(f" SDF {i+1} range: [{float(sdf.min()):.3f}, {float(sdf.max()):.3f}]")

    # Generate morph frames
    morph_frames = []

    for i in range(len(sdfs) - 1):
        sdf_a = sdfs[i]
        sdf_b = sdfs[i + 1]

        transition = np.linspace(0.0, 1.0, frames_per_transition)

        if i < len(trimeshes) - 2:
            # Prevent going from A->B B->C then D->D on the last frame 
            transition = transition[:-1] 

        for j, t in enumerate(transition):
            print(f"  Generating frame {j + 1}/{frames_per_transition} for transition {i + 1}/{len(trimeshes) - 1}...")

            # Keep interpolation within GPU

            t_torch = torch.tensor(float(t), device = device, dtype = torch.float32)

            sdf_interp = (1 - t_torch) * sdf_a + t_torch * sdf_b

            mesh = _mesh_from_volume_torch(
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

        print(f"\n{'='*50}")
        print(f"Total frames generated: {len(morph_frames)}")
        print(f"{'='*50}")

    return morph_frames







import numpy as np
import pyvista as pv

from mesh_to_sdf import mesh_to_sdf
from skimage import measure
from time import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from os import cpu_count

def compute_sdf(mesh_tri, query_points, resolution):
    """
    Compute the SDF values for a given mesh at specified query points and reshape to 3D grid.
    
    :param mesh_tri: The trimesh to compute SDF for. 
    :param query_points: The points at which to query the SDF (given from the np.stack of meshgrid with all of the ravelled points)
    :param resolution: The resolution of the 3D grid to reshape the SDF values into.
    """

    sdf = mesh_to_sdf(
        mesh_tri,
        query_points,
        surface_point_method='sample',
        sign_method='normal',
        sample_point_count=50000,  # Increasing will cause better accuracy. 
        normal_sample_count=11  # Odd number for better voting on sign determination
    )

    # Reshape to a 3D grid from 1D array (NumPy array reshaping)
    return sdf.reshape(resolution, resolution, resolution)


def compute_sdf_worker(args):
    """
    Worker function for parallel SDF computation. Unpacks arguments and calls compute_sdf.
    
    :param args: Tuple of (mesh_tri, query_points, resolution)
    :return: The computed SDF reshaped to 3D grid
    """
    mesh_tri, query_points, resolution = args
    return compute_sdf(mesh_tri, query_points, resolution)


def extract_isosurface_cpu(sdf, min_bounds, max_bounds, resolution, level = 0.0):  
    verts, faces, normals, values = measure.marching_cubes(
        sdf, 
        level=level,
        spacing=(
            (max_bounds[0] - min_bounds[0]) / (resolution - 1),
            (max_bounds[1] - min_bounds[1]) / (resolution - 1),
            (max_bounds[2] - min_bounds[2]) / (resolution - 1)
        ), # x, y, z spacing
        allow_degenerate=False  # Reject degenerate triangles that cause artifacts
    )

    # Accounts for mesh offset due to the bounding box and offsets by such.
    verts += min_bounds

    # Create PyVista mesh by accounting for the marching cubes output of verts (V, 3) and faces (F, 3). 
    # Faces results in an integer array representing the vertex indices for each triangle formed. 
    # PyVista mesh requires faces to be formatted as [3, i, j, k, 3, 3, i, j, k, ...] (essentially a repeating 1D array).
    # (I have absolutely no idea how to recreate this icl) (remember pv stands for pyvista please)
    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    morph_mesh = pv.PolyData(verts, faces_pv) # Intermediate mesh is created here.
    
    # Clean and repair the marching cubes output
    morph_mesh = morph_mesh.clean(tolerance=TOLERANCE)  # Merge very close vertices
    
    # Fill holes aggressively - use large hole_size to catch bigger gaps
    try:
        morph_mesh = morph_mesh.fill_holes(hole_size=1000)
    except:
        pass
    
    # Smooth the mesh slightly to reduce jagged artifacts from marching cubes
    try:
        morph_mesh = morph_mesh.smooth(n_iter=SMOOTH_ITER, relaxation_factor=RELAX_FACTOR)
    except:
        pass

    return morph_mesh


def create_morph_frames(sdf_a, sdf_b, min_bounds, max_bounds, resolution, frames=20):
    """
    Generate frames of morphed meshes via interpolation between two SDFs via marching cubes.
    
    :param sdf_a: Mesh A SDF values
    :param sdf_b: Mesh B SDF values
    :param min_bounds: The minimum bounds of the combined meshes
    :param max_bounds: The maximum bounds of the combined meshes
    :param resolution: The resolution of the 3D grid
    :param frames: The number of morph frames to generate
    """
    morph_frames = []

    for i, t in enumerate(np.linspace(0, 1, frames)):
        # print(f"Generating morph frame {i+1}/{frames} (t={t:.2f})...")

        # Interpolate between the two SDFs
        sdf_interp = (1 - t) * sdf_a + t * sdf_b

        try:
            morph_mesh = extract_isosurface_cpu(
                sdf_interp,
                min_bounds,
                max_bounds,
                resolution,
                level=0.0
            )

            morph_frames.append(morph_mesh)
            # print(f"Mesh has {len(verts)} vertices, {len(faces)} faces")

        except Exception as e:
            print(f"  X Failed for frame {i}: {e}")

            # Failcase in case the marching cubes fails (could be due to bad interpolation?
            # If the failcase is true, just utilise the last existing morph frame. (There shouldn't ever not be 1 existing frame due to the 
            # fact the intermediate mesh can just be the original mesh given t=0 surely)
            if morph_frames:
                morph_frames.append(morph_frames[-1])
                print(f"  → Using previous frame as fallback")

    return morph_frames

def create_morph_frames_worker(args):
    return create_morph_frames(*args)


def morph_meshes(trimeshes, resolution=64, num_frames=20):
    """
    Docstring for morph_meshes
    
    :param mesh_a_tri: Trimesh A to be interpolated
    :param mesh_b_tri: Trimesh B
    :param resolution: Resolution of 3D grid. 
    :param num_frames: Number of frames to generate
    """
    # Generating a bounding box which uses the xyz minimum and maximum alongside a minimum minus padding and a maximum positive padding
    # Changing this value has big effects on the computation time (Having like 0.05 causes LOTS of issues)
    min_bounds = np.minimum([trimesh.bounds[0] for trimesh in trimeshes], axis = 0) - 0.5
    max_bounds = np.maximum([trimesh.bounds[1] for trimesh in trimeshes], axis = 0) + 0.5

    # Generate 1D arrays where the starting point will be the minimum bound and then the end point is the maximum bound. The resolution 
    # is used to determine the spacing between the points i.e. [0, 10, 5] would give [0, 2.5, 5, 7.5, 10] for resolution = 5 (resolution - 1) 
    # Minus 1 for the resolution and each of the bounds are split into [x, y, z] so just take their increasing indices (if that makes any sense)
    # Then generate the grid by ij indexing for (x, y, z) order, whereas xy indexing would give (y,x,z)
    x = np.linspace(min_bounds[0], max_bounds[0], resolution)
    y = np.linspace(min_bounds[1], max_bounds[1], resolution)
    z = np.linspace(min_bounds[2], max_bounds[2], resolution)

    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

    # Ravelling to form a list of 3D points to convert to 1D array via stacking
    # For a meshgrid (N_x, N_y, N_z) with total grid size = N_x * N_y * N_z, and voxel indices of (i, j, k):
    # Deterministic mapping to 1D index = i * (N_y * N_z) + j * N_z + k and world coordinates = (x[i], y[j], z[k]).
    query_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

    print(f"""
            Grid resolution: {resolution}x{resolution}x{resolution}
            Total number of query points: {len(query_points)}
            Global Bounding Box: min={min_bounds}, max={max_bounds}
""")

    # Computing SDF values for both meshes at the query points and then reconverting to 3D grid from 1D array.(Debug code included via print statements)
    # Using parallel computation to speed up by computing both SDFs simultaneously via ThreadPoolExecutor
    print("\nComputing SDFs in parallel...")
    t0 = time()

    args = [(trimesh, query_points, resolution) for trimesh in trimeshes]

    # Use ThreadPoolExecutor to compute both SDFs in parallel. 
    # Dynamic coding for the number of max_workers based on the CPU count cores avilable. 
    # ThreadPool is preferred here as mesh_to_sdf releases the GIL (global interpreter lock) during computation
    with ThreadPoolExecutor(max_workers=min(len(trimeshes), cpu_count())) as executor:
        sdfs = list(executor.map(compute_sdf_worker, args))

    t1 = time()
    
    print(f"All SDFs computed in {t1 - t0:.1f}s")
    
    all_morph_frames = [] 

    for i in range(len(trimeshes) - 1):
        # print(f"\nGenerating transition {i+1}/{len(trimeshes)-1}: Mesh {i+1} → Mesh {i+2}")
        
        # For all transitions except the last, exclude the final frame to avoid duplicates
        # (the end of transition i is the start of transition i+1)
        if i < len(trimeshes) - 2:
            frames = create_morph_frames(
                sdfs[i], sdfs[i+1],
                min_bounds, max_bounds,
                resolution, frames
            )
            all_morph_frames.extend(frames[:-1])  # Exclude last frame
        else:
            # For the last transition, include all frames
            frames = create_morph_frames(
                sdfs[i], sdfs[i+1],
                min_bounds, max_bounds,
                resolution, frames
            )
            all_morph_frames.extend(frames)
    
    print(f"\n{'='*50}")
    print(f"Total frames generated: {len(all_morph_frames)}")
    print(f"{'='*50}")

    return all_morph_frames


def morph_mesh_sequence(trimeshes, resolution=64, frames_per_transition=20):
    """
    Morph through a sequence of multiple meshes (A → B → C → ...).
    
    :param trimeshes: List of trimesh objects to morph through in order
    :param resolution: Resolution of 3D grid
    :param frames_per_transition: Number of frames for each transition between consecutive meshes
    :return: List of all morph frames for the entire sequence
    """
    if len(trimeshes) < 2:
        raise ValueError("Need at least 2 meshes for morphing")
    
    print(f"\n{'='*50}")
    print(f"Multi-mesh morphing: {len(trimeshes)} meshes, {len(trimeshes)-1} transitions")
    print(f"{'='*50}")
    
    # Compute global bounding box that encompasses ALL meshes
    all_min_bounds = np.min([mesh.bounds[0] for mesh in trimeshes], axis=0) - PADDING
    all_max_bounds = np.max([mesh.bounds[1] for mesh in trimeshes], axis=0) + PADDING
    
    # Generate the shared grid for all meshes
    x = np.linspace(all_min_bounds[0], all_max_bounds[0], resolution)
    y = np.linspace(all_min_bounds[1], all_max_bounds[1], resolution)
    z = np.linspace(all_min_bounds[2], all_max_bounds[2], resolution)
    
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    query_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
    
    print(f"""
            Grid resolution: {resolution}x{resolution}x{resolution}
            Total query points: {len(query_points)}
            Global bounding box: min={all_min_bounds}, max={all_max_bounds}
""")
    
    # Pre-compute all SDFs in parallel
    print("Pre-computing SDFs for all meshes...")
    t0 = time()
    
    # Prepare arguments for all meshes
    all_args = [(mesh, query_points, resolution) for mesh in trimeshes]
    
    # Compute all SDFs in parallel
    with ThreadPoolExecutor(max_workers=min(len(trimeshes), 4)) as executor:
        all_sdfs = list(executor.map(compute_sdf_worker, all_args))
    
    t1 = time()
    print(f"  All {len(trimeshes)} SDFs computed in {t1 - t0:.1f}s")
    
    for i, sdf in enumerate(all_sdfs):
        print(f"  SDF {i+1} range: [{sdf.min():.3f}, {sdf.max():.3f}]")
    
    # Generate morph frames as well as the parameters for each consecutive pair
    all_morph_frames = []
    transistion_args = [] 

    for i in range(len(trimeshes) - 1):     
        transistion_args.append((all_sdfs[i], all_sdfs[i+1],
                                all_min_bounds, all_max_bounds,
                                resolution, frames_per_transition))
    
    # Limit number of cores to minimum number of cores avilable / needed. 
    with ProcessPoolExecutor(max_workers = min(len(trimeshes) - 1, cpu_count())) as executor:
        t_results = list(executor.map(create_morph_frames_worker, transistion_args))

    # For all transitions except the last, exclude the final frame to avoid duplicates
    # (the end of transition i is the start of transition i+1)
    for i, frames in enumerate(t_results):
        if i < len(trimeshes) - 2:
            all_morph_frames.extend(frames[:-1])  # Exclude last frame
        else: 
            all_morph_frames.extend(frames)
    
    print(f"\n{'='*50}")
    print(f"Total frames generated: {len(all_morph_frames)}")
    print(f"{'='*50}")
    
    return all_morph_frames
import numpy as np
import pyvista as pv

from mesh_to_sdf import mesh_to_sdf
from skimage import measure
from time import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing


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
        sign_method='depth', # I think this should be converted to normal, but there are no issues?
        sample_point_count=10000,
        normal_sample_count=5
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
        print(f"Generating morph frame {i+1}/{frames} (t={t:.2f})...")

        # Interpolate between the two SDFs
        sdf_interp = (1 - t) * sdf_a + t * sdf_b

        try:
            # Utilise marching cubes to create triangles where the SDF is zero (isosurface extraction)
            # By trial and error, a slight negative level gives better results for our SDFs because the surface looks a tiny bit better? 
            level = -0.01
            verts, faces, normals, values = measure.marching_cubes(
                sdf_interp, 
                level=level, # 
                spacing=(
                    (max_bounds[0] - min_bounds[0]) / (resolution - 1),
                    (max_bounds[1] - min_bounds[1]) / (resolution - 1),
                    (max_bounds[2] - min_bounds[2]) / (resolution - 1)
                ) # x, y, z spacing 
            )

            # Accounts for mesh offset due to the bounding box and offsets by such.
            verts += min_bounds

            # Create PyVista mesh by accounting for the marching cubes output of verts (V, 3) and faces (F, 3). 
            # Faces results in an integer array representing the vertex indices for each triangle formed. 
            # PyVista mesh requires faces to be formatted as [3, i, j, k, 3, 3, i, j, k, ...] (essentially a repeating 1D array).
            # (I have absolutely no idea how to recreate this icl) (remember pv stands for pyvista please)
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
            morph_mesh = pv.PolyData(verts, faces_pv) # Intermediate mesh is created here.
            morph_mesh = morph_mesh.clean()

            # Just fill holes regardless - newer PyVista removed is_watertight property
            # and it doesn't hurt to run fill_holes anyway
            try:
                morph_mesh = morph_mesh.fill_holes(hole_size=100)
            except:
                pass


            morph_frames.append(morph_mesh)
            print(f"Mesh has {len(verts)} vertices, {len(faces)} faces")

        except Exception as e:
            print(f"  X Failed for frame {i}: {e}")

            # Failcase in case the marching cubes fails (could be due to bad interpolation?
            # If the failcase is true, just utilise the last existing morph frame. (There shouldn't ever not be 1 existing frame due to the 
            # fact the intermediate mesh can just be the original mesh given t=0 surely)
            if morph_frames:
                morph_frames.append(morph_frames[-1])
                print(f"  → Using previous frame as fallback")

    return morph_frames


def morph_meshes(trimesh_a, trimesh_b, resolution=64, num_frames=20):
    """
    Docstring for morph_meshes
    
    :param mesh_a_tri: Trimesh A to be interpolated
    :param mesh_b_tri: Trimesh B
    :param resolution: Resolution of 3D grid. 
    :param num_frames: Number of frames to generate
    """
    # Generating a bounding box which uses the xyz minimum and maximum alongside a minimum minus padding and a maximum positive padding
    # Changing this value has big effects on the computation time (Having like 0.05 causes LOTS of issues)
    min_bounds = np.minimum(trimesh_a.bounds[0], trimesh_b.bounds[0]) - 0.5
    max_bounds = np.maximum(trimesh_a.bounds[1], trimesh_b.bounds[1]) + 0.5

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
            Bounding Box: min={min_bounds}, max={max_bounds}
""")

    # Computing SDF values for both meshes at the query points and then reconverting to 3D grid from 1D array.(Debug code included via print statements)
    # Using parallel computation to speed up by computing both SDFs simultaneously via ThreadPoolExecutor
    print("\nComputing SDFs in parallel...")
    t0 = time()

    # Prepare arguments for parallel computation (mesh, query_points, resolution) for each mesh
    args_a = (trimesh_a, query_points, resolution)
    args_b = (trimesh_b, query_points, resolution)

    # Use ThreadPoolExecutor to compute both SDFs in parallel (2 workers for 2 meshes)
    # ThreadPool is preferred here as mesh_to_sdf releases the GIL during computation
    with ThreadPoolExecutor(max_workers=2) as executor:
        print("Computing SDF for mesh A and mesh B simultaneously...")
        future_a = executor.submit(compute_sdf_worker, args_a)
        future_b = executor.submit(compute_sdf_worker, args_b)
        
        # Wait for both computations to complete and retrieve results
        sdf_a = future_a.result()
        sdf_b = future_b.result()

    t1 = time()
    print(f"  Both meshes done in {t1 - t0:.1f}s (parallel computation)")
    print(f"  SDF A range: [{sdf_a.min():.3f}, {sdf_a.max():.3f}]")
    print(f"  SDF B range: [{sdf_b.min():.3f}, {sdf_b.max():.3f}]")

    print(f"\nSDF computation complete! Total time: {t1 - t0:.1f}s")

    return create_morph_frames(sdf_a, sdf_b, min_bounds, max_bounds, resolution, num_frames)
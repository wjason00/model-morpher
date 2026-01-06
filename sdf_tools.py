import numpy as np
import pyvista as pv

from mesh_to_sdf import mesh_to_sdf
from skimage import measure
from time import time


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
        sign_method='depth',
        sample_point_count=10000,
        normal_sample_count=5
    )

    # Reshape to a 3D grid from 1D array (NumPy array reshaping)
    return sdf.reshape(resolution, resolution, resolution)


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
            # By trial and error, a slight negative level gives better results for our SDFs. 
            level = -0.01
            verts, faces, normals, values = measure.marching_cubes(
                sdf_interp, 
                level=level,
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
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()

            # Utilise verts as mesh point and faces_pv as what forms the triangles. 
            morph_mesh = pv.PolyData(verts, faces_pv)

            # Post-process each frame mesh to improve quality
            morph_mesh = morph_mesh.clean()
            morph_mesh = morph_mesh.fill_holes(hole_size=100)

            morph_frames.append(morph_mesh)
            print(f"  ✓ Generated mesh with {len(verts)} vertices, {len(faces)} faces")

        except Exception as e:
            print(f"  ✗ Failed for frame {i}: {e}")

            # Fall back case in frame is unable to be degenerated, reuse last valid frame. 
            if morph_frames:
                morph_frames.append(morph_frames[-1])
                print(f"  → Using previous frame as fallback")

    return morph_frames


def morph_meshes(mesh_a_tri, mesh_b_tri, resolution=64, num_frames=20):
    """
    Docstring for morph_meshes
    
    :param mesh_a_tri: Trimesh A to be interpolated
    :param mesh_b_tri: Trimesh B
    :param resolution: Resolution of 3D grid. 
    :param num_frames: Number of frames to generate
    """
    # A bounding box to account for both meshes
    min_bounds = np.minimum(mesh_a_tri.bounds[0], mesh_b_tri.bounds[0]) - 0.5
    max_bounds = np.maximum(mesh_a_tri.bounds[1], mesh_b_tri.bounds[1]) + 0.5

    # Create 3D grid coordinates
    x = np.linspace(min_bounds[0], max_bounds[0], resolution)
    y = np.linspace(min_bounds[1], max_bounds[1], resolution)
    z = np.linspace(min_bounds[2], max_bounds[2], resolution)

    # ij indexing for (x, y, z) order, whereas xy indexing would give (y,x,z)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

    # Ravelling to form a list of 3D points to convert to 1D array via stacking
    # For a meshgrid (N_x, N_y, N_z) with total grid size = N_x * N_y * N_z, and voxel indices of (i, j, k):
    # Deterministic mapping to 1D index = i * (N_y * N_z) + j * N_z + k and world coordinates = (x[i], y[j], z[k]).
    query_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

    print(f"\nGrid resolution: {resolution}x{resolution}x{resolution}")
    print(f"Total query points: {len(query_points)}")
    print(f"Grid bounds: min={min_bounds}, max={max_bounds}")

    # Computing SDF values for both meshes at the query points and then reconverting to 3D grid from 1D array.
    print("\nComputing SDFs...")
    t0 = time()

    print("Computing SDF for mesh A")
    sdf_a = compute_sdf(mesh_a_tri, query_points, resolution)
    t1 = time()
    print(f"  Mesh A done in {t1 - t0:.1f}s")
    print(f"  SDF A range: [{sdf_a.min():.3f}, {sdf_a.max():.3f}]")

    print("Computing SDF for mesh B...")
    sdf_b = compute_sdf(mesh_b_tri, query_points, resolution)
    t2 = time()
    print(f"  Mesh B done in {t2 - t1:.1f}s")
    print(f"  SDF B range: [{sdf_b.min():.3f}, {sdf_b.max():.3f}]")

    print(f"\nSDF computation complete! Total time: {t2 - t0:.1f}s")

    # Generate morph frames
    return create_morph_frames(sdf_a, sdf_b, min_bounds, max_bounds, resolution, num_frames)
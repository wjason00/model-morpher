import numpy as np
import pyvista as pv
import trimesh
from mesh_to_sdf import mesh_to_sdf
from skimage import measure


mesh_a = pv.read('test_models/halfpoly_suzanne.stl') 
mesh_b = pv.read('test_models/Hand_SUPERfinal.stl')  

# Converting from PyVista to Trimesh 
mesh_a_tri = trimesh.Trimesh(vertices=mesh_a.points, faces=mesh_a.faces.reshape(-1, 4)[:, 1:4])
mesh_b_tri = trimesh.Trimesh(vertices=mesh_b.points, faces=mesh_b.faces.reshape(-1, 4)[:, 1:4])

print(f"Mesh A: {len(mesh_a.points)} vertices, {len(mesh_a.faces)//4} faces")
print(f"Mesh B: {len(mesh_b.points)} vertices, {len(mesh_b.faces)//4} faces")


# Generate a bounding box capable of storing both meshes for SDF calculations
bounds_a = mesh_a.bounds
bounds_b = mesh_b.bounds
min_bounds = np.minimum(bounds_a[::2], bounds_b[::2]) - 0.1  # Add padding
max_bounds = np.maximum(bounds_a[1::2], bounds_b[1::2]) + 0.1

# (higher = more detail, but slower)
resolution = 64 

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

print(f"Grid resolution: {resolution}x{resolution}x{resolution}")
print(f"Total query points: {len(query_points)}")


# Computing SDF values for both meshes at the query points and then reconverting to 3D grid from 1D array.
print("Computing SDF for mesh A...")
sdf_a = mesh_to_sdf(mesh_a_tri, query_points, surface_point_method='scan', sign_method='depth')
sdf_a = sdf_a.reshape(resolution, resolution, resolution)

print("Computing SDF for mesh B...")
sdf_b = mesh_to_sdf(mesh_b_tri, query_points, surface_point_method='scan', sign_method='depth')
sdf_b = sdf_b.reshape(resolution, resolution, resolution)

print("SDF computation complete!")

num_frames = 10  # Number of intermediate morphing steps
morph_frames = []

for i, t in enumerate(np.linspace(0, 1, num_frames)):
    print(f"Generating morph frame {i+1}/{num_frames} (t={t:.2f})...")
    
    # Interpolate between the two SDFs
    sdf_interp = (1 - t) * sdf_a + t * sdf_b
    

    # Utilise marching cubes to create triangles where the SDF is zero (isosurface extraction)     
    try:
        # Extract the zero-level isosurface (where SDF = 0)
        verts, faces, normals, values = measure.marching_cubes(sdf_interp, level=0.0)
        
        # Scale vertices back to original coordinate space
        verts[:, 0] = verts[:, 0] / (resolution - 1) * (max_bounds[0] - min_bounds[0]) + min_bounds[0]
        verts[:, 1] = verts[:, 1] / (resolution - 1) * (max_bounds[1] - min_bounds[1]) + min_bounds[1]
        verts[:, 2] = verts[:, 2] / (resolution - 1) * (max_bounds[2] - min_bounds[2]) + min_bounds[2]
        
        # Create PyVista mesh from marching cubes output
        # Faces need to be formatted as [3, v0, v1, v2, 3, v3, v4, v5, ...]
        faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
        morph_mesh = pv.PolyData(verts, faces_pv)
        
        morph_frames.append(morph_mesh)
        print(f"  Generated mesh with {len(verts)} vertices")
        
    except Exception as e:
        print(f"  Warning: Could not generate mesh for frame {i}: {e}")

print(f"\nDisplaying {len(morph_frames)} morph frames...")
print("Close each window to see the next frame...")

for i, frame in enumerate(morph_frames):
    plotter = pv.Plotter()
    plotter.add_text(f"Mesh Morphing using SDFs - Frame {i+1}/{len(morph_frames)}", 
                     font_size=12, position='upper_edge')
    plotter.add_mesh(frame, color='lightblue', show_edges=False, smooth_shading=True)
    plotter.show()  # This will block until you close the window

print("Morphing animation complete!")


import numpy as np
import pyvista as pv
import trimesh
from time import time
from mesh_to_sdf import mesh_to_sdf
from skimage import measure

print("Loading and simplifying meshes...")
mesh_a = pv.read('test_models/hippocampus.stl')
mesh_b = pv.read('test_models/brain.stl')

# Use n_cells instead of n_faces
print(f"Original Mesh A: {mesh_a.n_cells} faces")
print(f"Original Mesh B: {mesh_b.n_cells} faces")

TARGET_FACES = 10000

if mesh_a.n_cells > TARGET_FACES:
    reduction = 1 - (TARGET_FACES / mesh_a.n_cells)
    mesh_a = mesh_a.decimate(reduction)
    print(f"Decimated Mesh A to: {mesh_a.n_cells} faces")

if mesh_b.n_cells > TARGET_FACES:
    reduction = 1 - (TARGET_FACES / mesh_b.n_cells)
    mesh_b = mesh_b.decimate(reduction)
    print(f"Decimated Mesh B to: {mesh_b.n_cells} faces")

# Converting from PyVista to Trimesh
mesh_a_tri = trimesh.Trimesh(
    vertices=mesh_a.points,
    faces=mesh_a.faces.reshape(-1, 4)[:, 1:4]
)
mesh_b_tri = trimesh.Trimesh(
    vertices=mesh_b.points,
    faces=mesh_b.faces.reshape(-1, 4)[:, 1:4]
)

# Generate a bounding box capable of storing both meshes for SDF calculations
bounds_a = mesh_a.bounds
bounds_b = mesh_b.bounds
min_bounds = np.minimum(bounds_a[0::2], bounds_b[0::2]) - 0.05  # Add padding
max_bounds = np.maximum(bounds_a[1::2], bounds_b[1::2]) + 0.05

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

print(f"\nGrid resolution: {resolution}x{resolution}x{resolution}")
print(f"Total query points: {len(query_points)}")

# Computing SDF values for both meshes at the query points and then reconverting to 3D grid from 1D array.
print("\nComputing SDFs...")
t0 = time()

print("Computing SDF for mesh A...")
sdf_a = mesh_to_sdf(
    mesh_a_tri,
    query_points,
    surface_point_method='sample',  # faster than 'scan'
    sign_method='normal',
    sample_point_count=10000
)
sdf_a = sdf_a.reshape(resolution, resolution, resolution)
t1 = time()
print(f"  Mesh A done in {t1 - t0:.1f}s")

print("Computing SDF for mesh B...")
sdf_b = mesh_to_sdf(
    mesh_b_tri,
    query_points,
    surface_point_method='sample',
    sign_method='normal',
    sample_point_count=10000
)
sdf_b = sdf_b.reshape(resolution, resolution, resolution)
t2 = time()
print(f"  Mesh B done in {t2 - t1:.1f}s")

print(f"\nSDF computation complete! Total time: {t2 - t0:.1f}s")

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

class ScrollViewer:
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.plotter = pv.Plotter()
        
        # Disable mouse wheel zoom so wheel only controls morphing
        self.plotter.iren.enable_terrain_style(mouse_wheel_zooms=False)
        
        self.actor = self.plotter.add_mesh(
            frames[0],
            color='lightblue',
            smooth_shading=True
        )
        self.text = self.plotter.add_text(
            f"Frame 1/{len(frames)} - Wheel: morph | Right-drag: zoom",
            position='upper_left',
            font_size=10
        )
        self.plotter.reset_camera()
        self.plotter.camera_position = 'iso'
        
        # Add mouse wheel control for morphing
        self.plotter.iren.add_observer(
            'MouseWheelForwardEvent',
            lambda obj, event: self.change(1)
        )
        self.plotter.iren.add_observer(
            'MouseWheelBackwardEvent',
            lambda obj, event: self.change(-1)
        )

    def change(self, delta):
        self.idx = np.clip(self.idx + delta, 0, len(self.frames) - 1)
        self.actor.GetMapper().SetInputData(self.frames[self.idx])
        self.text.SetText(
            0,
            f"Frame {self.idx+1}/{len(self.frames)} "
            f"({self.idx/(len(self.frames)-1)*100:.0f}%)"
        )
        self.plotter.render()

    def show(self):
        print("Controls:")
        print("  Mouse wheel: morph through frames")
        print("  Left drag: rotate | Middle drag: pan | Right drag: zoom")
        print("  Q: quit")
        self.plotter.show()

viewer = ScrollViewer(morph_frames)
viewer.show()

print("Morphing animation complete!")


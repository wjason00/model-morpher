import numpy as np
import pyvista as pv
import trimesh
from time import time
from mesh_to_sdf import mesh_to_sdf
from skimage import measure



print("Loading and validating meshes...")
mesh_a = pv.read('test_models/hippocampus.stl')
mesh_b = pv.read('test_models/brain.stl')


print(f"Original Mesh A: {mesh_a.n_cells} faces")
print(f"Original Mesh B: {mesh_b.n_cells} faces")



# Cleaning before trimesh conversion to improve reliability
print("\nCleaning meshes...")
mesh_a = mesh_a.clean()
mesh_a = mesh_a.fill_holes(hole_size=10000) 
mesh_a = mesh_a.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)


mesh_b = mesh_b.clean()
mesh_b = mesh_b.fill_holes(hole_size=10000)
mesh_b = mesh_b.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)


# Increase to improve the likelihood of watertight meshes.
TARGET_FACES = 5000  # Reduced for faster, more reliable processing


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

# Removing degenerate faces before repair 
mesh_a_tri.update_faces(mesh_a_tri.unique_faces())
mesh_a_tri.update_faces(mesh_a_tri.nondegenerate_faces())

mesh_b_tri.update_faces(mesh_b_tri.unique_faces())
mesh_b_tri.update_faces(mesh_b_tri.nondegenerate_faces())

# Repairing any holes / normals before SDF computation
if not mesh_a_tri.is_watertight:
    print("\nRepairing mesh A repair")

    mesh_a_tri.fill_holes()
    mesh_a_tri.fix_normals()
    mesh_a_tri.update_faces(mesh_a_tri.unique_faces())
    mesh_a_tri.update_faces(mesh_a_tri.nondegenerate_faces())

    mesh_a_tri.update_faces(mesh_a_tri.unique_faces())
    mesh_a_tri.update_faces(mesh_a_tri.nondegenerate_faces())

if not mesh_b_tri.is_watertight:
    print("Repairing mesh B repair")

    mesh_b_tri.fill_holes()
    mesh_b_tri.fix_normals()
    mesh_b_tri.update_faces(mesh_b_tri.unique_faces())
    mesh_b_tri.update_faces(mesh_b_tri.nondegenerate_faces())

    mesh_b_tri.update_faces(mesh_b_tri.unique_faces())
    mesh_b_tri.update_faces(mesh_b_tri.nondegenerate_faces())


print(f"Mesh A watertight: {mesh_a_tri.is_watertight}")
print(f"Mesh B watertight: {mesh_b_tri.is_watertight}")

# Recentering meshes to ensure smoother operation
mesh_a_tri.vertices -= mesh_a_tri.centroid
mesh_b_tri.vertices -= mesh_b_tri.centroid


# Generating bounds from trimesh to ensure consistency
bounds_a = mesh_a_tri.bounds
bounds_b = mesh_b_tri.bounds

# Computing diagonal lengths to account for scale factor (3D space)
diag_a = np.linalg.norm(bounds_a[1] - bounds_a[0])
diag_b = np.linalg.norm(bounds_b[1] - bounds_b[0])
scale_factor = diag_a / diag_b
mesh_b_tri.vertices *= scale_factor


print(f"\nMesh A bounds: {mesh_a_tri.bounds}")
print(f"Mesh B bounds (after scaling): {mesh_b_tri.bounds}")



# A bounding box to account for both meshes
min_bounds = np.minimum(mesh_a_tri.bounds[0], mesh_b_tri.bounds[0]) - 0.5
max_bounds = np.maximum(mesh_a_tri.bounds[1], mesh_b_tri.bounds[1]) + 0.5
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
print(f"Grid bounds: min={min_bounds}, max={max_bounds}")

# Computing SDF values for both meshes at the query points and then reconverting to 3D grid from 1D array.
print("\nComputing SDFs...")
t0 = time()


print("Computing SDF for mesh A")
sdf_a = mesh_to_sdf(
    mesh_a_tri,
    query_points,
    surface_point_method='sample', 
    sign_method='depth',
    sample_point_count=10000,  
    normal_sample_count=5       
)
sdf_a = sdf_a.reshape(resolution, resolution, resolution)
t1 = time()
print(f"  Mesh A done in {t1 - t0:.1f}s")
print(f"  SDF A range: [{sdf_a.min():.3f}, {sdf_a.max():.3f}]")


print("Computing SDF for mesh B (brain)...")
sdf_b = mesh_to_sdf(
    mesh_b_tri,
    query_points,
    surface_point_method='sample',
    sign_method='depth',
    sample_point_count=10000,  
    normal_sample_count=5       
)

sdf_b = sdf_b.reshape(resolution, resolution, resolution)
t2 = time()
print(f"  Mesh B done in {t2 - t1:.1f}s")
print(f"  SDF B range: [{sdf_b.min():.3f}, {sdf_b.max():.3f}]")


print(f"\nSDF computation complete! Total time: {t2 - t0:.1f}s")

num_frames = 20
morph_frames = []


for i, t in enumerate(np.linspace(0, 1, num_frames)):
    print(f"Generating morph frame {i+1}/{num_frames} (t={t:.2f})...")
    
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
            )
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


print(f"\n{'='*60}")
print(f"Successfully generated {len(morph_frames)} morph frames!")
print(f"{'='*60}\n")

# General Viewer class to handle mouse wheel scrolling through frames
class ScrollViewer:
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.plotter = pv.Plotter()
        
        self.plotter.iren.enable_terrain_style(mouse_wheel_zooms=False)
        
        self.actor = self.plotter.add_mesh(
            frames[0],
            color='lightblue',
            smooth_shading=True,
            show_edges=False  # Cleaner look
        )
        self.text = self.plotter.add_text(
            f"Controls | Left Click to Pan | Mouse Wheel to Morph | Right Click to Zoom | Q to Quit",
            position='upper_left',
            font_size=12
        )
        self.plotter.reset_camera()
        self.plotter.camera_position = 'iso'
        
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
        progress = self.idx / (len(self.frames) - 1) * 100 if len(self.frames) > 1 else 0
        self.text.SetText(
            0,
            f"Frame {self.idx+1}/{len(self.frames)} ({progress:.0f}%)"
        )
        self.plotter.render()

viewer = ScrollViewer(morph_frames)
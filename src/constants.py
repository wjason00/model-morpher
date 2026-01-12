# Main.py

# Loading settings
PREVIEW_RES = 50
PREVIEW_FRAMES = 10

# Quality Ver
RESOLUTION = 150 # Resolution increases cost cubically (meshgrid)
FRAME_COUNT = 20
TARGET_FACES = 15000 # Number of faces after decimating the mesh.

# sdf_tools.py
TOLERANCE = 1e-6 # Tolerance for cleaning meshes
SMOOTH_ITER = 50 # Number of smoothing iterations
RELAX_FACTOR = 0.15 # Relaxation factor for smoothing
PADDING = 0.02 # Padding around bounding box for SDF grid
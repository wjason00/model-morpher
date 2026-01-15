# Main.py

# Loading settings - Preview
PREVIEW_RES = 50
PREVIEW_FRAMES = 10

# Quality Ver
RESOLUTION = 256 # Resolution increases cost cubically (meshgrid)
FRAME_COUNT = 30
TARGET_FACES = 15000 # Number of faces after decimating the mesh.

# sdf_tools.py
TOLERANCE = 1e-6 # Tolerance for cleaning meshes
SMOOTH_ITER = 50 # Number of smoothing iterations
RELAX_FACTOR = 0.15 # Relaxation factor for smoothing

# Padding at 0.0001 gives a cool blocky effect - use for future? 
PADDING_PERCENT = 0.0001 

# Normalization ensures the zero-crossing gradient is consistent across different mesh scales
# Higher values = steeper gradient = sharper but potentially noisier surfaces
# Lower values = gentler gradient = smoother but potentially blobby surfaces
SDF_NORMALIZATION_SCALE = 0.3  # Normalizes SDF so values span roughly [-1, 1] near surface

# Mesh Quality Post-Processing
# Taubin smoothing alternates between positive/negative lambda to prevent shrinkage
# Unlike Laplacian smoothing which shrinks meshes, Taubin preserves volume
TAUBIN_ITERATIONS = 30  # Number of Taubin smoothing passes (more = smoother)
TAUBIN_PASS_BAND = 0.05  # Frequency cutoff - lower preserves more detail, higher smooths more

# Quadruples face count. 
SUBDIVIDE_ITERATIONS = 1 

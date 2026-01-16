import pyvista as pv

from config import (
    TOLERANCE,
    TAUBIN_ITERATIONS,
    TAUBIN_PASS_BAND,
    SUBDIVIDE_ITERATIONS
)

def postprocess_mesh(mesh: pv.PolyData) -> pv.PolyData:
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
    
    """# Ensure largest connected components are kept and therefore reduces blobbing.
    mesh = mesh.connectivity(extraction_mode = 'largest')
    if mesh.n_points == 0:
        return None"""
    
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
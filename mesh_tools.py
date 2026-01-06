import numpy as np 
import pyvista as pv 
import trimesh 

def load_and_clean_mesh(file_path, target_faces = 5000, hole_size = 10000): 
    """
    General loading of an STL file into a cleaned PyVista mesh.    
    
    :param file_path: Destination of the mesh file to load
    :param target_faces: Number of faces after decimation (increase to increase quality)
    :param hole_size: Maximum size of holes to fill in the mesh (increase to fill more holes)
    """

    print(f"Loading mesh from {file_path}...")
    mesh = pv.read(file_path)
    print(f"    Original Mesh: {mesh.n_cells} faces")
    
    # Cleaning and filling holes. 
    mesh = mesh.clean()
    mesh = mesh.fill_holes(hole_size=hole_size) 
    mesh = mesh.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)

    if mesh.n_cells > target_faces:   
        reduction = 1 - (target_faces / mesh.n_cells)
        mesh = mesh.decimate(reduction)
        print(f"    Decimated Mesh to: {mesh.n_cells} faces")

    return mesh 

def pyvista_to_trimesh(pv_mesh):
    """
    Conversion from Pyvista mesh to Trimesh format.
    
    :param pv_mesh: PyVista mesh to be converted
    """
    # Converting from PyVista to Trimesh
    return trimesh.Trimesh(
        vertices=pv_mesh.points,
        faces=pv_mesh.faces.reshape(-1, 4)[:, 1:4]
    )


def repair_mesh(mesh):
    """
    Detect if mesh needs to be repaired and perform repair if necessary via filling holes and fixing normals.
    
    :param mesh: Mesh to be repaired
    """
    # Removing degenerate faces before repair 
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    # Repairing any holes / normals before SDF computation
    if not mesh.is_watertight:
        print("Repairing mesh...")

        mesh.fill_holes()
        mesh.fix_normals()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())

        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
    else:
        print(" Already watertight - skipping repair")

    return mesh.is_watertight

def normalise_meshes(mesh_a, mesh_b):
    """
    Recenter and scale meshes to similar sizes for improved morphing results.
    
    :param mesh_a: Description
    :param mesh_b: Description
    """
    # Recentering meshes to ensure smoother operation
    mesh_a.vertices -= mesh_a.centroid
    mesh_b.vertices -= mesh_b.centroid

    # Generating bounds from trimesh to ensure consistency
    bounds_a = mesh_a.bounds
    bounds_b = mesh_b.bounds

    # Computing diagonal lengths to account for scale factor (3D space)
    diag_a = np.linalg.norm(bounds_a[1] - bounds_a[0])
    diag_b = np.linalg.norm(bounds_b[1] - bounds_b[0])
    scale_factor = diag_a / diag_b
    mesh_b.vertices *= scale_factor

    return mesh_a, mesh_b
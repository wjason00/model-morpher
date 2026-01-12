import numpy as np 
import pyvista as pv 
import trimesh 

from os import path

def load_and_clean_mesh(file_path, target_faces = 15000, hole_size = 10000): 
    """
    General loading of an STL file into a cleaned PyVista mesh.    
    
    :param file_path: Destination of the mesh file to load
    :param target_faces: Number of faces after decimation (increase to increase quality)
    :param hole_size: Maximum size of holes to fill in the mesh (increase to fill more holes)
    """
    
    if path.exists(file_path): 
        mesh = pv.read(file_path) 
    else:
        raise FileNotFoundError(f"Mesh file not found at: {file_path}")
        
    # Cleaning just merges duplicate points (near-duplicate to my knowledge) and deletes unreferenced vertices/ 
    # It also removes weird artifacts which may have a domino effect later on.
    # Computing normals is important because later on SDF computation utilises this for
    # sign estimation / repairing normals which were flipped.
    mesh = mesh.clean()
    mesh = mesh.fill_holes(hole_size=hole_size) 
    mesh = mesh.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)

    # Decimating essentially identifies the edges that are most likely to change the least (via looking at curvature and length and etc.)
    # and removes these edges / triangles alongside attempting to keep the important stuff. Then it worries about connectivity.
    if mesh.n_cells > target_faces:   
        reduction = 1 - (target_faces / mesh.n_cells)
        mesh = mesh.decimate(reduction)
        print(f"    Decimated Mesh to: {mesh.n_cells} faces")

    return mesh 

def pyvista_to_trimesh(pv_mesh):
    """
    Conversion from Pyvista mesh to Trimesh format by utilising vertices from the PyVista mesh and then storing as a flat array.
    reshape has use of -1 for automatic dimensions (maybe for future compatibility?) and rearranges into a length of 4.  
    
    :param pv_mesh: PyVista mesh to be converted
    """

    return trimesh.Trimesh(
        vertices=pv_mesh.points,
        faces=pv_mesh.faces.reshape(-1, 4)[:, 1:4] # For future ref: [:, 1:4] just drops the 3 vertices count for each face.
    )


def repair_mesh(mesh):
    """
    Detect if mesh needs to be repaired and perform repair if necessary via filling holes and fixing normals.
    Methodology behind updating the faces first is to ensure that any leaks can be identified.
    
    :param mesh: Mesh to be repaired
    """
    # This is different to cleaning in PyVista which was for its own meshes, this is utilising Trimesh cleaning. 
    # Firstly, unique_faces is used to remove duplicate faces and then nondegenerate_faces the degenerate faces (degenerate just means it doesn't really exist due to an area of 0)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    
    # Multiple repair passes for stubborn meshes
    for repair_pass in range(3):
        if mesh.is_watertight:
            break
            
        print(f"  Repair pass {repair_pass + 1}...")
        
        # Fix winding and normals first - critical for SDF sign computation
        mesh.fix_normals()
        mesh.fill_holes()
        
        # Clean up any artifacts created by hole filling
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        
        # Merge close vertices that might be causing gaps
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
    
    return mesh.is_watertight

def normalise_meshes(meshes):
    """
    Recenter and scale meshes to similar sizes for improved intermediate SDF computation
    
    :param meshes: List of trimesh objects to normalise
    """

    # Offsetting each meshs' vertices by its centre to appear at world space origin. (just think like voxel origin being at [3, 2, 1] would have to be offset by [-3, -2, -1])
    for mesh in meshes: 
        mesh.vertices -= mesh.centroid 

    reference_diag = np.linalg.norm(meshes[0].bounds[1] - meshes[0].bounds[0])

    # Start from the first mesh and scale all others relative to it. 
    # Simple volume scale factor formula i.e. big / small = scale then multiply the small one. 
    for i, mesh in enumerate(meshes[1: ], start=1): 
        # Calculate the normal of the diagonal to find scale factor 
        mesh_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

        scale_factor = reference_diag / mesh_diag
        mesh.vertices *= scale_factor

        print(f"{scale_factor} scale for mesh {i+1}") 

    return meshes
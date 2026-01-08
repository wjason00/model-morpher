from brainglobe_atlasapi import BrainGlobeAtlas
from mesh_tools import load_and_clean_mesh, pyvista_to_trimesh, repair_mesh, normalise_meshes
from sdf_tools import morph_meshes, morph_mesh_sequence
from viewer import ScrollViewer
from time import time 

t0 = time() 
TARGET_FACES = 15000 # Number of faces after decimating the mesh. 
RESOLUTION = 150 # Resolution increases cost cubically (meshgrid)
FRAME_COUNT = 20


# Using mouse brain atlas (can be swapped out in future)
atlas = BrainGlobeAtlas('allen_mouse_25um', check_latest = False)

object_a = atlas.meshfile_from_structure("DG")
object_b = atlas.meshfile_from_structure("CB")

# Loading the brain from the brain atlas instead of using file
# Multi-mesh morphing: Add more structures here for sequence morphing (A → B → C → ...)
# Example: morph through multiple brain regions
# REPLACE ATLAS PART WITH YOUR OWN MESH FILE PATHS IF WANTED
mesh_sequence_sources = [
    atlas.meshfile_from_structure("DG"),    # Dentate Gyrus
    "test_models\Hand_SUPERfinal.stl",   
    atlas.meshfile_from_structure("root"),  # Full brain
    "test_models\halfpoly_suzanne.stl", 
    atlas.meshfile_from_structure("DG"),  # DG again to loop back
]


def prepare_mesh(file_path):
    """Helper function to load, convert, and repair a single mesh."""
    pv_mesh = load_and_clean_mesh(file_path, target_faces=TARGET_FACES)
    tri_mesh = pyvista_to_trimesh(pv_mesh)
    repair_mesh(tri_mesh)
    return tri_mesh


def main():
    print("Loading and validating meshes...")
    
    # Load and prepare all meshes
    trimeshes = []
    for i, source in enumerate(mesh_sequence_sources):
        print(f" Preparing mesh {i+1}/{len(mesh_sequence_sources)}")
        tri_mesh = prepare_mesh(source)
        trimeshes.append(tri_mesh)
    
    # Normalising is just when the meshes are centered and scaled to make SDF computation easier
    print("\nNormalising mesh sequence...")
    trimeshes = normalise_meshes(trimeshes)
    
    # Generate the frames after all the big preprocessing is confirmed
    morph_frames = morph_mesh_sequence(
        trimeshes, 
        resolution=RESOLUTION, 
        frames_per_transition=FRAME_COUNT
    )

    print(f"Total time to reach viewer: {time() - t0:.2f} seconds")
    
    viewer = ScrollViewer(morph_frames)
    viewer.show()

    print("Morphing animation complete!")


if __name__ == "__main__":
    main()
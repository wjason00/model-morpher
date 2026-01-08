from brainglobe_atlasapi import BrainGlobeAtlas
from mesh_tools import load_and_clean_mesh, pyvista_to_trimesh, repair_mesh, normalise_meshes
from sdf_tools import morph_meshes
from viewer import ScrollViewer

TARGET_FACES = 15000 # Number of faces after decimating the mesh. 
RESOLUTION = 150 # Resolution increases cost cubically (meshgrid)
FRAME_COUNT = 20

# Using mouse brain atlas (can be swapped out in future)
atlas = BrainGlobeAtlas('allen_mouse_25um', check_latest = False)

# Loading the brain from the brain atlas instead of using file
# REPLACE ATLAS PART WITH YOUR OWN MESH FILE PATHS IF WANTED
object_a = atlas.meshfile_from_structure("DG")
object_b = atlas.meshfile_from_structure("root")

def main():

    print("Loading and validating meshes...") 

    mesh_a = load_and_clean_mesh(object_a, target_faces=TARGET_FACES)
    mesh_b = load_and_clean_mesh(object_b, target_faces=TARGET_FACES)
    trimesh_a = pyvista_to_trimesh(mesh_a)
    trimesh_b = pyvista_to_trimesh(mesh_b)

    # Repairing is where all of the holes and watertightness of each mesh is able to be checked and fixed if necessary.
    repair_mesh(trimesh_a)
    repair_mesh(trimesh_b)

    # Normailising is just when the meshes are centered and scaled to make SDF computation easier (less travelling across the world space?)
    print(f"Mesh B bounds before scaling: {trimesh_b.bounds}")
    trimesh_a, trimesh_b = normalise_meshes(trimesh_a, trimesh_b)
    print(f"Mesh B bounds (after scaling): {trimesh_b.bounds}")

    # Generate the frames after all the big preprocessing is confirmed? (increase resolution up to around 100 for best effect)
    # Increasing frame count past a certain point has diminishing returns (from experimentation)
    morph_frames = morph_meshes(trimesh_a, trimesh_b, resolution=RESOLUTION, num_frames=FRAME_COUNT)

    viewer = ScrollViewer(morph_frames)
    viewer.show() # Keeps the whole frame up and running until closed - might be a better way to do this?

    print("Morphing animation complete!")


if __name__ == "__main__":
    main()
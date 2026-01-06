from mesh_tools import load_and_clean_mesh, pyvista_to_trimesh, repair_mesh, normalise_meshes
from sdf_tools import morph_meshes
from viewer import ScrollViewer


def main():
    print("Loading and validating meshes...") 

    # Load meshes
    # Increase to improve the likelihood of watertight meshes.
    TARGET_FACES = 5000  # Reduced for faster, more reliable processing

    mesh_a = load_and_clean_mesh('test_models/hippocampus.stl', target_faces=TARGET_FACES)
    mesh_b = load_and_clean_mesh('test_models/brain.stl', target_faces=TARGET_FACES)

    # Convert to Trimesh
    mesh_a_tri = pyvista_to_trimesh(mesh_a)
    mesh_b_tri = pyvista_to_trimesh(mesh_b)

    # Repair meshes
    print("\nRepairing mesh A repair")
    watertight_a = repair_mesh(mesh_a_tri)

    print("Repairing mesh B repair")
    watertight_b = repair_mesh(mesh_b_tri)

    print(f"Mesh A watertight: {watertight_a}")
    print(f"Mesh B watertight: {watertight_b}")

    # Normalize (center and scale)
    mesh_a_tri, mesh_b_tri = normalise_meshes(mesh_a_tri, mesh_b_tri)

    print(f"\nMesh A bounds: {mesh_a_tri.bounds}")
    print(f"Mesh B bounds (after scaling): {mesh_b_tri.bounds}")

    # Morph meshes
    morph_frames = morph_meshes(mesh_a_tri, mesh_b_tri, resolution=64, num_frames=20)

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(morph_frames)} morph frames!")
    print(f"{'='*60}\n")

    # View results
    viewer = ScrollViewer(morph_frames)
    viewer.show()

    print("Morphing animation complete!")


if __name__ == "__main__":
    main()
# Model Morpher (Python-first draft)

A framework for generating smooth morphological transitions between 3D meshes using SDF interpolation and isosurface extraction via Marching Cubes formulae, before displaying in a PyVista scroller to allow for controlled morphing.

This is a personal passion project of mine.

---
## Loading Screen
![Recording 2026-01-12 at 23 41 37](https://github.com/user-attachments/assets/409f8347-e099-4562-93c6-8f19ac6a708e)
*Preview Mesh computation - 1.1s (delay is frame generation time)*

## Examples
![Recording 2026-01-08 at 13 27 08](https://github.com/user-attachments/assets/344589aa-08c4-428c-85fd-9b129692bbfb)
*Model Morph from brain model to hand model. Computed at resolution = 100 with a time of 6.5s and TARGET_FACES = 5000.*

![Recording 2026-01-08 at 13 30 48](https://github.com/user-attachments/assets/d2f6f64e-06fb-41a5-a455-3831474cf190)
*Model Morph from dentage gyrus (DG brainatlas code) to brain (root brainatlas code). Computed at resolution = 150 with a time of 39.2s and TARGET_FACES = 15,000.*

![Recording 2026-01-08 at 19 41 29](https://github.com/user-attachments/assets/75a6cd31-2b7c-462c-b599-31075d81a48c)
*Model Morph (dentate gyrus -> hand model -> brain -> suzanne (blender monkey) -> dentate gyrus). Computed at resolution = 150 and TARGET_FACES = 15,000* 
*Total Time = 84.52 seconds. SDF Computation = 50.9s*


## What it does

### 1. Preparing Meshes

- Cleans meshes (hole filling and computes normals)
- Decimates faces to simplify mesh
- Removes degenerate faces
- Merge nearby vertices together
- Centres and scales meshes

### 2. Grid Generation & Coordinate Mapping

- Generates a grid of query points for SDF computation using Torch tensors.
- Generate dynamic bounding relative to each mesh size utilising anisotropic padding.

For a grid with resolution $(N_x, N_y, N_z)$ and voxel indices $(i, j, k)$:
- **Flattened index:** $i \times N_y N_z + j \times N_z + k$ (done via ravelling of meshgrid by NumPy)
- **World position:** $(x_i, y_j, z_k)$ where coordinates are linearly spaced between bounds

### 3a. SDF Computation (CPU)

- Calculate signed distance for each grid point for pairs of the meshes.
- Utilises typical conventions (− for interior, + for exterior)
- Parallelised computation to reduce time complexity (will depend on user number of cores - accounted for)
- Map vertices from voxel coordinates back to world space using grid spacing (3D -> 1D array)

### 3b. SDF Computation (GPU)

- Temporary wavefront (.obj) files generated from trimesh
- PyTorch Volumetric MeshSDF utilising GPU-acceleration on CUDA device for calculating SDF. 
- Convert SDF to voxel units via element-wise cube rooting of voxel-spacing from world space coordinates. 

### 4. Interpolation of SDF

Generates intermediate SDF representations by linear blending:

$$\text{SDF}_{\text{morph}}(t) = (1-t) \times \text{SDF}_A + t \times \text{SDF}_B \quad t \in [0,1]$$

### 5. Surface Reconstruction 

- Multi-core parallelised transitioning of frames. 
- Utilise marching cubes to extract isosurface at each interpolation step (threshold: 0.0, with a spacing defined by resolution).
- Affine transformation applied to each axis to recover world position.

### 6. Postprocessing 

- Post-process the surface (clean, fill and smooth). 
- Utilise connectivity to ensure small blobs are removed. 
- Subdivide iterations used to increase face count (x4 for each iteration)
- Taubin smoothing used instead of Laplacian smoothing to decrease volume lost. 
- Normals are counted to ensure that shading is smooth and consistent via computer_normals()

### 7. Frame Generation 

- Parallelised frame generation.
- Preview generated whilst main loading occurs using wireframe mesh.
- Intermediate mesh generation from the vertices and faces. 
- Account for errors by using fallback frames (i.e. the last frame generated)
- Account for preventing duplicates by skipping last transistion frame of each transistion
  

### 8. Interactive Viewer

- PyVista-based 3D viewer with scroll-through frame navigation
- Mouse wheel controls morph progression between frames

## Prerequisites

- Python 3.9+
- System packages for VTK/PyVista may be required.

### GPU recommendations 

- CUDA

## Setup

If attempting to install PyTorch3D - good idea to check out: 

<https://github.com/MiroPsota/torch_packages_builder/issues/10> (helped me install)

### Option 1: Conda (Recommended)

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate model-morpher
```

### Option 2: pip/venv

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS or Linux
python -m venv .venv
source .venv/bin/activate 

pip install -r requirements.txt
```

## Running

- Add meshes to `test_models/` (e.g., `hippocampus.stl`, `brain.stl`).
- Execute:

```bash
python main.py
```

**Controls:** Scroll wheel navigates frames | Left-drag pans | Right-drag zooms | Q quits

## Configuration

Parameters in `config.py`:

```python
TARGET_FACES = 5000  # Target faces after decimation (lower means more aggressive removal of faces)
resolution = 64      # SDF meshgrid resolution (R³)
num_frames = 20      # Number of frames to generate
```

## Project Structure

```
mesh_morpher/
├── core/
│   ├── __init__.py
│   ├── device.py           # Device management - Check for CUDA
│   ├── grid.py             # Grid generation & bounds 
│   └── sdf.py              # SDF construction & queries 
├── processing/
│   ├── __init__.py
│   ├── isoextraction.py       # Marching cubes and isoextraction
│   └── postprocess.py      # Mesh cleaning/smoothing 
├── morphing/
│   ├── __init__.py
│   └── interpolator.py     # Main morph pipeline 
├── config.py               # Constants
├── main.py                 # Qt application
└── viewer.py               # Viewer
```


## Limitations

- Repairing meshes is limited to simple topological defects (i.e. simple holes) and therefore there will still be some weird artifacts. 
- Memory scales (and therefore the time taken) cubically with grid resolution
- Computational complexity (and the number of mistakes unfortunately) dependent on mesh density and SDF sampling rate.

## Notes for the timeline

Generally aiming to produce a stable Python-only version before considering other user experience aspects such as JS front end for ease-of-use (alongside a GIF export option?) or PyVista file-adding UX.

- Intended as a smaller milestone toward a larger neuroscience project and will likely be featured as a package within. 
- GUI is only intended for visualisation purposes, in reality, it'll be likely exported as a downloadable GIF. 

## Development Roadmap

- [X] GPU-accelerated SDF computation (In process - however really hard to understand correct packages / dependencies for PyTorch3D and PyTorch)
- [X] Batch processing capabilities (Accomplished for CPU only) 
- [ ] Animation export (video/GIF formats)
- [X] File insertion via GUI for high-level mesh morphing.
- [ ] Convert into a deployable and stable package.
- [X] Increase the number of meshes that are able to be morphed. (For however many meshes you want)
- [X] Loading Screen
- [ ] Textures implemented. 
- [ ] Texture morphing implemented
- [X] Preview implemented (wireframe preview generated)

## License

MIT License

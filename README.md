# Model Morpher (Python-first draft)

A framework for generating smooth morphological transitions between two 3D meshes using SDF interpolation and isosurface extraction via Marching Cubes formulae.

![Recording 2026-01-06 at 18 12 31](https://github.com/user-attachments/assets/50d7a3e8-b733-4518-be16-d062bc56182c)

## What it does

### 1. Preparing Meshes
- Cleans meshes (hole fillin and computes normals) 
- Decimates faces to simplify mesh 
- Removes degenerate faces
- Centres and scales meshes

### 2. SDF Computation
- Create 3D voxel grid bounding both meshes 
- Calculate signed distance for each vertex
- Utilises typical conventions (- for interior, + for exterior)

### 3. Interpolation of SDF

- Generates intermediate mesh representations: 

\[ \text{SDF}_{\text{morph}}(t) = (1-t) \cdot \text{SDF}_A + t \cdot \text{SDF}_B \quad t \in [0,1] \]

### 4. Reconstruction of Surface

- Utilise marching cubes to extract isosurface for each intermediate mesh (threshold: -0.01).
- Map vertices from voxel coordinates to world space utilising grid spacing (NumPy linalg)

### 5. Coordinate Mapping

For a grid with resolution \((N_x, N_y, N_z)\) and voxel indices \((i,j,k)\) therefore meaning that each voxel coordinate can be mapped to a real world space:
- **Flattened index:** \(i \cdot N_y N_z + j \cdot N_z + k\) (done via ravelling of meshgrid by NumPy)
- **World position:** \((x_i, y_j, z_k)\) 


## Prerequisites

- Python 3.9+
- System packages for VTK/PyVista may be required.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

## Running

- Add meshes to `test_models/` (e.g., `hippocampus.stl`, `brain.stl`).
- Execute:

```bash
python main.py
```

**Controls:** Scroll wheel navigates frames | Left-drag rotates | Right-drag zooms | Q quits

## Configuration

Parameters in `main.py`:

```python
TARGET_FACES = 5000  # Target faces after decimation
resolution = 64      # SDF grid resolution (R³)
num_frames = 20      # Number of frames to generate
```

## Project Structure

```
model-morpher/
├── main.py              # Main execution script
├── mesh_tools.py        # Mesh handling
├── sdf_tools.py       # SDF computation and frame generation
├── viewer.py            # Interactive 3D viewer 
├── requirements.txt     # Python dependencies
└── test_models/
    ├── hippocampus.stl
    └── brain.stl
```


## Limitations

- Repairing meshes is limited to simple topological defects (i.e. simple holes)
- Memory scales (and therefore the time taken) cubically with grid resolution
- Computational complexity (and the number of mistakes unfortunately) dependent on mesh density and SDF sampling rate.

## Notes for the timeline

Generally aiming to produce a stable Python-only version before considering other user experience aspects such as JS front end for ease-of-use (alongside a GIF export option?) or PyVista file-adding UX.

- Intended as a smaller milestone toward a larger neuroscience project and will likely be featured as a package within. 
- If meshes are large, reduce `TARGET_FACES` or `resolution` in `main.py`. (GPU version will come soon)

## Development Roadmap

- GPU-accelerated SDF computation
- Batch processing capabilities
- Animation export (video/GIF formats)
- File insertion via GUI for high-level mesh morphing.
- Convert into a deployable and stable package. 

## License

MIT License
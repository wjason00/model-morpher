# Model Morpher (Python-first draft)

This repo holds a quick Python-only prototype for a neuroscience timeline project later on: generating smooth morphs between two meshes (e.g., hippocampus → whole brain) before any JavaScript deployment or additional quality-of-life tooling is added.

## What it does

- Cleans and decimates two STL meshes utilising PyVista decimate function.
- Converts meshgrids to a 1D array via ravelling and stacking, utilising voxel indices mapping to world coordinates.
- Interpolates SDFs to build intermediate shapes.
- Utilises marching cubes to reconstruct meshes to display into a scrollable PyVista UI.

## Prerequisites

- Python 3.9+
- System packages for VTK/PyVista may be required.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pyvista trimesh mesh-to-sdf scikit-image
```

## Running

- Add meshes to `test_models/` (e.g., `hippocampus.stl`, `brain.stl`).
- Execute:

```bash
python main.py
```

- Scroll wheel morphs through frames; mouse drag/zoom controls remain available.

## Notes for the timeline

- This stage is Python-only: no JS front end, no extra file-adding UX.
- Intended as a smaller milestone toward a larger neuroscience project; future work will wrap this in a web UI and add nicer asset management.
- If meshes are large, reduce `TARGET_FACES` or `resolution` in `main.py`. (GPU version will come soon)

## Troubleshooting

- **Non-watertight meshes:** lower the hole-filling threshold in `main.py`.

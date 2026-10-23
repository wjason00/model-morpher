# Model Morpher

A Python desktop tool for exploring shape transitions between 3D meshes. Load a sequence of models, generate the intermediate surfaces, and scroll through the result in a PyVista viewer. Meshes can come from local files or the built-in BrainGlobe atlas browser.

This personal project explores morphing through **signed distance fields (SDFs)**: each mesh becomes a grid of distances to its surface, the grids are blended, and Marching Cubes reconstructs the intermediate shapes. The result is a visual shape transition; it does not establish anatomical correspondence or model biological development.

![Morphing sequence through dentate gyrus, hand, brain and Suzanne meshes](https://github.com/user-attachments/assets/707aa3f8-a590-46b0-a379-6e6e685866cf)

*A sequence through dentate gyrus, hand, whole-brain and Suzanne meshes, returning to dentate gyrus.*

## Try it

The current application targets an NVIDIA CUDA environment. The GPU pipeline is implemented; dependency compatibility and the CPU fallback remain limitations. See [requirements and setup](#requirements-and-setup) before installing.

Once the environment is ready, run from the repository root:

```bash
python main.py
```

1. Click **Load Meshes** and select at least two models. The included [hand](test_models/Hand_SUPERfinal.stl) and [Suzanne](test_models/halfpoly_suzanne.stl) STL files provide a starting pair. Files can stay anywhere on disk; they do not need to be copied into `test_models/`.
2. Alternatively, choose a BrainGlobe atlas, search its structures, and click **Add Selected**. The atlas panel can be hidden to give the preview more space.
3. Click **Start Morphing**. A lower-resolution wireframe preview appears before the full sequence finishes.
4. Explore the completed sequence in the viewer. The output is an interactive collection of mesh frames; video/GIF export is still planned.

| Control | Action |
| --- | --- |
| Mouse wheel | Move through frames |
| Left / Right arrow | Move one frame |
| Down / Up arrow | Move ten frames |
| Home / End | Jump to the first / last frame |
| Q | Quit the viewer |

## Loading and preview

![Model Morpher loading window with atlas selection and wireframe preview](https://github.com/user-attachments/assets/f1437f0a-8eeb-40ac-bd5f-9c7359bcf720)

*The loading window combines atlas selection, an adjustable split view and a wireframe preview.*

## Requirements and setup

- **Python 3.10** is specified in [environment.yml](environment.yml).
- **A desktop display with Qt and VTK/PyVista support** is required for the GUI and 3D viewer. Additional system graphics libraries may be needed for your platform.
- **NVIDIA GPU and a compatible CUDA/PyTorch/PyTorch3D installation** are the current target. The supplied dependency files specify PyTorch 2.9.1, CUDA 12.8 and PyTorch3D 0.7.9; compatible builds must be available for your operating system and Python version.
- **CPU-only use needs repair.** The worker selects a CPU fallback when CUDA is unavailable, but currently passes an unsupported `device` argument to that function. The CPU module also has unresolved configuration references. Installing CPU packages alone does not make this a working CPU-only application.
- **Network access and disk space for atlas data** may be needed: the GUI requests a BrainGlobe atlas on startup, and an uncached atlas must be downloaded.

Clone the repository and enter its directory:

```bash
git clone https://github.com/wjason00/model-morpher.git
cd model-morpher
```

The supplied environment files are setup recipes, not a portable lockfile. In particular, PyTorch3D availability varies by platform, and the pip requirements include CUDA-specific wheel versions. A default package index may not provide those builds. Match PyTorch, CUDA and PyTorch3D before running the application.

### Conda

```bash
conda env create -f environment.yml
conda activate model-morpher
```

If a pinned package is unavailable for your platform, adjust the recipe to a compatible build rather than mixing incompatible PyTorch and PyTorch3D versions.

### pip / venv

Create and activate an environment using Python 3.10:

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux shell
python3.10 -m venv .venv
source .venv/bin/activate
```

With a source for the matching CUDA wheels configured, install the requirements:

```bash
python -m pip install -r requirements.txt
```

The [PyTorch3D installation discussion](https://github.com/MiroPsota/torch_packages_builder/issues/10) helped with the original setup and may be useful when resolving build availability. It is an installation reference, rather than a guarantee that the supplied versions work on every platform.

## How it works

```text
Local meshes / atlas structures
    -> clean, repair, decimate, centre and scale
    -> sample signed distances on a shared 3D grid
    -> blend neighbouring SDFs
    -> extract surfaces with Marching Cubes
    -> clean and smooth the surfaces
    -> preview and scroll through mesh frames
```

For each neighbouring pair of meshes, the interpolation is:

$$\mathrm{SDF}(t) = (1-t)\,\mathrm{SDF}_A + t\,\mathrm{SDF}_B,\qquad t\in[0,1].$$

The CUDA path uses PyTorch Volumetric's `MeshSDF` for distance queries and PyTorch3D for batched Marching Cubes. Mesh preparation and postprocessing use PyVista and Trimesh. A Qt worker runs the morphing pipeline separately from the loading interface.

## Configuration

Edit [config.py](config.py) before launching. Current defaults include:

| Setting | Default | Purpose |
| --- | --- | --- |
| `PREVIEW_RES` | `50` | Grid resolution for the preview |
| `PREVIEW_FRAMES` | `10` | Preview frames per transition |
| `RESOLUTION` | `128` | Grid resolution for the full sequence |
| `FRAME_COUNT` | `30` | Frames per transition in the full sequence |
| `TARGET_FACES` | `15000` | Target face count when decimating input meshes |
| `SDF_QUERY_BATCH_SIZE` | `200000` | Points queried per SDF batch |
| `FRAME_BATCH_SIZE` | `5` | Surface extraction batch size |

Grid storage grows cubically with resolution. Frame count, mesh count and intermediate SDF tensors also affect memory use; batching does not remove that cost. Reduce resolution and frame count first when exploring a new model pair. No minimum RAM or VRAM specification has been established.

## Source guide

| File / directory | Responsibility |
| --- | --- |
| [main.py](main.py) | Qt application entry point |
| [ui/](ui/) | Mesh loading, atlas selection, loading animation and preview |
| [workers/morph_worker.py](workers/morph_worker.py) | Preprocessing and backend selection |
| [morphing/framer.py](morphing/framer.py) | CUDA morphing sequence and SDF blending |
| [core/](core/) | Device selection, shared grid and SDF queries |
| [processing/](processing/) | Surface extraction and postprocessing |
| [sdf_tools_cpu.py](sdf_tools_cpu.py) | CPU implementation requiring repair |
| [viewer.py](viewer.py) | Scrollable PyVista viewer |
| [services/atlas.py](services/atlas.py) | BrainGlobe atlas lookup and mesh paths |

## Limitations and next steps

Mesh repair handles only some defects. Open surfaces, inconsistent normals and coarse sampling can produce holes or other artefacts, and smoothing can remove detail. Inputs are centred and scaled to comparable bounding-box sizes, so original physical scale and placement are not preserved.

Implemented features include multiple-mesh sequences, GUI file selection, BrainGlobe atlas browsing, the loading animation, wireframe previews and batched CUDA processing. The next steps are:

- Repair and verify the CPU fallback.
- Make installation reproducible across supported platforms and package the application.
- Add video/GIF export.
- Explore texture support and texture morphing.

## License

The project is labelled MIT in its existing documentation. A standalone licence file is not yet included.

import os 
# Ignore KMP duplicate warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # To prevent macOS crashes due to libomp

import sys 
import numpy as np 
import torch
import pyvista as pv
import pandas as pd

from pyvistaqt import QtInteractor 
from PyQt5 import QtWidgets, QtCore 
from random import randint

from brainglobe_atlasapi import BrainGlobeAtlas
from config import PREVIEW_RES, PREVIEW_FRAMES, TARGET_FACES, RESOLUTION, FRAME_COUNT
from mesh_tools import load_and_clean_mesh, pyvista_to_trimesh, repair_mesh, normalise_meshes
from sdf_tools_cpu import morph_mesh_sequence
from sdf_tools_gpu import morph_mesh_sequence_torch
from viewer import ScrollViewer


class Loader(QtWidgets.QMainWindow): 
    """
    Loader for enabling file selection as well as 3D loading animation. 

    Brainglobe Atlas implementation is sandwiched in the loader for structure selection.
    3D morphing is handled in a separate thread to prevent UI blocking.
    """

    def __init__(self): 
        super().__init__() 

        self.mesh_paths = []
        self.morph_frames = [] 
        self.current_frame_idx = 0
        self.actor = None 
        self.worker = None 
        self.loading_timer = None
        self.loading_actor = None

        self.resize(800, 600)
        self.setWindowTitle("Model Morpher - Loader")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central) 
        vert_layout = QtWidgets.QVBoxLayout(central)

        # Implementing Controls 
        controls = QtWidgets.QHBoxLayout()
        vert_layout.addLayout(controls) 

        self.load_button = QtWidgets.QPushButton("Load Meshes")
        self.run_button = QtWidgets.QPushButton("Run Morphing")
        self.run_button.setEnabled(False)

        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.status_label.setMinimumWidth(200) # Prevent Clipping

        controls.addWidget(self.load_button)
        controls.addWidget(self.run_button)
        controls.addWidget(self.status_label)
        controls.addStretch(1)

        bga_layout = QtWidgets.QHBoxLayout()
        vert_layout.addLayout(bga_layout)

        # Boiler for BrainGlobe Atlas Selection
        self.bga_altas_label = QtWidgets.QLabel("Atlas :")
        self.bga_atlas_combo = QtWidgets.QComboBox()
        self.bga_region_search = QtWidgets.QLineEdit()
        self.bga_region_search.setPlaceholderText("Search Structure...")

        self.bga_region_list = QtWidgets.QListWidget()
        self.bga_region_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.bga_add_button = QtWidgets.QPushButton("Add Selected Structures")
        self.bga_add_button.setEnabled(False)

        bga_layout.addWidget(self.bga_altas_label)
        bga_layout.addWidget(self.bga_atlas_combo)
        bga_layout.addWidget(self.bga_region_search)
        bga_layout.addWidget(self.bga_add_button)

        bga_layout.addStretch(1)
        vert_layout.addWidget(self.bga_region_list, stretch = 1)

        # Init BrainGlobe Atlas 

        self.bga_atlas = None 
        self.bga_structures_df = None 

        self._init_bga_atlases() 
        self.bga_atlas_combo.currentIndexChanged.connect(self._on_bga_atlas_change)
        self.bga_region_search.textChanged.connect(self._filter_bga_regions)
        self.bga_add_button.clicked.connect(self._add_bga_structures)

        self.plotter = QtInteractor(self) 
        vert_layout.addWidget(self.plotter.interactor)

        self._show_loading()
        self.load_button.clicked.connect(self.add_mesh_file)
        self.run_button.clicked.connect(self.run_morphing)


    def _init_bga_atlases(self): 
        """
        Initialize BrainGlobe Atlas selection dropdown.
        
        :param self: Description
        """

        self.bga_atlas_combo.addItem("allen_mouse_25um")
        self.bga_atlas_combo.addItem("mpin_zfish_1um")

        self._on_bga_atlas_change(0)  # Load the first atlas by default


    def _on_bga_atlas_change(self, index):
        """
        Handle atlas change event.
        
        :param self: Description
        :param index: Selected index
        """

        atlas = self.bga_atlas_combo.currentText()

        if not atlas:
            return 
        
        self.status_label.setText(f"Status: Loading atlas {atlas}...")
        QtWidgets.QApplication.processEvents()

        try:
            self.bga_atlas = BrainGlobeAtlas(atlas, check_latest = False)

            # Utilise Pandas DataFrame for relevant columns
            self.bga_structures_df = self.bga_atlas.lookup_df[["acronym", "name"]].copy()
        
        except Exception as e:
            self.status_label.setText(f"Error loading atlas {atlas}: {str(e)}")
            self.bga_atlas = None 
            self.bga_structures_df = None 

            self.bga_region_list.clear()
            self.bga_add_button.setEnabled(False)
            return

        
        self._refresh_region_list(self.bga_structures_df)
        self.bga_add_button.setEnabled(True)

        self.status_label.setText(f"Status: Atlas {atlas} loaded.")


    def _filter_bga_regions(self, text): 
        """
        Detect based on search box input by checking if the message is contained within.
        
        :param self: Description
        :param text: Description
        """
        if self.bga_structures_df is None:
            return 
        
        text = text.strip().lower() 

        if not text:
            filtered_df = self.bga_structures_df
        else: 
            mask = (
                self.bga_structures_df['acronym'].str.lower().str.contains(text) |
                self.bga_structures_df['name'].str.lower().str.contains(text)
            )
            filtered_df = self.bga_structures_df[mask]
        
        self._refresh_region_list(filtered_df)
    

    def _refresh_region_list(self, df):
        """
        Refresh the region list based on the provided DataFrame.
        
        :param self: Description
        :param df: Description
        """

        self.bga_region_list.clear()

        for _, row in df.iterrows():
            item = QtWidgets.QListWidgetItem(f"{row['acronym']} - {row['name']}")
            item.setData(QtCore.Qt.UserRole, row['acronym'])
            self.bga_region_list.addItem(item)


    def _add_bga_structures(self):
        """
        Add selected structures from BrainGlobe Atlas to mesh paths.
        
        :param self: Description
        """

        if not self.bga_atlas:
            return 
        
        selected_items = self.bga_region_list.selectedItems()
        if not selected_items:
            return 
        
        i = 0
        for item in selected_items:
            acro = item.data(QtCore.Qt.UserRole)
            try:
                mesh_path = str(self.bga_atlas.meshfile_from_structure(acro))
                self.mesh_paths.append(mesh_path)
                i += 1
            except Exception as e:
                self.status_label.setText(f"Error loading structure {acro}: {str(e)}")

        # If any meshes were successfully added   
        if i: 
            self.status_label.setText(f"Added {i} structures from atlas. Total meshes: {len(self.mesh_paths)}")

            # Allow to run when more than 2 meshes are loaded. 
            if len(self.mesh_paths) >= 2:
                self.run_button.setEnabled(True)


    def _show_loading(self): 
        """
        3d loading animation while the SDF and morph frames are calculated
        
        :param self: Description
        """

        self.plotter.clear() 
        self.loading_actor = self.plotter.add_mesh(
            pv.Sphere(radius = 1.0).points, 
            color = 'lightblue', 
            point_size = 5, 
            render_points_as_spheres = True
        )

        # Rotation of Actor 

        self.loading_timer = QtCore.QTimer()
        self.loading_timer.timeout.connect(self._rotate_loading)
        self.loading_timer.start(35)  # Rotate every 35 ms


    def _rotate_loading(self):
        """
        Rotate the loading actor for animation.
        
        :param self: Description
        """
        if self.loading_actor: 
            self.loading_actor.RotateY(randint(1, 10)) 
            self.loading_actor.RotateX(randint(1, 10)) 
            self.loading_actor.RotateZ(randint(1, 10)) 
            self.plotter.render()
        else:
            self.loading_timer.stop() 

    
    def add_mesh_file(self): 
        """
        Open file dialog to select mesh files.
        
        :param self: Description
        """
        file_button = QtWidgets.QFileDialog(self, "Select Meshes")
        file_button.setFileMode(QtWidgets.QFileDialog.ExistingFiles)

        file_button.setNameFilters([
            "3D Mesh Files (*.stl *.obj *.ply *.off, *.vtk)",
            "All Files (*)"
        ])

        if file_button.exec_():
            selected = file_button.selectedFiles()

            if selected: 
                self.mesh_paths.extend(selected)
                self.status_label.setText(f"Status: {len(self.mesh_paths)} meshes loaded")
                self.run_button.setEnabled(True)

                if len(self.mesh_paths) >= 2: 
                    self.run_button.setEnabled(True)

    # Morphing handling
    def run_morphing(self): 
        """
        Run the morphing process after meshes are loaded.
        
        :param self: Description
        """
        if not self.mesh_paths:
            self.status_label.setText("No meshes loaded to morph.")
            return 
        
        # Preventing repeated events
        self.status_label.setText("Status: Generating Preview Morph...")
        self.run_button.setEnabled(False)
        self.load_button.setEnabled(False)
        
        # Generating Preview
        self.preview_worker = Morpher(
            self.mesh_paths, 
            target_faces = TARGET_FACES,
            resolution = PREVIEW_RES,
            frames_per_transition = PREVIEW_FRAMES,
            mode = "preview"
        )

        self.preview_worker.success.connect(self._on_morph_success)
        self.preview_worker.error.connect(self._on_morph_error)
        self.preview_worker.start()

    
    def _on_morph_success(self, mode, morph_frames):
        """
        Handle successful morphing completion.
        """

        if mode == "preview":
            if hasattr(self, "loading_timer") and self.loading_timer:
                self.loading_timer.stop()
                self.loading_timer = None

            # Preview Frame Generation
            self.morph_frames = morph_frames 
            self.current_frame_idx = 0

            # Ensure that the entire plot is clearer
            self.plotter.clear()
            self.leading_actor = None 

            if self.morph_frames:

                # First frame 
                self.actor = self.plotter.add_mesh(
                    self.morph_frames[0],
                    color = 'lightblue',
                    show_edges = True,
                    opacity = 1.0
                )

                self.plotter.reset_camera()
                self.plotter.render() # Initial Render

                # Automatic playback of preview 
                if not hasattr(self, "playback_timer"):
                    self.playback_timer = QtCore.QTimer()
                    self.playback_timer.timeout.connect(self._update_preview_frame)
                self.playback_timer.start(100)  # Update every 100 ms

                self.status_label.setText("Preview Morph Generated. Computing Full Morph...")

                # Generation of full morph after preview
                self.worker = Morpher(
                    self.mesh_paths,
                    target_faces = TARGET_FACES,
                    resolution = RESOLUTION,
                    frames_per_transition = FRAME_COUNT,
                    mode = "full"
                )

                self.worker.success.connect(self._on_morph_success)
                self.worker.error.connect(self._on_morph_error)
                self.worker.finished.connect(self._on_worker_finished)
                self.worker.start()

                # Clean up Preview Worker ref
                self.preview_worker = None

        else: # Full morph replace
            if hasattr(self, "playback_timer") and self.playback_timer:
                self.playback_timer.stop()
                self.playback_timer = None

            if hasattr(self, "loading_timer") and self.loading_timer: 
                self.loading_timer.stop()
                self.loading_timer = None

            self.viewer = ScrollViewer(morph_frames)
            self.viewer.show()

            # Loader window closed. 
            self.close()


    def _on_morph_error(self, mode, error_msg):
        """
        Handle morphing errors.
        """

        try:
            self.loading_timer.stop()
        except Exception:
            pass 

        if mode == "preview":
           # If preview fails, try going straight to full morph
            self.status_label.setText(f"Preview error, computing full morph: {error_msg}")
            
            # Clean up preview worker
            self.preview_worker = None
            
            self.worker = Morpher(
                self.mesh_paths,
                target_faces=TARGET_FACES,
                resolution=RESOLUTION,
                frames_per_transition=FRAME_COUNT,
                mode="full"
            )
            self.worker.success.connect(self._on_morph_success)
            self.worker.error.connect(self._on_morph_error)
            self.worker.finished.connect(self._on_worker_finished)
            self.worker.start() 

        else:
            self.plotter.clear()
            self._show_loading()  # Show loading screen again
            self.status_label.setText(f"Error during morphing: {error_msg}")
            print(error_msg)
            self.load_button.setEnabled(True)
            if len(self.mesh_paths) >= 2:
                self.run_button.setEnabled(True)

    def _on_worker_finished(self):
        """
        Handle worker thread completion.
        """

        self.worker = None

    def _update_preview_frame(self):
        """
        Update the preview frame during playback.
        """

        if not self.morph_frames or self.actor is None:
            return 
        
        self.current_frame_idx = (self.current_frame_idx + 1) % len(self.morph_frames)
        current_frame = self.morph_frames[self.current_frame_idx]
        self.actor.GetMapper().SetInputData(current_frame)
        self.plotter.render()
        

    def closeEvent(self, event):
        """
        Ensure background threads are stopped before closing the window.
        """
        try:
            if hasattr(self, "playback_timer") and self.playback_timer:
                self.playback_timer.stop()
        except Exception:
            pass

        try:
            if hasattr(self, "preview_worker") and self.preview_worker and self.preview_worker.isRunning():
                self.preview_worker.quit()
                self.preview_worker.wait()
        except Exception:
            pass

        try:
            if self.worker and self.worker.isRunning():
                self.worker.quit()
                self.worker.wait()
        except Exception:
            pass

        event.accept()


    
class Morpher(QtCore.QThread):
    """
    Docstring for Morpher
    """

    # Mode, Frames
    success = QtCore.pyqtSignal(str, list)
    error = QtCore.pyqtSignal(str, str) # Mode, Err Message

    
    def __init__(self, mesh_paths, target_faces, resolution, frames_per_transition, mode = "full", parent = None):
        super().__init__(parent)
        self.mesh_paths = mesh_paths
        self.target_faces = target_faces
        self.resolution = resolution
        self.frames_per_transition = frames_per_transition
        self.mode = mode 


    def run(self):
        """
        Docstring for run
        
        :param self: Description
        """
        try: 
            # Load and prepare all the meshes
            trimeshes = [] 

            for source in self.mesh_paths:
                pv_mesh = load_and_clean_mesh(source, target_faces=self.target_faces)
                tri_mesh = pyvista_to_trimesh(pv_mesh)
                repair_mesh(tri_mesh)
                trimeshes.append(tri_mesh)

            # Normalising is just when the meshes are centered and scaled to make SDF computation easier
            trimeshes = normalise_meshes(trimeshes)

            # Step 3: Verify normalization
            for i, mesh in enumerate(trimeshes):
                diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
                print(f"  Mesh {i+1}: diagonal={diag:.2f}, center={mesh.centroid}")
                
            # GPU 
            if torch.cuda.is_available():
                morph_frames = morph_mesh_sequence_torch(
                    trimeshes=trimeshes,
                    resolution=self.resolution,
                    frames_per_transition=self.frames_per_transition,
                    device="cuda"
                )

            # CPU
            else:
                # Generate the frames after all the big preprocessing is confirmed
                morph_frames = morph_mesh_sequence(
                    trimeshes,
                    resolution = self.resolution,
                    frames_per_transition= self.frames_per_transition
                )

            self.success.emit(self.mode, morph_frames)

        except Exception as e:
            self.error.emit(self.mode, str(e))



def main():
    app = QtWidgets.QApplication(sys.argv)
    loader = Loader()
    loader.show()
    loader._show_loading()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
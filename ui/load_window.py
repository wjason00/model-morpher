"""
Docstring for ui.load_window
"""

from PyQt5 import QtWidgets
from pyvistaqt import QtInteractor

from config import PREVIEW_RES, PREVIEW_FRAMES, TARGET_FACES, RESOLUTION, FRAME_COUNT
from ui.atlas_widget import AtlasWidget
from ui.load_anim import LoadingAnimation
from ui.preview_widget import Previewer
from workers.morph_worker import MorphWorker
from viewer import ScrollViewer


class LoadWindow(QtWidgets.QMainWindow):
    """
    Docstring for LoadWindow
    """

    def __init__(self):
        """
        Docstring for __init__
        
        :param self: Description
        :param parent: Description
        """

        super().__init__()
        self.mesh_paths = []
        self.preview_worker = None
        self.full_worker = None 

        self._setup_ui()
        self.loading_animation.start()

    
    def _setup_ui(self):
        """
        Docstring for _setup_ui
        
        :param self: Description
        """

        self.setWindowTitle("Model Morpher")
        self.resize(1000, 800)


        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)


        # Control Buttons 
        print("creating layouts")
        button_layout = QtWidgets.QHBoxLayout()

        self.load_button = QtWidgets.QPushButton("Load Meshes")
        self.load_button.clicked.connect(self._on_load_meshes)
        button_layout.addWidget(self.load_button)

        print("adding load button")

        self.run_button = QtWidgets.QPushButton("Start Morphing")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._on_start_morphing)
        button_layout.addWidget(self.run_button)

        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.status_label.setMinimumWidth(200)
        button_layout.addWidget(self.status_label)
        button_layout.addStretch(1)

        layout.addLayout(button_layout)
        
        # Atlas Widget
        self.atlas_widget = AtlasWidget()
        self.atlas_widget.setMinimumHeight(300)
        self.atlas_widget.structures_selected.connect(self._on_atlas_structure_selected)
        layout.addWidget(self.atlas_widget)

        # Preview Plotter
        self.plotter = QtInteractor(self)
        self.plotter.setMinimumHeight(400)
        layout.addWidget(self.plotter.interactor)

        # Animating 
        self.loading_animation = LoadingAnimation(self.plotter)
        self.previewer = Previewer(self.plotter) 


    def _on_load_meshes(self):
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
                self._update_status()
                self.status_label.setText(f"Status: {len(self.mesh_paths)} meshes loaded")


    def _on_atlas_structure_selected(self, mesh_paths: list):
        """
        Handle structure selection from atlas widget.
        
        :param self: Description
        :param mesh_paths: Description
        """

        self.mesh_paths.extend(mesh_paths)
        self._update_status()

    
    def _update_status(self):
        """
        Update the status label with the number of loaded meshes.
        
        :param self: Description
        """
        count = len(self.mesh_paths)
        self.status_label.setText(f"Status: {count} meshes loaded")
        self.run_button.setEnabled(count >= 2)

    
    def _on_start_morphing(self):
        """
        Start the morphing process in a separate thread.
        
        :param self: Description
        """
        
        # Edge case where < 2 is not accounted for due to other accounting
        if not self.mesh_paths or len(self.mesh_paths) < 2:
            self.status_label.setText("Status: Please load at least two meshes to morph.")
            return 

        self.run_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.atlas_widget.setEnabled(False)
        self.status_label.setText("Status: Morphing in progress...")

        self.loading_animation.start()

        # Generating Preview
        self.preview_worker = MorphWorker(
            mesh_paths=self.mesh_paths,
            resolution=PREVIEW_RES,
            target_faces=TARGET_FACES,
            frames_per_transition=PREVIEW_FRAMES,
            mode="preview",
            device="cuda"
        )

        self.preview_worker.success.connect(self._on_morph_success)
        self.preview_worker.error.connect(self._on_morph_error)
        self.preview_worker.start()


    def _on_morph_success(self, mode: str, frames: list):
        """
        Handle successful morphing completion.
        
        :param self: Description
        :param mode: Description
        :param frames: Description
        """

        if mode == "preview":
            # Preview is complete - therefore start render
            self.loading_animation.stop()
            self.previewer.load_frames(frames)
            self.status_label.setText("Status: Preview generated. Starting full morph...")

            # Start full morphing
            self.full_worker = MorphWorker(
                mesh_paths=self.mesh_paths,
                resolution=RESOLUTION,
                target_faces=TARGET_FACES,
                frames_per_transition=FRAME_COUNT,
                mode="full",
                device="cuda"
            )

            self.full_worker.success.connect(self._on_morph_success)
            self.full_worker.error.connect(self._on_morph_error)
            self.full_worker.start()

        elif mode == "full":
            self.status_label.setText("Status: Morphing complete.")
            
            self.previewer.stop()
            self.loading_animation.stop()

            # Viewer Window launched

            viewer = ScrollViewer(frames)
            viewer.show()

            # Close loader window
            self.close()

    def _on_morph_error(self, mode: str, error_message: str):
        """
        Handle morphing errors.
        
        :param self: Description
        :param mode: Description
        :param error_message: Description
        """

        self.loading_animation.stop()
        self.previewer.stop()

        self.status_label.setText(f"Status: Error during {mode} morphing - {error_message}")

        # Re-enable UI
        self.load_button.setEnabled(True)
        self.atlas_widget.setEnabled(True)
        self.run_button.setEnabled(len(self.mesh_paths) >= 2)

    def closeEvent(self, event):
        """
        Handle window close event.
        
        :param self: Description
        :param event: Description
        """

        self.loading_animation.stop()
        self.previewer.stop()

        # Waiting for worker threads. 
        if self.preview_worker and self.preview_worker.isRunning():
            self.preview_worker.quit()
            self.preview_worker.wait()

        if self.full_worker and self.full_worker.isRunning():
            self.full_worker.quit()
            self.full_worker.wait()

        event.accept()
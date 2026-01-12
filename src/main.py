import sys 
import numpy 
import pyvista as pv

from pyvistaqt import QtInteractor 
from PyQt5 import QtWidgets, QtCore 
from random import randint

from constants import TARGET_FACES, RESOLUTION, FRAME_COUNT
from mesh_tools import load_and_clean_mesh, pyvista_to_trimesh, repair_mesh, normalise_meshes
from sdf_tools import morph_mesh_sequence
from viewer import ScrollViewer


class Loader(QtWidgets.QMainWindow): 
    """
    Loader for enabling file selection as well as 3D loading animation. 
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

        self.plotter = QtInteractor(self) 
        vert_layout.addWidget(self.plotter.interactor)

        self._show_idle()
        self.load_button.clicked.connect(self.add_mesh_file)
        self.run_button.clicked.connect(self.run_morphing)


    # Scene Handling 
    def _show_idle(self): 
        """
        Handling the idle scene while no meshes are selected. 
        
        :param self: Description
        """

        self.plotter.clear()
        self.plotter.add_text("Idle - Load meshes to begin",
                              position = "upper_left",
                              font_size = 12
                    )
                                         

        sphere = pv.Sphere(radius = 1.0).points
        self.plotter.add_mesh(sphere, 
                              color = 'lightblue', 
                              point_size = 5, 
                              render_points_as_spheres = True
                    )      

        self.plotter.reset_camera() 

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

    
    def run_morphing(self): 
        """
        Run the morphing process after meshes are loaded.
        
        :param self: Description
        """
        if not self.mesh_paths:
            return 
        
        # Preventing repeated events
        self.status_label.setText("Loading meshes.")
        self.run_button.setEnabled(False)
        self.load_button.setEnabled(False)
        
        self._show_loading()

        self.worker = Morpher(
           self.mesh_paths,
           target_faces = TARGET_FACES,
           resolution = RESOLUTION, 
           frames_per_transition = FRAME_COUNT
            )
    
        self.worker.success.connect(self._on_morph_success)
        self.worker.error.connect(self._on_morph_error)
        self.worker.start()
    

    def _on_morph_success(self, morph_frames):
        """
        Handle successful morphing completion.
        """

        try:
            self.loading_timer.stop()
        except Exception:
            pass 

        # Reusing old viewer (PyVista) for general handling of the mesh sequencing.
        viewer = ScrollViewer(morph_frames)
        viewer.show() 


        # Close the loader and clean the worker thread 
        self.worker = None
        self.close() 


    def _on_morph_error(self, error_msg):
        """
        Handle morphing errors.
        """

        try:
            self.loading_timer.stop()
        except Exception:
            pass 

        self._show_idle()
        self.status_label.setText(f"Error during morphing: {error_msg}")
        self.load_button.setEnabled(True)

        if len(self.mesh_paths) >= 2:
            self.run_button.setEnabled(True)

    
class Morpher(QtCore.QThread):
    """
    Docstring for Morpher
    """

    success = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)

    
    def __init__(self, mesh_paths, target_faces, resolution, frames_per_transition, parent = None):
        super().__init__(parent)
        self.mesh_paths = mesh_paths
        self.target_faces = target_faces
        self.resolution = resolution
        self.frames_per_transition = frames_per_transition


    def run(self):
        try: 
            # Load and prepare all the meshes
            trimeshes = [] 
            for i, source in enumerate(self.mesh_paths):
                pv_mesh = load_and_clean_mesh(source, target_faces=self.target_faces)
                tri_mesh = pyvista_to_trimesh(pv_mesh)
                repair_mesh(tri_mesh)
                trimeshes.append(tri_mesh)

            # Normalising is just when the meshes are centered and scaled to make SDF computation easier
            trimeshes = normalise_meshes(trimeshes)

            # Generate the frames after all the big preprocessing is confirmed
            morph_frames = morph_mesh_sequence(
                trimeshes,
                resolution = self.resolution,
                frames_per_transition= self.frames_per_transition
            )

            self.success.emit(morph_frames)

        except Exception as e:
            self.error.emit(str(e))

def main():
    app = QtWidgets.QApplication(sys.argv)
    loader = Loader()
    loader.show()
    loader._show_idle()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

"""
Docstring for workers.morph_worker
"""

from turtle import st
import numpy as np 
import torch

from PyQt5 import QtCore

from mesh_tools import load_and_clean_mesh, pyvista_to_trimesh, repair_mesh, normalise_meshes
from morphing.framer import morph_mesh_sequence_torch
from sdf_tools_cpu import morph_mesh_sequence


class MorphWorker(QtCore.QThread):
    """
    Worker class for performing mesh morphing in a separate thread.
    """

    # (mode, frames / error_messagae)
    success = QtCore.pyqtSignal(str, list)
    error = QtCore.pyqtSignal(str, str)

    def __init__(
            self, mesh_paths: list, resolution: int, target_faces: int, 
            frames_per_transition: int, mode: str = "full", device: str = "cuda"
            ):
        """
        Docstring for __init__
        
        :param self: Description
        :param mesh_paths: Description
        :type mesh_paths: list
        :param resolution: Description
        :type resolution: int
        :param target_faces: Description
        :type target_faces: int
        :param frames_per_transition: Description
        :type frames_per_transition: int
        :param mode: Description
        :type mode: str
        """

        super().__init__()
        self.mesh_paths = mesh_paths
        self.target_faces = target_faces
        self.resolution = resolution
        self.frames_per_transition = frames_per_transition
        self.mode = mode
        self.device = device

    
    def run(self):
        """
        Docstring for run
        
        :param self: Description
        """

        try:
            # Load and preprocess meshes
            trimeshes = []
            for i, path in enumerate(self.mesh_paths):
                mesh = load_and_clean_mesh(path, target_faces=self.target_faces)
                trimesh = pyvista_to_trimesh(mesh)
                repair_mesh(trimesh)
                trimeshes.append(trimesh)

            trimeshes = normalise_meshes(trimeshes)

            # Morphing
            if self.device == "cuda" and torch.cuda.is_available():
                frames = morph_mesh_sequence_torch(
                    trimeshes,
                    resolution=self.resolution,
                    frames_per_transition=self.frames_per_transition,
                    device="cuda" 
                )

            else:
                frames = morph_mesh_sequence(
                    trimeshes,
                    resolution=self.resolution,
                    frames_per_transition=self.frames_per_transition,
                    device = "cpu"
                )

            self.success.emit(self.mode, frames)
        except Exception as e:
            self.error.emit(self.mode, str(e))


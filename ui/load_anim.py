"""
Docstring for ui.load_anim
"""

from random import randint 
from PyQt5 import QtCore 

import pyvista as pv 


class LoadingAnimation:

    def __init__(self, plotter): 
        self.plotter = plotter
        self.actor = None
        self.timer = None 

    
    def start(self):
        self.plotter.clear() 

        # Spinning Pointcloud 
        self.actor = self.plotter.add_mesh(
            pv.Sphere(radius=1).points,
            color='lightgreen',
            point_size=5,
            render_points_as_spheres=True
        )

        # Animate Rotation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._rotate)
        self.timer.start(35)  # Rotate every 50 ms

    
    def stop(self):
        if self.timer:
            self.timer.stop()
            self.timer = None 
        
        if self.actor:
            self.plotter.remove_actor(self.actor)
            self.actor = None

    
    def _rotate(self):
        if self.actor:
            self.actor.RotateX(randint(1, 10))
            self.actor.RotateY(randint(1, 10))
            self.actor.RotateZ(randint(1, 10))
            self.plotter.render()

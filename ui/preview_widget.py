"""
Docstring for ui.preview_widget
"""

from PyQt5 import QtCore


class Previewer:

    def __init__(self, plotter):
        """
        Docstring for __init__
        
        :param self: Description
        :param plotter: Description
        """

        self.plotter = plotter
        self.frames = []
        self.timer = None
        self.current_frame_index = 0
        self.actor = None

    
    def load_frames(self, frames):
        """
        Docstring for load_frames
        
        :param self: Description
        :param frames: Description
        """

        self.stop()
        self.frames = frames 
        self.current_frame_index = 0 

        if not frames:
            return 
        
        # First Frame display

        self.plotter.clear()
        self.actor = self.plotter.add_mesh(
            frames[0],
            color='lightblue',
            show_edges=True, 
            opacity = 1.0
        )

        self.plotter.reset_camera()
        self.plotter.render()

        # Playback 

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._advance_frame)
        self.timer.start(100)  # 10 FPS


    def stop(self):
        """
        Docstring for stop
        
        :param self: Description
        """

        if self.timer:
            self.timer.stop()
            self.timer = None 
        
    
    def clear(self):
        """
        Docstring for clear
        
        :param self: Description
        """

        self.stop()
        self.frames = []
        self.plotter.clear()
        self.actor = None
    

    def _advance_frame(self):
        """
        Docstring for _advance_frame
        
        :param self: Description
        """

        if not self.frames or not self.actor:
            return 
        
        self.current_frame_index = (self.current_frame_index + 1) % len(self.frames)
        self.actor.GetMapper().SetInputData(self.frames[self.current_frame_index])
        self.plotter.render()



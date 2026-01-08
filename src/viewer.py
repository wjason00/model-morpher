import numpy as np
import pyvista as pv


# General Viewer class to handle mouse wheel scrolling through frames
class ScrollViewer:
    def __init__(self, frames, scroll_step=None):

        self.frames = frames
        self.idx = 0
        self.plotter = pv.Plotter()
        
        # Scale the scroll speed based on the total number of frames (to avoid the slow feeling with large frame scrolling)
        if scroll_step is None:
            self.scroll_step = max(1, len(frames) // 30) if len(frames) > 30 else 1
        else:
            self.scroll_step = scroll_step

        self.plotter.iren.enable_terrain_style(mouse_wheel_zooms=False)

        self.actor = self.plotter.add_mesh(
            frames[0],
            color='lightblue',
            smooth_shading=True,
            show_edges=False  # Cleaner look
        )
        self.text = self.plotter.add_text(
            f"Scroll: ±{self.scroll_step} | L / R Arrow Keys: ±1 | U / D Arrow Keys: ±10 | Left Click: Pan | Right Click Zoom | Q: Quit",
            position='upper_left',
            font_size=12
        )
        self.plotter.reset_camera()
        self.plotter.camera_position = 'iso'

        # Mouse wheel for normal scrolling (uses scroll_step)
        self.plotter.iren.add_observer(
            'MouseWheelForwardEvent',
            lambda obj, event: self.change(self.scroll_step)
        )
        self.plotter.iren.add_observer(
            'MouseWheelBackwardEvent',
            lambda obj, event: self.change(-self.scroll_step)
        )

        
        # Keyboard controls for fine/coarse navigation
        self.plotter.add_key_event('Right', lambda: self.change(1))      
        self.plotter.add_key_event('Left', lambda: self.change(-1))      
        self.plotter.add_key_event('Up', lambda: self.change(10))       
        self.plotter.add_key_event('Down', lambda: self.change(-10))     
        self.plotter.add_key_event('Home', lambda: self.jump_to(0))      
        self.plotter.add_key_event('End', lambda: self.jump_to(len(self.frames) - 1)) 


    def change(self, delta):
        # Ensures that the index doesn't exceed target bounds i.e. will always be an indexable frame 
        self.idx = np.clip(self.idx + delta, 0, len(self.frames) - 1)
        self._update_display()
    
    def jump_to(self, frame_idx):
        """Jump directly to a specific frame index."""
        self.idx = np.clip(frame_idx, 0, len(self.frames) - 1)
        self._update_display()
    
    def _update_display(self):
        """Update the mesh and text display."""
        self.actor.GetMapper().SetInputData(self.frames[self.idx])
        progress = self.idx / (len(self.frames) - 1) * 100 if len(self.frames) > 1 else 0
        self.text.SetText(
            0,
            f"Frame {self.idx+1}/{len(self.frames)} ({progress:.0f}%)"
        )
        self.plotter.render()

    def show(self):
        self.plotter.show()
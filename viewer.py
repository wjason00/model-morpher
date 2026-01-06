import numpy as np
import pyvista as pv


# General Viewer class to handle mouse wheel scrolling through frames
class ScrollViewer:
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.plotter = pv.Plotter()

        self.plotter.iren.enable_terrain_style(mouse_wheel_zooms=False)

        self.actor = self.plotter.add_mesh(
            frames[0],
            color='lightblue',
            smooth_shading=True,
            show_edges=False  # Cleaner look
        )
        self.text = self.plotter.add_text(
            f"Controls | Left Click to Pan | Mouse Wheel to Morph | Right Click to Zoom | Q to Quit",
            position='upper_left',
            font_size=12
        )
        self.plotter.reset_camera()
        self.plotter.camera_position = 'iso'

        self.plotter.iren.add_observer(
            'MouseWheelForwardEvent',
            lambda obj, event: self.change(1)
        )
        self.plotter.iren.add_observer(
            'MouseWheelBackwardEvent',
            lambda obj, event: self.change(-1)
        )

    def change(self, delta):
        self.idx = np.clip(self.idx + delta, 0, len(self.frames) - 1)
        self.actor.GetMapper().SetInputData(self.frames[self.idx])
        progress = self.idx / (len(self.frames) - 1) * 100 if len(self.frames) > 1 else 0
        self.text.SetText(
            0,
            f"Frame {self.idx+1}/{len(self.frames)} ({progress:.0f}%)"
        )
        self.plotter.render()

    def show(self):
        self.plotter.show()
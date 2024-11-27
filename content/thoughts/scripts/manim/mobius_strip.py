from manim import *
import numpy as np


class MobiusStrip(ThreeDScene):
  def construct(self):
    self.set_camera_orientation(phi=70 * DEGREES, theta=20* DEGREES)

    ax = ThreeDAxes(x_range=(-10, 10, 1), y_range=(-10, 10, 1), z_range=(-10, 10, 1))

    mobius = Surface(
      lambda u, v: np.array([
        (4 + v / 2 * np.cos(u / 2)) * np.cos(u),
        (4 + v / 2 * np.cos(u / 2)) * np.sin(u),
        v / 2 * np.sin(u / 2),
      ]),
      v_range=[-1, 1],
      u_range=[0, 2 * PI],
    )
    mobius.set_color_by_gradient(ORANGE, BLUE)
    mobius.set_opacity(0.5)

    self.play(DrawBorderThenFill(ax), run_time=1.5)

    self.play(Write(mobius), run_time=4)
    self.move_camera(phi=160 * DEGREES, theta=55 * DEGREES, run_time=5, zoom=1.25)
    self.wait()

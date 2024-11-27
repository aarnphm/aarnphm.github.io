from manim import *


class HyperplaneComparison(Scene):
  def construct(self):
    # Create coordinate axes
    axes = Axes(x_range=[-5, 5, 1], y_range=[-5, 5, 1])
    self.add(axes)

    # Data points for two classes
    class1_points = [  # Class 1 (e.g., blue circles)
      [-4, -2],
      [-3, -3],
      [-3, -1],
      [-2, -2],
      [-1, -4],
    ]
    class2_points = [  # Class 2 (e.g., red triangles)
      [1, 2],
      [2, 3],
      [3, 1],
      [2, 2],
      [4, 3],
    ]

    # Plot data points
    class1_dots = VGroup(*[Dot(axes.coords_to_point(x, y), color=BLUE) for x, y in class1_points])
    class2_dots = VGroup(*[Dot(axes.coords_to_point(x, y), color=RED) for x, y in class2_points])
    self.play(FadeIn(class1_dots), FadeIn(class2_dots))

    # Draw an arbitrary hyperplane: y = x + 0.5
    arbitrary_line = axes.get_graph(lambda x: x + 0.5, color=YELLOW, x_range=[-5, 5])
    arbitrary_label = axes.get_graph_label(arbitrary_line, label='Arbitrary Hyperplane', x=2, direction=UP)
    self.play(ShowCreation(arbitrary_line), Write(arbitrary_label))
    self.wait(2)

    # Remove the arbitrary hyperplane
    self.play(FadeOut(arbitrary_line), FadeOut(arbitrary_label))

    # Draw the maximum margin hyperplane: y = x
    margin_line = axes.get_graph(lambda x: x, color=GREEN, x_range=[-5, 5])
    margin_label = axes.get_graph_label(margin_line, label='Maximum Margin Hyperplane', x=2, direction=UP)
    self.play(ShowCreation(margin_line), Write(margin_label))

    # Draw the margin lines (dotted lines parallel to the hyperplane)
    margin_offset = 1 * np.sqrt(2)  # Margin distance based on line y = x
    upper_margin_line = axes.get_graph(lambda x: x + margin_offset, color=GREEN, x_range=[-5, 5])
    lower_margin_line = axes.get_graph(lambda x: x - margin_offset, color=GREEN, x_range=[-5, 5])
    self.play(ShowCreation(upper_margin_line), ShowCreation(lower_margin_line))

    # Highlight the support vectors (points closest to the hyperplane)
    support_vectors_coords = [[-2, -2], [2, 2]]
    support_vectors = VGroup(*[
      Dot(axes.coords_to_point(x, y), color=YELLOW).set_z_index(1) for x, y in support_vectors_coords
    ])
    self.play(Indicate(support_vectors))

    # Indicate the margin distance
    margin_line_point = axes.coords_to_point(1, 1)
    upper_margin_point = axes.coords_to_point(1, 1 + margin_offset)
    margin_distance = Line(margin_line_point, upper_margin_point, color=WHITE, stroke_width=2)
    self.play(ShowCreation(margin_distance), Write(margin_text))
    self.wait(2)

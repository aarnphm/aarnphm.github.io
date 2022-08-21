from manimlib import *

class JumpReLU(Scene):
    def construct(self):
        # Create the axes
        axes = Axes(x_range=[-4, 4, 1], y_range=[-1, 4, 1])
        axes_labels = axes.get_axis_labels(x_label_tex="z", y_label_tex="J(z)")

        # Create the JumpReLU function graph
        jump_relu_graph = VGroup()

        # Add the horizontal line at y=0 for x≤1
        jump_relu_graph.add(
            Line(
                axes.c2p(-4, 0),
                axes.c2p(1, 0),
                color=ORANGE
            )
        )

        # Add the diagonal line for x>1
        jump_relu_graph.add(
            axes.get_graph(
                lambda x: x,
                color=ORANGE,
                x_range=[1, 4]
            )
        )

        # Add open and closed dots at the jump point
        jump_point_lower = Dot(axes.c2p(1, 0), color=RED, fill_opacity=0)
        jump_point_lower.set_stroke(ORANGE, 2)  # Make it an open circle
        jump_point_upper = Dot(axes.c2p(1, 1), color=RED)  # Closed circle

        jump_relu_graph.add(jump_point_lower, jump_point_upper)

        # Add κ label at x=1
        kappa_label = Tex(r"\kappa").scale(0.8)
        kappa_label.next_to(axes.c2p(1, 0), DOWN)

        # Animations
        self.play(ShowCreation(axes), ShowCreation(axes_labels))
        self.wait(0.5)

        # Show JumpReLU
        self.play(ShowCreation(jump_relu_graph))
        self.wait()

        self.play(Write(kappa_label))
        self.wait()

        # Highlight the jump points
        self.play(
            Flash(jump_point_lower, color=RED, flash_radius=0.3),
            Flash(jump_point_upper, color=RED, flash_radius=0.3),
        )
        self.wait()

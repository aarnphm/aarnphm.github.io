from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 10
b = 8 / 3
r = 28


# Lorenz equations
def lorenz(t: list[int], y: list[int]) -> list[float]:
  return [sigma * (y[1] - y[0]), r * y[0] - y[1] - y[0] * y[2], y[0] * y[1] - b * y[2]]


# Initial conditions
y0 = [15, 15, 36]

# Time span
t_span = [0, 100]


def solve(diff=False) -> None:
  # Solve ODE
  sol = solve_ivp(lorenz, t_span, y0, t_eval=np.linspace(0, 100, 10000))

  # Plotting
  plt.figure(figsize=(15, 10))

  # y1, y2, y3 as functions of t
  plt.subplot(2, 2, 1)
  plt.plot(sol.t, sol.y[0], label='y1')
  plt.plot(sol.t, sol.y[1], label='y2')
  plt.plot(sol.t, sol.y[2], label='y3')
  plt.title('Time Series of y1, y2, y3')
  plt.xlabel('Time')
  plt.ylabel('Values')
  plt.legend()

  # y1 vs y2
  plt.subplot(2, 2, 2)
  plt.plot(sol.y[0], sol.y[1])
  plt.title('y1 vs y2')
  plt.xlabel('y1')
  plt.ylabel('y2')

  # y1 vs y3
  plt.subplot(2, 2, 3)
  plt.plot(sol.y[0], sol.y[2])
  plt.title('y1 vs y3')
  plt.xlabel('y1')
  plt.ylabel('y3')

  # y2 vs y3
  plt.subplot(2, 2, 4)
  plt.plot(sol.y[1], sol.y[2])
  plt.title('y2 vs y3')
  plt.xlabel('y2')
  plt.ylabel('y3')

  plt.tight_layout()
  plt.show()

  if diff:
    show_diff(sol)


def show_diff(sol):
  # Slight modification in initial conditions
  y0_modified = [y + 1e-10 for y in y0]

  # Solve ODE with modified initial conditions
  sol_modified = solve_ivp(lorenz, t_span, y0_modified, t_eval=np.linspace(0, 100, 10000))

  # Plotting differences
  plt.figure(figsize=(15, 5))

  # Difference in y1, y2, y3 as functions of t
  plt.subplot(1, 3, 1)
  plt.plot(sol.t, sol.y[0] - sol_modified.y[0], label='Δy1')
  plt.plot(sol.t, sol.y[1] - sol_modified.y[1], label='Δy2')
  plt.plot(sol.t, sol.y[2] - sol_modified.y[2], label='Δy3')
  plt.title('Difference in Time Series of y1, y2, y3')
  plt.xlabel('Time')
  plt.ylabel('Difference in Values')
  plt.legend()

  # Difference in y1 vs y2
  plt.subplot(1, 3, 2)
  plt.plot(sol.y[0] - sol_modified.y[0], sol.y[1] - sol_modified.y[1])
  plt.title('Difference in y1 vs y2')
  plt.xlabel('Δy1')
  plt.ylabel('Δy2')

  # Difference in y1 vs y3
  plt.subplot(1, 3, 3)
  plt.plot(sol.y[0] - sol_modified.y[0], sol.y[2] - sol_modified.y[2])
  plt.title('Difference in y1 vs y3')
  plt.xlabel('Δy1')
  plt.ylabel('Δy3')

  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  solve(diff=True)

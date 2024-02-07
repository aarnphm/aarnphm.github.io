import os
import numpy as np
import scipy.optimize as opt

PATH = os.path.dirname(os.path.abspath(__file__))


def find_period(file_name):
  # Load data from the file
  data = np.loadtxt(file_name)
  # Extract time, x, and y coordinates
  t = data[:, 0]
  x = data[:, 1]
  y = data[:, 2]
  print(data)

  # Define a function to find the difference between positions at different times
  def position_difference(t2, t1):
    # Interpolate to find x and y at t2
    x2 = np.interp(t2, t, x)
    y2 = np.interp(t2, t, y)

    # Get x and y at t1
    x1 = np.interp(t1, t, x)
    y1 = np.interp(t1, t, y)

    # Return the square of the distance between the two points
    return (x2 - x1) ** 2 + (y2 - y1) ** 2

  # Find the period by solving for when the position difference is zero
  # Initial guess for period is the difference between the first and last time values
  initial_guess = t[-1] - t[0]
  period = opt.fsolve(position_difference, initial_guess, args=(t[0]))[0]

  return period


if __name__ == '__main__':
  print(find_period(os.path.join(PATH, 'data.txt')))

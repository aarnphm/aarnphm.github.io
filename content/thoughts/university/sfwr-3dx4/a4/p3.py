from scipy.signal import place_poles
import numpy as np

# System matrices
A = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 9.8, 0]])
B = np.array([[0], [1], [0], [-1]])

# Desired poles
desired_poles = np.array([-2 + 1j, -2 - 1j, -5, -5.000000000000001])

# Calculate the feedback matrix K
place_result = place_poles(A, B, desired_poles)
print(place_result.gain_matrix)

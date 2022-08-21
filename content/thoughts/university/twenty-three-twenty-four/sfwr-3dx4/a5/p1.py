import numpy as np
from scipy.signal import place_poles

# Define the system matrices
A = np.array([[0, 1, 0], [0, 0, 1], [-5, -6, 0]])

B = np.array([[0], [0], [1]])

C = np.array([[1, 0, 0]])


# Step 1: Check the observability of the system
def check_observability(A, C):
  n = A.shape[0]
  O = np.zeros((n, n))
  for i in range(n):
    O[i, :] = C @ np.linalg.matrix_power(A, i)
  return np.linalg.matrix_rank(O) == n


observable = check_observability(A, C)
print(f"The system is {'observable' if observable else 'not observable'}.")

# Step 2: Define the observer dynamics
# Observer gain matrix L will be computed later

# Step 3: Compute the desired characteristic equation
desired_poles = np.array([-10, -10, -15])
char_eq_coeffs = np.poly(desired_poles)
print('Desired characteristic equation coefficients:')
print(char_eq_coeffs)

# Step 4: Compute the observer gain matrix L
L = place_poles(A.T, C.T, np.array([-10, -10.001, -15])).gain_matrix.T
print('Observer gain matrix L:')
print(L)

# Step 5: Construct the observer dynamics
A_obs = A - L @ C
print('Observer dynamics matrix:')
print(A_obs)

import control
import numpy as np


# Define the transfer function
num = [3, 4, -2]
den = [1, 3, 7, 5]
sys = control.tf(num, den)

# Convert to state-space representation
sys_ss = control.ss(sys)

# Print the state-space matrices
print('A =')
print(sys_ss.A)
print('B =')
print(sys_ss.B)
print('C =')
print(sys_ss.C)
print('D =')
print(sys_ss.D)

# Extract state-space matrices
A = sys_ss.A
B = sys_ss.B

# Compute the controllability matrix
n = A.shape[0]  # Number of states
C = np.hstack([B] + [np.linalg.matrix_power(A, i) @ B for i in range(1, n)])

# Check rank of controllability matrix
rank = np.linalg.matrix_rank(C)

print('Controllability Matrix:')
print(C)
print(f'\nRank of Controllability Matrix: {rank}')
print(f'Number of States: {n}')

if rank == n:
  print('The system is controllable.')
else:
  print('The system is not controllable.')


# Desired closed-loop pole locations
poles = [-4, -4, -5]

# Compute the state feedback gain vector K using Ackermann's formula
K = control.acker(A, B, poles)

print('State Feedback Gain Vector K:')
print(K)

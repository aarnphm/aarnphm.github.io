import control as ctl
import numpy as np
import matplotlib.pyplot as plt

# Define the transfer function
G = ctl.tf(1, [1, 4, 13, 0])
# zeta = 0.2588
zeta = 0.2588

# Plot the root locus
rlist, klist = ctl.root_locus(G, plot=True, grid=True)

# Calculate the roots for a range of K values
k_values = np.linspace(0, 100, 5000)  # Range of K values to test
theta = np.arccos(zeta)  # angle from the negative real axis
print(f'Angle from negative real axis: {np.rad2deg(theta):.2f} degrees')

# Plot the root locus
plt.figure()
ctl.rlocus(G)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Root Locus')
plt.grid(True)

# Find the gain K for a damping ratio of 0.2588
K_range = np.linspace(0, 10, 1000)
for K in K_range:
  poles = np.roots(G.den[0][0] + K * G.num[0][0])
  zeta_actual = -np.cos(np.angle(poles[0]))
  if np.abs(zeta_actual - zeta) < 0.001:
    break

# Print the gain value
print(f'The gain K for a damping ratio of 0.2588 is approximately: {K:.3f}')
print(f'The poles are: {poles}')
print(f'the roots of the transfer function with K={K:.3f} are: {np.roots([1, 4, 13, K])}')

# Extract the imaginary part of the complex poles
wn = np.abs(poles[0].imag)

# Create the closed-loop transfer function with the found gain K
G_cl = ctl.feedback(K * G, 1)

# Create an ideal second-order system with the same damping ratio and natural frequency
G_ideal = ctl.tf([wn**2], [1, 2 * zeta * wn, wn**2])

# Generate time vector for simulation
t = np.linspace(0, 10, 1000)

# Simulate the step response of the closed-loop system and the ideal system
_, y_cl = ctl.step_response(G_cl, t)
_, y_ideal = ctl.step_response(G_ideal, t)

# Plot the step responses
plt.figure()
plt.plot(t, y_cl, label='Closed-loop System')
plt.plot(t, y_ideal, '--', label='Ideal Second-Order System')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Step Response Comparison')
plt.legend()
plt.grid()
plt.show()

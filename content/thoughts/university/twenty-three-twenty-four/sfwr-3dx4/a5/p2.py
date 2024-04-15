import numpy as np
import control

# Given requirements
OS = 0.10  # 10% overshoot
Ts = 0.5  # 0.5 second settling time

# Calculating damping ratio (zeta) from overshoot
zeta = np.sqrt(np.log(OS) ** 2 / (np.pi**2 + np.log(OS) ** 2))

# Calculating natural frequency (omega_n) from settling time and damping ratio
wn = 4 / (zeta * Ts)

print('zeta:', zeta, 'omega_n:', wn)

# Define the plant model
A = np.array([[-1, 1], [0, 2]])
B = np.array([[0], [1]])
C = np.array([[1, 1]])
D = 0

# Create state space model
plant = control.ss(A, B, C, D)

# Augment the plant with an integrator
A_aug = np.block([[A, np.zeros((2, 1))], [-C, 0]])
B_aug = np.block([[B], [0]])
C_aug = np.block([[C, 0]])
D_aug = np.array([[0]])
plant_aug = control.ss(A_aug, B_aug, C_aug, D_aug)

# Desired closed-loop poles for 10% overshoot and 0.5s settling time
desired_poles = np.roots([1, 2 * zeta * wn, wn**2])
desired_poles = np.append(desired_poles, -10 * np.max(np.abs(desired_poles)))

print('Desired poles:', desired_poles)

# Design the state feedback gain matrix
K = control.place(plant_aug.A, plant_aug.B, desired_poles)

# Extract the integral gain
ki = K[0, 2]

# Define the controller transfer function
controller_tf = control.tf([ki], [1, 0])

# Compute the open-loop and closed-loop transfer functions
open_loop_tf = control.series(controller_tf, plant)
closed_loop_tf = control.feedback(open_loop_tf, 1)

# Print the mathematical models
print('Plant model:')
print(plant)

print('\nAugmented plant model:')
print(plant_aug)

print('\nState feedback gains:')
print(f'K = {K}')

print('\nIntegral controller transfer function:')
print(controller_tf)

print('\nOpen-loop transfer function:')
print(open_loop_tf)

print('\nClosed-loop transfer function:')
print(closed_loop_tf)

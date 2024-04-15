import numpy as np
import matplotlib.pyplot as plt
import control as ctl

# Desired specifications
OS = 0.20  # 20% overshoot
Ts = 1  # 1 second settling time
num = [1, 10]
den = [1, 15, 38, 24]

# Define the system components
# s^3 + 15s^2 + 38s + 24
G = ctl.TransferFunction(num, den)  # Original G(s)

# Calculations for desired pole locations
zeta = -np.log(OS) / np.sqrt(np.pi**2 + np.log(OS) ** 2)  # damping ratio
wn = 4 / (zeta * Ts)  # real part of poles

# Desired closed-loop pole locations
s1 = -zeta * wn + 1j * wn * np.sqrt(1 - zeta**2)
s2 = -zeta * wn - 1j * wn * np.sqrt(1 - zeta**2)

# Angle condition to find controller zero
angles = np.angle([1, 2, 12, 10], deg=True)
z = np.roots([1, -np.sum(angles) / 180 * np.pi])

print(z)

# Magnitude condition to find controller gain
lengths_open = np.abs([s1 + 1, s1 + 2, s1 + 12, s1 + 10])
lengths_closed = np.abs([s1 + z])
Kd = 1 / (np.prod(lengths_open) / np.prod(lengths_closed))

# PD controller transfer function
Kp = Kd * z
print(f'Kp = {Kp}, Kd = {Kd}')

# Plot root locus
plt.figure()
ctl.rlocus(G, plot=True)
plt.axvline(-zeta * wn, color='r', linestyle='--')
plt.axhline(wn * np.sqrt(1 - zeta**2), color='r', linestyle='--')
plt.axhline(-wn * np.sqrt(1 - zeta**2), color='r', linestyle='--')
plt.title('Root Locus')
plt.show()

from scipy.signal import TransferFunction, step
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

OS, Ts = 0.10, 1.0
zeta = fsolve(lambda z: np.exp(-z*np.pi/np.sqrt(1-z**2)) - OS, 0.5)[0]
wn = 4 / (zeta * Ts)

# Coefficients from the standard second-order system
a1 = 2 * zeta * wn  # coefficient of s
a0 = wn**2          # constant coefficient

# Equating the coefficients to solve for Kp and Kd
# 7 + Kd = a1 and 5 + Kp = a0
Kp = a0 - 5
Kd = (a1 - 7) / Kp

# Confirm the design by plotting the step response
# First, define the transfer function of the closed-loop system with the calculated Kp and Kd
G = TransferFunction([Kd, Kp], [1, 7+Kd*Kp, 5+Kp])

# Now, generate the step response of the system
time = np.linspace(0, 5, 500)
time, response = step(G, T=time)

print(Kp, Kd, zeta, wn)
# Plot the step response
plt.figure(figsize=(10, 6))
plt.plot(time, response)
plt.title('Step Response of the Designed PD Controlled System')
plt.xlabel('Time (seconds)')
plt.ylabel('Output')
plt.grid(True)
plt.show()

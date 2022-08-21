import numpy as np

OS = 0.05
Ts = 3

zeta = -np.log(OS) / np.sqrt(np.pi**2 + np.log(OS) ** 2)
wn = 4 / (zeta * Ts)
print(zeta, wn)

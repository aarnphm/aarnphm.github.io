import numpy as np
import matplotlib.pyplot as plt


# Define the transfer function
def G(s):
  return 300 * (s + 100) / ((s + 1) * (s + 10) * (s + 40))


# Generate frequency vector (in rad/s)
w = np.logspace(-2, 3, 1000)

# Compute magnitude and phase
mag = 300 * np.abs(1j * w + 100) / (np.abs(1j * w + 1) * np.abs(1j * w + 10) * np.abs(1j * w + 40))
phase = np.angle(G(1j * w), deg=True)

# Asymptotic magnitude approximation
asymp_mag = np.zeros_like(w)
asymp_mag[w < 1] = 300 * 100 / (1 * 10 * 40)  # DC gain
asymp_mag[(w >= 1) & (w < 10)] = 300 * 100 / (w[np.where((w >= 1) & (w < 10))] * 10 * 40)  # -20 dB/dec slope
asymp_mag[(w >= 10) & (w < 40)] = 300 * 100 / (w[np.where((w >= 10) & (w < 40))] ** 2 * 40)  # -40 dB/dec slope
asymp_mag[w >= 40] = 300 * 100 / (w[w >= 40] ** 3)  # -60 dB/dec slope

# Asymptotic phase approximation
asymp_phase = np.zeros_like(w)
asymp_phase[w < 0.1] = 0  # 0 deg
asymp_phase[(w >= 0.1) & (w < 1)] = -45  # -45 deg
asymp_phase[(w >= 1) & (w < 10)] = -90  # -90 deg
asymp_phase[(w >= 10) & (w < 40)] = -180  # -180 deg
asymp_phase[w >= 40] = -270  # -270 deg

# Plot Bode diagram
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.loglog(w, mag)
ax1.loglog(w, asymp_mag, '--')
ax1.set_ylabel('Magnitude (dB)')
ax1.set_title('Asymptotic Bode Plot')
ax1.grid(which='both')

ax2.semilogx(w, phase)
ax2.semilogx(w, asymp_phase, '--')
ax2.set_xlabel('Frequency (rad/s)')
ax2.set_ylabel('Phase (deg)')
ax2.grid(which='both')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
mu = 0.012277471
mu_hat = 1 - mu


# Differential equations
def equations(t, u):
  u1, u1_prime, u2, u2_prime = u
  delta1 = ((u1 + mu) ** 2 + u2**2) ** 1.5
  delta2 = ((u1 - mu_hat) ** 2 + u2**2) ** 1.5

  du1dt = u1_prime
  du1_primedt = u1 + 2 * u2_prime - mu_hat * (u1 + mu) / delta1 - mu * (u1 - mu_hat) / delta2
  du2dt = u2_prime
  du2_primedt = u2 - 2 * u1_prime - mu_hat * u2 / delta1 - mu * u2 / delta2

  return [du1dt, du1_primedt, du2dt, du2_primedt]


# Initial conditions
u0 = [0.994, 0, 0, -2.001585106379082522420537862224]
# Time span
t_span = (0, 17.1)


def solve(step=100):
  t_eval = np.linspace(t_span[0], t_span[1], step)
  # Solve using Runge-Kutta method of order 4
  sol = solve_ivp(equations, t_span, u0, method='RK45', t_eval=t_eval)
  # Plotting the orbit
  plt.figure(figsize=(10, 6))
  plt.plot(sol.y[0], sol.y[2], label=f'Orbit with {step} steps')
  plt.xlabel('u1')
  plt.ylabel('u2')
  plt.title(f'Orbit of the Third Body ({step} Steps)')
  plt.legend()
  plt.grid(True)
  plt.show()


if __name__ == '__main__':
  for it in [100, 1000, 10000, 20000]:
    solve(it)

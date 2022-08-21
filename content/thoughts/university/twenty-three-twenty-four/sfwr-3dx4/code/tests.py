import numpy as np
import control
import sympy as sp
from scipy import signal
from scipy.signal import lti, step
import matplotlib.pyplot as plt
from sympy import symbols, solve, S

from math import exp, pi, sqrt

K_p = symbols('K_p')
K_d = symbols('K_d')
K_i = symbols('K_i')


def routh_table(coefficients):
  # Start the Routh-Hurwitz table using the coefficients of the polynomial
  routh = [coefficients[0:2], coefficients[1:3]]  # Initialize the first two rows
  rows = len(coefficients) - 1

  for i in range(2, rows):
    row = []
    for j in range(len(routh[0]) - 1):
      # Calculate the Routh-Hurwitz table elements using the formula
      numerator = routh[i - 1][0] * routh[i - 2][j + 1] - routh[i - 2][0] * routh[i - 1][j + 1]
      denominator = routh[i - 1][0]
      # Check if the denominator is zero to avoid division by zero
      if denominator == 0:
        row.append(0)
      else:
        row.append(numerator / denominator)
    # Fill the rest of the row if necessary
    if len(row) == 1:
      row.append(0)
    routh.append(row)
    # If the entire row is zero, use the special rule (add epsilon if necessary)
    if all([sp.simplify(term) == 0 for term in row]):
      row = [(sp.symbols('s') ** (rows - i - 1 - k)) for k in range(len(row) + 1)]
      routh.append(row)

  return routh


def q1():
  K = symbols('K', real=True)
  conditions_p0 = [
    5 > 0,  # a_3
    25 + 5 * K > 0,  # a_2
    6 * K > 0,  # a_1
    5 * K > 0,  # a_0
  ]
  conditions_p2 = [
    5 > 0,  # a_3
    35 + 5 * K > 0,  # a_2
    50 + 16 * K > 0,  # a_1
    5 * K > 0,  # a_0
  ]
  stability_range_p0 = solve(conditions_p0, K)
  stability_range_p2 = solve(conditions_p2, K)
  print(stability_range_p0, stability_range_p2)

  # K, P, s = sp.symbols('K P s')
  # coefficients = [5, 25 + 5 * K + 5 * P, 25 * P + 6 * K + 5 * K * P, 5 * K]
  # routh_table_result = routh_table(coefficients)
  # for row in routh_table_result: print(row)


def q2():
  # Redefine symbols if necessary
  K2, K3 = symbols('K2 K3', real=True)

  # Coefficients
  a1 = 8 + 2 * K3
  a2 = 26 + 10 * K3 + 2 * K2
  a3 = 26 + 12 * K3 + 6 * K2

  # Routh-Hurwitz additional term
  b1 = (1 * a2 - a1 * a3) / a1

  # Conditions
  cond1 = a1 > 0
  cond2 = a3 > 0
  cond3 = b1 > 0

  # Solve the inequalities
  solution_cond1 = solve(cond1, K3, domain=S.Reals)
  solution_cond2 = solve(cond2, K2, domain=S.Reals)
  solution_cond3 = solve(cond3, K2, domain=S.Reals)

  print(solution_cond1, solution_cond2, solution_cond3)


def q3():
  num = [1]
  den = [1, 12, 20]
  sys = control.tf(num, den)
  A, B, C, D = control.ssdata(sys)

  poles = [-20, -30]

  # Compute the observer gain matrix L
  L = control.place(A.T, C.T, poles).T

  # Print the observer gain matrix
  print('Observer gain matrix L:', L)


def q4(): ...


def q5():
  # part a
  def steady_state_error(K):
    s = control.tf('s')
    G = K * (s + 6) / (s**3 + 15 * s**2 + (62 + K) * s + (72 + 6 * K))
    Kp = control.evalfr(G, 0)
    ess = 1 / (1 + Kp)

    return ess

  # Find the minimum value of K that ensures steady-state error is at most 5%
  K_min = 1
  while steady_state_error(K_min) > 0.05:
    K_min += 1

  print(f'The minimum value of K that ensures a steady-state error of at most 5% is: {K_min}')

  # Verify the steady-state error for the minimum K value
  ess_min = steady_state_error(K_min)
  print(f'Steady-state error for K = {K_min}: {ess_min:.4f}')

  # Recalculate the steady-state value with corrected understanding
  K = 228
  ss_value_corrected = 6 * K / (6 * K + 72)

  # Recalculate the steady-state error for a step disturbance
  ss_error_corrected = 1 - ss_value_corrected
  ss_error_corrected


def q6(): ...


def q7():
  # Continuous-time transfer function
  num_c = [1]
  den_c = [1, 10]
  sys_c = signal.TransferFunction(num_c, den_c)

  # Discretize using ZOH method
  dt = 0.01  # sampling time
  sys_d = signal.cont2discrete((num_c, den_c), dt, method='zoh')

  # Extract numerator and denominator coefficients
  num_d, den_d, _ = signal.tfdata(sys_d)
  num_d = num_d.flatten()
  den_d = den_d.flatten()

  print('Discrete-time transfer function:')
  print(f'num = {num_d}')
  print(f'den = {den_d}')

  # Difference equation coefficients
  b = num_d
  a = den_d

  print('\nDifference equation:')
  print(f'y[k] = {a[1]:.4f}*y[k-1] + {a[2]:.4f}*y[k-2] + {b[0]:.4f}*u[k] + {b[1]:.4f}*u[k-1]')

  # Simulate the discrete-time system
  t = np.arange(0, 1, dt)
  u = np.ones_like(t)  # step input

  _, y = signal.dlsim(sys_d, u, t)

  # Plot the step response
  plt.figure()
  plt.plot(t, y, label='Discrete-time')
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  plt.title('Step Response')
  plt.grid()
  plt.legend()
  plt.show()


def q8():
  # part a
  s = symbols('s')
  # Coefficients for the second order polynomial in the denominator
  zeta_omega_n = 8  # This is 2*zeta*omega_n
  omega_n_squared = 36  # This is omega_n^2

  # Solve for natural frequency (omega_n) and damping ratio (zeta)
  omega_n = omega_n_squared**0.5
  zeta = zeta_omega_n / (2 * omega_n)
  print(omega_n, zeta)

  # Calculate percent overshoot (%OS)
  percent_overshoot = 100 * exp(-pi * zeta / sqrt(1 - zeta**2))

  # Calculate settling time (Ts) with 2% criterion
  settling_time = 4 / (zeta * omega_n)

  print(percent_overshoot, settling_time)

  real_part_poles = -4  # Real part of the complex poles
  # Values of z to test
  z_values = [-4, 0, -10]
  # Time vector for simulation
  t = np.linspace(0, 10, 1000)

  # Plotting setup
  plt.figure(figsize=(10, 6))

  for z in z_values:
    # Define the transfer function
    num = [96, 96 * z]  # Coefficients of the numerator
    den = [1, 16, 116, 288]  # Coefficients of the denominator
    system = lti(num, den)

    # Time response
    t, y = step(system, T=t)

    # Plot
    plt.plot(t, y, label=f'z = {z}')

  plt.title('Step Response for Different Values of z')
  plt.xlabel('Time (seconds)')
  plt.ylabel('Response')
  plt.grid(True)
  plt.legend()
  plt.show()


# vim: set tw=119 ts=2 sw=2 ai et :

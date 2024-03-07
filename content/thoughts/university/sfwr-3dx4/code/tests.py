from __future__ import annotations

import sympy
import numpy as np
import sympy as sp
from sympy import symbols, apart, inverse_laplace_transform, simplify, expand, apart, lcm, together, solve, Poly, Matrix
from sympy.abc import s, t
from scipy.signal import TransferFunction, step as sci_step, StateSpace
from scipy.optimize import fsolve

K_p = symbols("K_p")
K_d = symbols("K_d")
K_i = symbols("K_i")

def q1():
  K, K_v = symbols("K K_v")
  G_s = (2 * K / s) * ((1/s) / (1-K_v/s))
  # Define the equation for the poles
  equation = s**2 - K_v

  # Solve the equation for s
  poles = solve(equation, s)
  print(poles)
  return poles

def q2():
  G_s = (s-6)/((s+2)*(s+5)**2)
  L_input = 1/s
  Y_s = G_s*L_input
  Y_s_apart = apart(Y_s)
  output_time_domain = inverse_laplace_transform(Y_s_apart, s, t)
  print(output_time_domain)
  return output_time_domain


def q3(): ...

def q4(): ...

def q5(): ...

def q6():
  A = np.array([[1, 1, 1], [0, 1, 1], [3, -2, 4]])
  B = np.array([[1], [0], [1]])
  C = np.array([[4, 5, 2]])
  D = np.array([[0]])

  # Calculate poles and zeros
  system = StateSpace(A, B, C, D)
  poles = system.poles
  zeros = system.zeros
  print(zeros, poles)
  return zeros, poles

# vim: set tw=119 ts=2 sw=2 ai et :

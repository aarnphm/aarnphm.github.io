from scipy.optimize import linprog

# coefficients of objective function
c = [-5, -4]

# inequality constraints
A = [[2, 1], [5, 8]]

# right-hand side of inequality constraints
b = [30, 120]

# bounds for variables
x0_bounds = (0, None)
x1_bounds = (0, None)

res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='highs')

print(res.x, -res.fun)


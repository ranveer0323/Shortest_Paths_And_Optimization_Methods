from scipy.optimize import fsolve
import numpy as np

# Define the equations
def equations(vars):
    x1, x2, x3, x4, lambd = vars
    eq1 = -4*x1 + 20 - lambd
    eq2 = 9*x2**2 - 60*x2 + 90 - lambd
    eq3 = 10*np.cos(x3) - lambd
    eq4 = -10*np.exp(-0.5*x4) - lambd
    eq5 = x1 + x2 + x3 + x4 - 20
    return [eq1, eq2, eq3, eq4, eq5]

# Initial guess
initial_guess = [1, 1, 1, 1, 1]

# Solve the equations
results = fsolve(equations, initial_guess)

# Extract the results
x1_opt, x2_opt, x3_opt, x4_opt, lambda_opt = results

# Calculate the total value
total_value = -2*x1_opt**2 + 20*x1_opt + 3*x2_opt**3 - 30*x2_opt**2 + 90*x2_opt + 10*np.sin(x3_opt) + 20*np.exp(-0.5*x4_opt)

print("Optimal Weights:")
print("Element A (x1):", x1_opt)
print("Element B (x2):", x2_opt)
print("Element C (x3):", x3_opt)
print("Element D (x4):", x4_opt)
print("Total Value of Alloy:", total_value)

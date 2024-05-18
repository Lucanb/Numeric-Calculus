import numpy as np
import cmath
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 3*x**2 + 2*x - 5

def df(x):
    return 3*x**2 - 6*x + 2

def horner(coeffs, x):
    result = coeffs[0]
    for coeff in coeffs[1:]:
        result = result * x + coeff
    return result

def calculate_bounds(coeffs):
    A = max(abs(coeff) for coeff in coeffs[1:])
    R = (abs(coeffs[0]) + A) / abs(coeffs[0])
    return R

def muller_method(coeffs, x0, x1, x2, eps, max_iter=100):
    h1 = x1 - x0
    h2 = x2 - x1
    delta1 = (horner(coeffs, x1) - horner(coeffs, x0)) / h1
    delta2 = (horner(coeffs, x2) - horner(coeffs, x1)) / h2
    d = (delta2 - delta1) / (h2 + h1)
    iteration = 0
    roots = []
    
    while iteration < max_iter:
        b = delta2 + h2 * d
        D = cmath.sqrt(b**2 - 4 * horner(coeffs, x2) * d)
        if abs(b - D) < abs(b + D):
            E = b + D
        else:
            E = b - D
        h = -2 * horner(coeffs, x2) / E
        x3 = x2 + h
        if abs(h) < eps:
            roots.append(x3)
            break
        x0, x1, x2 = x1, x2, x3
        h1, h2 = x1 - x0, x2 - x1
        delta1, delta2 = (horner(coeffs, x1) - horner(coeffs, x0)) / h1, (horner(coeffs, x2) - horner(coeffs, x1)) / h2
        d = (delta2 - delta1) / (h2 + h1)
        iteration += 1
    return roots

def newton_fourth_order(f, df, x0, tol=1e-10, max_iter=100):
    xn = x0
    for n in range(max_iter):
        fxn = f(xn)
        if abs(fxn) < tol:
            return xn, n
        dfxn = df(xn)
        if dfxn == 0:
            return None, n
        xn = xn - fxn / dfxn
    return None, max_iter

def newton_fifth_order(f, df, x0, tol=1e-10, max_iter=100):
    xn = x0
    for n in range(max_iter):
        fxn = f(xn)
        if abs(fxn) < tol:
            return xn, n
        dfxn = df(xn)
        if dfxn == 0:
            return None, n
        xn = xn - fxn / dfxn
    return None, max_iter

x_initial = 0.5
n4_root, n4_iterations = newton_fourth_order(f, df, x_initial)
n5_root, n5_iterations = newton_fifth_order(f, df, x_initial)

print(f"Newton Fourth Order: Root at {n4_root}, found in {n4_iterations} iterations")
print(f"Newton Fifth Order: Root at {n5_root}, found in {n5_iterations} iterations")

coeffs = [1, -6, 13, -12, 4]

R = calculate_bounds(coeffs)
print(f"All roots are in the interval [-{R}, {R}]")

epsilon = 0.001
x0, x1, x2 = -5, 0, 5

roots = muller_method(coeffs, x0, x1, x2, epsilon)
print("Roots found using Muller's method:", roots)

x_vals = np.linspace(-R, R, 400)
y_vals = [horner(coeffs, x) for x in x_vals]
root_vals = [horner(coeffs, root) for root in roots]

print("Polynomial values at the roots:")
for root, value in zip(roots, root_vals):
    print(f"Root: {root}, Value: {value}")

plt.plot(x_vals, y_vals, label='Polynomial')
plt.scatter(roots, root_vals, color='red', label='Muller Roots')
plt.scatter([n4_root], [f(n4_root)], color='blue', label='Newton 4th Order Root')
plt.scatter([n5_root], [f(n5_root)], color='green', label='Newton 5th Order Root')
plt.title('Roots of Polynomial')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.grid(True)
plt.show()

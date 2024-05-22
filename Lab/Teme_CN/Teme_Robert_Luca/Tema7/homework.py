import numpy as np
import cmath
import matplotlib.pyplot as plt

class PolyFunction:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def evaluate(self, x):
        result = self.coefficients[0]
        for coeff in self.coefficients[1:]:
            result = result * x + coeff
        return result

    def derivative(self, x):
        return 3*x**2 - 6*x + 2

    def find_bounds(self):
        max_coeff = max(abs(coeff) for coeff in self.coefficients[1:])
        bound = (abs(self.coefficients[0]) + max_coeff) / abs(self.coefficients[0])
        return bound

class RootAprox:
    def __init__(self, polynomial):
        self.polynomial = polynomial

    def muller_method(self, start, mid, end, tolerance, max_iter=100):
        h_start = mid - start
        h_end = end - mid
        delta_start = (self.polynomial.evaluate(mid) - self.polynomial.evaluate(start)) / h_start
        delta_end = (self.polynomial.evaluate(end) - self.polynomial.evaluate(mid)) / h_end
        divisor = (delta_end - delta_start) / (h_end + h_start)
        iterations = 0
        roots = []
        
        while iterations < max_iter:
            b = delta_end + h_end * divisor
            D = cmath.sqrt(b**2 - 4 * self.polynomial.evaluate(end) * divisor)
            E = b + D if abs(b - D) < abs(b + D) else b - D
            h = -2 * self.polynomial.evaluate(end) / E
            new_x = end + h
            if abs(h) < tolerance:
                roots.append(new_x)
                break
            start, mid, end = mid, end, new_x
            h_start, h_end = mid - start, end - mid
            delta_start = (self.polynomial.evaluate(mid) - self.polynomial.evaluate(start)) / h_start
            delta_end = (self.polynomial.evaluate(end) - self.polynomial.evaluate(mid)) / h_end
            divisor = (delta_end - delta_start) / (h_end + h_start)
            iterations += 1
        return roots

    def newton_method(self, initial_guess, tolerance=1e-10, max_iter=100):
        xn = initial_guess
        for n in range(max_iter):
            fxn = self.polynomial.evaluate(xn)
            if abs(fxn) < tolerance:
                return xn, n
            dfxn = self.polynomial.derivative(xn)
            if dfxn == 0:
                return None, n
            xn = xn - fxn / dfxn
        return None, max_iter

coefficients = [1, -6, 13, -12, 4]
polynomial = PolyFunction(coefficients)
root_finder = RootAprox(polynomial)

initial_guess = 0.5
root_4th, iterations_4th = root_finder.newton_method(initial_guess)
root_5th, iterations_5th = root_finder.newton_method(initial_guess)

if root_4th is not None:
    print(f"Newton's 4th order method found a root at {root_4th} after {iterations_4th} iterations")
else:
    print("Newton's 4th order method did not converge to a root")

if root_5th is not None:
    print(f"Newton's 5th order method found a root at {root_5th} after {iterations_5th} iterations")
else:
    print("Newton's 5th order method did not converge to a root")

bound_R = polynomial.find_bounds()
print(f"All roots lie within the interval [-{bound_R}, {bound_R}]")

tolerance = 0.001
start, mid, end = -5, 0, 5

muller_roots = root_finder.muller_method(start, mid, end, tolerance)
print("Roots found using Muller's method:", muller_roots)

x_vals = np.linspace(-bound_R, bound_R, 400)
y_vals = [polynomial.evaluate(x) for x in x_vals]
muller_y_vals = [polynomial.evaluate(root) for root in muller_roots]

print("Evaluated polynomial values at the roots:")
for root, value in zip(muller_roots, muller_y_vals):
    print(f"Root: {root}, Value: {value}")

plt.plot(x_vals, y_vals, label='Polynomial Curve')
plt.scatter([root.real for root in muller_roots], [root.real for root in muller_y_vals], color='red', label='Muller Roots')
if root_4th is not None:
    plt.scatter([root_4th], [polynomial.evaluate(root_4th)], color='blue', label='Newton 4th Order Root')
if root_5th is not None:
    plt.scatter([root_5th], [polynomial.evaluate(root_5th)], color='green', label='Newton 5th Order Root')
plt.title('Polynomial Roots Visualization')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.grid(True)
plt.show()

from numpy import array as arr, linalg as la, dot
import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, func, grad_func):
        self.func = func
        self.grad_func = grad_func

    def evaluate_function(self, x, y):
        return self.func(x, y)

    def compute_grad(self, x, y):
        grad_x = 2 * (x - 2)
        grad_y = 2 * (y - 3)
        return arr([grad_x, grad_y])

    def numerical_gradient(self, x, y, epsilon=1e-5):
        f = self.func
        grad = np.zeros(2)
        grad[0] = (f(x + epsilon, y) - f(x - epsilon, y)) / (2 * epsilon)
        grad[1] = (f(x, y + epsilon) - f(x, y - epsilon)) / (2 * epsilon)
        return grad

    def descent_mechanism(self, init_pos, rate=0.1, eps=1e-8, max_steps=10000):
        path = [init_pos]
        change = 1
        pos = init_pos
        iteration = 0
        while iteration < max_steps and change > eps:
            gradient = self.grad_func(pos[0], pos[1])
            next_position = pos - rate * gradient
            change = la.norm(next_position - pos)
            pos = next_position
            path.append(pos)
            iteration += 1
        return pos, arr(path)

    def find_optimal_step(self, pos, decrease_factor=0.8, alpha=0.3, iterations=50):
        step_size = 1
        for _ in range(iterations):
            grad = self.grad_func(pos[0], pos[1])
            new_position = pos - step_size * grad
            if self.func(*new_position) < self.func(*pos) - alpha * step_size * dot(grad, grad):
                break
            step_size *= decrease_factor
        return step_size

def target_function(x, y):
    return (x - 2)**2 + (y - 3)**2

initial = arr([0.0, 0.0])
gd = GradientDescent(target_function, GradientDescent(target_function, None).compute_grad)

result, trajectory = gd.descent_mechanism(initial, rate=0.1)
print("Computed position with constant rate:", result)

adaptive_trajectory = [initial]
current = initial
while True:
    gradient_here = gd.compute_grad(current[0], current[1])
    optimal_rate = gd.find_optimal_step(current)
    new_current = current - optimal_rate * gradient_here
    adaptive_trajectory.append(new_current)
    if la.norm(new_current - current) < 1e-8:
        break
    current = new_current

print("Computed position with adaptive rate:", current)

fig, ax = plt.subplots(figsize=(10, 6))
X, Y = np.meshgrid(np.linspace(-2, 5, 400), np.linspace(-1, 6, 400))
Z = gd.evaluate_function(X, Y)
contour_plot = ax.contour(X, Y, Z, levels=50)
ax.plot(*zip(*trajectory), marker='o', color='red', label='Constant Rate Path')
ax.plot(*zip(*adaptive_trajectory), marker='o', color='blue', linestyle='dashed', label='Adaptive Rate Path')
ax.scatter([2], [3], color='green', label='Minimum (2,3)')
ax.legend()
ax.set_title('Gradient Descent Paths')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
plt.show()

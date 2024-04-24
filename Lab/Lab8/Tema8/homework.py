from numpy import array as arr, linalg as la, dot
import numpy as np
import matplotlib.pyplot as plt

def evaluate_function(coord1, coord2):
    return (coord1 - 2)**2 + (coord2 - 3)**2

def compute_grad(coord1, coord2, dim=2):
    return arr([(2 * coord1 - 4) if i == 0 else (2 * coord2 - 6) for i in range(dim)])

def gradient_estimate(coord1, coord2, small_step=1e-5):
    func = evaluate_function
    grad_x = (func(coord1 + small_step, coord2) - func(coord1 - small_step, coord2)) / (2 * small_step)
    grad_y = (func(coord1, coord2 + small_step) - func(coord1, coord2 - small_step)) / (2 * small_step)
    return arr([grad_x, grad_y])

def descent_mechanism(grad_func, init_pos, rate=0.1, eps=1e-5, max_steps=10000):
    path = [init_pos]
    change = 1
    pos = init_pos
    iteration = 0
    while iteration < max_steps and change > eps:
        gradient = grad_func(pos[0], pos[1])
        next_position = pos - rate * gradient
        change = la.norm(next_position - pos)
        pos = next_position
        path.append(pos)
        iteration += 1
    return pos, arr(path)

def find_optimal_step(pos, grad_func, decrease_factor=0.8, alpha=0.3, iterations=50):
    step = 1
    iteration = 0
    while iteration < iterations:
        grad = grad_func(pos[0], pos[1])
        new_position = pos - step * grad
        if evaluate_function(*new_position) < evaluate_function(*pos) - alpha * step * dot(grad, grad):
            break
        step *= decrease_factor
        iteration += 1
    return step

initial = arr([0.0, 0.0])

result, trajectory = descent_mechanism(compute_grad, initial, rate=0.1)
print("Computed position with constant rate:", result)

adaptive_trajectory = [initial]
current = initial
while True:
    gradient_here = compute_grad(current[0], current[1])
    optimal_rate = find_optimal_step(current, compute_grad)
    new_current = current - optimal_rate * gradient_here
    adaptive_trajectory.append(new_current)
    if la.norm(new_current - current) < 1e-5:
        break
    current = new_current

print("Computed position with adaptive rate:", current)

plt.figure(figsize=(10, 6))
X, Y = np.meshgrid(np.linspace(-2, 5, 400), np.linspace(-1, 6, 400))
Z = evaluate_function(X, Y)
plt.contour(X, Y, Z, levels=50)
plt.plot(*zip(*trajectory), marker='o', color='red', label='Constant Rate Path')
plt.plot(*zip(*adaptive_trajectory), marker='o', color='blue', linestyle='dashed', label='Adaptive Rate Path')
plt.scatter([2], [3], color='green', label='Minimum (2,3)')
plt.legend()
plt.title('Gradient Descent Paths')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

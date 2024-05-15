import numpy as np
import matplotlib.pyplot as plt
import math

def my_inputs():
    f = [
        [1, 5, lambda x: x**4 - 12*x**3 + 30*x**2 + 12, 1],
        [2, 4, lambda x: math.exp(x), 2],
        [-2, 4, lambda x: math.sin(x), 3]
    ]
    try:
        num_points = int(input('Enter number of points: '))
        func_index = int(input('Enter index of function kind - 0 to 2 - : '))
        if num_points < 1 or func_index not in range(len(f)):
            raise ValueError("Invalid number of points or index.")
        return num_points, func_index, f
    except ValueError as error_message:
        print(f'Error: {str(error_message)}')
        exit(1)

def atkienDelata_calc(Ox, Oy):
    points_num = len(Ox)
    delta = np.array(Oy)
    for k in range(1, points_num):
        delta[k:points_num] = (delta[k:points_num] - delta[k-1:points_num-1]) / (Ox[k:points_num] - Ox[0:points_num-k])
    return delta

def Newton_polynom_calc(a_i, Ox, Valx):
    order = len(a_i) - 1
    Oy = a_i[order]
    for idx in range(1, order+1)[::-1]:
        Oy = a_i[idx-1] + (Valx - Ox[idx-1]) * Oy
    return Oy

def lsc_method(Ox, Oy, degree):
    A = np.vander(Ox, N=degree+1, increasing=True)
    A_tA = np.dot(A.T, A)
    A_ty = np.dot(A.T, Oy)
    x = np.linalg.solve(A_tA, A_ty)
    return x

def horner_method(coeffs, eval_point):
    final_val = 0
    for coeff in reversed(coeffs):
        final_val = final_val * eval_point + coeff
    return final_val

def start_interpolate(index, n, f_container):
    x_0, x_n, selected_func, poly_order = f_container[index]
    x_points = np.linspace(x_0, x_n, n)
    y_points = np.array([selected_func(val) for val in x_points])

    print('x:', x_points)
    print('y:', y_points)

    delta_atkiens = atkienDelata_calc(x_points, y_points)
    lsc_x = lsc_method(x_points, y_points, poly_order)

    precise_x = np.linspace(x_0, x_n, 300)
    O_y = np.array([selected_func(point) for point in precise_x])
    newton_array = np.array([Newton_polynom_calc(delta_atkiens, x_points, point) for point in precise_x])
    least_squares_y = np.array([horner_method(lsc_x, point) for point in precise_x])

    test_point = x_n - 0.5
    print(f"Test point (x̄): {test_point}")
    print("Newton Interpolation result and error at x̄:")
    print(test_point, Newton_polynom_calc(delta_atkiens, x_points, test_point), abs(Newton_polynom_calc(delta_atkiens, x_points, test_point) - selected_func(test_point)))
    print("Least Squares result and error at x̄:")
    print(test_point, horner_method(lsc_x, test_point), abs(horner_method(lsc_x, test_point) - selected_func(test_point)))

    plt.figure(figsize=(10, 6))
    plt.plot(precise_x, O_y, label='Absolute Function', color='black')
    plt.plot(precise_x, newton_array, '--', label='Lagrange Polynom', color='red')
    plt.plot(precise_x, least_squares_y, '-.', label='LSC Polynom', color='blue')
    plt.scatter(x_points, y_points, color='green', label='Sample Points')
    plt.title('Interpolation Comparison')
    plt.xlabel('x')
    plt.ylabel('Function Value')
    plt.legend()
    plt.grid(True)
    plt.show()

n, idx, f = my_inputs()
start_interpolate(idx, n, f)

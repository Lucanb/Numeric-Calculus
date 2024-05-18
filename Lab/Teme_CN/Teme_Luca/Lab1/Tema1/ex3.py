import math
import random
import matplotlib.pyplot as plt
import numpy as np

def TFunction(n,a):
    if n == 1:
        return a
    elif n == 2:
        return 3 * a / (3 - a**2)
    elif n == 3:
        return 15 * a - a**3/ (15 - 6 * a**2)
    elif n == 4:
        return (105 * a - 10 * a**3)/(105 - 45 * a**2 + a**4)
    elif n == 5:
        return (945 * a - 105 * a**3 + a**5) / (945 - 420 * a**2 + 15 * a**4)
    elif n == 6:
        return (10395 * a - 1260 * a**3 + 21 * a**5) / (10395 - 4725 * a**2 + 210 * a**4 - a**6)
    elif n == 7:
        return (135135 * a - 17325 * a**3 + 378 * a**5 - a**7) / (135135 - 62370 * a**2 + 3150 * a**4 - 28*a**6)
    elif n == 8:
        return (2027025 * a - 270270 * a**3 + 6930 * a**5 - 36 * a**7) / (2027025 - 945945 * a**2 + 51975 * a**4 - 630*a**6 + a**8)
    elif n == 9:
        return (34459425 * a - 4729725 * a**3 + 135135 * a**5 - 990 * a**7 + a**9) / (34459425 - 16216200 * a**2 + 945945 * a**4 - 13860*a**6 + 45*a**8)
    else:
        return 0

def sinusAprox(n,a):  # formulele 6,7
    return 2*TFunction(n,a/2)/(1+TFunction(n,a/2)**2)

def cosinusAprox(n,a):        
    return (1-TFunction(n,a/2)**2) /(1+TFunction(n,a/2)**2)  

def generateTanSubplot(ax, i, x, function, function_compare):
    y_tan_approx = function(i, x)
    ax.plot(x, function_compare(x), label='Functia exacta', color='red')
    if i == 6:
        ax.plot(x, y_tan_approx, label=f'Aproximare functie (i={i})', color='green', linestyle='--')
    else:
        if i == 7:
            ax.plot(x, y_tan_approx, label=f'Aproximare functie (i={i})', color='blue', linestyle='--')
        else:
            ax.plot(x, y_tan_approx, label=f'Aproximare functie (i={i})', color='yellow', linestyle='--')
            
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.axvline(-np.pi/2, color='gray', linestyle='--')
    ax.axvline(np.pi/2, color='gray', linestyle='--')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-10, 10)
    ax.set_title(f'Aproximare pentru i = {i}')
    ax.legend()

# graficele pentru erori la tangenta :

random_numbers = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 10000)

error_sums = {}
counts = {}
for i in range(4, 10):
    error_sum = 0
    count = 0
    for number in random_numbers:
        value = TFunction(i, number)
        exact_value = math.tan(number)
        error_sum += abs(value - exact_value)
        count += 1
    error_sums[i] = error_sum
    counts[i] = count

mean_errors = {i: error_sums[i] / counts[i] for i in range(4, 10)}
top_functions = sorted(mean_errors.items(), key=lambda x: x[1])
functions, mean_error_values = zip(*top_functions)

print(top_functions)

plt.figure(figsize=(10, 6))
plt.bar(functions, mean_error_values, color='skyblue')
plt.title('Top funcții după media erorilor')
plt.xlabel('Funcție T(i, a)')
plt.ylabel('Media erorilor')
plt.xticks(range(1, 10))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# graficele de comparatie tangenta:

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
x = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 500)
i_values = [4, 5, 6, 7, 8, 9]
subplot_counter = 0
for i in i_values:
    generateTanSubplot(axs[subplot_counter // 3, subplot_counter % 3], i, x, TFunction, np.tan)
    subplot_counter += 1

plt.tight_layout()
plt.show()

# part 2 - graficele pentru erori sinus

i_values_sin = [6, 7]
error_values_sin = []

for i in i_values_sin:
    error_sum = 0
    count = 0
    for number in random_numbers:
        value = sinusAprox(i, number)
        exact_value = np.sin(number)
        error_sum += abs(value - exact_value)
        count += 1
    mean_error = error_sum / count
    error_values_sin.append(mean_error)

plt.figure(figsize=(8, 6))
plt.bar(i_values_sin, error_values_sin, color='skyblue')
plt.title('Eroare medie pentru aproximarea sinusului')
plt.xlabel('Valoare i')
plt.ylabel('Eroare medie')
plt.xticks(i_values_sin)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# part 2 - graficele pentru erori cosinus

i_values_cos = [6, 7]
error_values_cos = []

for i in i_values_cos:
    error_sum = 0
    count = 0
    for number in random_numbers:
        value = cosinusAprox(i, number)
        exact_value = np.cos(number)
        error_sum += abs(value - exact_value)
        count += 1
    mean_error = error_sum / count
    error_values_cos.append(mean_error)

plt.figure(figsize=(8, 6))
plt.bar(i_values_cos, error_values_cos, color='skyblue')
plt.title('Eroare medie pentru aproximarea cosinusului')
plt.xlabel('Valoare i')
plt.ylabel('Eroare medie')
plt.xticks(i_values_cos)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# part 3 - graficele pentru sinus/cosinus

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Aproximarea sinusului
i_values_sin = [6, 7]
x_sin = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 500)
for i in i_values_sin:
    generateTanSubplot(axs[0], i, x_sin, sinusAprox, np.sin)

# Aproximarea cosinusului
i_values_cos = [6, 7]
x_cos = np.linspace(-np.pi/2 + 0.01, np.pi/2 - 0.01, 500)
for i in i_values_cos:
    generateTanSubplot(axs[1], i, x_cos, cosinusAprox, np.cos)

plt.tight_layout()
plt.show()
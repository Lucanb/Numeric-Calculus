import numpy as np

def jacobiDivision(A_vec, n, eps=1e-10, K_max=1000):
    k = 0
    while k < K_max:
        maximum = 0
        p, q = 0, 0
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if abs(A_vec[idx]) > maximum:
                    maximum = abs(A_vec[idx])
                    p, q = i, j
                idx += 1
        
        if maximum < eps:
            break  

        alpha = (A_vec[p*(p+1)//2 + p] - A_vec[q*(q+1)//2 + q]) / (2 * A_vec[q*(q+1)//2 + p])
        if alpha >= 0:
            t = -alpha + np.sqrt(alpha**2 + 1)
        else:
            t = -alpha - np.sqrt(alpha**2 + 1)
        
        const = 1 / np.sqrt(1 + t**2)
        s = const * t

        idx = 0
        for i in range(n):
            if i == p:
                A_vec[idx] = const * (A_vec[p*(p+1)//2 + p] + t * A_vec[q*(q+1)//2 + p])
            elif i == q:
                A_vec[idx] = const * (A_vec[q*(q+1)//2 + q] - t * A_vec[q*(q+1)//2 + p])
            else:
                A_vec[idx] = A_vec[idx] - s * (A_vec[q*(q+1)//2 + i] - A_vec[p*(p+1)//2 + i])
            idx += 1

        k += 1
    
    return A_vec

def verifEigenvaluesNorm(A_init, U, Lambda):
    approximation = np.dot(A_init, U)
    rez = np.dot(U, Lambda)
    norm = np.linalg.norm(approximation - rez)
    print("Norma matrice ||A_init * U - U * Lambda||:", norm)

def generate_symmetric_matrix(n):
    A = np.random.rand(n, n)
    A = (A + A.T) / 2  # Facem matricea să fie simetrică
    return A

# Testarea algoritmului adaptat
p = 5
n = 5
eps = 1e-10

# Test 1: Matrice simetrică aleatoare
A_init_vec = np.random.rand(p * (p + 1) // 2)
A_init = np.zeros((n, n))
idx = 0
for i in range(n):
    for j in range(i+1):
        A_init[i, j] = A_init[j, i] = A_init_vec[idx]
        idx += 1

A_result_vec = jacobiDivision(A_init_vec, n, eps=eps)
A_result = np.zeros((n, n))
idx = 0
for i in range(n):
    for j in range(i+1):
        A_result[i, j] = A_result[j, i] = A_result_vec[idx]
        idx += 1
Lambda = np.diag(np.diag(A_result))
U, _, _ = np.linalg.svd(A_result)
print("Test 1:")
print("U Matrice:")
print(U)
print("Jacobi Matrice:")
print(A_result)
verifEigenvaluesNorm(A_init, U, Lambda)

# Test 2: Matrice simetrică generată aleator
A = generate_symmetric_matrix(n)
A_final = jacobiDivision(A.flatten(), n, eps=eps)
A_result = np.zeros((n, n))
idx = 0
for i in range(n):
    for j in range(i+1):
        A_result[i, j] = A_result[j, i] = A_final[idx]
        idx += 1
Lambda = np.diag(np.diag(A_result))
U, _, _ = np.linalg.svd(A_result)
print("\nTest 2:")
print("U Matrice:")
print(U)
print("Jacobi Matrice:")
print(A_result)
verifEigenvaluesNorm(A, U, Lambda)

# Test 3: Matrice simetrică de dimensiune mai mare
n = 10
A = generate_symmetric_matrix(n)
A_final = jacobiDivision(A.flatten(), n, eps=eps)
A_result = np.zeros((n, n))
idx = 0
for i in range(n):
    for j in range(i+1):
        A_result[i, j] = A_result[j, i] = A_final[idx]
        idx += 1
Lambda = np.diag(np.diag(A_result))
U, _, _ = np.linalg.svd(A_result)
print("\nTest 3:")
print("U Matrice:")
print(U)
print("Jacobi Matrice:")
print(A_result)
verifEigenvaluesNorm(A, U, Lambda)

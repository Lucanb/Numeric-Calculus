import numpy as np

def jacobi(A, epsilon=1e-10, max_iter=1000):
    n = A.shape[0]
    U = np.eye(n) 
    k = 0
    
    while k < max_iter:

        max_elem = 0
        p, q = 0, 0
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i, j]) > max_elem:
                    max_elem = abs(A[i, j])
                    p, q = i, j
        
        if max_elem < epsilon:
            break  

        alpha = (A[p, p] - A[q, q]) / (2 * A[p, q])
        if alpha >= 0:
            t = -alpha + np.sqrt(alpha**2 + 1)
        else:
            t = -alpha - np.sqrt(alpha**2 + 1)
        
        c = 1 / np.sqrt(1 + t**2)
        s = c * t

        R = np.eye(n)
        R[p, p] = c
        R[p, q] = -s
        R[q, p] = s
        R[q, q] = c
        
        A = np.dot(R.T, np.dot(A, R))
        U = np.dot(U, R)
        
        k += 1
    
    return A, U


def calculate_eigenvalues(A_init):
    A, U = jacobi(A_init)  # Apelăm funcția Jacobi pentru a obține matricea diagonalizată și matricea de transformare
    Lambda = np.diag(np.diag(A))  # Valorile proprii sunt pe diagonala matricei A
    return Lambda

def verify_eigenvalues(A_init, U, Lambda):
    approx = np.dot(A_init, U)
    exact = np.dot(U, Lambda)
    norm = np.linalg.norm(approx - exact)
    print("Norma matrice ||A_init * U - U * Lambda||:", norm)


def cholesky_iterative(A, epsilon=1e-10, max_iter=10000):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    
    for i in range(n):
        for j in range(i + 1):
            temp_sum = np.sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - temp_sum)
            else:
                L[i][j] = (1.0 / L[j][j]) * (A[i][j] - temp_sum)
    
    L = np.tril(L)
    
    A_new = np.dot(L, L.T)
    
    k = 1
    while k < max_iter:
        for i in range(n):
            for j in range(i + 1):
                temp_sum = np.sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = np.sqrt(A_new[i][i] - temp_sum)
                else:
                    L[i][j] = (1.0 / L[j][j]) * (A_new[i][j] - temp_sum)

        A_new = np.dot(L, L.T)

        difference = np.linalg.norm(A_new - A)
        if difference < epsilon:
            break
        
        k += 1
    
    return A_new

p = 5
n = 5
epsilon = 1e-10
A_init = np.random.rand(p, n)
A_init = np.triu(A_init) + np.triu(A_init, 1).T
A_init = np.dot(A_init.T, A_init)
A, U = jacobi(A_init, epsilon=epsilon)
Lambda = calculate_eigenvalues(A_init) 
print('Jacobi : ',A,U)
verify_eigenvalues(A_init, U, Lambda)

# Exemplu de utilizare
p = 5
n = 5
epsilon = 1e-10
A = np.random.rand(p, n)
A = np.dot(A.T, A) 
A_final = cholesky_iterative(A, epsilon=epsilon)
print("A_final - Choleski:")
print(A_final)


U, S, VT = np.linalg.svd(A)
singular_values = S
rank = np.linalg.matrix_rank(A)
condition_number = np.linalg.cond(A)
pseudoinv = np.dot(VT.T, np.dot(np.diag(1/S), U.T))
pseudo_least_squares = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)

print("Valorile singulare ale matricei A:")
print(singular_values)
print("Rangul matricei A:", rank)
print("Nr de condiționare al matricei A:", condition_number)
print("Pseudoinversa Moore-Penrose a matricei A:")
print(pseudoinv)
norm_pseudo_diff = np.linalg.norm(pseudoinv - pseudo_least_squares, ord=1)
print("Norma ||AI - AJ||1:", norm_pseudo_diff)
import numpy as np

def jacobiDivision(A, eps=1e-10, K_max=1000):
    n = A.shape[0]
    U = np.eye(n) 
    k = 0
    
    while k < K_max:
        maximum = 0
        p, q = 0, 0
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i, j]) > maximum:
                    maximum = abs(A[i, j])
                    p, q = i, j
        
        if maximum < eps:
            break  

        alpha = (A[p, p] - A[q, q]) / (2 * A[p, q])
        if alpha >= 0:
            t = -alpha + np.sqrt(alpha**2 + 1)
        else:
            t = -alpha - np.sqrt(alpha**2 + 1)
        
        const = 1 / np.sqrt(1 + t**2)
        s = const * t
        R = np.eye(n)
        R[p, p] = const
        R[p, q] = -s
        R[q, p] = s
        R[q, q] = const
        A = np.dot(R.T, np.dot(A, R))
        U = np.dot(U, R)
        k += 1
    
    return A, U

def verifEigenvaluesNorm(A_init, U, Lambda):
    approximation = np.dot(A_init, U)
    rez = np.dot(U, Lambda)
    norm = np.linalg.norm(approximation - rez)
    print("Norma matrice ||A_init * U - U * Lambda||:", norm)


def choleskyMethod(A, eps=1e-10, K_max=10000):
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
    A_Mat = np.dot(L, L.T)
    
    k = 1
    while k < K_max:
        for i in range(n):
            for j in range(i + 1):
                temp_sum = np.sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = np.sqrt(A_Mat[i][i] - temp_sum)
                else:
                    L[i][j] = (1.0 / L[j][j]) * (A_Mat[i][j] - temp_sum)

        A_Mat = np.dot(L, L.T)
        norm = np.linalg.norm(A_Mat - A)
        if norm < eps:
            break 
        k += 1
    
    return A_Mat

p = 5
n = 5
eps = 1e-10

A_init = np.random.rand(p, n)
A_init = np.triu(A_init) + np.triu(A_init, 1).T
A_init = np.dot(A_init.T, A_init)
A, U = jacobiDivision(A_init, eps=eps)

Lambda = np.diag(np.diag(A)) 
print(f"Jacobi Matrice : \n {A} \n {U}")
verifEigenvaluesNorm(A_init, U, Lambda)

p = 5
n = 5
eps = 1e-10
A = np.random.rand(p, n)
A = np.dot(A.T, A) 

A_final = choleskyMethod(A, eps=eps)
print(f" Choleski Matrice : \n  {A_final}")


U, S, VT = np.linalg.svd(A)
val_singulare = S
rank = np.linalg.matrix_rank(A)
condition_number = np.linalg.cond(A)
MoorePenInv = np.dot(VT.T, np.dot(np.diag(1/S), U.T))
LSC = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)

print(f"Valorile singulare ale matricei A : \n {val_singulare}")

print(f"Rangul matricei A: \n {rank}")

print(f"Nr de conditionare :  {condition_number}")

print(f"Pseudoinversa Moore-Penrose a matricei A : \n {MoorePenInv}")

norm_pseudo_diff = np.linalg.norm(MoorePenInv - LSC, ord=1)
print(f"Norma ||AI - AJ|| : {norm_pseudo_diff}")
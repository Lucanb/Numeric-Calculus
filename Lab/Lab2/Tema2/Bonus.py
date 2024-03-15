import numpy as np

def LU_CorutAlgo(A, eps):
    n = A.shape[0]
    dim_L_U = n * (n + 1) // 2
    L_U_lower = np.zeros(dim_L_U)
    L_U_upper = np.zeros(dim_L_U)
    index = 0

    for i in range(n):
        L_U_lower[index] = 1
        index += 1

    for p in range(n):
        for i in range(p, n):
            if index >= dim_L_U:
                break
            L_U_lower[index] = A[i][p] - sum(L_U_lower[L_index] * L_U_upper[U_index] for L_index, U_index in zip(range(p), range(index - p, index)))
            if abs(L_U_lower[index]) < eps and i == p:
                print('Nu se poate calcula o astfel de descompunere')
                return None, None
            index += 1

        for j in range(p, n):
            if index >= dim_L_U:
                break
            index_j = index - (n - p) + (j - p)
            L_U_upper[index_j] = (A[p][j] - sum(L_U_lower[L_index] * L_U_upper[U_index] for L_index, U_index in zip(range(p), range(index_j - (j - p), index_j)))) / L_U_lower[index_j - (j - p)]

    L = np.zeros((n, n))
    U = np.zeros((n, n))
    index_L = 0
    index_U = 0
    
    for i in range(n):
        for j in range(n):
            if i < j:
                U[i][j] = L_U_upper[index_U]
                index_U += 1
            elif i == j:
                L[i][j] = 1
                U[i][j] = L_U_lower[index_L]
                index_L += 1
            else:
                L[i][j] = L_U_lower[index_L]
                index_L += 1

    return L, U

def CalculateDeterminat(L, U):
    n = L.shape[0]
    det_L = 1
    det_U = 1
    for p in range(n):
        det_L *= L[p][p]
        det_U *= U[p][p]
    final_det = det_U * det_L
    return final_det

def direct_substitution(L, b, eps):
    n = L.shape[0]
    x = np.zeros(n)

    for i in range(n):
        if abs(L[i][i]) < eps:
            print("Sistem fără soluție.")
            return None
        x[i] = (b[i] - np.dot(L[i][:i], x[:i])) / L[i][i]

    return x

def inverse_substitution(U, b, eps):
    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < eps:
            print("Sistem fără soluție.")
            return None
        x[i] = (b[i] - np.dot(U[i][i + 1:], x[i + 1:])) / U[i][i]

    return x

def calculate_SolutionAx(L, U, b, eps):
    y = direct_substitution(L, b, eps)
    x = inverse_substitution(U, y, eps)
    return x

n = int(input("Introduceți dimensiunea matricilor sistemului: "))
t = int(input("Introduceți precizia calculelor (puterea de 10 pentru eps): "))
eps = 10 ** (-t)

# A_init = np.random.rand(n, n)
# b_init = np.random.rand(n)
A_init = np.array([[2.5,2,2],[5,6,5],[5,6,6.5]])
b_init = np.array([2,2,2])
A = np.copy(A_init)

L, U = LU_CorutAlgo(A, eps)

if L is not None and U is not None:
    print('Matricea L:', L)
    print('Matricea U:', U)

    det_LU = CalculateDeterminat(L, U)
    print('Determinantul LU:', det_LU)

    x = calculate_SolutionAx(L, U, b_init, eps)
    if x is not None:
        print('Soluția x a sistemului Ax cu substituții este:', x)

        A_inverse = np.linalg.inv(A_init)
        x_absolute = np.dot(A_inverse, b_init)

        norm = np.linalg.norm(np.dot(A, x) - b_init, 2)
        print('Norma rezultatului:', norm)

        norm_x = np.linalg.norm(x - x_absolute, 2)
        norm_x_A = np.linalg.norm(x - np.dot(A_inverse, b_init), 2)
        print(f'Norma x-x_lib: {norm_x}')
        print(f'Norma x-A^-1b: {norm_x_A}')
else:
    print('Nu se poate calcula deoarece sistemul nu are soluție.')

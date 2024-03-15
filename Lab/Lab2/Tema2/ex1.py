import numpy as np

def LU_CorutAlgo(A, eps):
    n = A.shape[0]

    for p in range(n):
        for i in range(p, n):
            A[i][p] = A[i][p] - sum(A[i][k] * A[k][p] for k in range(p))
            if abs(A[i][p]) < eps and i == p:
                print('Nu se poate calcula o astfel de descompunere')
                return None
        for i in range(p + 1, n):
            A[p][i] = (A[p][i] - sum(A[p][k] * A[k][i] for k in range(p))) / A[p][p]
    return A

def CalculateDeterminat(A):
    n = A.shape[0]
    det_A = 1
    for p in range(n):
        det_A *= A[p][p]
    return det_A

def direct_substitution(A, b, eps):
    n = A.shape[0]
    x = np.zeros(n)

    for i in range(n):
        if abs(A[i][i]) < eps:
            print("sistem fara solutie")
            return None
        x[i] = (b[i] - np.dot(A[i][:i], x[:i])) / A[i][i]

    return x

def inverse_substitution(A, b, eps):
    n = A.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < eps:
            print("sistem fara solutie")
            return None
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) #/ A[i][i]

    return x

def calculate_SolutionAx(A, b, eps):
    x = direct_substitution(A, b, eps)
    if x is not None:
        x = inverse_substitution(A, x, eps)
    return x

n = int(input("dimensiunea matricilor sistem : "))
t = int(input("precizia calculelor : "))
eps = 10 ** (-t)

A_init = np.random.rand(n, n)
b_init = np.random.rand(n)
# A_init = np.array([[2.5,2,2],[5,6,5],[5,6,6.5]])
# b_init = np.array([2,2,2])
A = np.copy(A_init)

A = LU_CorutAlgo(A, eps)

if A is not None:
    print('Matricea A cu descompunerea LU este :', A)

    det_A = CalculateDeterminat(A)
    print('determinantul matricei A este : ', det_A)

    x = calculate_SolutionAx(A, b_init, eps)
    if x is not None:
        print('solutia x a sistemului Ax cu substitutii este : ', x)

        A_inverse = np.linalg.inv(A_init)
        x_absolute = np.dot(A_inverse, b_init)
        print('solutia absolta este : ', x_absolute)
        print(A_init)
        norm = np.linalg.norm(np.dot(A_init, x) - b_init, 2)
        print('Norma rezultatului este : ', norm)

        norm_x = np.linalg.norm(x - x_absolute, 2)
        norm_x_A = np.linalg.norm(x - np.dot(A_inverse, b_init), 2)
        print(f'Norma x-x_lib : {norm_x}')
        print(f'Norma x-A^-1b : {norm_x_A}')
else:
    print('Nu se poate calcula deoarece sistemul nu are solutie : ')

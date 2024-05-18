import numpy as np

class LU_Decomposition_Solver:
    def __init__(self, n, t):
        self.n = n
        self.eps = 10 ** (-t)
        self.A_init = np.random.rand(n, n)
        self.b_init = np.random.rand(n)
        self.Lv = np.zeros(n * (n + 1) // 2, dtype=np.float64)
        self.Uv = np.zeros(n * (n + 1) // 2, dtype=np.float64)

    def indexL(self, i, j):
        """ Indexing function for accessing elements in the lower triangular vector """
        return i * (i + 1) // 2 + j

    def indexU(self, i, j):
        """ Indexing function for accessing elements in the upper triangular vector """
        return j * (j + 1) // 2 + i

    def LU_decompose(self):
        for p in range(self.n):
            for i in range(p, self.n):
                sum_LU = sum(self.Lv[self.indexL(i, k)] * self.Uv[self.indexU(k, p)] for k in range(p))
                self.Lv[self.indexL(i, p)] = self.A_init[i, p] - sum_LU
            for i in range(p + 1, self.n):
                if abs(self.Lv[self.indexL(p, p)]) < self.eps:
                    print("Matrix is singular!")
                    return False
                sum_LU = sum(self.Lv[self.indexL(p, k)] * self.Uv[self.indexU(k, i)] for k in range(p))
                self.Uv[self.indexU(p, i)] = (self.A_init[p, i] - sum_LU) / self.Lv[self.indexL(p, p)]
            for i in range(self.n):
                self.Uv[self.indexU(i, i)] = 1
        return True

    def solve(self):
        if not self.LU_decompose():
            return None
        y = self.forward_substitution()
        if y is None:
            return None
        x = self.backward_substitution(y)
        return x

    def forward_substitution(self):
        y = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            sum_Ly = sum(self.Lv[self.indexL(i, j)] * y[j] for j in range(i))
            y[i] = (self.b_init[i] - sum_Ly) / self.Lv[self.indexL(i, i)]
        return y

    def backward_substitution(self, y):
        x = np.zeros(self.n, dtype=np.float64)
        for i in reversed(range(self.n)):
            sum_Ux = sum(self.Uv[self.indexU(i, j)] * x[j] for j in range(i + 1, self.n))
            x[i] = y[i] - sum_Ux
        return x

    def print_results(self, x):
        print('Matrix A:', self.A_init)
        print('Vector b:', self.b_init)
        print('LU decomposition (stored in vectors):')
        print('Lv:', self.Lv)
        print('Uv:', self.Uv)
        print('Solution vector x:', x)
        norm = np.linalg.norm(np.dot(self.A_init, x) - self.b_init)
        print('Norm ||Ax - b||:', norm)
        if norm < 10 ** -9:
            print('Solution is correct!')
        else:
            print('Solution is incorrect!')

        A_inv = np.linalg.inv(self.A_init)
        x_exact = np.dot(A_inv, self.b_init)
        norm_diff = np.linalg.norm(x - x_exact)
        print('Norm ||x_LU - x_exact||:', norm_diff)

    def compute_solution(self):
        x = self.solve()
        if x is not None:
            self.print_results(x)
        else:
            print('Failed to find a valid solution.')

n = int(input("Enter the dimension of the matrix: "))
t = int(input("Enter the precision level for calculations: "))
solver = LU_Decomposition_Solver(n, t)
solver.compute_solution()

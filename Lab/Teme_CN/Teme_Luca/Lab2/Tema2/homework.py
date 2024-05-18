import numpy as np

class LU_Decomposition_Solver:
    def __init__(self, n, t):
        self.n = n
        self.eps = 10 ** (-t)
        self.A_init = np.random.rand(n, n)
        self.b_init = np.random.rand(n)
        self.A = np.copy(self.A_init)

    def LU_descomp(self):
        try:
            for i in range(self.n):
                self.Lower_triangular(i)
                self.Upper_triangular(i)
            return self.A
        except Exception as e:
            print(e)
            return None

    def Lower_triangular(self, index):
        for i in range(index, self.n):
            self.A[i][index] -= sum(self.A[i][k] * self.A[k][index] for k in range(index))
            if abs(self.A[i][index]) < self.eps and i == index:
                raise ValueError('Matrix decomposition failed.')

    def Upper_triangular(self, index):
        for i in range(index + 1, self.n):
            if abs(self.A[index][index]) < self.eps:
                raise ZeroDivisionError('Division by zero in matrix decomposition.')
            self.A[index][i] = (self.A[index][i] - sum(self.A[index][k] * self.A[k][i] for k in range(index))) / self.A[index][index]

    def get_det_value(self):
        if self.A is None:
            return None
        return np.prod(np.diag(self.A))

    def system_solver(self):
        y = self.forward_method_substitution()
        if y is None:
            return None
        return self.backward_method_substitution(y)

    def forward_method_substitution(self):
        y = np.zeros(self.n)
        for i in range(self.n):
            if abs(self.A[i][i]) < self.eps:
                print("No solution exists.")
                return None
            y[i] = (self.b_init[i] - np.dot(self.A[i][:i], y[:i])) / self.A[i][i]
        return y

    def backward_method_substitution(self, y):
        x = np.zeros(self.n)
        for i in reversed(range(self.n)):
            if abs(self.A[i][i]) < self.eps:
                print("No solution exists.")
                return None
            x[i] = (y[i] - np.dot(self.A[i][i + 1:], x[i + 1:]))
        return x

    def print_results(self, x):
        print('Matrix A after LU decomposition:', self.A)
        print('Determinant of the matrix:', self.get_det_value())
        print('Solution vector x:', x)
        A_inv = np.linalg.inv(self.A_init)
        x_sol = np.dot(A_inv, self.b_init)
        print('Exact solution x from A⁻¹b:', x_sol)
        print('||Ax - b|| : ', np.linalg.norm(np.dot(self.A_init, x) - self.b_init))
        print('||x - x_sol|| :', np.linalg.norm(x - x_sol))

    def compute_solution(self):
        self.A = self.LU_descomp()
        if self.A is not None:
            x = self.system_solver()
            if x is not None:
                self.print_results(x)
            else:
                print('Failed to find a valid solution.')
        else:
            print('LU decomposition failed.')

n = int(input("Enter the dimension of the matrix: "))
t = int(input("Enter the precision level for calculations: "))
solver = LU_Decomposition_Solver(n, t)
solver.compute_solution()

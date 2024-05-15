import numpy as np

class JacobiClass:
    def __init__(self, n, eps):
        self.n = n
        self.eps = eps
        self.A_init = np.random.rand(n, n)
        self.A_init = (self.A_init + self.A_init.T) / 2
        self.v = np.zeros(n * (n + 1) // 2, dtype=np.float64)
        self.U = np.eye(n)
        self.add_Elements_vec()

    def add_Elements_vec(self):
        idx = 0
        for i in range(self.n):
            for j in range(i + 1):
                self.v[idx] = self.A_init[i, j]
                idx += 1

    def jacobiDivision(self):
        A = self.A_init.copy()
        k = 0
        while True:
            p, q = np.unravel_index(np.argmax(np.abs(np.tril(A, -1))), A.shape)
            if abs(A[p, q]) < self.eps:
                break
            alpha = (A[q, q] - A[p, p]) / (2 * A[p, q])
            t = np.sign(alpha) / (abs(alpha) + np.sqrt(alpha ** 2 + 1))
            c = 1 / np.sqrt(t ** 2 + 1)
            s = t * c

            for i in range(self.n):
                if i != p and i != q:
                    A_ip = A[i, p]
                    A_iq = A[i, q]
                    A[i, p] = A_ip * c - A_iq * s
                    A[p, i] = A[i, p]
                    A[i, q] = A_iq * c + A_ip * s
                    A[q, i] = A[i, q]

            A_pp = A[p, p]
            A_qq = A[q, q]
            A[p, p] = c ** 2 * A_pp + s ** 2 * A_qq - 2 * s * c * A[p, q]
            A[q, q] = s ** 2 * A_pp + c ** 2 * A_qq + 2 * s * c * A[p, q]
            A[p, q] = 0
            A[q, p] = 0

            for i in range(self.n):
                U_ip = self.U[i, p]
                U_iq = self.U[i, q]
                self.U[i, p] = U_ip * c - U_iq * s
                self.U[i, q] = U_iq * c + U_ip * s

            k += 1
            if k > 1000:
                break

        self.Lambda = np.diag(A)
        return self.Lambda, self.U

    def verify_eigenvalues(self):
        A_U = np.dot(self.A_init, self.U)
        U_Lambda = np.dot(self.U, np.diag(self.Lambda))
        norm = np.linalg.norm(A_U - U_Lambda)
        print("Norma matrice ||A_init * U - U * Lambda||:", norm)

if __name__ == "__main__":
    n = int(input("Enter the matrix dim : "))
    t = int(input("Enter the precision level for calculations: "))
    eps = 10 ** -t
    solver = JacobiClass(n, eps)
    Lambda, U = solver.jacobiDivision()
    print(f"Eigenvalues (Lambda): \n{Lambda}")
    print(f"Eigenvectors (U): \n{U}")
    solver.verify_eigenvalues()

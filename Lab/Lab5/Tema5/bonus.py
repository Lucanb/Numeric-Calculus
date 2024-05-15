import numpy as np

class JacobiClass:
    def __init__(self, n, eps):
        self.n = n
        self.eps = eps
        # self.A_init = [1,2,3,4,2,3,4,5,3,4,5,6,4,5,6,7]
        # self.A_init = (self.A_init + self.A_init.T) / 2
        self.v = np.zeros(n * (n + 1) // 2, dtype=np.float64)
        self.U = np.eye(n)
        self.add_Elements_vec()

    def add_Elements_vec(self):
        # idx = 0
        # for i in range(self.n):
        #     for j in range(i + 1):
        #         self.v[idx] = self.A_init[i, j]
        #         idx += 1
        self.v = [1,2,3,3,4,5,4,5,6,7]

    def idx_to_v(self, i, j):
        if i < j:
            i, j = j, i
        return (i * (i + 1)) // 2 + j

    def jacobiDivision(self):
        v = self.v.copy()
        k = 0
        while True:
            max_val = 0
            p, q = -1, -1
            for i in range(self.n):
                for j in range(i):
                    idx = self.idx_to_v(i, j)
                    if abs(v[idx]) > max_val:
                        max_val = abs(v[idx])
                        p, q = i, j
            
            if max_val < self.eps:
                print(v)
                break

            idx_pq = self.idx_to_v(p, q)
            idx_pp = self.idx_to_v(p, p)
            idx_qq = self.idx_to_v(q, q)
            alpha = (v[idx_pp] - v[idx_qq]) / (2 * v[idx_pq])
            t = -alpha + np.sign(alpha) * np.sqrt(alpha**2 + 1)
            if(abs(alpha)<eps):
                t = 1
            c = 1 / np.sqrt(t**2 + 1)
            s = t * c

            new_pp = c**2 * v[idx_pp] + s**2 * v[idx_qq] + 2 * s * c * v[idx_pq]
            new_qq = s**2 * v[idx_pp] + c**2 * v[idx_qq] - 2 * s * c * v[idx_pq]
            v[idx_pp] = new_pp
            v[idx_qq] = new_qq

            v[idx_pq] = 0

            for i in range(self.n):
                if i != p and i != q:
                    idx_ip = self.idx_to_v(i, p)
                    idx_iq = self.idx_to_v(i, q)
                    old_ip = v[idx_ip]
                    old_iq = v[idx_iq]
                    v[idx_ip] = c * old_ip + s * old_iq
                    v[idx_iq] = -s * old_ip + c * old_iq

            for i in range(self.n):
                U_ip = self.U[i, p]
                U_iq = self.U[i, q]
                self.U[i, p] = U_ip * c - U_iq * s
                self.U[i, q] = U_iq * c + U_ip * s

            k += 1
            print(k)
            if k > 1000:
                print('a')
                break

        self.Lambda = np.array([v[self.idx_to_v(i, i)] for i in range(self.n)])
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

import numpy as np

class MatrixDecompositions:
    def __init__(self, epsilon=1e-10, max_iterations=1000):
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def jacobi_rotation(self, mat):
        n = mat.shape[0]
        ortho_mat = np.eye(n)
        iter_count = 0
        
        while iter_count < self.max_iterations:
            max_val = 0
            p, q = 0, 0
            for i in range(n):
                for j in range(i+1, n):
                    if abs(mat[i, j]) > max_val:
                        max_val = abs(mat[i, j])
                        p, q = i, j
            
            if max_val < self.epsilon:
                break  

            alpha = (mat[p, p] - mat[q, q]) / (2 * mat[p, q])
            t = (-alpha + np.sqrt(alpha**2 + 1)) if alpha >= 0 else (-alpha - np.sqrt(alpha**2 + 1))
            
            cos_theta = 1 / np.sqrt(1 + t**2)
            sin_theta = cos_theta * t
            rot_matrix = np.eye(n)
            rot_matrix[p, p] = cos_theta
            rot_matrix[p, q] = -sin_theta
            rot_matrix[q, p] = sin_theta
            rot_matrix[q, q] = cos_theta
            mat = np.dot(rot_matrix.T, np.dot(mat, rot_matrix))
            ortho_mat = np.dot(ortho_mat, rot_matrix)
            iter_count += 1
        
        return mat, ortho_mat

    def check_eigen_norm(self, init_mat, ortho_mat, eigen_vals):
        approx_mat = np.dot(init_mat, ortho_mat)
        result_mat = np.dot(ortho_mat, eigen_vals)
        norm_val = np.linalg.norm(approx_mat - result_mat)
        print("Norma matricei ||A_init * U - U * Lambda||:", norm_val)

    def cholesky_decomp(self, mat):
        n = mat.shape[0]
        lower_tri = np.zeros_like(mat, dtype=float)
        
        for i in range(n):
            for j in range(i + 1):
                sum_temp = np.sum(lower_tri[i][k] * lower_tri[j][k] for k in range(j))
                if i == j:
                    lower_tri[i][j] = np.sqrt(mat[i][i] - sum_temp)
                else:
                    lower_tri[i][j] = (1.0 / lower_tri[j][j]) * (mat[i][j] - sum_temp)
        
        lower_tri = np.tril(lower_tri)
        approx_chol = np.dot(lower_tri, lower_tri.T)
        
        iter_count = 1
        while iter_count < self.max_iterations:
            for i in range(n):
                for j in range(i + 1):
                    sum_temp = np.sum(lower_tri[i][k] * lower_tri[j][k] for k in range(j))
                    if i == j:
                        lower_tri[i][j] = np.sqrt(approx_chol[i][i] - sum_temp)
                    else:
                        lower_tri[i][j] = (1.0 / lower_tri[j][j]) * (approx_chol[i][j] - sum_temp)

            approx_chol = np.dot(lower_tri, lower_tri.T)
            norm_val = np.linalg.norm(approx_chol - mat)
            if norm_val < self.epsilon:
                break 
            iter_count += 1
        
        return approx_chol

# Setări inițiale și generarea matricilor
epsilon = 1e-10
max_iterations = 1000
matrix_decomp = MatrixDecompositions(epsilon, max_iterations)

p = 5
n = 5

init_mat = np.random.rand(p, n)
init_mat = np.triu(init_mat) + np.triu(init_mat, 1).T
init_mat = np.dot(init_mat.T, init_mat)
diag_mat, ortho_mat = matrix_decomp.jacobi_rotation(init_mat)

eigen_vals = np.diag(np.diag(diag_mat)) 
print(f"Matrice Jacobi : \n {diag_mat} \n {ortho_mat}")
matrix_decomp.check_eigen_norm(init_mat, ortho_mat, eigen_vals)

mat = np.random.rand(p, n)
mat = np.dot(mat.T, mat)

final_chol = matrix_decomp.cholesky_decomp(mat)
print(f"Matrixe Cholesky : \n  {final_chol}")

U, S, VT = np.linalg.svd(mat)
sing_vals = S
mat_rank = np.linalg.matrix_rank(mat)
cond_num = np.linalg.cond(mat)
pseudo_inv = np.dot(VT.T, np.dot(np.diag(1/S), U.T))
least_sq_sol = np.dot(np.linalg.inv(np.dot(mat.T, mat)), mat.T)

print(f"Valorile singulare : \n {sing_vals}")

print(f"Rangul matricei : \n {mat_rank}")

print(f"Nr conditionare :  {cond_num}")

print(f"Pseudo Inversa Moore Penrose: \n {pseudo_inv}")

norm_diff = np.linalg.norm(pseudo_inv - least_sq_sol, ord=1)
print(f"Norma ||AI - AJ|| : {norm_diff}")

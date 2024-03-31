import numpy as np
import urllib.request
import time
from copy import deepcopy

class SparseMatrix:
    def __init__(self,n):
        self.n = n
        self.mat = [{} for _ in range(n)]
    
    # def __add__(self,other): 

    def addMyElement(self, i, j, x):
        if x != 0: 
            self.mat[i][j] = self.mat[i].setdefault(j, 0) + x

    def has_null_diagonal(self):
        for i in range(self.n):
            if i in self.mat[i] and self.mat[i][i] == 0:
                return True
        return False

    def printNonZeroElements(self):
        for i, row in enumerate(self.mat):
            for j, val in row.items():
                print(f"Element nenul la poziția ({i}, {j}): {val}")


    def gauss_seidel(self, b, tol=1e-6, max_iter=1000):
        x = np.zeros(self.n)

        for _ in range(max_iter):
            x_new = x.copy()

            for i in range(self.n):
                sigma = 0
                for j in range(self.n):
                    if (i, j) != i and (i, j) in self.mat:
                        sigma += self.mat[(i, j)] * x_new[j]

                x[i] = (b[i] - sigma) / self.mat.get((i, i), 0)

            if all(abs(x[i] - x_new[i]) < tol for i in range(self.n)):
                break

        return x, _

    def calculate_residual_norm(self, x, b):
        Ax = [sum(self.mat[i][j] * x[j] for j in range(self.n)) for i in range(self.n)]

        residual = np.abs(np.array(Ax) - np.array(b))
        norm_inf = np.linalg.norm(residual, np.inf)
            
        return norm_inf                      

def loadData(index):
    try:
        with open(f"data/a_{index}.txt",'r') as file:
            n1 = int(file.readline().strip())
            file_a = file.read().strip().split('\n', 1)[1]
            a = SparseMatrix(n1)
            file.seek(0)
            for line in file:
                if line:
                    val = line.split(',')
                    if len(val) != 3:
                        continue
                    x = float(val[0].strip())
                    i = int(val[1].strip())
                    j = int(val[2].strip())
                    a.addMyElement(i, j, x)
                else:
                    continue

        with open(f"data/b_{index}.txt",'r') as file:
            n2 = int(file.readline().strip())
            b = []
            
            for _ in range(n2):
                line = file.readline().strip()
                if line:
                    b_val = float(line)
                    b.append(b_val)
                else:
                    continue
        
        return n1,n2,a,b
  
    except Exception as e:
        print(f"Eroare la incarcarea fisierului {index}:", e)
        return None, None, None, None

n1,n2,a,b = loadData(2)
if a.has_null_diagonal():
    print("Matricea are un element diagonal nul. Sistemul nu poate fi rezolvat folosind metoda iterativă Gauss-Seidel.")
else:
    print('e oke')
    x_approx, num_iter = a.gauss_seidel(b)
    print("Soluția aproximată:", x_approx)
    print("Numărul de iterații:", num_iter)
        
    residual_norm = a.calculate_residual_norm(x_approx, b)
    print("Norma reziduului:", residual_norm)
